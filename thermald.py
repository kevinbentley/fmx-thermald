#!/usr/bin/env python3
"""Thermal camera daemon for ICI FMX 400P (InfiRay OEM, Dahua-derived RTSP stack).

Holds the single allowed RTSP session to 192.168.1.123, parses the custom HYAV
chunks to recover raw 16-bit radiometric frames, and serves MJPEG preview +
snapshot endpoints over HTTP.

Endpoints:
  GET  /                   dashboard page with embedded preview
  GET  /preview.mjpg       live MJPEG (multipart/x-mixed-replace)
  GET  /snapshot.tiff      most recent raw 16-bit frame as TIFF
  POST /snapshot           write TIFF + PNG + JSON sidecar to disk
  POST /focus?cmd=step|auto  manual focus step or one-shot autofocus (port 36399)
  GET  /pixel?x=&y=        temperature at pixel (°C/°F/K)
  GET  /status             daemon + last-frame stats + calibration info as JSON

Raw 16-bit values are always preserved in the TIFF. Temperatures are derived
via the loaded calibration (polynomial fit from thermal-calibrate). If no
calibration file is found the daemon falls back to an approximate formula
(T_C = raw16/90 - 29) chosen to map the FMX's 16-bit raw range onto its
published operating range. Accuracy in this mode is roughly ±2 °C; use only
as a ballpark until a real calibration is available.
"""
import argparse, io, json, os, queue, signal, socket, struct, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image

try:
    import av          # PyAV — H.264 decoding for the camera's cooked pseudocolor feed
    HAVE_AV = True
except ImportError:
    HAVE_AV = False

# ---- Camera protocol constants --------------------------------------------
W, H = 384, 288
PIX = W * H
PLANE = PIX                    # 110 592 bytes per plane
FRAME_DATA = 2 * PLANE         # 221 184 bytes of pixels per frame
DEFAULT_HOST = '192.168.1.123'
DEFAULT_PATH = '/cam/realmonitor?channel=1&subtype=0'
DEFAULT_CALIB_PATH = '/home/kbentley/thermal/calib.json'

# ---- Focus/lens control (port 36399) --------------------------------------
# Framing:  7D FF 00 00 | AA | LEN | <body> | CHK | EB AA
#   LEN = len(body) + 1  (covers body + CHK)
#   CHK = (0xAA + LEN + sum(body)) & 0xFF
# Camera ACKs with the fixed 4-byte frame 7D FF 00 7B.
FOCUS_PORT = 36399
FOCUS_COMMANDS = {
    'step': bytes.fromhex('0821010201'),   # manual focus step, one motor increment
    'auto': bytes.fromhex('082f0100'),     # one-shot autofocus
}

def _focus_frame(body):
    ln = len(body) + 1
    chk = (0xAA + ln + sum(body)) & 0xFF
    return b'\x7d\xff\x00\x00\xaa' + bytes([ln]) + body + bytes([chk]) + b'\xeb\xaa'

def send_focus(host, body, timeout=2.0):
    """Open a short-lived TCP connection to port 36399, send one framed
    command, and return the camera's reply bytes."""
    frame = _focus_frame(body)
    with socket.create_connection((host, FOCUS_PORT), timeout=timeout) as s:
        s.sendall(frame)
        resp = b''
        while len(resp) < 4:
            chunk = s.recv(4 - len(resp))
            if not chunk:
                break
            resp += chunk
    return resp

# ---- Calibration ----------------------------------------------------------
# Loaded at startup. When present, raw_to_C evaluates the polynomial from
# calib.json; otherwise falls back to the FMX-shape approximation below.
CALIB = None  # dict or None

# Fallback formula for when no calib.json is present. Derived from observing
# that this camera emits ~90 raw counts per Kelvin with uint16 zero anchored
# near T_K ≈ 244 K (≈ -29 °C). That mapping puts the 16-bit range across the
# FMX 400P's published operating span (~ -29 °C to +700 °C). Unit-to-unit
# variation is typically ±2 °C so this is a ballpark only.
FALLBACK_SLOPE  = 1.0 / 90.0   # K per raw count
FALLBACK_OFFSET = -29.0        # °C (value at raw = 0)
FALLBACK_DESC   = (f'T_C = raw16/{1/FALLBACK_SLOPE:.0f} + ({FALLBACK_OFFSET:+.0f})   '
                   f'(UNCALIBRATED BALLPARK for FMX 400P, ±2 °C)')

def load_calib(path):
    global CALIB
    if not path:
        sys.stderr.write(f'[calib] disabled; using fallback: {FALLBACK_DESC}\n')
        return
    try:
        with open(path) as f:
            c = json.load(f)
    except FileNotFoundError:
        sys.stderr.write(f'[calib] {path} not found; using fallback: {FALLBACK_DESC}\n')
        return
    except Exception as e:
        sys.stderr.write(f'[calib] failed to load {path}: {e}; using fallback\n')
        return
    if c.get('model') != 'polynomial' or 'coef_highest_first' not in c:
        sys.stderr.write(f'[calib] {path} unrecognized schema; using fallback\n')
        return
    c['_path'] = os.path.abspath(path)
    CALIB = c
    sys.stderr.write(
        f'[calib] loaded order-{c.get("order")} fit from {path}: '
        f'coef={c["coef_highest_first"]} RMSE={c.get("rmse_C"):.3f}°C '
        f'R²={c.get("r_squared"):.6f} (n={c.get("n_points")})\n'
    )

def raw_to_C(raw):
    """Convert raw16 scalar or array to °C using loaded calibration, or
    the FMX-shape fallback if no calibration is loaded."""
    if CALIB is not None:
        return np.polyval(CALIB['coef_highest_first'], raw)
    return np.asarray(raw, dtype=np.float64) * FALLBACK_SLOPE + FALLBACK_OFFSET

def calib_info():
    """Short dict describing the current temperature model (for /status and sidecars)."""
    if CALIB is None:
        return {
            'calibrated': False,
            'formula': FALLBACK_DESC,
            'accuracy_C': 2.0,
        }
    return {
        'calibrated': True,
        'path': CALIB.get('_path'),
        'order': CALIB.get('order'),
        'coef_highest_first': CALIB.get('coef_highest_first'),
        'formula': CALIB.get('formula'),
        'rmse_C': CALIB.get('rmse_C'),
        'r_squared': CALIB.get('r_squared'),
        'n_points': CALIB.get('n_points'),
    }

# ---- Pseudocolor lookup table (ironbow) -----------------------------------
def _ironbow_lut():
    # control points: (t, r, g, b) with t in [0,1] and channels in [0,255]
    stops = [
        (0.00,   0,   0,   0),
        (0.15,  30,   0,  80),
        (0.30,  90,   0, 140),
        (0.45, 180,  20, 110),
        (0.60, 240,  60,  40),
        (0.75, 255, 150,   0),
        (0.90, 255, 230,  60),
        (1.00, 255, 255, 255),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for a, b in zip(stops, stops[1:]):
            if a[0] <= t <= b[0]:
                u = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
                lut[i] = [int(round(a[k+1] + u * (b[k+1] - a[k+1]))) for k in range(3)]
                break
    return lut

LUT = _ironbow_lut()

# Preview render modes:
#   cam    — decode the camera's H.264 pseudocolor (chan 0) and recolor with our LUT.
#            Matches vendor fidelity (NUC/BPR already applied) but is 8-bit & lossy.
#   clahe  — raw radiometric (chan 8) → despeckle → CLAHE → LUT. 16-bit source, all
#            sensor artifacts included; tunable contrast.
#   linear — raw → despeckle → percentile stretch → LUT. Brightness ∝ temperature.
# The hover readout, /snapshot.tiff, and calibration always use the raw plane.
VALID_MODES = ('cam', 'clahe', 'linear')
DEFAULT_MODE = os.environ.get('THERMAL_MODE', 'cam').lower()
if DEFAULT_MODE not in VALID_MODES:
    DEFAULT_MODE = 'cam'

# Despeckle: the FMX's sensor/firmware emits rare pixels where the MSB and
# LSB planes get out of sync by ±1 MSB, producing raw values ≈ ±256 off
# from neighbors. These are visible as bright/dark speckles along high-
# gradient edges in the scene. Replace any pixel that deviates from its
# 3x3 neighborhood median by more than this threshold (in raw counts).
DESPECKLE_THRESHOLD = 200

# Temporal median over the last N frames, computed at publish time and used
# as the canonical raw frame. Kills the MSB/LSB-sync glitches per-pixel
# because the glitch lands on different pixels each frame, so a 3-frame
# median recovers the true value. Set SMOOTH_N=1 to disable. Latency cost
# is (SMOOTH_N-1) * frame_period (~133 ms at N=3, 15 fps).
SMOOTH_N = max(1, int(os.environ.get('THERMAL_SMOOTH', '3')))

def despeckle(raw16):
    """Replace MSB/LSB-sync-error pixels with their 3x3 neighborhood median."""
    src = raw16.astype(np.int32)
    pad = np.pad(src, 1, mode='edge')
    nbrs = np.stack([
        pad[:-2, :-2], pad[:-2, 1:-1], pad[:-2, 2:],
        pad[1:-1, :-2],                 pad[1:-1, 2:],
        pad[2:,  :-2], pad[2:,  1:-1], pad[2:,  2:],
    ])
    nmed = np.median(nbrs, axis=0)
    bad = np.abs(src - nmed) > DESPECKLE_THRESHOLD
    out = raw16.copy()
    out[bad] = nmed[bad].astype(raw16.dtype)
    return out

def _stretch_linear(raw16, lo_pct=1.0, hi_pct=99.0):
    lo, hi = np.percentile(raw16, [lo_pct, hi_pct])
    if hi <= lo: hi = lo + 1
    return np.clip((raw16.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

def _clahe_u8(raw16, tiles=(6, 8), clip_limit=2.5, nbins=256):
    """Contrast-limited adaptive histogram equalization, numpy-only.

    Compact range into `nbins` via a wide percentile clip, build per-tile
    clipped-histogram CDFs, then bilinearly blend the 4 nearest tile LUTs
    at each pixel. Returns uint8 (H, W).
    """
    H, W = raw16.shape
    th, tw = tiles
    tile_h, tile_w = H // th, W // tw

    lo, hi = np.percentile(raw16, [0.5, 99.5])
    if hi <= lo: hi = lo + 1
    q = np.clip((raw16.astype(np.float32) - lo) * ((nbins - 1) / (hi - lo)),
                0, nbins - 1).astype(np.int32)

    luts = np.empty((th, tw, nbins), dtype=np.float32)
    for i in range(th):
        y0 = i * tile_h
        y1 = (i + 1) * tile_h if i < th - 1 else H
        for j in range(tw):
            x0 = j * tile_w
            x1 = (j + 1) * tile_w if j < tw - 1 else W
            tile = q[y0:y1, x0:x1]
            hist = np.bincount(tile.ravel(), minlength=nbins).astype(np.float32)
            n_pix = tile.size
            clip = max(1.0, clip_limit * n_pix / nbins)
            excess = np.maximum(hist - clip, 0).sum()
            np.minimum(hist, clip, out=hist)
            hist += excess / nbins
            cdf = np.cumsum(hist)
            span = max(1.0, float(cdf[-1] - cdf[0]))
            luts[i, j] = (cdf - cdf[0]) * (255.0 / span)

    # Bilinear blend of the 4 nearest tile LUTs. Tile (i, j) center is at
    # pixel (i*tile_h + tile_h/2, j*tile_w + tile_w/2); compute fractional
    # tile coordinate per pixel and pick floor-corner + neighbor.
    ys = np.arange(H)
    xs = np.arange(W)
    ty = (ys - tile_h / 2) / tile_h
    tx = (xs - tile_w / 2) / tile_w
    i0 = np.clip(np.floor(ty).astype(np.int32), 0, th - 2)
    j0 = np.clip(np.floor(tx).astype(np.int32), 0, tw - 2)
    fy = np.clip(ty - i0, 0, 1).astype(np.float32).reshape(-1, 1)
    fx = np.clip(tx - j0, 0, 1).astype(np.float32).reshape(1, -1)
    i0g = i0.reshape(-1, 1)
    j0g = j0.reshape(1, -1)
    L00 = luts[i0g,     j0g,     q]
    L01 = luts[i0g,     j0g + 1, q]
    L10 = luts[i0g + 1, j0g,     q]
    L11 = luts[i0g + 1, j0g + 1, q]
    out = (L00 * (1 - fy) * (1 - fx) + L01 * (1 - fy) * fx +
           L10 *      fy  * (1 - fx) + L11 *      fy  * fx)
    return np.clip(out, 0, 255).astype(np.uint8)

def render_jpeg(mode, *, raw=None, gray=None, quality=85):
    """Colorize the frame with ironbow and JPEG-encode.
      mode='cam'    — recolor the camera's grayscale H.264 frame (must pass gray)
      mode='clahe'  — raw → despeckle → CLAHE → LUT (must pass raw)
      mode='linear' — raw → despeckle → percentile stretch → LUT (must pass raw)
    Returns bytes, or None if the requested source isn't available yet."""
    if mode == 'cam':
        if gray is None:
            return None
        rgb = LUT[gray]
    else:
        if raw is None:
            return None
        cleaned = despeckle(raw)
        norm = _stretch_linear(cleaned) if mode == 'linear' else _clahe_u8(cleaned)
        rgb = LUT[norm]
    buf = io.BytesIO()
    Image.fromarray(rgb, 'RGB').save(buf, 'JPEG', quality=quality)
    return buf.getvalue()

# ---- Shared frame store ---------------------------------------------------
class FrameStore:
    """Holds the latest raw (radiometric) and gray (H.264-decoded) frames.
    Raw and gray are published by different sources on the RTSP socket and
    arrive at slightly different rates, so they have independent sequence
    numbers. Consumers wait on whichever they care about."""
    def __init__(self):
        self.cond = threading.Condition()
        self.raw = None       # np.uint16 (H, W) — canonical (post-smoothing) radiometric
        self.md = None        # 28-byte HYAV metadata header
        self.raw_seq = 0
        self.raw_ts = 0.0
        # Ring buffer of the last SMOOTH_N raw frames used for temporal median.
        self._raw_hist = []
        self.gray = None      # np.uint8 (H, W) — camera pseudocolor (chan 0)
        self.gray_seq = 0
        self.gray_ts = 0.0

    def publish_raw(self, raw, md):
        with self.cond:
            # Temporal median over the last SMOOTH_N frames. Before we have
            # a full window, use what we've got (median is still well-defined).
            if SMOOTH_N <= 1:
                smoothed = raw
            else:
                self._raw_hist.append(raw)
                if len(self._raw_hist) > SMOOTH_N:
                    self._raw_hist.pop(0)
                if len(self._raw_hist) == 1:
                    smoothed = raw
                else:
                    smoothed = np.median(np.stack(self._raw_hist), axis=0).astype(np.uint16)
            self.raw = smoothed
            self.md = md
            self.raw_seq += 1
            self.raw_ts = time.time()
            self.cond.notify_all()

    def publish_gray(self, gray):
        with self.cond:
            self.gray = gray
            self.gray_seq += 1
            self.gray_ts = time.time()
            self.cond.notify_all()

    def wait_frame(self, last_seq, which='gray', timeout=5.0):
        """Wait for the next 'raw' or 'gray' frame past last_seq.
        Returns (seq, frame) or None on timeout."""
        with self.cond:
            deadline = time.monotonic() + timeout
            while True:
                seq  = self.gray_seq if which == 'gray' else self.raw_seq
                data = self.gray     if which == 'gray' else self.raw
                if data is not None and seq != last_seq:
                    return seq, data
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cond.wait(remaining)

    def snapshot(self):
        """Latest frame pair. Returns None if no raw frame received yet."""
        with self.cond:
            if self.raw is None:
                return None
            return {
                'raw': self.raw.copy(),
                'md': self.md,
                'seq': self.raw_seq,
                'ts': self.raw_ts,
                'gray': self.gray.copy() if self.gray is not None else None,
                'gray_seq': self.gray_seq,
            }

STORE = FrameStore()

# Set on SIGINT/SIGTERM to ask background threads to exit.
STOP = threading.Event()
# Holds a reference to the current live RTSP client so the shutdown path can
# close its socket (interrupting any blocked recv) and send TEARDOWN.
CAMERA_CLIENT = {'ref': None}

# ---- RTSP client with async dispatcher ------------------------------------
class RTSPClient:
    """Single-socket RTSP client that multiplexes interleaved HYAV frames and
    RTSP control responses after PLAY has started."""

    def __init__(self, host=DEFAULT_HOST, port=554, path=DEFAULT_PATH):
        self.host, self.port, self.path = host, port, path
        self.url = f'rtsp://{host}:{port}{path}'
        self.sock = None
        self.rbuf = b''
        self.cseq = 0
        self.session = None
        self.base = self.url + '/'
        self.send_lock = threading.Lock()
        self.resp_queue = queue.Queue()
        self._closed = False
        # H.264 decoder for chan 0 (camera's cooked pseudocolor). Lazily
        # constructed — we need a fresh decoder per RTSP session so it re-
        # acquires SPS/PPS from the first keyframe after reconnect.
        self.h264 = None
        self._h264_errors = 0

    # -- low-level socket helpers --
    def _read_exact(self, n):
        while len(self.rbuf) < n:
            d = self.sock.recv(max(4096, n - len(self.rbuf)))
            if not d:
                raise ConnectionError('RTSP server closed')
            self.rbuf += d
        out, self.rbuf = self.rbuf[:n], self.rbuf[n:]
        return out

    def _send(self, method, *, url=None, extra=()):
        with self.send_lock:
            self.cseq += 1
            cseq = self.cseq
            lines = [f'{method} {url or self.url} RTSP/1.0',
                     f'CSeq: {cseq}',
                     'User-Agent: thermald/1.0']
            if self.session:
                lines.append(f'Session: {self.session}')
            lines += list(extra)
            lines += ['', '']
            self.sock.sendall('\r\n'.join(lines).encode())
        return cseq

    # -- handshake phase: synchronous recv --
    def _recv_rtsp_sync(self):
        while b'\r\n\r\n' not in self.rbuf:
            d = self.sock.recv(4096)
            if not d:
                raise ConnectionError('closed during handshake')
            self.rbuf += d
        i = self.rbuf.index(b'\r\n\r\n') + 4
        hdr = self.rbuf[:i].decode('latin-1')
        self.rbuf = self.rbuf[i:]
        clen = 0
        for line in hdr.split('\r\n'):
            if line.lower().startswith('content-length:'):
                clen = int(line.split(':', 1)[1].strip())
        body = self._read_exact(clen) if clen else b''
        return hdr, body

    def connect_and_play(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=10)
        self.sock.settimeout(60)
        self._send('OPTIONS')
        self._recv_rtsp_sync()

        self._send('DESCRIBE', extra=['Accept: application/sdp'])
        hdr, _ = self._recv_rtsp_sync()
        for line in hdr.split('\r\n'):
            if line.lower().startswith('content-base:'):
                self.base = line.split(':', 1)[1].strip()

        self._send('SETUP', url=self.base + 'trackID=0',
                   extra=['Transport: DH/AVP/TCP;unicast;interleaved=0-1;mode=play'])
        hdr, _ = self._recv_rtsp_sync()
        for line in hdr.split('\r\n'):
            if line.lower().startswith('session:'):
                self.session = line.split(':', 1)[1].strip().split(';')[0]
        if not self.session:
            raise RuntimeError('no Session header in SETUP response')

        self._send('SETUP', url=self.base + 'trackID=4',
                   extra=['Transport: DH/AVP/TCP;unicast;interleaved=2-3;mode=play'])
        self._recv_rtsp_sync()

        self._send('PLAY', url=self.base, extra=['Range: npt=0.000-'])
        self._recv_rtsp_sync()

    # -- streaming phase: async dispatcher reads everything --
    def dispatch_loop(self):
        """Read the socket forever, feeding HYAV frames into STORE and RTSP
        responses into self.resp_queue. Raises on disconnect."""
        while not self._closed:
            first = self._read_exact(1)
            b = first[0]
            if b == 0x24:  # '$' interleaved frame
                head = self._read_exact(5)  # channel(1) + length(4 BE)
                length = struct.unpack('>I', head[1:5])[0]
                body = self._read_exact(length)
                self._handle_interleaved(head[0], body)
            elif 0x41 <= b <= 0x5A:  # 'A'..'Z' -> likely "RTSP/..."
                # Accumulate until CRLFCRLF
                line = first
                while b'\r\n\r\n' not in line:
                    line += self._read_exact(1)
                clen = 0
                for l in line.split(b'\r\n'):
                    if l.lower().startswith(b'content-length:'):
                        clen = int(l.split(b':', 1)[1].strip())
                body = self._read_exact(clen) if clen else b''
                self.resp_queue.put((line.decode('latin-1'), body))
            else:
                # unknown byte; try to resync
                sys.stderr.write(f'[rtsp] unexpected byte 0x{b:02x}, resyncing\n')

    def _handle_interleaved(self, chan, body):
        # Both video (chan 0) and radiometric (chan 8) arrive as HYAV chunks.
        # Layout: "HYAV" + 28-byte header + payload + 8-byte trailer ("hyav" + LE size).
        if len(body) < 32 + 8 or body[:4] != b'HYAV':
            return
        md = body[4:32]
        payload = body[32:-8]
        if chan == 8:
            # Raw radiometric: MSB plane then LSB plane, row-major 384x288.
            if len(payload) < FRAME_DATA:
                return
            msb = np.frombuffer(payload[:PLANE], dtype=np.uint8).reshape(H, W)
            lsb = np.frombuffer(payload[PLANE:2*PLANE], dtype=np.uint8).reshape(H, W)
            raw = (msb.astype(np.uint16) << 8) | lsb.astype(np.uint16)
            STORE.publish_raw(raw, bytes(md))
        elif chan == 0 and HAVE_AV:
            # Camera's cooked pseudocolor (grayscale H.264). Feed each chunk
            # to the decoder; SPS/PPS/IDR arrive with the first keyframe.
            if self.h264 is None:
                try:
                    self.h264 = av.CodecContext.create('h264', 'r')
                except Exception as e:
                    sys.stderr.write(f'[cam] H.264 decoder init failed: {e}\n')
                    self.h264 = False   # sentinel: never retry
                    return
            if self.h264 is False:
                return
            try:
                for frame in self.h264.decode(av.Packet(payload)):
                    gray = frame.to_ndarray(format='gray8')
                    STORE.publish_gray(gray)
            except av.error.InvalidDataError:
                # First couple of P-slices before the initial keyframe land here.
                self._h264_errors += 1
            except Exception as e:
                self._h264_errors += 1
                if self._h264_errors <= 3:
                    sys.stderr.write(f'[cam] H.264 decode error: {e}\n')

    def keepalive(self):
        try:
            self._send('GET_PARAMETER', url=self.base)
        except Exception as e:
            sys.stderr.write(f'[rtsp] keepalive send failed: {e}\n')

    def teardown(self):
        self._closed = True
        try:
            if self.sock and self.session:
                self._send('TEARDOWN', url=self.base)
        except Exception:
            pass
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass

# ---- Camera supervisor (reconnect with backoff) ---------------------------
def camera_supervisor(host):
    backoff = 1.0
    while not STOP.is_set():
        client = RTSPClient(host=host)
        CAMERA_CLIENT['ref'] = client
        try:
            sys.stderr.write(f'[cam] connecting to {host}...\n')
            client.connect_and_play()
            sys.stderr.write(f'[cam] PLAY ok (session {client.session}); streaming\n')
            backoff = 1.0
            stop_ka = threading.Event()
            def keepalive_loop():
                while not stop_ka.wait(30):
                    if STOP.is_set(): return
                    client.keepalive()
            ka = threading.Thread(target=keepalive_loop, daemon=True)
            ka.start()
            client.dispatch_loop()   # blocks until error or teardown
        except Exception as e:
            if STOP.is_set():
                break
            sys.stderr.write(f'[cam] {type(e).__name__}: {e}; reconnect in {backoff:.0f}s\n')
        finally:
            stop_ka_ref = locals().get('stop_ka')
            if stop_ka_ref: stop_ka_ref.set()
            CAMERA_CLIENT['ref'] = None
            client.teardown()
        if STOP.wait(backoff): break
        backoff = min(backoff * 2, 30.0)
    sys.stderr.write('[cam] supervisor exiting\n')

# ---- Camera admin client (HTTP /v1/...) -----------------------------------
# Thin wrapper around the camera's REST API used to proxy image-control
# settings (brightness/contrast/AGC/DDE) from our dashboard. Thread-safe:
# the JWT is cached under a lock and refreshed on 401.
class CameraClient:
    def __init__(self, host, user='admin', password='admin'):
        self.host = host
        self.user = user
        self.password = password
        self.token = None
        self.lock = threading.Lock()

    def _login(self):
        import urllib.request, urllib.parse
        url = f'http://{self.host}/v1/user/login?' + urllib.parse.urlencode(
            {'date': int(time.time()*1000), 'username': self.user, 'password': self.password})
        with urllib.request.urlopen(url, timeout=5) as r:
            body = json.loads(r.read().decode())
        self.token = body['Data']['Token']

    def request(self, method, path, params=None, data=None):
        import urllib.request, urllib.parse, urllib.error
        params = dict(params or {}); params.setdefault('date', int(time.time()*1000))
        body_bytes = json.dumps(data).encode() if data is not None else None
        for attempt in range(2):
            with self.lock:
                if not self.token:
                    self._login()
                token = self.token
            url = f'http://{self.host}/v1{path}?' + urllib.parse.urlencode(params)
            headers = {'X-Token': token, 'Accept': 'application/json'}
            if body_bytes is not None:
                headers['Content-Type'] = 'application/json'
            req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 401 and attempt == 0:
                    with self.lock: self.token = None
                    continue
                raise
        raise RuntimeError('camera auth failed')

CAMERA = None   # CameraClient instance, created in main()

# ---- HTTP server ----------------------------------------------------------
SNAPSHOT_DIR = os.environ.get('THERMAL_SNAPSHOTS', '/home/kbentley/thermal/snapshots')
CAMERA_HOST = {'ref': DEFAULT_HOST}  # filled in by main() from --host

# Image-control settings we proxy. Each entry is:
#   (ui-key,  GET path,  SET path,  PUT-body-key,  slider min/max/step or None for enum)
IMAGE_SETTINGS = [
    ('brightness', '/cmd01/agcmenu/GetBrightness', '/cmd01/agcmenu/SetBrightness', 'bright',   (0, 255, 1)),
    ('contrast',   '/cmd01/agcmenu/GetContrast',   '/cmd01/agcmenu/SetContrast',   'contrast', (0, 100, 1)),
    ('ddeGears',   '/cmd01/agcmenu/GetDDEGears',   '/cmd01/agcmenu/SetDDEGears',   'gear',     (0,   8, 1)),
    ('agcMode',    '/cmd01/agcmenu/GetAGCMode',    '/cmd01/agcmenu/SetAGCMode',    'mode',     None),  # enum 0..3
]
AGC_MODE_LABELS = {0: 'manual', 1: 'auto', 2: 'histogram', 3: 'plateau'}

INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>thermald</title>
<style>
 body{font-family:sans-serif;margin:24px;background:#111;color:#eee}
 #wrap{position:relative;display:inline-block;line-height:0}
 #view{image-rendering:pixelated;border:1px solid #333;width:768px;height:576px;cursor:crosshair;display:block}
 #xh{position:absolute;pointer-events:none;display:none;width:24px;height:24px;margin:-12px 0 0 -12px;
     border:1px solid #fff;border-radius:50%;box-shadow:0 0 0 1px rgba(0,0,0,.6)}
 #xh::before,#xh::after{content:'';position:absolute;background:#fff}
 #xh::before{left:-10px;top:11px;width:44px;height:1px}
 #xh::after{left:11px;top:-10px;width:1px;height:44px}
 #readout{font-family:monospace;font-size:15px;margin-top:12px;min-height:22px;
          padding:8px 12px;background:#222;border:1px solid #333;display:inline-block;min-width:420px}
 button{padding:8px 16px;font-size:15px;margin:4px 4px 4px 0}
 a{color:#8cf}
 .big{font-size:22px;font-weight:600}
 #calib{display:inline-block;font-family:monospace;font-size:13px;
        padding:4px 10px;border-radius:4px;margin-left:12px;vertical-align:middle}
 #calib.ok{background:#0c2;color:#000}
 #calib.warn{background:#c80;color:#000}
 #calib.bad{background:#c22;color:#fff}
</style></head><body>
<h1>ICI FMX 400P live<span id="calib">…</span></h1>
<div id="wrap">
 <img id="view" src="/preview.mjpg" width="768" height="576"/>
 <div id="xh"></div>
</div>
<div id="readout">move cursor over image &rarr; temperature</div>
<p>
 <button onclick="snap()">save snapshot</button>
 <span style="margin-left:12px;padding-left:12px;border-left:1px solid #444"></span>
 focus:
 <button onclick="focusCmd('step')" title="one manual focus step">step</button>
 <button onclick="focusCmd('auto')" title="one-shot autofocus">auto</button>
 <span id="focusmsg" style="font-family:monospace;font-size:13px;margin-left:8px;color:#aaa"></span>
 <span style="margin-left:12px;padding-left:12px;border-left:1px solid #444"></span>
 mode:
 <button id="mode-cam"    onclick="setMode('cam')"    title="camera's pseudocolor H.264 + our ironbow LUT (clean, matches IRFlash)">camera</button>
 <button id="mode-clahe"  onclick="setMode('clahe')"  title="raw radiometric + despeckle + CLAHE (16-bit source; sensor edge artifacts visible)">CLAHE</button>
 <button id="mode-linear" onclick="setMode('linear')" title="raw radiometric + despeckle + linear stretch (brightness ∝ temperature)">linear</button>
 <br><span id="snapmsg"></span>
</p>
<div id="camcfg" style="margin-top:14px;padding:12px;background:#1a1a1a;border:1px solid #333;max-width:768px;font-size:14px">
 <div style="font-weight:600;margin-bottom:8px">Camera image settings <span id="camcfg-status" style="font-weight:400;color:#888;margin-left:8px"></span></div>
 <div style="display:grid;grid-template-columns:110px 1fr 60px;gap:8px 12px;align-items:center">
  <label>brightness</label>
  <input type="range" id="s-brightness" min="0" max="255" step="1">
  <span id="v-brightness" style="font-family:monospace">—</span>

  <label>contrast</label>
  <input type="range" id="s-contrast" min="0" max="100" step="1">
  <span id="v-contrast" style="font-family:monospace">—</span>

  <label>DDE gears</label>
  <input type="range" id="s-ddeGears" min="0" max="8" step="1">
  <span id="v-ddeGears" style="font-family:monospace">—</span>

  <label>AGC mode</label>
  <select id="s-agcMode" style="grid-column:span 2">
   <option value="0">0 — manual</option>
   <option value="1">1 — auto</option>
   <option value="2">2 — histogram</option>
   <option value="3">3 — plateau</option>
  </select>
 </div>
 <div style="color:#888;font-size:12px;margin-top:6px">These control the camera's internal pipeline → visible in <b>camera</b> mode only. Raw radiometry (CLAHE/linear modes) is unaffected.</div>
</div>
<p>Endpoints:
<a href="/snapshot.tiff">/snapshot.tiff</a> &middot;
<a href="/status">/status</a> &middot;
<a href="/pixel?x=192&y=144">/pixel?x=192&y=144</a></p>
<script>
const W=384, H=288;
const view=document.getElementById('view');
const xh=document.getElementById('xh');
const readout=document.getElementById('readout');
const calib=document.getElementById('calib');
let pending=null, inflight=false;

const MODES = ['cam','clahe','linear'];
let currentMode = localStorage.getItem('thermalMode') || 'cam';
if(!MODES.includes(currentMode)) currentMode = 'cam';
function setMode(m){
  currentMode = m;
  localStorage.setItem('thermalMode', m);
  for(const k of MODES){
    document.getElementById('mode-'+k).style.fontWeight = (m===k?'700':'400');
  }
  view.src = '/preview.mjpg?mode=' + m + '&t=' + Date.now();
}
setMode(currentMode);

// Calibration badge from /status
fetch('/status').then(r=>r.json()).then(s=>{
  const c=s.calibration||{};
  if(c.calibrated){
    calib.className='ok';
    calib.title='source: '+(c.path||'?');
    calib.textContent='calibrated &#10003; RMSE '+c.rmse_C.toFixed(2)+'°C '+
                      'R²='+c.r_squared.toFixed(4)+' n='+c.n_points;
    calib.innerHTML=calib.textContent.replace('&#10003;','✓');
  }else{
    calib.className='bad';
    calib.title=c.formula||'';
    calib.textContent='UNCALIBRATED ballpark ±2°C';
  }
}).catch(e=>{ calib.className='warn'; calib.textContent='status unavailable'; });

async function send(){
  if(!pending || inflight) return;
  const p=pending; pending=null; inflight=true;
  try{
    const r=await fetch(`/pixel?x=${p.x}&y=${p.y}`);
    if(r.ok){
      const j=await r.json();
      readout.innerHTML=
        `<span class="big">${j.T_C.toFixed(2)}&deg;C</span> `+
        `(${j.T_F.toFixed(1)}&deg;F / ${j.T_K.toFixed(2)} K) &nbsp; `+
        `x=${j.x} y=${j.y} &nbsp; raw16=${j.raw16} &nbsp; seq=${j.seq}`;
    }
  }catch(e){}
  inflight=false;
  if(pending) send();
}

view.addEventListener('mousemove', e=>{
  const r=view.getBoundingClientRect();
  const px=(e.clientX-r.left)/r.width;
  const py=(e.clientY-r.top)/r.height;
  const x=Math.max(0,Math.min(W-1,Math.floor(px*W)));
  const y=Math.max(0,Math.min(H-1,Math.floor(py*H)));
  xh.style.display='block';
  xh.style.left=(e.clientX-r.left)+'px';
  xh.style.top=(e.clientY-r.top)+'px';
  pending={x,y};
  send();
});
view.addEventListener('mouseleave',()=>{
  xh.style.display='none';
  pending=null;
});

async function snap(){
  const name=new Date().toISOString().replace(/[:.]/g,'-');
  document.getElementById('snapmsg').textContent=' …saving';
  try{
    const r=await fetch('/snapshot?name='+name,{method:'POST'});
    const j=await r.json();
    document.getElementById('snapmsg').textContent=
      j.error ? (' error: '+j.error) : (' saved seq='+j.seq+' -> '+j.tiff);
  }catch(e){
    document.getElementById('snapmsg').textContent=' error: '+e;
  }
}

async function focusCmd(cmd){
  const msg=document.getElementById('focusmsg');
  msg.style.color='#aaa'; msg.textContent=cmd+' …';
  try{
    const r=await fetch('/focus?cmd='+cmd,{method:'POST'});
    const j=await r.json();
    if(j.error){ msg.style.color='#f88'; msg.textContent=cmd+': '+j.error; }
    else{ msg.style.color=j.ack?'#8f8':'#fc0'; msg.textContent=cmd+(j.ack?' ✓':' reply='+j.reply); }
  }catch(e){
    msg.style.color='#f88'; msg.textContent=cmd+': '+e;
  }
}

// Camera image settings — load current state, wire sliders/select to PUT on change.
const CAM_KEYS = ['brightness','contrast','ddeGears','agcMode'];
const camStatus = document.getElementById('camcfg-status');

async function camcfgLoad(){
  camStatus.textContent='loading…';
  try{
    const r = await fetch('/camera/image');
    const j = await r.json();
    for(const k of CAM_KEYS){
      const v = j.settings?.[k];
      const el = document.getElementById('s-'+k);
      const out = document.getElementById('v-'+k);
      if(v==null){ if(out) out.textContent='—'; continue; }
      el.value = v;
      if(out) out.textContent = v;
    }
    camStatus.textContent = j.errors ? ('some reads failed: '+Object.keys(j.errors).join(',')) : 'ok';
    camStatus.style.color = j.errors ? '#fc0' : '#8f8';
  }catch(e){
    camStatus.textContent = 'load error: '+e;
    camStatus.style.color = '#f88';
  }
}

async function camcfgSet(name, value){
  camStatus.textContent = name+' = '+value+' …';
  camStatus.style.color = '#aaa';
  try{
    const r = await fetch('/camera/image', {
      method:'PUT',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, value: Number(value)}),
    });
    const j = await r.json();
    if(j.error){ camStatus.textContent = name+' error: '+j.error; camStatus.style.color = '#f88'; return; }
    const out = document.getElementById('v-'+name);
    if(out) out.textContent = value;
    camStatus.textContent = name+' ← '+value; camStatus.style.color = '#8f8';
  }catch(e){
    camStatus.textContent = name+' error: '+e; camStatus.style.color = '#f88';
  }
}

// debounce slider input → send when user pauses for 150ms
function debounce(fn, ms){ let t; return (...a)=>{ clearTimeout(t); t=setTimeout(()=>fn(...a), ms); }; }

for(const k of ['brightness','contrast','ddeGears']){
  const el = document.getElementById('s-'+k);
  const out = document.getElementById('v-'+k);
  const send = debounce(v => camcfgSet(k, v), 150);
  el.addEventListener('input', () => { out.textContent = el.value; send(el.value); });
}
document.getElementById('s-agcMode').addEventListener('change', e => camcfgSet('agcMode', e.target.value));

camcfgLoad();
</script></body></html>""".encode('utf-8')

def _save_snapshot(name):
    snap = STORE.snapshot()
    if snap is None:
        return None
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    if not name:
        name = time.strftime('%Y%m%d_%H%M%S', time.localtime(snap['ts']))
    tiff_path = os.path.join(SNAPSHOT_DIR, f'{name}.tiff')
    png_path  = os.path.join(SNAPSHOT_DIR, f'{name}.png')
    json_path = os.path.join(SNAPSHOT_DIR, f'{name}.json')
    raw = snap['raw']
    # TIFF preserves the raw radiometric data exactly as received (no despeckle).
    Image.fromarray(raw, mode='I;16').save(tiff_path)
    # PNG uses whichever mode the preview defaults to.
    gray = snap.get('gray')
    if DEFAULT_MODE == 'cam' and gray is not None:
        rgb = LUT[gray]
    else:
        cleaned = despeckle(raw)
        norm = _stretch_linear(cleaned) if DEFAULT_MODE == 'linear' else _clahe_u8(cleaned)
        rgb = LUT[norm]
    Image.fromarray(rgb, 'RGB').save(png_path)
    t_c = raw_to_C(raw)
    stats_key = 'stats_C' if CALIB is not None else 'stats_uncalibrated_C'
    meta = {
        'seq': snap['seq'],
        'timestamp': snap['ts'],
        'timestamp_iso': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(snap['ts'])),
        'width': W, 'height': H,
        'hyav_metadata_hex': snap['md'].hex() if snap['md'] else None,
        'source': f'rtsp://{DEFAULT_HOST}:554{DEFAULT_PATH}',
        'calibration': calib_info(),
        stats_key: {
            'min': float(t_c.min()),
            'mean': float(t_c.mean()),
            'max': float(t_c.max()),
        },
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    return {'tiff': tiff_path, 'png': png_path, 'json': json_path, 'seq': snap['seq']}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f'[http] {self.client_address[0]} {fmt % args}\n')

    def _json(self, obj, status=200):
        body = json.dumps(obj, indent=2).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(INDEX_HTML)))
            self.end_headers()
            self.wfile.write(INDEX_HTML)
            return
        if u.path == '/preview.mjpg':
            return self._mjpeg(parse_qs(u.query))
        if u.path == '/snapshot.tiff':
            return self._tiff()
        if u.path == '/pixel':
            return self._pixel(parse_qs(u.query))
        if u.path == '/status':
            return self._status()
        if u.path == '/camera/image':
            return self._camera_image_get()
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == '/snapshot':
            qs = parse_qs(u.query)
            name = qs.get('name', [None])[0]
            out = _save_snapshot(name)
            if out is None:
                return self._json({'error': 'no frame yet'}, status=503)
            return self._json(out)
        if u.path == '/focus':
            return self._focus(parse_qs(u.query))
        # Also accept POST on /camera/image for convenience (same body shape as PUT).
        if u.path == '/camera/image':
            return self._camera_image_set()
        self.send_error(404)

    def do_PUT(self):
        u = urlparse(self.path)
        if u.path == '/camera/image':
            return self._camera_image_set()
        self.send_error(404)

    def _camera_image_get(self):
        """Read current brightness/contrast/AGC/DDEGears from the camera."""
        if CAMERA is None:
            return self._json({'error': 'camera client not initialized'}, 503)
        out = {'settings': {}, 'ranges': {}, 'labels': {'agcMode': AGC_MODE_LABELS}}
        for key, get_path, _set, _body_key, rng in IMAGE_SETTINGS:
            try:
                r = CAMERA.request('GET', get_path)
                out['settings'][key] = r.get('Data')
            except Exception as e:
                out['settings'][key] = None
                out.setdefault('errors', {})[key] = f'{type(e).__name__}: {e}'
            if rng is not None:
                out['ranges'][key] = {'min': rng[0], 'max': rng[1], 'step': rng[2]}
        return self._json(out)

    def _camera_image_set(self):
        """PUT a single setting. Body: {name: <ui-key>, value: <int>}."""
        if CAMERA is None:
            return self._json({'error': 'camera client not initialized'}, 503)
        length = int(self.headers.get('Content-Length', '0'))
        try:
            body = json.loads(self.rfile.read(length).decode()) if length else {}
        except Exception as e:
            return self._json({'error': f'bad JSON: {e}'}, 400)
        name  = body.get('name')
        value = body.get('value')
        if value is None:
            return self._json({'error': 'value required'}, 400)
        row = next((s for s in IMAGE_SETTINGS if s[0] == name), None)
        if row is None:
            return self._json({'error': f'unknown setting; expected one of {[s[0] for s in IMAGE_SETTINGS]}'}, 400)
        _key, _get, set_path, body_key, rng = row
        try:
            value = int(value)
        except (TypeError, ValueError):
            return self._json({'error': 'value must be an integer'}, 400)
        if rng is not None and not (rng[0] <= value <= rng[1]):
            return self._json({'error': f'{name} out of range [{rng[0]},{rng[1]}]'}, 400)
        # The cmd01 SetXxx endpoints take their payload as QUERY PARAMS, not
        # JSON body (axios 'params:e' in the vendor SPA). JSON body is
        # silently accepted with Code=200 but ignored.
        try:
            r = CAMERA.request('PUT', set_path, params={body_key: value})
        except Exception as e:
            return self._json({'error': f'camera call failed: {type(e).__name__}: {e}'}, 502)
        return self._json({'name': name, 'value': value, 'camera_response': r})

    def _focus(self, qs):
        cmd = (qs.get('cmd', [''])[0] or '').lower()
        if cmd not in FOCUS_COMMANDS:
            return self._json({'error': f'cmd must be one of {sorted(FOCUS_COMMANDS)}'}, 400)
        try:
            resp = send_focus(CAMERA_HOST['ref'], FOCUS_COMMANDS[cmd])
        except OSError as e:
            return self._json({'error': f'{type(e).__name__}: {e}'}, 502)
        ack = b'\x7d\xff\x00\x7b'
        return self._json({
            'cmd': cmd,
            'sent': _focus_frame(FOCUS_COMMANDS[cmd]).hex(),
            'reply': resp.hex(),
            'ack': resp == ack,
        })

    def _mjpeg(self, qs):
        mode = (qs.get('mode', [DEFAULT_MODE])[0] or DEFAULT_MODE).lower()
        if mode not in VALID_MODES:
            mode = DEFAULT_MODE
        # 'cam' mode waits on grayscale updates; 'clahe'/'linear' wait on raw.
        which = 'gray' if mode == 'cam' else 'raw'
        bnd = b'thermalframe'
        self.send_response(200)
        self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={bnd.decode()}')
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.end_headers()
        last = -1
        try:
            while not STOP.is_set():
                r = STORE.wait_frame(last, which=which, timeout=1)
                if r is None:
                    continue
                last, data = r
                jpeg = render_jpeg(mode,
                                   gray=data if mode == 'cam' else None,
                                   raw =data if mode != 'cam' else None)
                if jpeg is None:
                    continue
                hdr = (b'\r\n--' + bnd + b'\r\nContent-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n')
                self.wfile.write(hdr); self.wfile.write(jpeg); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _tiff(self):
        snap = STORE.snapshot()
        if snap is None:
            return self._json({'error': 'no frame yet'}, status=503)
        buf = io.BytesIO()
        Image.fromarray(snap['raw'], mode='I;16').save(buf, 'TIFF')
        data = buf.getvalue()
        self.send_response(200)
        self.send_header('Content-Type', 'image/tiff')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Frame-Seq', str(snap['seq']))
        self.send_header('X-Frame-Time', f"{snap['ts']:.3f}")
        self.end_headers()
        self.wfile.write(data)

    def _pixel(self, qs):
        try:
            x = int(qs.get('x', ['0'])[0]); y = int(qs.get('y', ['0'])[0])
        except ValueError:
            return self._json({'error': 'x and y must be ints'}, 400)
        snap = STORE.snapshot()
        if snap is None:
            return self._json({'error': 'no frame yet'}, 503)
        if not (0 <= x < W and 0 <= y < H):
            return self._json({'error': f'out of bounds ({W}x{H})'}, 400)
        r = int(snap['raw'][y, x])
        t_c = float(raw_to_C(r))
        return self._json({
            'x': x, 'y': y, 'raw16': r,
            'T_C': round(t_c, 3),
            'T_F': round(t_c * 9/5 + 32, 3),
            'T_K': round(t_c + 273.15, 3),
            'calibrated': CALIB is not None,
            'seq': snap['seq'], 'timestamp': snap['ts'],
        })

    def _status(self):
        snap = STORE.snapshot()
        out = {
            'ready': snap is not None,
            'width': W, 'height': H,
            'snapshot_dir': SNAPSHOT_DIR,
            'calibration': calib_info(),
            'default_mode': DEFAULT_MODE,
            'h264_available': HAVE_AV,
            'gray_ready': snap is not None and snap.get('gray') is not None,
            'gray_seq': snap['gray_seq'] if snap is not None else 0,
        }
        if snap is not None:
            raw = snap['raw']
            t_c = raw_to_C(raw)
            out.update({
                'seq': snap['seq'],
                'timestamp': snap['ts'],
                'age_seconds': round(time.time() - snap['ts'], 3),
                'stats_C': {
                    'min': round(float(t_c.min()), 2),
                    'mean': round(float(t_c.mean()), 2),
                    'max': round(float(t_c.max()), 2),
                },
            })
        return self._json(out)

# ---- entry point ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default=DEFAULT_HOST, help='camera IP')
    ap.add_argument('--bind', default='0.0.0.0', help='HTTP bind address')
    ap.add_argument('--port', type=int, default=8080, help='HTTP port')
    ap.add_argument('--calib', default=DEFAULT_CALIB_PATH,
                    help=f'path to calib.json (default {DEFAULT_CALIB_PATH}). '
                         f'Use "" to disable and fall back to uncalibrated.')
    args = ap.parse_args()

    load_calib(args.calib)
    CAMERA_HOST['ref'] = args.host
    global CAMERA
    CAMERA = CameraClient(args.host)

    # Camera reader — not a daemon thread, so we can join it on shutdown
    # and guarantee a TEARDOWN is sent before the process exits.
    cam_thread = threading.Thread(target=camera_supervisor, args=(args.host,),
                                  name='camera', daemon=False)
    cam_thread.start()

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    # Request-handler threads should not block process exit; MJPEG loops
    # cooperate via the STOP event, but daemon=True is belt-and-suspenders.
    server.daemon_threads = True
    sys.stderr.write(f'[http] serving on http://{args.bind}:{args.port}\n')

    # Install signal handlers BEFORE we block. The handler just sets STOP;
    # all real cleanup happens after we fall through the wait below.
    def on_signal(signum, _frame):
        sys.stderr.write(f'[shutdown] signal {signum} received\n')
        STOP.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Run the HTTP loop in a side thread so the main thread can wait for STOP
    # and then orchestrate cleanup in a known order.
    http_thread = threading.Thread(target=server.serve_forever,
                                   name='http', daemon=True)
    http_thread.start()

    STOP.wait()

    # 1) Close the camera socket so any blocked recv() fails immediately.
    client = CAMERA_CLIENT.get('ref')
    if client is not None:
        sys.stderr.write('[shutdown] TEARDOWN to camera\n')
        client.teardown()
    # 2) Wait for the supervisor to exit (sends TEARDOWN again if needed).
    cam_thread.join(timeout=5)
    if cam_thread.is_alive():
        sys.stderr.write('[shutdown] camera thread still alive after 5s\n')
    # 3) Stop the HTTP server: stops accepting, closes listening socket, and
    #    joins the serve_forever loop. MJPEG handlers already saw STOP and
    #    will exit their loops on the next condition wakeup (<=1s).
    server.shutdown()
    server.server_close()
    sys.stderr.write('[shutdown] done\n')

if __name__ == '__main__':
    main()
