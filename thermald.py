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

# ---- Camera protocol constants --------------------------------------------
W, H = 384, 288
PIX = W * H
PLANE = PIX                    # 110 592 bytes per plane
FRAME_DATA = 2 * PLANE         # 221 184 bytes of pixels per frame
DEFAULT_HOST = '192.168.1.123'
DEFAULT_PATH = '/cam/realmonitor?channel=1&subtype=0'
DEFAULT_CALIB_PATH = '/home/kbentley/thermal/calib.json'

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

def render_jpeg(raw16, quality=80):
    """Percentile-stretch the raw 16-bit frame and ironbow-color it to JPEG."""
    lo, hi = np.percentile(raw16, [2, 98])
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((raw16.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    rgb = LUT[norm]
    buf = io.BytesIO()
    Image.fromarray(rgb, 'RGB').save(buf, 'JPEG', quality=quality)
    return buf.getvalue()

# ---- Shared frame store ---------------------------------------------------
class FrameStore:
    def __init__(self):
        self.cond = threading.Condition()
        self.raw = None       # np.uint16 array (H, W)
        self.md = None        # 28-byte HYAV metadata header
        self.seq = 0
        self.ts = 0.0
        self.jpeg = None

    def publish(self, raw, md):
        jpeg = render_jpeg(raw)
        with self.cond:
            self.raw = raw
            self.md = md
            self.seq += 1
            self.ts = time.time()
            self.jpeg = jpeg
            self.cond.notify_all()

    def wait_next(self, last_seq, timeout=5.0):
        with self.cond:
            deadline = time.monotonic() + timeout
            while self.seq == last_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cond.wait(remaining)
            return self.seq, self.jpeg

    def snapshot(self):
        with self.cond:
            if self.raw is None:
                return None
            return {
                'raw': self.raw.copy(),
                'md': self.md,
                'seq': self.seq,
                'ts': self.ts,
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
        if len(body) < 32 or body[:4] != b'HYAV':
            return  # not a radiometric frame — ignore
        md = body[4:32]
        if len(body) < 32 + FRAME_DATA:
            return
        data = body[32:32 + FRAME_DATA]
        msb = np.frombuffer(data[:PLANE], dtype=np.uint8).reshape(H, W)
        lsb = np.frombuffer(data[PLANE:], dtype=np.uint8).reshape(H, W)
        raw = (msb.astype(np.uint16) << 8) | lsb.astype(np.uint16)
        STORE.publish(raw, bytes(md))

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

# ---- HTTP server ----------------------------------------------------------
SNAPSHOT_DIR = os.environ.get('THERMAL_SNAPSHOTS', '/home/kbentley/thermal/snapshots')

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
<p><button onclick="snap()">save snapshot</button><span id="snapmsg"></span></p>
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
    Image.fromarray(raw, mode='I;16').save(tiff_path)
    lo, hi = np.percentile(raw, [2, 98])
    if hi <= lo: hi = lo + 1
    norm = np.clip((raw.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    Image.fromarray(LUT[norm], 'RGB').save(png_path)
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
            return self._mjpeg()
        if u.path == '/snapshot.tiff':
            return self._tiff()
        if u.path == '/pixel':
            return self._pixel(parse_qs(u.query))
        if u.path == '/status':
            return self._status()
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
        self.send_error(404)

    def _mjpeg(self):
        bnd = b'thermalframe'
        self.send_response(200)
        self.send_header('Content-Type', f'multipart/x-mixed-replace; boundary={bnd.decode()}')
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.end_headers()
        last = -1
        try:
            while not STOP.is_set():
                r = STORE.wait_next(last, timeout=1)
                if r is None:
                    continue
                last, jpeg = r
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
