# thermald — radiometric gateway for the ICI FMX 400P

A single-file Python daemon that turns an **ICI FMX 400P** thermal camera
(InfiRay-OEM hardware running a Dahua-derived RTSP stack) into something
you can actually build with: a browser-viewable live preview, a JSON
pixel-temperature API, and on-demand snapshots that preserve the camera's
raw 16-bit radiometric output for offline analysis.

Also probably works on other InfiRay-OEM cameras that negotiate Dahua's
`DH/AVP/TCP` transport and stream the proprietary `HYAV` container on
port 554. If you have one, the protocol notes below will tell you whether
you're in the same family.

## What's here

| File | What it is |
|---|---|
| `thermald.py` | The daemon. Holds the (single) RTSP session, parses HYAV frames into 16-bit radiometric arrays, serves the HTTP endpoints. |
| `thermal-snap` | Small client that POSTs to `/snapshot`. Saves `<name>.tiff` (16-bit raw), `<name>.png` (pseudocolor), and `<name>.json` (sidecar w/ calibration). |
| `thermal-focus` | CLI for the port-36399 lens-control protocol: `thermal-focus step [N]` for manual focus steps, `thermal-focus auto` for one-shot autofocus. Talks directly to the camera; doesn't need `thermald.py` running. |
| `thermal-calibrate` | Fits a linear (or quadratic) `T_C = f(raw16)` model from black-body snapshots. Produces `calib.json`. |
| `rtsp_probe.py` | Reference implementation of the handshake and HYAV parse. Good for diagnosing new cameras. |
| `calib.example.json` | Schema reference for `calib.json`. The coefficients are from one unit — don't use them on yours. |

## The protocol, briefly

This is what `thermald.py` speaks. It was reverse-engineered from a
Wireshark capture of the vendor app (`IRFlash`) streaming the camera; no
official spec was used. Verified live on an ICI FMX 400P (MAC OUI
`90:74:9D` = IRay Technology / InfiRay).

1. Standard RTSP handshake (OPTIONS, DESCRIBE, SETUP×2, PLAY) against
   `rtsp://CAMERA:554/cam/realmonitor?channel=1&subtype=0`. **No authentication.**
2. Crucially, the `SETUP` transport line is Dahua's proprietary profile:
   `Transport: DH/AVP/TCP;unicast;interleaved=0-1;mode=play`. With
   standard `RTP/AVP/TCP`, the camera falls back to serving **lossy
   H.264 pseudocolor video** — which is what ffmpeg/VLC see. With
   `DH/AVP/TCP`, it streams raw radiometric frames.
3. After PLAY, the socket delivers a repeating binary chunk:
   ```
   $ 0x08  <4-byte BE length = 221224>              # 6-byte packet header
   HYAV <28-byte frame metadata>                    # 32-byte chunk header
   <110592 bytes: MSB plane, row-major 384×288>     # high byte of each pixel
   <110592 bytes: LSB plane, row-major 384×288>     # low byte of each pixel
   hyav <10-byte trailer>
   ```
   Per-pixel `raw16 = MSB[y,x]*256 + LSB[y,x]`, unsigned. The 28-byte
   metadata header turned out to be frame counter + timestamp + length —
   **no calibration constants embedded**. You have to calibrate yourself.
4. The 28-byte HYAV metadata has no calibration constants. The camera's
   uint16 output is roughly `raw = 90 * (T_K − 244)`, which maps the
   16-bit range onto the sensor's ~ −29 °C to +697 °C operating span.
   Use that as a ballpark only; unit-to-unit variation is several degrees.
5. RTSP-level keepalive (`GET_PARAMETER`) must be sent on the same socket
   every 30 s or the camera tears down the session after 60 s.
6. Only **one RTSP session at a time**. If thermald dies without sending
   TEARDOWN, the camera refuses new connections for up to ~60 s.

### Focus / lens control (port 36399)

The RTSP stream is read-only. Focus commands ride on a separate TCP port
(`36399`), using a tiny framed request/response protocol reverse-engineered
from a Wireshark capture of `IRFlash` and confirmed live.

Frame format (both directions):

```
7D FF 00 00 | AA | LEN | <body> | CHK | EB AA
  LEN = len(body) + 1   # covers body + CHK
  CHK = (0xAA + LEN + sum(body)) & 0xFF
```

The camera ACKs every command with the fixed 4-byte frame `7D FF 00 7B`.
Both observed commands belong to class `08` (lens/PTZ):

| command | body | effect |
|---|---|---|
| manual focus step | `08 21 01 02 01` — `{class=08, sub=21, ch=01, dir=02, speed=01}` | one motor increment per call |
| one-shot autofocus | `08 2F 01 00` — `{class=08, sub=2F, ch=01, mode=00}` | camera hunts on its own |

`thermal-focus` (and `POST /focus?cmd=step|auto` on the daemon) are the
thin wrappers around this. The step direction byte (`0x02`) is what the
vendor app uses; flipping it to reach the other direction is not yet
confirmed.

## Requirements

- Python 3.10+
- `numpy`, `Pillow`
- A camera that negotiates `DH/AVP/TCP` and emits the HYAV framing above

```bash
pip install numpy Pillow
```

## Running

```bash
python3 thermald.py --host 192.168.1.123 --port 8080
```

Open `http://<host>:8080/` in a browser — you'll see a live pseudocolor
preview. Mousing over the image shows the temperature at the pixel under
the cursor; hit **save snapshot** to dump the current frame to disk. The
**step** and **auto** focus buttons drive the camera's lens motor via the
port-36399 protocol described below.

### Command-line options

| flag | default | notes |
|---|---|---|
| `--host` | `192.168.1.123` | camera IP |
| `--port` | `8080` | HTTP bind port |
| `--bind` | `0.0.0.0` | HTTP bind address |
| `--calib` | `./calib.json` | path to calibration JSON. Pass `""` to disable and force the ballpark fallback (±2 °C). |

### Environment variables

- `THERMAL_SNAPSHOTS` — directory for saved snapshots (default `./snapshots`)

### HTTP endpoints

| method+path | description |
|---|---|
| `GET /` | Dashboard page with preview, mouseover, snapshot + focus buttons, calibration badge |
| `GET /preview.mjpg` | Live MJPEG stream (ironbow pseudocolor) |
| `GET /snapshot.tiff` | Most recent frame as 16-bit TIFF (no sidecar) |
| `POST /snapshot?name=…` | Save `<name>.tiff`, `<name>.png`, `<name>.json` to snapshot dir. Auto-names by timestamp if `name` omitted. |
| `POST /focus?cmd=step\|auto` | Send a lens-control command on port 36399. `step` = one manual focus step; `auto` = one-shot autofocus. Returns the framed bytes sent and the camera's ACK. |
| `GET /pixel?x=&y=` | JSON: `raw16`, `T_C`, `T_F`, `T_K`, `calibrated` |
| `GET /status` | JSON: readiness, frame stats, loaded calibration info |

### Snapshot file formats

Each call to `POST /snapshot` produces three files:

- `<name>.tiff` — 16-bit grayscale. **Raw sensor counts**, not temperatures.
  Calibration is never baked into saved files, so you can always re-fit
  later. `numpy.array(PIL.Image.open(path))` gives you a `uint16` array
  of shape `(288, 384)`.
- `<name>.png` — pseudocolor render (for human viewing only).
- `<name>.json` — sidecar with `seq`, timestamps, HYAV metadata hex, the
  loaded calibration, and min/mean/max °C using that calibration.

## Calibration

Without calibration, the daemon uses a FMX-shaped ballpark formula:

```
T_C ≈ raw16 / 90 − 29      (±2 °C, good enough for "is it hot?")
```

For anything where accuracy matters, use a black body at known setpoints.
The workflow:

1. Aim the camera at the black body in a stable position.
2. For each setpoint, wait ~30 s for the BB to settle, then:
   ```bash
   ./thermal-snap bb_25c
   ./thermal-snap bb_35c
   ./thermal-snap bb_50c
   ./thermal-snap bb_70c
   ```
3. Identify the BB region's pixel extents in the image (hover over the
   dashboard to find corners). Example: `x0=278, y0=122, x1=293, y1=135`.
4. Fit:
   ```bash
   ./thermal-calibrate \
       --roi 278,122,293,135 \
       snapshots/bb_25c.tiff 25.0 \
       snapshots/bb_35c.tiff 35.0 \
       snapshots/bb_50c.tiff 50.0 \
       snapshots/bb_70c.tiff 70.0 \
       --out calib.json
   ```
   Output reports RMSE, R², and per-point residuals. On a well-behaved
   camera you should see RMSE well under 1 °C for a linear fit across a
   reasonable range. If residuals look cubic (pos/neg/pos/neg or
   opposite), re-run with `--order 2`.
5. Restart `thermald.py`. It auto-loads `./calib.json`; the dashboard
   badge will turn green and show RMSE/R².

Notes:
- One-point calibration only fixes bias, not slope. Use at least two.
- Keep the BB in exactly the same pixel region between setpoints so the
  ROI is consistent.
- Don't stare the BB aperture into the camera's own IR window — specular
  reflections will corrupt the measurement.
- Recalibrate if you change optics, move to a wildly different ambient
  temperature, or update camera firmware.

## Architecture notes

- One background thread holds the RTSP session and parses frames into a
  shared `FrameStore` (latest-wins; clients read, no backlog).
- MJPEG requests are served from the same store — each subscriber gets
  every new frame once.
- Calibration is applied at read time, never to stored files.
- Clean SIGTERM/SIGINT path: signal → set stop event → close camera
  socket → join supervisor (sends RTSP TEARDOWN) → stop HTTP server. So
  `systemctl stop` / `kill` leaves the camera reusable immediately,
  instead of locked out for the 60 s session timeout.

## Running as a service

Tmux is the 10-second solution:

```bash
tmux new -d -s thermal 'cd /path/to/thermal && python3 thermald.py'
```

For a real service, a user systemd unit works fine — nothing in the
daemon needs root. (PRs welcome.)

## License

MIT. See `LICENSE`.

## Credits

Protocol reverse-engineered by observing `IRFlash` (ICI's Windows tool)
streaming an FMX 400P. HYAV framing and the DH/AVP/TCP transport hint
are Dahua conventions; not the author's invention.
