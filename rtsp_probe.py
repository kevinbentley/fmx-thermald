#!/usr/bin/env python3
"""One-shot: do the RTSP handshake against the camera and read one HYAV frame.
Prints summary stats to confirm the protocol works live."""
import socket, struct, sys, time
import numpy as np

HOST = '192.168.1.123'
PORT = 554
PATH = '/cam/realmonitor?channel=1&subtype=0'
URL = f'rtsp://{HOST}:{PORT}{PATH}'
W, H = 384, 288
PIX = W*H

def send(sock, method, cseq, session=None, url=URL, extra=()):
    lines = [f'{method} {url} RTSP/1.0',
             f'CSeq: {cseq}',
             'User-Agent: thermald-probe/1.0']
    if session: lines.append(f'Session: {session}')
    lines += list(extra)
    lines += ['', '']
    sock.sendall('\r\n'.join(lines).encode())

def recv_response(sock, buf):
    while b'\r\n\r\n' not in buf:
        d = sock.recv(4096)
        if not d: raise ConnectionError('closed')
        buf += d
    i = buf.index(b'\r\n\r\n') + 4
    headers = buf[:i].decode('latin-1')
    buf = buf[i:]
    clen = 0
    for line in headers.split('\r\n'):
        if line.lower().startswith('content-length:'):
            clen = int(line.split(':',1)[1].strip())
    while len(buf) < clen:
        d = sock.recv(4096)
        if not d: raise ConnectionError('closed')
        buf += d
    body = buf[:clen]
    buf = buf[clen:]
    return headers, body, buf

def read_exact(sock, buf, n):
    while len(buf) < n:
        d = sock.recv(max(4096, n - len(buf)))
        if not d: raise ConnectionError('closed')
        buf += d
    return buf[:n], buf[n:]

def main():
    s = socket.create_connection((HOST, PORT), timeout=10)
    buf = b''
    send(s, 'OPTIONS', 1)
    h, _, buf = recv_response(s, buf)
    print('OPTIONS ->', h.splitlines()[0])

    send(s, 'DESCRIBE', 2, extra=['Accept: application/sdp'])
    h, body, buf = recv_response(s, buf)
    print('DESCRIBE ->', h.splitlines()[0], f'SDP {len(body)} bytes')
    base = URL + '/'
    for line in h.split('\r\n'):
        if line.lower().startswith('content-base:'):
            base = line.split(':',1)[1].strip()

    send(s, 'SETUP', 3,
         url=base + 'trackID=0',
         extra=['Transport: DH/AVP/TCP;unicast;interleaved=0-1;mode=play'])
    h, _, buf = recv_response(s, buf)
    print('SETUP trackID=0 ->', h.splitlines()[0])
    session = None
    for line in h.split('\r\n'):
        if line.lower().startswith('session:'):
            session = line.split(':',1)[1].strip().split(';')[0]
    print('  session =', session)

    send(s, 'SETUP', 4, session=session,
         url=base + 'trackID=4',
         extra=['Transport: DH/AVP/TCP;unicast;interleaved=2-3;mode=play'])
    h, _, buf = recv_response(s, buf)
    print('SETUP trackID=4 ->', h.splitlines()[0])

    send(s, 'PLAY', 5, session=session, url=base, extra=['Range: npt=0.000-'])
    h, _, buf = recv_response(s, buf)
    print('PLAY ->', h.splitlines()[0])

    print('\nReading HYAV chunks...')
    t0 = time.time()
    for i in range(5):
        # expect '$' <chan> <4-byte BE len>
        hdr, buf = read_exact(s, buf, 6)
        if hdr[0] != 0x24:
            print(f'unexpected byte 0x{hdr[0]:02x}')
            return
        chan = hdr[1]
        body_len = struct.unpack('>I', hdr[2:6])[0]
        body, buf = read_exact(s, buf, body_len)
        magic = body[:4]
        md = body[4:32]
        data_size = 2*PIX
        data = body[32:32+data_size]
        trailer = body[32+data_size:]
        msb = np.frombuffer(data[:PIX], dtype=np.uint8).reshape(H,W)
        lsb = np.frombuffer(data[PIX:], dtype=np.uint8).reshape(H,W)
        raw = (msb.astype(np.uint16)<<8) | lsb.astype(np.uint16)
        t_c = raw.astype(np.float32)/16.0 - 273.15
        print(f'  frame {i}: chan={chan} len={body_len} magic={magic} '
              f'T_C min/mean/max = {t_c.min():.2f}/{t_c.mean():.2f}/{t_c.max():.2f}')
    print(f'\n5 frames in {time.time()-t0:.2f}s')

    # TEARDOWN cleanly
    send(s, 'TEARDOWN', 6, session=session, url=base)
    try:
        h, _, buf = recv_response(s, buf)
        print('TEARDOWN ->', h.splitlines()[0])
    except Exception:
        pass
    s.close()

if __name__ == '__main__':
    main()
