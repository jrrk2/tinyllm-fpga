#!/usr/bin/env python3
"""Dump all 64 lanes of the FPGA's layer hidden_out (regmap 0x1D0..0x1EF)
as one signed-int per line.  Pipe to host/decode_hidden.py to see what
tokens the FPGA predicts."""
import socket, struct, sys

PEER = ("192.168.1.42", 19783)

def reg_read(s, addr, nwords=1, seq=1):
    body = struct.pack("<BBHBBBB", 0x02, seq & 0xFF, addr & 0xFFFF, nwords & 0xFF, 0, 0, 0)
    s.sendto(body, PEER)
    deadline = __import__("time").time() + 2.0
    while __import__("time").time() < deadline:
        try: buf, _ = s.recvfrom(2048)
        except socket.timeout: continue
        if len(buf) < 8 or buf[0] != 0x03 or buf[1] != (seq & 0xFF):
            continue
        _, n = struct.unpack_from("<HB", buf, 2)
        return [struct.unpack_from("<I", buf, 8 + 4*i)[0] for i in range(n)]

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(1.0)
words = []
for i in range(32):
    words += reg_read(s, 0x1D0 + i, seq=(i+1) & 0xFF)
for w in words:
    lo = w & 0xFFFF; hi = (w >> 16) & 0xFFFF
    if lo & 0x8000: lo -= 0x10000
    if hi & 0x8000: hi -= 0x10000
    print(lo); print(hi)
