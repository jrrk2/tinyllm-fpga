#!/usr/bin/env python3
"""Direct poke of factor RAM — write a known wild value, confirm hidden_out
changes drastically.  If it doesn't, the override-RAM write path is broken."""
import socket, struct, sys, time

PEER = ("192.168.1.42", 19783)
NL = 30

_seq = [0]
def nxt():
    _seq[0] = (_seq[0] + 1) & 0xFF
    return _seq[0] or 1

def reg_write(s, addr, value):
    pkt = struct.pack("<BBB", 0x01, nxt(), 1) + b"\x00" * 5 + \
          struct.pack("<HIH", addr & 0xFFFF, value & 0xFFFFFFFF, 0)
    s.sendto(pkt, PEER)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(1.0)

# Write attn_factor = 0 to ALL layers (should completely zero out attention,
# making hidden_out = h_in only at every layer's scale).
print("zeroing attn_factor on every layer...", file=sys.stderr)
for L in range(NL):
    reg_write(s, 0x180 + L, 0)
    time.sleep(0.001)

# Also zero gate/up/mlp on all layers.
print("zeroing swiglu factors on every layer...", file=sys.stderr)
for L in range(NL):
    reg_write(s, 0x100 + L, 0)       # gate, up both 0
    reg_write(s, 0x140 + L, 0)       # mlp 0
    time.sleep(0.001)
print("done.  Now run host/fpga_per_layer_dump.py", file=sys.stderr)
