#!/usr/bin/env python3
"""Write a per-row brom_SCALE_* entry at runtime via the Ethernet bus.

Protocol:
  reg 0x01C: stage {kind[3:0] << 16, addr[15:0]}
  reg 0x01D: write data (16-bit) — fires a one-cycle toggle on the eth side,
             CDC'd into core_clk, fires scale_wr_en.  smollm_layer routes
             to the brom whose kind matches; that brom writes its port B.

Kind mapping (matches smollm_layer.sv decode):
   0=Q  1=K  2=V  3=O  4=GATE  5=UP  6=DOWN  7=GAMMA1  8=GAMMA2

Address layout per matrix (entries are interleaved by layer):
   Q,O,DOWN:        layer*D + row   (D=576, NL=30, max=17280)
   K,V:             layer*H_KV*HD + row  (H_KV=3, HD=64, max=5760)
   GATE,UP:         layer*FFN + row (FFN=1536, max=46080)
   GAMMA1,GAMMA2:   layer*D + lane  (D=576, NL=30, max=17280)
"""
import argparse, socket, struct, sys, time

PEER = ("192.168.1.42", 19783)

KINDS = {"Q":0, "K":1, "V":2, "O":3, "GATE":4, "UP":5, "DOWN":6,
         "GAMMA1":7, "GAMMA2":8}

_seq = [0]
def nxt():
    _seq[0] = (_seq[0] + 1) & 0xFF
    return _seq[0] or 1

def reg_write(s, addr, value):
    pkt = struct.pack("<BBB", 0x01, nxt(), 1) + b"\x00" * 5 + \
          struct.pack("<HIH", addr & 0xFFFF, value & 0xFFFFFFFF, 0)
    s.sendto(pkt, PEER)

def scale_write(s, kind: str, addr: int, data: int):
    k = KINDS[kind.upper()]
    reg_write(s, 0x01C, (k << 16) | (addr & 0xFFFF))
    time.sleep(0.001)   # let addr+kind settle in eth-clk regs
    reg_write(s, 0x01D, data & 0xFFFF)  # triggers the toggle
    time.sleep(0.001)   # let CDC + write complete in core_clk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(KINDS.keys()))
    ap.add_argument("addr", type=lambda x: int(x, 0))
    ap.add_argument("data", type=lambda x: int(x, 0))
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(1.0)
    scale_write(s, args.kind, args.addr, args.data)
    print(f"wrote brom_SCALE_{args.kind.upper()}[0x{args.addr:04x}] = 0x{args.data:04x}", file=sys.stderr)

if __name__ == "__main__":
    main()
