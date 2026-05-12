#!/usr/bin/env python3
"""Trigger one FPGA forward pass, then read the per-layer snapshot for
every layer 0..NL-1 and save each as fpga_layer_NN.txt.  Diff each
against a Python sim reference to localise the first divergent layer.

Workflow:
  1. trigger forward pass (REG_RESTART)
  2. wait for done
  3. for each layer:
       write snapshot_layer_sel (REG_SNAP_SEL = 0x00A)
       sleep enough for CDC settle (~1ms is plenty)
       read 288 words from 0x200..0x31F → 576 Q1.15 lanes

Output: fpga_layer_00.txt … fpga_layer_29.txt in CWD.
"""
import os, socket, struct, sys, time

PEER  = ("192.168.1.42", 19783)
NL    = 30

REG_RESTART  = 0x1F1
REG_ML_STATE = 0x017
REG_SNAP_SEL = 0x00A
REG_RESULT   = 0x200


_seq_ctr = [0]
def _next_seq():
    _seq_ctr[0] = (_seq_ctr[0] + 1) & 0xFF
    return _seq_ctr[0] or 1

def reg_read(s, addr, nwords=1, seq=None, retries=3):
    for attempt in range(retries):
        sq = (seq if seq is not None else _next_seq()) & 0xFF
        body = struct.pack("<BBHBBBB", 0x02, sq, addr & 0xFFFF,
                           nwords & 0xFF, 0, 0, 0)
        s.sendto(body, PEER)
        deadline = time.time() + 0.5
        while time.time() < deadline:
            try: buf, _ = s.recvfrom(2048)
            except socket.timeout: continue
            if len(buf) < 8 or buf[0] != 0x03 or buf[1] != sq: continue
            _, n = struct.unpack_from("<HB", buf, 2)
            return [struct.unpack_from("<I", buf, 8 + 4*i)[0] for i in range(n)]
    raise TimeoutError(f"no REG_RSP for 0x{addr:03x} after {retries} attempts")


def reg_write(s, addr, value, seq=1):
    # FT_REG_WRITE format (eth_ctrl line 18):
    #   [0]=op [1]=seq [2]=n [3..7]=pad [8..]=n*{addr16, data32, pad16}
    pkt = struct.pack("<BBB", 0x01, seq & 0xFF, 1) + b"\x00" * 5 + \
          struct.pack("<HIH", addr & 0xFFFF, value & 0xFFFFFFFF, 0)
    s.sendto(pkt, PEER)


def read_576_lanes(s, seq_base=None):
    words = []
    for i in range(288):
        words += reg_read(s, REG_RESULT + i)
    lanes = []
    for w in words:
        lo = w & 0xFFFF; hi = (w >> 16) & 0xFFFF
        if lo & 0x8000: lo -= 0x10000
        if hi & 0x8000: hi -= 0x10000
        lanes.append(lo); lanes.append(hi)
    return lanes


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(1.0)

    print("triggering forward pass …", file=sys.stderr)
    reg_write(s, REG_RESTART, 1, seq=2)
    time.sleep(0.05)

    deadline = time.time() + 30
    while time.time() < deadline:
        v = reg_read(s, REG_ML_STATE, seq=3)[0]
        if (v >> 8) & 1: break
        time.sleep(0.05)
    else:
        print("WARNING: FSM never reported done", file=sys.stderr)

    for L in range(NL):
        reg_write(s, REG_SNAP_SEL, L, seq=_next_seq())
        time.sleep(0.010)  # CDC + refresh settle (~12 µs needed, 10 ms is plenty)
        lanes = read_576_lanes(s)
        path = f"fpga_layer_{L:02d}.txt"
        with open(path, "w") as f:
            f.write(f"# FPGA hidden_state after layer {L}, 576 lanes Q1.15 signed dec\n")
            for v in lanes:
                f.write(f"{v}\n")
        nz = sum(1 for x in lanes if x != 0)
        print(f"  layer {L:2d}: {nz}/576 non-zero, range [{min(lanes)}..{max(lanes)}]  → {path}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
