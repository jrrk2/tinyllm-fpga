#!/usr/bin/env python3
"""Dump the FPGA's per-layer hidden state to localise the first divergent layer.

The on-chip snapshot (smollm_multilayer_tm_bfp) captures ONE layer's hidden_out
per run: the layer whose index == snapshot_layer_sel (reg 0x00A) at token
position == snapshot_step_sel (reg 0x00B).  So this tool re-runs the autoregress
once per layer, then reads the captured hidden back via wr_kind 12 (mantissa) /
13 (per-tile exponent) through the BRAM peek port (0x060 target / 0x062 read).

This matches the ~1-BRAM-budget design: a single 1-layer snapshot buffer, host
re-runs per layer.  Slower than a one-pass dump, but it fits and is correct.

Usage:  python3 fpga_per_layer_dump.py [--nl 32] [--d 960] [--step 0]
        (smollm360 defaults; use --nl 30 --d 576 for smollm135)
Output: fpga_layer_00.txt … fpga_layer_<NL-1>.txt
        D signed Q1.15 mantissas (one per line) + NT_D tile exponents in a footer.
"""
import argparse, socket, struct, sys, time

PEER = ("192.168.1.42", 19783)

REG_SNAP_SEL    = 0x00A   # per-layer snapshot: layer select
REG_STEP_SEL    = 0x064   # per-layer snapshot: token-step select (0x00B was taken)
REG_RESTART     = 0x1F1   # pulse [0]=1 to re-run the autoregress
REG_DONE        = 0x1F0   # {.., lay_done_latched(bit1), lay_done(bit0)}
REG_BRAM_TARGET = 0x060   # write {inc[31], kind[22:18], addr[17:0]}
REG_BRAM_READ   = 0x062   # registered port-A read-back

SNAP_KIND_M = 12          # snap_m (mantissa) read kind
SNAP_KIND_E = 13          # snap_e (per-tile exponent) read kind

_seq = [0]
def nseq():
    _seq[0] = (_seq[0] + 1) & 0xFF
    return _seq[0] or 1

def reg_read(s, addr, retries=4):
    for _ in range(retries):
        sq = nseq()
        s.sendto(struct.pack("<BBHBBBB", 0x02, sq, addr & 0xFFFF, 1, 0, 0, 0), PEER)
        deadline = time.time() + 0.5
        while time.time() < deadline:
            try: buf, _a = s.recvfrom(2048)
            except socket.timeout: continue
            if len(buf) < 12 or buf[0] != 0x03 or buf[1] != sq: continue
            return struct.unpack_from("<I", buf, 8)[0]
    raise TimeoutError(f"no REG_RSP for 0x{addr:03x}")

def reg_write(s, addr, value):
    s.sendto(struct.pack("<BBB", 0x01, nseq(), 1) + b"\x00"*5 +
             struct.pack("<HIH", addr & 0xFFFF, value & 0xFFFFFFFF, 0), PEER)

def read_bram(s, kind, addr):
    target = ((kind & 0x1f) << 18) | (addr & 0x3ffff)   # inc=0
    reg_write(s, REG_BRAM_TARGET, target)
    return reg_read(s, REG_BRAM_READ) & 0xFFFF

def s16(v): return v - 0x10000 if v & 0x8000 else v
def s8(v):  return v - 0x100   if v & 0x80   else v

def wait_done(s, timeout=30):
    # Let the just-issued restart clear the latched-done from the prior run,
    # then poll until it re-asserts.
    time.sleep(0.05)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if reg_read(s, REG_DONE) & 0x2:
            return True
        time.sleep(0.02)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl",   type=int, default=32, help="number of layers")
    ap.add_argument("--d",    type=int, default=960, help="hidden dim D")
    ap.add_argument("--step", type=int, default=0,  help="token-step (pos) to snapshot")
    args = ap.parse_args()
    NL, D, STEP = args.nl, args.d, args.step
    NT_D = D // 16

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(0.5)
    print(f"per-layer dump: NL={NL} D={D} step={STEP} (re-running per layer)", file=sys.stderr)

    for L in range(NL):
        reg_write(s, REG_SNAP_SEL, L)
        reg_write(s, REG_STEP_SEL, STEP)
        time.sleep(0.005)               # CDC settle for the sels before restart
        reg_write(s, REG_RESTART, 1)    # re-run with this (layer, step) selected
        if not wait_done(s):
            print(f"  layer {L}: WARNING run never reported done", file=sys.stderr)
        mant = [s16(read_bram(s, SNAP_KIND_M, i)) for i in range(D)]
        expo = [s8(read_bram(s, SNAP_KIND_E, i)) for i in range(NT_D)]
        path = f"fpga_layer_{L:02d}.txt"
        with open(path, "w") as f:
            f.write(f"# FPGA hidden after layer {L}, step {STEP}: "
                    f"{D} Q1.15 mantissas, then {NT_D} tile exponents\n")
            for v in mant:
                f.write(f"{v}\n")
            f.write("# tile exponents\n")
            for v in expo:
                f.write(f"# e {v}\n")
        nz = sum(1 for x in mant if x != 0)
        print(f"  layer {L:2d}: {nz}/{D} non-zero, range [{min(mant)}..{max(mant)}]  -> {path}",
              file=sys.stderr)

if __name__ == "__main__":
    main()
