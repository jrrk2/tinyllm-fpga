#!/usr/bin/env python3
"""Unified verifier for the FPGA selftests (matvec, rmsnorm, rope, swiglu, softmax).

Reads BUILD_VERSION first, optionally retriggers the chosen selftest, then
reads the result registers and compares to the host-computed reference.

Usage:
    python3 host/selftest_verify.py rmsnorm
    python3 host/selftest_verify.py rope --restart
    python3 host/selftest_verify.py all              # run all five
"""
import argparse
import os
import socket
import struct
import sys
import time

PEER         = ("192.168.1.42", 19783)
FT_REG_WRITE = 0x01
FT_REG_READ  = 0x02
FT_REG_RSP   = 0x03

# op -> (result_base, n_words, done_addr, restart_addr, expected_file, tolerance, n_lanes)
OPS = {
    "matvec":  (0x100,  8, 0x108, 0x109, "matvec_selftest_expected.txt",   0, 16),
    "rmsnorm": (0x110, 32, 0x130, 0x131, "rmsnorm_selftest_expected.txt",  1, 64),
    "rope":    (0x140, 32, 0x160, 0x161, "rope_selftest_expected.txt",     1, 64),
    "swiglu":  (0x170, 32, 0x190, 0x191, "swiglu_selftest_expected.txt",   1, 64),
    "softmax": (0x1A0, 32, 0x1C0, 0x1C1, "softmax_selftest_expected.txt", 10, 64),
    # Multilayer (NL=30 SmolLM2) block-FP architecture: 64 lanes ×
    # 16-bit Q1.15 (at the LAST layer's h_out_p2 scale, packed 2 lanes
    # per regmap word).  Tolerance is wide because cumulative noise
    # across 30 quantized layers is significant — text validation comes
    # from decoding through lm_head, not strict lane comparison.
    "layer":   (0x1D0, 32, 0x1F0, 0x1F1, "tm_layer_expected.txt", 5000, 64),
}


def read_expected(name, n_lanes):
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "generated", name))
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            out.append(int(parts[1]))     # signed decimal
    if len(out) < n_lanes:
        raise RuntimeError(f"{name}: expected at least {n_lanes} lines, got {len(out)}")
    # Truncate if reference has more lanes than the FPGA emits — common when
    # the regmap window is narrower than the layer's full hidden_out (e.g.,
    # SmolLM2 D=576 layer, FPGA exports first 64 lanes via 0x1D0..0x1EF).
    return out[:n_lanes]


def reg_read(s, addr, nwords=1, seq=1):
    body = struct.pack("<BBHBBBB", FT_REG_READ, seq & 0xFF,
                       addr & 0xFFFF, nwords & 0xFF, 0, 0, 0)
    s.sendto(body, PEER)
    deadline = time.time() + 1.0
    while time.time() < deadline:
        try: buf, _ = s.recvfrom(2048)
        except socket.timeout: continue
        if len(buf) < 8 or buf[0] != FT_REG_RSP or buf[1] != (seq & 0xFF):
            continue
        _, n = struct.unpack_from("<HB", buf, 2)
        return [struct.unpack_from("<I", buf, 8 + 4*i)[0] for i in range(n)]
    raise TimeoutError(f"no REG_RSP for 0x{addr:03x} nwords={nwords}")


def reg_write(s, addr, value, seq=1):
    pkt  = struct.pack("<BBB", FT_REG_WRITE, seq & 0xFF, 1)
    pkt += b"\x00" * 5
    pkt += struct.pack("<HIH", addr & 0xFFFF, value & 0xFFFFFFFF, 0)
    s.sendto(pkt, PEER)


def read_words_chunked(s, base, n, seq_base=10):
    """Reg_read auto-splits since MAX_REG_READS=19."""
    out = []
    chunk = 19
    seq = seq_base
    a = base
    rem = n
    while rem > 0:
        m = min(chunk, rem)
        out += reg_read(s, a, m, seq=seq)
        a += m; rem -= m; seq += 1
    return out


def words_to_lanes(words, n_lanes, lane_bits=16):
    """16-bit  → 2 lanes per word; 24-bit (sign-ext to 32-bit) → 1 lane per word."""
    lanes = []
    if lane_bits == 16:
        for w in words:
            lo = w & 0xFFFF; hi = (w >> 16) & 0xFFFF
            if lo & 0x8000: lo -= 0x10000
            if hi & 0x8000: hi -= 0x10000
            lanes.append(lo); lanes.append(hi)
    else:
        # 24-bit Q15.9 sign-extended into 32-bit
        for w in words:
            v = w & 0xFFFFFFFF
            if v & 0x80000000: v -= 0x100000000
            lanes.append(v)
    return lanes[:n_lanes]


def run_one(s, op, do_restart):
    base, nw, done_addr, restart_addr, exp_file, tol, n_lanes = OPS[op]
    expected = read_expected(exp_file, n_lanes)

    if do_restart:
        print(f"  retriggering {op} (write 0x{restart_addr:03x}[0]=1) ...")
        reg_write(s, restart_addr, 1, seq=2)
        time.sleep(0.05)

    words = read_words_chunked(s, base, nw, seq_base=10)
    done  = reg_read(s, done_addr, 1, seq=20)[0] & 1
    # All ops pack 2 lanes per regmap word (16-bit Q1.15 each).
    lanes = words_to_lanes(words, n_lanes, lane_bits=16)

    fails = 0; worst = 0
    for l in range(n_lanes):
        diff = lanes[l] - expected[l]
        if abs(diff) > abs(worst): worst = diff
        if abs(diff) > tol:
            print(f"    lane {l:2d}:  fpga={lanes[l]:+7d}   ref={expected[l]:+7d}   diff={diff:+5d}  FAIL")
            fails += 1
    if fails == 0 and done:
        print(f"  PASS: {op:7s}  {n_lanes} lanes within ±{tol} LSB  (worst {worst:+d})  done={done}")
        return True
    else:
        print(f"  FAIL: {op:7s}  {fails}/{n_lanes} mismatched  worst {worst:+d}  done={done}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=list(OPS.keys()) + ["all"])
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(2.0)

    bv = reg_read(s, 0x10F, 1)[0]
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(bv))
    print(f"FPGA BUILD_VERSION = 0x{bv:08x}  ({ts})")

    ops = list(OPS.keys()) if args.op == "all" else [args.op]
    all_ok = True
    for op in ops:
        if not run_one(s, op, args.restart):
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
