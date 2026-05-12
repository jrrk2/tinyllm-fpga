#!/usr/bin/env python3
"""Read the FPGA rmsnorm_selftest result via UDP REG_READ and compare to the
host-computed reference in generated/rmsnorm_selftest_expected.txt.

Register map:
  0x110..0x12F : 32 × 32-bit words = 64 Q1.15 lanes (lane 2k in [15:0],
                 lane 2k+1 in [31:16] of word 0x110+k).
  0x130[0]     : rms_done flag
  0x131[0]     : write 1 to retrigger the selftest
  0x10F        : BUILD_VERSION (read first to confirm which firmware is running)
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
LANES        = 64

EXPECTED_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "generated", "rmsnorm_selftest_expected.txt"))


def read_expected():
    out = []
    with open(EXPECTED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            hex16, signed = line.split()[:2]
            out.append(int(signed))
    if len(out) != LANES:
        raise RuntimeError(f"expected {LANES} ref lines, got {len(out)}")
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true",
                    help="Write 0x131[0]=1 to retrigger before reading.")
    args = ap.parse_args()

    expected = read_expected()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(2.0)

    bv = reg_read(s, 0x10F, 1)[0]
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(bv))
    print(f"FPGA BUILD_VERSION = 0x{bv:08x}  ({ts})")

    if args.restart:
        print("retriggering rmsnorm_selftest (write 0x131[0]=1) ...")
        reg_write(s, 0x131, 1)
        time.sleep(0.05)

    # Read result: 32 words, but max per frame is 19 — split into two reads.
    words  = reg_read(s, 0x110, 19, seq=2)
    words += reg_read(s, 0x110 + 19, 32 - 19, seq=3)
    done   = reg_read(s, 0x130, 1, seq=4)[0] & 1
    if not done:
        print("WARNING: rms_done=0 — selftest may not have completed",
              file=sys.stderr)

    lanes = []
    for w in words:
        lo = w & 0xFFFF
        hi = (w >> 16) & 0xFFFF
        if lo & 0x8000: lo -= 0x10000
        if hi & 0x8000: hi -= 0x10000
        lanes.append(lo); lanes.append(hi)

    print(f"FPGA rmsnorm_selftest (done={done}):")
    fails = 0; worst = 0
    for l in range(LANES):
        diff = lanes[l] - expected[l]
        if abs(diff) > abs(worst): worst = diff
        tag  = "ok" if abs(diff) <= 1 else "FAIL"
        if abs(diff) > 1:
            print(f"  lane {l:2d}:  fpga={lanes[l]:+7d}   ref={expected[l]:+7d}  diff={diff:+5d}  {tag}")
            fails += 1
    print()
    if fails:
        print(f"FAIL: {fails}/{LANES} lanes mismatch (>1 LSB), worst diff {worst:+d}")
        sys.exit(1)
    print(f"PASS: rmsnorm_selftest on VC707 within ±1 LSB on all {LANES} lanes "
          f"(worst {worst:+d})")
    sys.exit(0)


if __name__ == "__main__":
    main()
