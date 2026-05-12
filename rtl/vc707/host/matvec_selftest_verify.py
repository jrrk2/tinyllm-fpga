#!/usr/bin/env python3
"""Read the FPGA matvec_selftest result via UDP REG_READ and compare to the
host-computed reference in generated/matvec_selftest_expected.txt.

Reports the BUILD_VERSION at register 0x10F so we know which firmware is
actually running on the board.  With --restart, writes 0x109[0]=1 to
retrigger the selftest before reading the result."""
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
LANES        = 16

EXPECTED_PATH = os.path.join(os.path.dirname(__file__), "..", "generated",
                             "matvec_selftest_expected.txt")
EXPECTED_PATH = os.path.abspath(EXPECTED_PATH)


def read_expected():
    out = []
    with open(EXPECTED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            hex16, signed = line.split()[:2]
            out.append(int(signed))
    if len(out) != LANES:
        raise RuntimeError(f"expected {LANES} reference lines, got {len(out)}")
    return out


def reg_read(s, addr, nwords=1, seq=1):
    """REG_READ: 8-byte header (type, seq, addr LE u16, nwords, 3 pad).
       REG_RSP:  8-byte header + nwords × LE u32 starting at byte 8.
       Skips stray ACK / heartbeat frames left in the recv queue."""
    body = struct.pack("<BBHBBBB", FT_REG_READ, seq & 0xFF,
                       addr & 0xFFFF, nwords & 0xFF, 0, 0, 0)
    s.sendto(body, PEER)
    deadline = time.time() + 1.0
    while time.time() < deadline:
        try:
            buf, _ = s.recvfrom(2048)
        except socket.timeout:
            continue
        if len(buf) < 8 or buf[0] != FT_REG_RSP or (buf[1] != (seq & 0xFF)):
            continue   # ACK / heartbeat / unrelated reply
        _, n = struct.unpack_from("<HB", buf, 2)
        return [struct.unpack_from("<I", buf, 8 + 4*i)[0] for i in range(n)]
    raise TimeoutError(f"no REG_RSP for addr=0x{addr:03x} nwords={nwords}")


def reg_write(s, addr, value, seq=1):
    """FT_REG_WRITE format (mirrors host/microgpt_eth.py):
       byte 0   = type (0x01)
       byte 1   = seq
       byte 2   = n_writes (1..16)
       bytes 3..7 = pad
       bytes 8.. = n × {addr16 LE, data32 LE, pad16}"""
    pkt = struct.pack("<BBB", FT_REG_WRITE, seq & 0xFF, 1)
    pkt += b"\x00" * 5                      # pad bytes 3..7
    pkt += struct.pack("<HIH", addr & 0xFFFF, value & 0xFFFFFFFF, 0)
    s.sendto(pkt, PEER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true",
                    help="Write 0x109[0]=1 to retrigger selftest before reading.")
    args = ap.parse_args()

    expected = read_expected()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(2.0)

    # 1. Build version
    try:
        bv = reg_read(s, 0x10F, nwords=1)[0]
        epoch = bv
        ts    = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))
        print(f"FPGA BUILD_VERSION = 0x{bv:08x}  ({ts})")
    except Exception as e:
        print(f"WARN: could not read BUILD_VERSION (0x10F): {e}", file=sys.stderr)

    # 2. Optional restart
    if args.restart:
        print("retriggering selftest (write 0x109[0]=1) ...")
        reg_write(s, 0x109, 0x00000001)
        time.sleep(0.05)   # FSM finishes in ~70 cycles; plenty of time at 125 MHz

    # 3. Read result + done flag
    words = reg_read(s, 0x100, nwords=9)
    done  = words[8] & 1
    if not done:
        print(f"WARNING: mvst_done=0 — selftest may not have completed", file=sys.stderr)

    lanes = []
    for i in range(8):
        lo = words[i] & 0xFFFF
        hi = (words[i] >> 16) & 0xFFFF
        if lo & 0x8000: lo -= 0x10000
        if hi & 0x8000: hi -= 0x10000
        lanes.append(lo)
        lanes.append(hi)

    print(f"FPGA result (done={done}):")
    fails = 0
    for l in range(LANES):
        diff = lanes[l] - expected[l]
        tag  = "ok" if diff == 0 else "FAIL"
        print(f"  lane {l:2d}:  fpga={lanes[l]:+7d}   ref={expected[l]:+7d}   "
              f"diff={diff:+5d}   {tag}")
        if diff != 0: fails += 1

    if fails == 0:
        print(f"\nPASS: matvec_int8_engine on VC707 produced bit-exact results "
              f"on all {LANES} lanes")
        sys.exit(0)
    else:
        print(f"\nFAIL: {fails}/{LANES} lanes mismatch")
        sys.exit(1)


if __name__ == "__main__":
    main()
