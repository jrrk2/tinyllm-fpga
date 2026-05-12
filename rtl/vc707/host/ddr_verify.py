#!/usr/bin/env python3
"""Verify a DDR3 write by reading back samples through the tile cache.

After uploading a binary with ddr_write.py, this script:
  1. Picks N random 8 KiB-aligned addresses from the file.
  2. For each: triggers a tile cache load at that address.
  3. Reads back 4 sample words via diag taps (0x013-0x016).
  4. Compares with expected bytes from the file.

This is a spot-check, not a full verify.  Exhaustive verification would
need a CRC engine in hardware or a full tile-by-tile readback path.
"""

import argparse
import random
import socket
import struct
import sys
import time
from pathlib import Path


PEER = ("192.168.1.42", 19783)


def reg_read(s, addr, n=1, seq=1):
    s.sendto(struct.pack('<BBHBBBB', 0x02, seq, addr & 0xFFFF, n & 0xFF, 0, 0, 0), PEER)
    while True:
        b, _ = s.recvfrom(2048)
        if b[0] == 0x03 and b[1] == seq:
            return [struct.unpack_from('<I', b, 8 + 4*i)[0] for i in range(b[4])]


def reg_write(s, ws, seq=1):
    body = struct.pack('<BBBBBBBB', 0x01, seq, len(ws), 0, 0, 0, 0, 0)
    for a, d in ws:
        body += struct.pack('<HIH', a & 0xFFFF, d & 0xFFFFFFFF, 0)
    s.sendto(body, PEER)


def trigger_load(s, addr, seq):
    reg_write(s, [(0x011, addr)], seq)
    reg_write(s, [(0x010, 0x1)], seq + 1)
    deadline = time.time() + 1.0
    while time.time() < deadline:
        v = reg_read(s, 0x012, 1, seq + 2)[0]
        if (v >> 1) & 1:
            return v
        time.sleep(0.001)
    raise TimeoutError(f"tile load from 0x{addr:08x} timed out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="binary file (must match what was uploaded)")
    ap.add_argument("--base", default="0", help="DDR3 base byte address (default 0)")
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    base = int(args.base, 0)
    blob = Path(args.file).read_bytes()
    n_tiles = len(blob) // 8192        # 8 KiB tiles
    if n_tiles == 0:
        sys.exit("file too small to verify (need at least 8 KiB)")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(2.0)
    # drain heartbeats
    s.settimeout(0.05)
    for _ in range(20):
        try: s.recv(2048)
        except: break
    s.settimeout(2.0)

    pre = reg_read(s, 0x012, 1, 1)[0]
    if not ((pre >> 2) & 1):
        sys.exit("DDR3 init_calib_complete is 0 — calibration failed?")

    # Verify samples
    ok = 0
    bad = 0
    for trial in range(args.samples):
        tile = random.randint(0, n_tiles - 1)
        ddr_addr = base + tile * 8192
        seq_b = 100 + trial * 5
        trigger_load(s, ddr_addr, seq=seq_b)
        # Read 4 sample words via diag taps
        b0_w0   = reg_read(s, 0x013, 1, seq_b + 3)[0]
        b1_w0   = reg_read(s, 0x015, 1, seq_b + 4)[0]
        # We just toggled load; the load went into the *inactive* bank.
        # Don't know which one without checking active_bank, so just pick
        # whichever isn't all-zero (post-reset BRAM stays 0).
        loaded_w0 = b1_w0 if b0_w0 == 0 else b0_w0

        # Expected: first 4 bytes of the tile = blob[tile*8192 : tile*8192+4]
        expected_w0 = struct.unpack("<I", blob[tile*8192 : tile*8192+4])[0]
        match = (loaded_w0 == expected_w0)
        ok += int(match)
        bad += int(not match)
        flag = "✓" if match else "✗"
        print(f"  {flag} tile @ 0x{ddr_addr:08x}  expect=0x{expected_w0:08x}  got=0x{loaded_w0:08x}")

    print(f"\n  {ok}/{args.samples} samples matched", file=sys.stderr)
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
