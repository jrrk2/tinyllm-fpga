#!/usr/bin/env python3
"""End-to-end smoke test for the BFP autoregress bitstream
(USE_BFP=1 BFP_STREAM=1).

Sequence:
  1. (optional) upload generated/lbfp_full_DDR3.bin to DDR3 via ddr_write.
  2. Pulse restart  (write 1 to 0x1F1)
  3. Poll done bit  (read 0x1F0)
  4. Read 10 × 32-bit result words from 0x1D0 → 19 × 16-bit token IDs
  5. Print "RTL_TOKENS: …" and decode via decode_rtl_tokens.py

Pass --skip-upload if DDR3 is already loaded with the current weight image.

Usage:
    python3 host/bfp_smoke_test.py
    python3 host/bfp_smoke_test.py --skip-upload
    python3 host/bfp_smoke_test.py --peer 192.168.1.42:19783
"""
import argparse, os, socket, struct, subprocess, sys, time
from pathlib import Path

PEER         = ("192.168.1.42", 19783)
REG_RESTART  = 0x1F1
REG_DONE     = 0x1F0
REG_RESULT   = 0x1D0
N_PROMPT     = 4
N_GEN        = 15
N_STEPS      = N_PROMPT + N_GEN          # 19
NWORDS_TOKENS = (N_STEPS * 16 + 31) // 32  # 10 words covers 19×16 bits


def reg_read(s, addr, nwords=1, seq=1):
    body = struct.pack("<BBHBBBB", 0x02, seq & 0xFF, addr & 0xFFFF,
                       nwords & 0xFF, 0, 0, 0)
    s.sendto(body, PEER)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try: buf, _ = s.recvfrom(2048)
        except socket.timeout: continue
        if len(buf) < 8 or buf[0] != 0x03 or buf[1] != (seq & 0xFF): continue
        _, n = struct.unpack_from("<HB", buf, 2)
        return [struct.unpack_from("<I", buf, 8 + 4*i)[0] for i in range(n)]
    raise TimeoutError(f"no REG_RSP for 0x{addr:03x}")


def reg_write(s, addr, value, seq=1):
    pkt = struct.pack("<BBB", 0x01, seq & 0xFF, 1) + \
          struct.pack("<HI", addr & 0xFFFF, value & 0xFFFFFFFF)
    s.sendto(pkt, PEER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", default="192.168.1.42:19783",
                    help="FPGA host:port")
    ap.add_argument("--skip-upload", action="store_true",
                    help="DDR3 already has the weight image")
    ap.add_argument("--bin",
                    default=str(Path(__file__).resolve().parent.parent /
                                "generated" / "lbfp_full_DDR3.bin"),
                    help="path to lbfp_full_DDR3.bin")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="seconds to wait for `done`")
    args = ap.parse_args()

    global PEER
    host, port = args.peer.split(":")
    PEER = (host, int(port))

    here = Path(__file__).resolve().parent

    # 1. Upload DDR3 image
    if not args.skip_upload:
        if not os.path.exists(args.bin):
            sys.exit(f"FAIL: {args.bin} not found.  Run:\n"
                     f"  make -C {here.parent}    "
                     f"# to bake the .bin via gen_smollm_blockfp_ddr.py")
        print(f"[smoke] uploading {args.bin} ({os.path.getsize(args.bin)/1024/1024:.1f} MB) "
              f"to DDR3 @ {PEER[0]} …", file=sys.stderr)
        subprocess.check_call([sys.executable, str(here / "ddr_write.py"),
                               args.bin])
    else:
        print("[smoke] --skip-upload set, assuming DDR3 already loaded",
              file=sys.stderr)

    # 2. Connect + identify build
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(1.0)
    try:
        bv = reg_read(s, 0x10F)[0]
        print(f"[smoke] FPGA BUILD_VERSION = 0x{bv:08x}", file=sys.stderr)
    except TimeoutError:
        sys.exit(f"FAIL: no response from FPGA at {PEER}")

    # 3. Restart pulse
    reg_write(s, REG_RESTART, 1, seq=2)
    print(f"[smoke] restart pulsed, waiting up to {args.timeout:.0f}s for done …",
          file=sys.stderr)
    time.sleep(0.05)

    # 4. Poll done bit
    deadline = time.time() + args.timeout
    last_v = None
    while time.time() < deadline:
        v = reg_read(s, REG_DONE, seq=3)[0]
        if v != last_v:
            print(f"  REG_DONE = 0x{v:08x}", file=sys.stderr)
            last_v = v
        if v & 1:
            elapsed = args.timeout - (deadline - time.time())
            print(f"[smoke] done @ t≈{elapsed:.2f}s", file=sys.stderr)
            break
        time.sleep(0.2)
    else:
        sys.exit("FAIL: FPGA never asserted done within timeout")

    # 5. Read packed tokens
    words = reg_read(s, REG_RESULT, nwords=NWORDS_TOKENS, seq=4)
    if len(words) != NWORDS_TOKENS:
        sys.exit(f"FAIL: read {len(words)} words, expected {NWORDS_TOKENS}")

    tokens = []
    bits = 0
    accum = 0
    for w in words:
        accum |= (w << bits)
        bits += 32
        while bits >= 16 and len(tokens) < N_STEPS:
            tokens.append(accum & 0xFFFF)
            accum >>= 16
            bits  -= 16

    print(f"\nRTL_TOKENS: " + " ".join(str(t) for t in tokens))

    # 6. Decode (best-effort — needs lbfp_full_PROMPT_TOKENS.txt + transformers)
    decode = here / "decode_rtl_tokens.py"
    if decode.exists():
        # Pipe via a temp log so decode_rtl_tokens can find the RTL_TOKENS line.
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as tf:
            tf.write("RTL_TOKENS: " + " ".join(str(t) for t in tokens) + "\n")
            log_path = tf.name
        try:
            subprocess.call([sys.executable, str(decode), log_path])
        finally:
            os.unlink(log_path)


if __name__ == "__main__":
    main()
