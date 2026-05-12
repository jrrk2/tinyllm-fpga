#!/usr/bin/env python3
"""UDP host for vc707_microgpt_eth.

The FPGA listens at 192.168.1.42:19783 (UDP). It also responds to ARP
requests for that IP from MAC 02:00:00:4d:47:31 — so once the host can
ARP-resolve the FPGA, ordinary UDP just works.

No raw socket / no special privileges required.

Usage:
  ./microgpt_eth.py probe
  ./microgpt_eth.py gen --steps 15 --temperature 0.5 --seed 2
  ./microgpt_eth.py --fpga 192.168.1.42 --port 19783 probe
"""

import argparse
import socket
import struct
import sys
import time

DEFAULT_FPGA_IP   = "192.168.1.42"
DEFAULT_FPGA_PORT = 19783

FT_REG_WRITE = 0x01
FT_REG_READ  = 0x02
FT_REG_RSP   = 0x03
FT_HEARTBEAT = 0x04
FT_NAK       = 0x05
FT_ACK       = 0x06

# 32-bit register-word offsets — must match vc707_microgpt_eth.sv.
REG_MAGIC       = 0x000
REG_VERSION     = 0x001
REG_CTRL        = 0x002
REG_STATUS      = 0x003
REG_GEN_CFG     = 0x004
REG_SEED        = 0x005
REG_TOP_LOGIT   = 0x006
REG_OUT_BASE    = 0x018
REG_PERF_CYC    = 0x036
REG_LOGITS_BASE = 0x040


def open_socket(timeout: float = 0.5) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    return s


def reg_write(sock, peer, writes, seq: int = 0) -> None:
    """writes = [(addr16, data32), ...] (1..16 entries)."""
    assert 1 <= len(writes) <= 16
    # payload byte 0=type, 1=seq, 2=n, 3-7=pad, then n*8 bytes
    body = struct.pack("<BBBBBBBB", FT_REG_WRITE, seq, len(writes), 0, 0, 0, 0, 0)
    for addr, data in writes:
        body += struct.pack("<HIH", addr & 0xFFFF, data & 0xFFFFFFFF, 0)
    sock.sendto(body, peer)


def reg_read(sock, peer, start_addr: int, nwords: int, seq: int = 1):
    """Returns list of nwords ints. Blocks until REG_RSP arrives (or raises)."""
    assert 1 <= nwords <= 19
    # payload byte 0=type, 1=seq, 2-3=start_addr LE, 4=nwords, 5-7=pad
    body = struct.pack("<BBHBBBB", FT_REG_READ, seq,
                       start_addr & 0xFFFF, nwords & 0xFF, 0, 0, 0)
    sock.sendto(body, peer)

    deadline = time.time() + 1.0
    while time.time() < deadline:
        try:
            buf, src = sock.recvfrom(2048)
        except socket.timeout:
            continue
        if len(buf) < 8 or buf[0] != FT_REG_RSP or buf[1] != seq:
            continue
        r_start, r_n = struct.unpack_from("<HB", buf, 2)
        return [struct.unpack_from("<I", buf, 8 + 4 * i)[0] for i in range(r_n)]
    raise TimeoutError(f"no REG_RSP for {nwords} words @ {start_addr:#x}")


def cmd_probe(args, sock, peer):
    vals = reg_read(sock, peer, REG_MAGIC, 4)
    magic_ascii = bytes((vals[0] >> i) & 0xFF for i in (0, 8, 16, 24)).decode("ascii", "replace")
    print(f"  magic   = {vals[0]:#010x}  ({magic_ascii!r})")
    print(f"  version = {vals[1]:#010x}")
    print(f"  ctrl    = {vals[2]:#010x}")
    print(f"  status  = {vals[3]:#010x}")


def cmd_gen(args, sock, peer):
    cfg = ((args.temperature_q88 & 0xFFFF) << 16) | ((args.steps & 0xFF) << 8)
    reg_write(sock, peer, [
        (REG_GEN_CFG, cfg),
        (REG_SEED,    args.seed & 0xFFFFFFFF),
    ], seq=10)
    reg_write(sock, peer, [(REG_CTRL, 0x1)], seq=11)

    t0 = time.time()
    while time.time() - t0 < 5.0:
        s = reg_read(sock, peer, REG_STATUS, 1, seq=12)[0]
        if s & 0x4:
            break
        time.sleep(0.005)
    else:
        raise TimeoutError("generation didn't complete in 5s")

    out_len = (s >> 24) & 0xFF
    print(f"  status   = {s:#010x}  out_len={out_len}")
    tokens = reg_read(sock, peer, REG_OUT_BASE, max(1, out_len), seq=13)
    chars = "abcdefghijklmnopqrstuvwxyz"
    name = "".join(chars[t & 0x1F] for t in tokens if t < 26)
    print(f"  tokens   = {tokens}")
    print(f"  name     = {name!r}")


def cmd_listen(args, sock, peer):
    """Just print HEARTBEAT frames as they arrive."""
    sock.settimeout(2.0)
    print(f"listening for HEARTBEAT from {peer[0]}:{peer[1]} ...")
    try:
        while True:
            try:
                buf, src = sock.recvfrom(2048)
            except socket.timeout:
                continue
            if not buf or buf[0] != FT_HEARTBEAT:
                continue
            state, last_token, out_len, done_flags = struct.unpack_from("BBBB", buf, 2)
            uptime, = struct.unpack_from("<I", buf, 8)
            print(f"hb: state={state} last_token={last_token} out_len={out_len} "
                  f"flags={done_flags:#04x} uptime={uptime}s")
    except KeyboardInterrupt:
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fpga", default=DEFAULT_FPGA_IP, help=f"FPGA IP (default {DEFAULT_FPGA_IP})")
    p.add_argument("--port", type=int, default=DEFAULT_FPGA_PORT,
                   help=f"FPGA UDP port (default {DEFAULT_FPGA_PORT})")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe")
    sub.add_parser("listen")

    g = sub.add_parser("gen")
    g.add_argument("--steps", type=int, default=15)
    g.add_argument("--temperature", type=float, default=0.5)
    g.add_argument("--seed", type=lambda x: int(x, 0), default=2)

    args = p.parse_args()
    if hasattr(args, "temperature"):
        args.temperature_q88 = max(1, min(0xFFFF, int(round(args.temperature * 256))))

    sock = open_socket()
    peer = (args.fpga, args.port)

    if args.cmd == "probe":   cmd_probe(args, sock, peer)
    elif args.cmd == "gen":   cmd_gen(args, sock, peer)
    elif args.cmd == "listen": cmd_listen(args, sock, peer)


if __name__ == "__main__":
    main()
