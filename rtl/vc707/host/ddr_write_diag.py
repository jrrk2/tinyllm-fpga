#!/usr/bin/env python3
"""Poll DDR3 write-path stage counters from the FPGA.

Reads regmap 0x019..0x01C every <interval> seconds and prints deltas, so
you can see exactly where the upload pipeline stalls:

  rx   = FT_DDR_WRITE frames accepted by eth_ctrl parser
  done = ddr_wr_req toggles (write dispatched into MIG)
  ack  = ddr_ack_sync1 toggles seen (MIG completed the write)
  tx   = FT_ACK frames dispatched back to host

Run in one terminal while `host/ddr_write.py` runs in another; compare the
counters' rate of progress.  If `rx` < host's `sent`, RX is dropping
frames (eth-side parser too slow / FIFO full).  If `tx` < `ack`, the
ACK frames are not making it onto the wire (TX FIFO full).  Etc.
"""
import argparse, socket, struct, time, sys

PEER         = ("192.168.1.42", 19783)
FT_REG_READ  = 0x02
FT_REG_RSP   = 0x03


def reg_read(s, addr, nwords=1, seq=1):
    body = struct.pack("<BBHBBBB", FT_REG_READ, seq & 0xFF,
                       addr & 0xFFFF, nwords & 0xFF, 0, 0, 0)
    s.sendto(body, PEER)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try: buf, _ = s.recvfrom(2048)
        except socket.timeout: continue
        if len(buf) < 8 or buf[0] != FT_REG_RSP or buf[1] != (seq & 0xFF):
            continue
        _, n = struct.unpack_from("<HB", buf, 2)
        return [struct.unpack_from("<I", buf, 8 + 4*i)[0] for i in range(n)]
    raise TimeoutError(f"no REG_RSP for 0x{addr:03x}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.5,
                    help="seconds between samples (default 0.5)")
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(1.0)

    bv = reg_read(s, 0x10F)[0]
    print(f"BUILD_VERSION = 0x{bv:08x}")

    print(f"{'time':>6}  {'rx':>10}  {'done':>10}  {'ack':>10}  {'tx':>10}  "
          f"{'rx/s':>8}  {'tx/s':>8}  rx-tx")
    print("-" * 90)

    t0 = time.time()
    last_t = t0
    last_rx = last_tx = 0
    while True:
        try:
            words = []
            for a in (0x019, 0x01A, 0x01B, 0x01C):
                words.append(reg_read(s, a, seq=(a & 0xFF))[0])
            rx, done, ack, tx = words
            now = time.time()
            dt = now - last_t
            rx_rate = (rx - last_rx) / dt if dt > 0 else 0
            tx_rate = (tx - last_tx) / dt if dt > 0 else 0
            elapsed = now - t0
            print(f"{elapsed:6.1f}  {rx:>10}  {done:>10}  {ack:>10}  {tx:>10}  "
                  f"{rx_rate:>8.0f}  {tx_rate:>8.0f}  {rx-tx:>+5}",
                  flush=True)
            last_t, last_rx, last_tx = now, rx, tx
            time.sleep(args.interval)
        except (KeyboardInterrupt, BrokenPipeError):
            break
        except TimeoutError as e:
            print(f"  poll timeout: {e}", file=sys.stderr)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
