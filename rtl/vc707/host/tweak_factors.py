#!/usr/bin/env python3
"""Runtime-tune the per-layer SwiGLU + Attn-AV scale factors on the FPGA
without rebuilding the bitstream.

Address map (host word addresses, all 32-bit writes):
  0x100..0x11D : SwiGLU lo per layer (NL=30):
                 [31:16] = up_in_factor   (Q1.15)
                 [15:0]  = gate_in_factor (Q1.15)
  0x140..0x15D : mlp_out_factor per layer (Q16.8 in [23:0])
  0x180..0x19D : attn_factor    per layer (Q16.8 in [23:0])

Usage:
  # multiply every layer's mlp_out_factor by 0.25 (try a 4× knock-down):
  python3 host/tweak_factors.py --mlp-scale 0.25
  # then re-run inference:
  python3 host/fpga_per_layer_dump.py
  python3 host/decode_layer29.py
"""
import argparse, socket, struct, sys, time, re

PEER = ("192.168.1.42", 19783)
NL   = 30

_seq = [0]
def nxt():
    _seq[0] = (_seq[0] + 1) & 0xFF
    return _seq[0] or 1

def reg_write(s, addr, value):
    # Same packet format as snap_sel_probe.py — 5-byte header pad + addr16+data32+pad16.
    pkt = struct.pack("<BBB", 0x01, nxt(), 1) + b"\x00" * 5 + \
          struct.pack("<HIH", addr & 0xFFFF, value & 0xFFFFFFFF, 0)
    s.sendto(pkt, PEER)


def reg_read(s, addr):
    sq = nxt()
    body = struct.pack("<BBHBBBB", 0x02, sq, addr & 0xFFFF, 1, 0, 0, 0)
    s.sendto(body, PEER)
    deadline = time.time() + 0.5
    while time.time() < deadline:
        try: buf, _ = s.recvfrom(2048)
        except socket.timeout: continue
        if len(buf) < 8 or buf[0] != 0x03 or buf[1] != sq: continue
        return struct.unpack_from("<I", buf, 8)[0]
    return None


def read_factor(s, layer, kind):
    """kind: 0=SwiGLU lo (32-bit packed {up,gate}), 1=mlp_out, 2=attn."""
    reg_write(s, 0x00B, (kind << 5) | (layer & 0x1F))
    time.sleep(0.001)   # CDC settle
    return reg_read(s, 0x00F)


def parse_calibration():
    """Read tm_layer_swiglu_attn.svh to get the current calibrated factors."""
    factors = []
    attn = []
    with open("../generated/tm_layer_swiglu_attn.svh") as f:
        for ln in f:
            m = re.match(r"\s*64'h([0-9a-fA-F]+),?\s*//\s*L(\d+):", ln)
            if m:
                v = int(m.group(1), 16)
                factors.append({"gate": v & 0xFFFF,
                                "up":   (v >> 16) & 0xFFFF,
                                "mlp":  (v >> 32) & 0xFFFFFF})
                continue
            m = re.match(r"\s*24'h([0-9a-fA-F]+),?\s*//\s*L(\d+):", ln)
            if m:
                attn.append(int(m.group(1), 16))
    return factors, attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-scale", type=float, default=1.0,
                    help="Multiply every layer's gate_in_factor by this.")
    ap.add_argument("--up-scale",   type=float, default=1.0)
    ap.add_argument("--mlp-scale",  type=float, default=1.0,
                    help="Multiply every layer's mlp_out_factor by this — primary "
                         "MLP-amplitude tuning knob.")
    ap.add_argument("--attn-scale", type=float, default=1.0)
    ap.add_argument("--reset", action="store_true",
                    help="Restore all factors to calibrated defaults.")
    args = ap.parse_args()

    sw, at = parse_calibration()
    if len(sw) != NL or len(at) != NL:
        sys.exit(f"FATAL: expected NL={NL} entries, got sw={len(sw)} attn={len(at)}")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(1.0)

    if args.reset:
        gs = us = ms = atts = 1.0
        print("Restoring calibrated defaults.", file=sys.stderr)
    else:
        gs, us, ms, atts = args.gate_scale, args.up_scale, args.mlp_scale, args.attn_scale
        print(f"Applying scales: gate={gs} up={us} mlp={ms} attn={atts}", file=sys.stderr)

    for L in range(NL):
        g = max(0, min(0xFFFF, int(round(sw[L]["gate"] * gs))))
        u = max(0, min(0xFFFF, int(round(sw[L]["up"]   * us))))
        m = max(1, min(0xFFFFFF, int(round(sw[L]["mlp"]   * ms))))
        a = max(1, min(0xFFFFFF, int(round(at[L]        * atts))))
        # One factor type per round; sleep between every reg_write so the
        # CDC toggle handshake reliably captures each write (back-to-back
        # writes were lost in earlier tests, only the last few layers
        # actually updating).
        reg_write(s, 0x100 + L, (u << 16) | g); time.sleep(0.002)
        reg_write(s, 0x140 + L, m);             time.sleep(0.002)
        reg_write(s, 0x180 + L, a);             time.sleep(0.002)

    print(f"Wrote {NL} layers' factors.  Verifying via readback ...",
          file=sys.stderr)
    bad = 0
    for L in range(3):
        sw_lo = read_factor(s, L, 0)
        mlp   = read_factor(s, L, 1)
        attn  = read_factor(s, L, 2)
        g_got = sw_lo & 0xFFFF
        u_got = (sw_lo >> 16) & 0xFFFF
        m_got = mlp & 0xFFFFFF
        a_got = attn & 0xFFFFFF
        g_exp = max(0, min(0xFFFF, int(round(sw[L]["gate"] * gs))))
        u_exp = max(0, min(0xFFFF, int(round(sw[L]["up"]   * us))))
        m_exp = max(1, min(0xFFFFFF, int(round(sw[L]["mlp"]   * ms))))
        a_exp = max(1, min(0xFFFFFF, int(round(at[L]        * atts))))
        ok = (g_got == g_exp and u_got == u_exp and m_got == m_exp and a_got == a_exp)
        if not ok: bad += 1
        print(f"  L{L:2d}: gate {g_got}/{g_exp} up {u_got}/{u_exp} "
              f"mlp {m_got}/{m_exp} attn {a_got}/{a_exp}  "
              f"{'OK' if ok else 'MISMATCH'}", file=sys.stderr)
    if bad: print(f"  {bad}/{3} mismatches — runtime path broken.", file=sys.stderr)
    else:   print(f"  readback OK — run host/fpga_per_layer_dump.py", file=sys.stderr)

if __name__ == "__main__":
    main()
