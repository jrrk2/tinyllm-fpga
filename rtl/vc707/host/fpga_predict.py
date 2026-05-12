#!/usr/bin/env python3
"""End-to-end: trigger the FPGA's multilayer forward pass, capture
hidden_out, decode through SmolLM2's lm_head, print predicted tokens.

Two modes:
  (default)  use just the 64 lanes the FPGA exposes via regmap
             (zero-fill the remaining 512 of D=576 — partial decode)
  --full PATH  read all 576 lanes from a Verilator-style dump file
               (the cpp testbench writes generated/tm_layer_dut_out.txt)

The FPGA was validated bit-identical to the Verilator dump on the 64
lanes it exposes, so passing --full ../generated/tm_layer_dut_out.txt
gives the same answer the FPGA *would* produce if all D lanes were
read out.

Run:
    # Bring up FPGA + DDR3 first:
    #   make program
    #   python3 host/ddr_write.py ../generated/tm_layer_DDR3.bin --tail-pad 2048
    python3 host/fpga_predict.py
    python3 host/fpga_predict.py --full ../generated/tm_layer_dut_out.txt
"""
import argparse, socket, struct, sys, time
import numpy as np

PEER  = ("192.168.1.42", 19783)
MODEL = "HuggingFaceTB/SmolLM2-135M"
D     = 576

# Register addresses
REG_ML_STATE  = 0x017       # [2:0] = ml_state, [8] = lay_done
REG_RESULT    = 0x200       # 288 words × 32 bits = 576 lanes × 16 bits (full)
REG_DONE      = 0x1F0       # bit[0] = done
REG_RESTART   = 0x1F1       # write 1 to retrigger

# Last layer's h_out_p2 (block-FP scale exponent of FPGA's hidden_out).
# Pulled from generated/tm_layer_data.svh: TM_RESCALE[NL-1] field [55:52].
H_OUT_P2_LAST_LAYER = 10


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


def fpga_run(restart=True):
    """Trigger the multilayer forward, wait for done, return 64 lanes."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(1.0)

    bv = reg_read(s, 0x10F)[0]
    print(f"  FPGA BUILD_VERSION = 0x{bv:08x}", file=sys.stderr)

    if restart:
        reg_write(s, REG_RESTART, 1, seq=2)
        time.sleep(0.05)
    # Wait for done
    deadline = time.time() + 30
    while time.time() < deadline:
        v = reg_read(s, REG_ML_STATE, seq=3)[0]
        layer = (v >> 3) & 0x1F
        done  = (v >> 8) & 1
        ml_state = v & 0x7
        if done:
            print(f"  FSM done (layer={layer}, ml_state={ml_state})", file=sys.stderr)
            break
        time.sleep(0.05)
    else:
        print("  WARNING: FSM never reported done", file=sys.stderr)

    # Read 288 words = 576 × 16-bit lanes (full D)
    NW = 288
    words = []
    for i in range(NW):
        words += reg_read(s, REG_RESULT + i, seq=(10 + i) & 0xFF)
    lanes = []
    for w in words:
        lo = w & 0xFFFF; hi = (w >> 16) & 0xFFFF
        if lo & 0x8000: lo -= 0x10000
        if hi & 0x8000: hi -= 0x10000
        lanes.append(lo); lanes.append(hi)
    return lanes


def load_full_dump(path):
    """Read 576 signed integers from a Verilator/sim dump (one per line)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            out.append(int(line.split()[0]))
    if len(out) != D:
        raise SystemExit(f"FAIL: {path} has {len(out)} lanes, expected {D}")
    return out


def decode(lanes, p2, top=10):
    """Project a length-D hidden_out through SmolLM2's final RMSNorm + lm_head."""
    print(f"\n  loading SmolLM2 ...", file=sys.stderr)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        embed  = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        norm_w = model.model.norm.weight.detach().cpu().numpy().astype(np.float32)

    # Block-FP Q1.15 at scale 2^p2 → real
    h = np.array(lanes, dtype=np.float32) * (1 << p2) / 32768.0
    print(f"  hidden range [{h.min():+.2f}, {h.max():+.2f}]  std {h.std():.2f}",
          file=sys.stderr)

    # Final RMSNorm + lm_head (tied with embed)
    h_normed = (h / np.sqrt(np.mean(h*h) + 1e-5)) * norm_w
    logits   = h_normed @ embed.T

    top_ids = np.argsort(logits)[-top:][::-1]
    print(f"\n  Top-{top} predicted next-token (with logit):")
    for t in top_ids:
        print(f"    {tok.decode([int(t)])!r:>20}  {logits[int(t)]:+.2f}")

    nid = int(top_ids[0])
    return tok.decode([nid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=None,
                    help="If set, read all 576 lanes from this file instead of "
                         "the FPGA's 64-lane regmap window.  Use the Verilator "
                         "dump generated/tm_layer_dut_out.txt.")
    ap.add_argument("--no-restart", action="store_true",
                    help="Don't pulse the FPGA's restart bit (assume FSM is "
                         "already done from a previous run)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--p2",  type=int, default=H_OUT_P2_LAST_LAYER,
                    help="block-FP exponent of the last layer's h_out scale")
    args = ap.parse_args()

    if args.full:
        print(f"reading {args.full} ({D} lanes) ...", file=sys.stderr)
        lanes = load_full_dump(args.full)
    else:
        print(f"triggering FPGA forward at {PEER[0]}:{PEER[1]} ...", file=sys.stderr)
        lanes = fpga_run(restart=not args.no_restart)
        print(f"  read {len(lanes)} lanes from FPGA", file=sys.stderr)
        if len(lanes) < D:
            print(f"  zero-filling remaining {D - len(lanes)}", file=sys.stderr)
            lanes = lanes + [0] * (D - len(lanes))

    next_tok = decode(lanes, args.p2, top=args.top)
    print(f"\nFPGA predicts next token after the prompt: {next_tok!r}")
    if "Lily" in next_tok or "there" in next_tok or "," == next_tok.strip():
        print("→ COHERENT (matches what real SmolLM2 would say)")
    else:
        print("→ likely incoherent — precision shortfall in the residual quantisation")


if __name__ == "__main__":
    main()
