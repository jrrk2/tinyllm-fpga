#!/usr/bin/env python3
"""Project an FPGA/Verilator hidden_out vector through SmolLM2's final
RMSNorm + lm_head and print top-K predicted tokens.

Input file format: one signed integer per line (Q15.9 24-bit). lines
beginning with '#' are ignored.  Path is given on the command line."""
import argparse, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--p2",  type=int, default=10,
                    help="block-FP scale exponent of the last layer (h_out_p2). "
                         "Real = stored * 2^p2 / 32768")
    args = ap.parse_args()

    # Load values
    vals = []
    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            tok = line.split()[0]
            vals.append(int(tok))
    print(f"loaded {len(vals)} lanes from {args.path}", file=sys.stderr)

    # Load model
    tok   = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        embed = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        norm_w = model.model.norm.weight.detach().cpu().numpy().astype(np.float32)

    # block-FP Q1.15 at scale 2^p2 → real
    scale = float(1 << args.p2)
    h = np.array(vals, dtype=np.float32) * scale / 32768.0
    print(f"hidden range [{h.min():.2f},{h.max():.2f}]  std {h.std():.2f}", file=sys.stderr)

    # Final norm + lm_head
    h_normed = (h / np.sqrt(np.mean(h*h) + 1e-5)) * norm_w
    logits = h_normed @ embed.T   # tied lm_head
    top = np.argsort(logits)[-args.top:][::-1]
    print(f"\nTop-{args.top} predicted tokens:")
    for t in top:
        print(f"  {tok.decode([int(t)])!r}  (logit {logits[int(t)]:.2f})")


if __name__ == "__main__":
    main()
