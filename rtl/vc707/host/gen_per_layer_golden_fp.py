#!/usr/bin/env python3
"""Float (PyTorch) per-layer hidden reference for the FPGA per-layer dump.

The FPGA's on-chip snapshot (fpga_per_layer_dump.py) captures one layer's hidden
state at token position `--step` (default 0 = first prompt token).  This produces
the matching golden: a plain float forward of SmolLM2-360M over the SAME prompt
ids, emitting the hidden after each layer at that position.

It is NOT block-FP — so it will not match the FPGA lane-for-lane (BFP rounding),
but the per-layer MAGNITUDE / range tracks the correct trajectory, so a layer
where the FPGA hidden collapses (->0), explodes, or diverges in magnitude is
immediately visible against this reference.

Usage:  MODEL=HuggingFaceTB/SmolLM2-360M python3 host/gen_per_layer_golden_fp.py \
            --ids 6403 1980 253 655 --step 0
Output: py_layer_fp_00.txt … py_layer_fp_<NL-1>.txt  (one float per line)
        plus a per-layer norm/range summary on stderr.
"""
import argparse, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-360M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=[6403, 1980, 253, 655],
                    help="prompt token ids (must match the FPGA PROMPT_TOKENS)")
    ap.add_argument("--step", type=int, default=0,
                    help="token position to snapshot (matches FPGA --step)")
    args = ap.parse_args()

    print(f"[golden] loading {MODEL} ...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    NL = model.config.num_hidden_layers
    D = model.config.hidden_size
    print(f"[golden] NL={NL} D={D} ids={args.ids} step={args.step}", file=sys.stderr)
    if args.step >= len(args.ids):
        sys.exit(f"step {args.step} >= prompt length {len(args.ids)}")

    ids = torch.tensor([args.ids], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    # hidden_states: tuple len NL+1; [0]=embedding, [L+1]=output of layer L.
    hs = out.hidden_states
    print(f"{'layer':>5} {'norm':>12} {'min':>12} {'max':>12}", file=sys.stderr)
    for L in range(NL):
        h = hs[L + 1][0, args.step, :].float().cpu().numpy()
        with open(f"py_layer_fp_{L:02d}.txt", "w") as f:
            f.write(f"# float golden hidden after layer {L}, pos {args.step}: {D} lanes\n")
            for v in h:
                f.write(f"{float(v):.6e}\n")
        print(f"{L:>5} {np.linalg.norm(h):>12.4f} {h.min():>12.4f} {h.max():>12.4f}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
