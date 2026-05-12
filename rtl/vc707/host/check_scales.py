#!/usr/bin/env python3
"""Print the per-tensor calibration scales for the stages that need scale-aware
RTL fixes (SwiGLU gate/up/mlp, Attn V/attn).  Used to size the fixed-point
arithmetic for swiglu.sv + smollm_layer.sv Attn-AV patches."""
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("sb", "host/sim_blockfp.py")
sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok   = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float32).eval()
sc = sb.calibrate(model, tok, "Once upon a time there was a princess.", margin=1.5)

print(f"{'L':>3} {'lsc[gate]':>10} {'lsc[up]':>10} {'lsc[mlp]':>10}   {'lsc[v]':>10} {'lsc[attn]':>10}")
for li in range(min(8, len(sc))):
    c = sc[li]
    print(f"{li:>3} {c['gate']:>10.4f} {c['up']:>10.4f} {c['mlp']:>10.4f}   "
          f"{c['v']:>10.4f} {c['attn']:>10.4f}")
# Print sample ratio:
print()
for li in range(min(5, len(sc))):
    c = sc[li]
    print(f"L{li}: silu input range ≈ [-{c['gate']:.2f}, {c['gate']:.2f}]; "
          f"mlp_factor = gate*up/mlp = {c['gate']*c['up']/c['mlp']:.4f}; "
          f"attn_factor = v/attn = {c['v']/c['attn']:.4f}")
