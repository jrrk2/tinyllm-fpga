#!/usr/bin/env python3
"""Compare FPGA's baked tm_layer_HIDDEN_IN.hex (layer-0 input) against
Python sim's embedding for the last prompt token at h_in_p2=1 scale.
If these differ, the FPGA and sim are running on different starting points
and no per-layer diff downstream is meaningful."""
import sys, numpy as np

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "Once upon a time"
H_IN_P2_L0 = 1   # TM_RESCALE[0] field [51:48] = 0x1

fpga_hex = []
with open("../generated/tm_layer_HIDDEN_IN.hex") as f:
    for ln in f:
        ln = ln.strip()
        if not ln: continue
        v = int(ln, 16)
        if v & 0x8000: v -= 0x10000
        fpga_hex.append(v)
fpga = np.array(fpga_hex, dtype=np.int32)
print(f"FPGA HIDDEN_IN: {len(fpga)} lanes, range [{fpga.min()}..{fpga.max()}]")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()
last_id = ids[-1]
print(f"prompt {PROMPT!r} → ids={ids}; last token id={last_id} ({tok.decode([last_id])!r})")

with torch.no_grad():
    e = model.model.embed_tokens.weight[last_id].detach().cpu().numpy().astype(np.float32)

# Quantize at h_in_p2=1: x_int = round(x_real * 32768 / 2^p2)
scale = float(1 << H_IN_P2_L0)
py = np.clip(np.round(e * 32768.0 / scale), -32768, 32767).astype(np.int32)
print(f"Python embed quantised: range [{py.min()}..{py.max()}]")

diff = fpga - py
err = np.abs(diff)
print(f"\nlane-by-lane diff: max|err|={err.max()}  mean|err|={err.mean():.2f}")
print("first 10 lanes:")
for i in range(10):
    print(f"  lane {i:3d}: fpga={fpga[i]:6d}  py={py[i]:6d}  diff={diff[i]:6d}")
