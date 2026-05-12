#!/usr/bin/env python3
"""Decode fpga_layer_29.txt (= final hidden_state) through SmolLM2 lm_head."""
import sys, numpy as np

MODEL = "HuggingFaceTB/SmolLM2-135M"
H_OUT_P2_LAST = 10          # TM_RESCALE[NL-1] field [55:52]

path = sys.argv[1] if len(sys.argv) > 1 else "fpga_layer_29.txt"
lanes = []
with open(path) as f:
    for ln in f:
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        lanes.append(int(ln))
print(f"loaded {len(lanes)} lanes from {path}", file=sys.stderr)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
with torch.no_grad():
    embed  = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
    norm_w = model.model.norm.weight.detach().cpu().numpy().astype(np.float32)

h = np.array(lanes, dtype=np.float32) * (1 << H_OUT_P2_LAST) / 32768.0
print(f"hidden range [{h.min():+.2f}, {h.max():+.2f}]  std {h.std():.2f}", file=sys.stderr)

h_normed = (h / np.sqrt(np.mean(h*h) + 1e-5)) * norm_w
logits   = h_normed @ embed.T

top_ids = np.argsort(logits)[-10:][::-1]
print("\nTop-10 predicted next-token after 'Once upon a time':")
for t in top_ids:
    print(f"  {tok.decode([int(t)])!r:>20}  {logits[int(t)]:+.2f}")
