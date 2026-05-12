#!/usr/bin/env python3
"""Compare Python sim's calibration against FPGA's baked TM_RESCALE."""
import sys, re, numpy as np

# Read FPGA TM_RESCALE values from the generated .svh
def parse_fpga_rescale():
    out = []
    with open("../generated/tm_layer_data.svh") as f:
        for ln in f:
            m = re.match(r"\s*64'h([0-9a-fA-F]+),?\s*//\s*L(\d+):", ln)
            if not m: continue
            v = int(m.group(1), 16)
            r1     = v        & 0xFFFFFF
            r2     = (v>>24)  & 0xFFFFFF
            h_in   = (v>>48)  & 0xF
            h_out  = (v>>52)  & 0xF
            sh1    = ((v>>56) & 0xF); sh1 = sh1 - 16 if sh1 >= 8 else sh1
            sh2    = ((v>>60) & 0xF); sh2 = sh2 - 16 if sh2 >= 8 else sh2
            out.append({"h_in":h_in, "h_out":h_out, "r1":r1, "r2":r2, "sh1":sh1, "sh2":sh2})
    return out

fpga = parse_fpga_rescale()
print(f"FPGA TM_RESCALE: NL={len(fpga)}")
print("Layer 0:", fpga[0])

# Python sim's calibration
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util
spec = importlib.util.spec_from_file_location("sb", "host/sim_blockfp.py")
sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)

MODEL = "HuggingFaceTB/SmolLM2-135M"
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
sc = sb.calibrate(model, tok, "Once upon a time there was a princess.", margin=1.5)

print("\nPython calibration vs FPGA (first 5 layers):")
print(f"{'L':>3} {'fpga h_in':>10} {'py h_in':>10} {'fpga h_out':>11} {'py h_out':>10}")
for li in range(min(5, len(fpga))):
    py_h_in  = sb.pow2(sc[li].get("hidden_in",  1.0))
    py_h_out = sb.pow2(sc[li].get("hidden_out", 1.0))
    f_in, f_out = fpga[li]["h_in"], fpga[li]["h_out"]
    ok_in  = "OK" if py_h_in  == f_in  else "DIFF"
    ok_out = "OK" if py_h_out == f_out else "DIFF"
    print(f"{li:>3} {f_in:>10} {py_h_in:>10} {f_out:>11} {py_h_out:>10}  ({ok_in},{ok_out})")
