#!/usr/bin/env python3
"""Drive the existing tb_matvec testbench with real SmolLM2-135M data.

Picks 16 rows of a chosen Linear layer's weight, picks 64 inputs from a
real activation (the post-RMSNorm input to that layer at token 0), and
writes them into the same test_matvec.bin format the testbench reads.

The reference output is computed with a bit-exact integer mimic of
matvec_int8_engine — so success means the SV is correctly implementing
the quantization scheme on actual model data, not just random noise.
"""
import argparse
import struct
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
LANES  = 16
IN_DIM = 64                          # subset of the full hidden=576

def saturate_q15(x): return np.clip(x, -32768, 32767).astype(np.int16)

def matvec_int8_engine_pyref(w_int8, scale_q15, x_q15):
    """Bit-exact integer mimic of rtl/vc707/src/smollm/matvec_int8_engine.sv."""
    L, IN = w_int8.shape
    acc = np.zeros(L, dtype=np.int64)
    for k in range(IN):
        acc += x_q15[k].astype(np.int64) * w_int8[:, k].astype(np.int64)
    out = np.zeros(L, dtype=np.int16)
    for l in range(L):
        scaled  = int(acc[l]) * int(scale_q15[l])
        shifted = scaled >> 15
        out[l]  = max(-32768, min(32767, shifted))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0,
                    help="transformer layer index")
    ap.add_argument("--proj",  default="self_attn.q_proj",
                    choices=["self_attn.q_proj", "self_attn.k_proj",
                             "self_attn.v_proj", "self_attn.o_proj",
                             "mlp.gate_proj",    "mlp.up_proj",
                             "mlp.down_proj"])
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--row-offset", type=int, default=0,
                    help="which 16-row block of the weight to use")
    ap.add_argument("--act-scale", type=float, default=8.0,
                    help="full-scale magnitude for Q-format activation quantization")
    args = ap.parse_args()

    print(f"loading {MODEL_NAME} ...", file=sys.stderr)
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 torch_dtype=torch.float32)
    model.eval()

    # 1. Pick weight rows and quantize INT8 per-row.
    layer = model.model.layers[args.layer]
    parts = args.proj.split(".")
    proj = layer
    for p in parts: proj = getattr(proj, p)
    w = proj.weight.detach().cpu().numpy().astype(np.float32)   # [out, in]
    w_block = w[args.row_offset : args.row_offset + LANES, :IN_DIM]
    amax = np.maximum(np.abs(w_block).max(axis=1), 1e-8)
    weight_scale_real = amax / 127.0
    w_int8 = np.round(w_block / weight_scale_real[:, None]).clip(-128, 127).astype(np.int8)

    # 2. Capture the actual activation that feeds this layer at token 0.
    #    For self_attn.q_proj, that's the output of input_layernorm (RMSNorm).
    #    For mlp.gate_proj, that's post_attention_layernorm.
    capture = {}
    target_norm = layer.input_layernorm if parts[0] == "self_attn" else layer.post_attention_layernorm
    def cap(_m, _i, o):
        capture["x"] = (o[0] if isinstance(o, tuple) else o).detach().cpu().numpy()
    h = target_norm.register_forward_hook(cap)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        model(ids)
    h.remove()
    x_real = capture["x"][0, 0, :IN_DIM].astype(np.float64)        # token 0, first IN_DIM channels

    # 3. Quantize activation to 16-bit fixed-point with chosen full-scale.
    x_q15 = saturate_q15(np.round(x_real * (32768.0 / args.act_scale)))

    # 4. Build scale_q15 so the engine output is Q1.15 of (x_real @ w_real / FS_OUT)
    #    where x_real = x_q15 * (act_scale/32768) and w_real = w_int8 * weight_scale_real.
    #    acc                = sum(x_q15 * w_int8)
    #    real_acc_real      = (act_scale/32768) * weight_scale_real * acc
    #    we want out_q15    = real_acc_real * 32768 / FS_OUT
    #                       = acc * (act_scale * weight_scale_real / FS_OUT)
    #    engine produces    = (acc * scale_q15) >> 15
    #    so scale_q15        = round(act_scale * weight_scale_real * 32768 / FS_OUT)
    FS_OUT = max(1.0, args.act_scale * 4.0)        # heuristic; just to avoid satn
    scale_q15 = np.round(args.act_scale * weight_scale_real * 32768.0 / FS_OUT).astype(np.int64)
    scale_q15 = np.clip(scale_q15, -32768, 32767).astype(np.int16)

    # 5. Reference (integer-exact mimic of the SV)
    out_ref = matvec_int8_engine_pyref(w_int8, scale_q15, x_q15)

    # 6. Also a "pure FP" reference for sanity (real_x @ real_w / FS_OUT), in Q1.15
    real_w = w_int8.astype(np.float32) * weight_scale_real[:, None]
    fp_out = (x_real @ real_w.T) / FS_OUT
    fp_q15 = saturate_q15(np.round(fp_out * 32768.0))

    print(f"layer={args.layer}  proj={args.proj}  rows[{args.row_offset}:{args.row_offset+LANES}]",
          file=sys.stderr)
    print(f"x_real range [{x_real.min():.3f},{x_real.max():.3f}]   "
          f"act_scale={args.act_scale}", file=sys.stderr)
    print(f"x_q15  range [{x_q15.min()},{x_q15.max()}]", file=sys.stderr)
    print(f"weight_scale_real range [{weight_scale_real.min():.4f},{weight_scale_real.max():.4f}]",
          file=sys.stderr)
    print(f"FS_OUT={FS_OUT}   scale_q15 range [{scale_q15.min()},{scale_q15.max()}]",
          file=sys.stderr)
    print(f"fp_q15  = {fp_q15.tolist()}", file=sys.stderr)
    print(f"out_ref = {out_ref.tolist()}", file=sys.stderr)

    # 7. Write the test vector in the format tb_matvec.cpp expects.
    with open("test_matvec.bin", "wb") as f:
        f.write(struct.pack("<II", LANES, IN_DIM))
        f.write(x_q15.tobytes())
        for k in range(IN_DIM):
            for l in range(LANES):
                f.write(struct.pack("b", int(w_int8[l, k])))
        f.write(scale_q15.tobytes())
        f.write(out_ref.tobytes())
    print("wrote test_matvec.bin (real SmolLM2 data)", file=sys.stderr)

if __name__ == "__main__":
    main()
