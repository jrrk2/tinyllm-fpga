#!/usr/bin/env python3
"""Convert SmolLM2-135M-Instruct FP32 weights → INT8 + FP16 scales binary.

Output: weights.bin matching the DDR3 layout in src/smollm/PLAN.md.

Quantization: per-channel symmetric INT8.
  For each output row of a weight matrix:
    scale = max(|row|) / 127
    int8  = round(row / scale)
  Stored: int8 weights (1 byte each), then FP16 scales table at end.

Usage:
  pip install transformers safetensors numpy
  ./convert.py --out weights.bin

This script is the "compile step" for the FPGA. The resulting
weights.bin is uploaded to DDR3 via the host UDP-write path (Phase 2 in
PLAN.md).
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    sys.exit("pip install transformers safetensors")


# ----------------------------------------------------------------------
# DDR3 layout — mirrors src/smollm/PLAN.md.
# ----------------------------------------------------------------------
HIDDEN, FFN, N_LAYERS, VOCAB = 576, 1536, 30, 49152
N_HEADS, N_KV_HEADS, HEAD_DIM = 9, 3, 64

BASE_EMBED  = 0x0000_0000
BASE_LAYERS = 0x01B0_0000
BASE_SCALES = 0x07F0_0000

SZ_WQ    = HIDDEN * HIDDEN
SZ_WK    = HIDDEN * (N_KV_HEADS * HEAD_DIM)
SZ_WV    = HIDDEN * (N_KV_HEADS * HEAD_DIM)
SZ_WO    = HIDDEN * HIDDEN
SZ_WG    = HIDDEN * FFN
SZ_WU    = HIDDEN * FFN
SZ_WD    = FFN    * HIDDEN
SZ_GAM   = HIDDEN * 2
SZ_LAYER = SZ_WQ + SZ_WK + SZ_WV + SZ_WO + SZ_WG + SZ_WU + SZ_WD + 2 * SZ_GAM


def quant_per_channel_int8(w_fp32: np.ndarray):
    """Per-row symmetric INT8 quantization.

    `w` shape: (out_dim, in_dim).  Each row gets its own scale.
    Returns: (int8 array same shape, fp16 scales of length out_dim).
    """
    assert w_fp32.ndim == 2
    abs_max = np.max(np.abs(w_fp32), axis=1, keepdims=True)
    abs_max[abs_max == 0] = 1e-8
    scale = (abs_max / 127.0).astype(np.float32)
    q = np.round(w_fp32 / scale).clip(-127, 127).astype(np.int8)
    return q, scale.flatten().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--out", required=True, help="output weights.bin path")
    ap.add_argument("--max-mem", default="2GiB")
    args = ap.parse_args()

    print(f"loading {args.model} …", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="float32")
    sd = model.state_dict()

    # Allocate sparse output buffer covering up to BASE_SCALES + scales.
    out = bytearray(BASE_SCALES + 2 * 1024 * 1024)   # 1 MB headroom for scales

    # ---- embed_tokens / lm_head (tied) ----
    embed = sd["model.embed_tokens.weight"].cpu().float().numpy()  # [49152, 576]
    assert embed.shape == (VOCAB, HIDDEN)
    q_embed, s_embed = quant_per_channel_int8(embed)
    out[BASE_EMBED:BASE_EMBED + q_embed.size] = q_embed.tobytes()
    print(f"  embed_tokens: {embed.shape}  scale [min,max] = "
          f"[{s_embed.min():.3e}, {s_embed.max():.3e}]", file=sys.stderr)

    scales_blob = bytearray()
    scales_blob += s_embed.tobytes()

    # ---- per-layer ----
    for li in range(N_LAYERS):
        base = BASE_LAYERS + li * SZ_LAYER
        cur = base
        for tag, key, expected in [
            ("Wq", f"model.layers.{li}.self_attn.q_proj.weight", (HIDDEN, HIDDEN)),
            ("Wk", f"model.layers.{li}.self_attn.k_proj.weight", (N_KV_HEADS*HEAD_DIM, HIDDEN)),
            ("Wv", f"model.layers.{li}.self_attn.v_proj.weight", (N_KV_HEADS*HEAD_DIM, HIDDEN)),
            ("Wo", f"model.layers.{li}.self_attn.o_proj.weight", (HIDDEN, HIDDEN)),
            ("Wg", f"model.layers.{li}.mlp.gate_proj.weight",    (FFN, HIDDEN)),
            ("Wu", f"model.layers.{li}.mlp.up_proj.weight",      (FFN, HIDDEN)),
            ("Wd", f"model.layers.{li}.mlp.down_proj.weight",    (HIDDEN, FFN)),
        ]:
            w = sd[key].cpu().float().numpy()
            assert w.shape == expected, f"{key}: {w.shape} vs {expected}"
            q, s = quant_per_channel_int8(w)
            out[cur:cur + q.size] = q.tobytes()
            cur += q.size
            scales_blob += s.tobytes()

        # gamma (FP16, no quantization)
        gamma_attn = sd[f"model.layers.{li}.input_layernorm.weight"].cpu().half().numpy()
        gamma_mlp  = sd[f"model.layers.{li}.post_attention_layernorm.weight"].cpu().half().numpy()
        out[cur:cur + 2*HIDDEN] = gamma_attn.tobytes(); cur += 2*HIDDEN
        out[cur:cur + 2*HIDDEN] = gamma_mlp.tobytes();  cur += 2*HIDDEN

        if li % 5 == 0 or li == N_LAYERS - 1:
            print(f"  layer {li:2d}  packed @ 0x{base:08x} ({SZ_LAYER:,} B)",
                  file=sys.stderr)

    # Final RMSNorm gamma (kept FP16 next to embed_tokens scales for now)
    gamma_final = sd["model.norm.weight"].cpu().half().numpy()
    scales_blob += gamma_final.tobytes()

    # Append scales table at BASE_SCALES
    out[BASE_SCALES:BASE_SCALES + len(scales_blob)] = scales_blob

    # Trim trailing padding
    final_size = BASE_SCALES + len(scales_blob)
    Path(args.out).write_bytes(bytes(out[:final_size]))
    print(f"\nwrote {final_size:,} bytes ({final_size/1024/1024:.1f} MiB) to {args.out}",
          file=sys.stderr)
    print(f"map:  embed @ 0x{BASE_EMBED:08x}", file=sys.stderr)
    print(f"      layers @ 0x{BASE_LAYERS:08x} (each = {SZ_LAYER:,} B)", file=sys.stderr)
    print(f"      scales @ 0x{BASE_SCALES:08x} ({len(scales_blob):,} B)", file=sys.stderr)


if __name__ == "__main__":
    main()
