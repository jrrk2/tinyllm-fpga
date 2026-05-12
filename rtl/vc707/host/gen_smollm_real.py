#!/usr/bin/env python3
"""Generate FPGA test data from REAL SmolLM2-135M weights (HuggingFace).

Produces the same set of files as `gen_multilayer_test.py --mode tm`
(tm_layer_*.hex + tm_layer_data.svh + tm_layer_expected.txt + per-layer
tm_layer_L<n>_W_*.hex), but using real model weights quantised to the
FPGA's INT8/Q1.15 scheme instead of random data.

Reuses gen_layer_test.forward_layer for the fixed-point reference, so
expected.txt matches what the FPGA implements bit-for-bit (modulo
saturation-flip noise at NL=30 — see selftest_verify tolerance notes).

Usage:
    python3 host/gen_smollm_real.py --prompt "Once upon a time"
                                    --pos 3
"""

import argparse
import os
import sys
import math
import numpy as np

# Reuse the layer math + I/O helpers (same fixed-point semantics the
# FPGA implements).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_layer_test import (
    quant_q15,
    quant_int8_per_row,
    forward_layer,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "generated")

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def hex_word(v): return f"{int(v) & 0xFFFF:04x}"


def write_hex(name, values, hexer):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        for v in values:
            f.write(f"{hexer(v)}\n")


def write_packed_w(out_dir, name, mat):
    """Pack int8 weights as the FPGA expects: rows of 16 lanes × in_dim,
    grouped into 128-bit lines.  Mirror of gen_multilayer_test.write_packed_w."""
    out_dim, in_dim = mat.shape
    assert out_dim % 16 == 0, f"{name}: out_dim {out_dim} not /16"
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        for chunk in range(out_dim // 16):
            for k in range(in_dim):
                # 16 lanes packed LSB-first as 128-bit hex word.
                line = 0
                for lane in range(16):
                    b = int(mat[chunk*16 + lane, k]) & 0xFF
                    line |= b << (lane * 8)
                f.write(f"{line:032x}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--pos",    type=int, default=3,
                    help="Position to test (kv_pos = pos).  Tokens 0..pos-1 "
                         "fill the KV cache; token pos's hidden_in is what "
                         "the FPGA processes.")
    ap.add_argument("--max-ctx", type=int, default=4,
                    help="MAX_CTX (matches RTL).  pos must be < max-ctx.")
    args = ap.parse_args()

    assert args.pos < args.max_ctx, f"pos {args.pos} must be < MAX_CTX {args.max_ctx}"

    print(f"loading {MODEL_NAME} ...", file=sys.stderr)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32).eval()

    cfg = model.config
    D       = cfg.hidden_size                                     # 576
    NL      = cfg.num_hidden_layers                               # 30
    H_Q     = cfg.num_attention_heads                             # 9
    H_KV    = cfg.num_key_value_heads                             # 3
    HD      = D // H_Q                                            # 64
    FFN     = cfg.intermediate_size                               # 1536
    MAX_CTX = args.max_ctx
    POS     = args.pos
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))

    print(f"  D={D}  H_Q={H_Q}  H_KV={H_KV}  HD={HD}  FFN={FFN}  NL={NL}  "
          f"rope_theta={rope_theta}", file=sys.stderr)

    # ----------------------------------------------------------------------
    # Tokenize prompt; pad/truncate to MAX_CTX positions.
    # ----------------------------------------------------------------------
    ids = tok(args.prompt, add_special_tokens=False).input_ids
    print(f"  prompt = {args.prompt!r}  → {len(ids)} tokens: {ids}",
          file=sys.stderr)
    while len(ids) < MAX_CTX:
        ids.append(0)        # pad with token 0 (visible in trace)
    ids = ids[:MAX_CTX]
    print(f"  using first {MAX_CTX} tokens (pos {POS} = id {ids[POS]} = "
          f"{tok.decode([ids[POS]])!r})", file=sys.stderr)

    # ----------------------------------------------------------------------
    # Extract per-layer weights, quantise to INT8 + Q1.15.  Layout matches
    # what gen_layer_test.forward_layer expects.
    # ----------------------------------------------------------------------
    layers = []
    with torch.no_grad():
        embed_w = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        for li in range(NL):
            L = model.model.layers[li]
            Wq    = L.self_attn.q_proj.weight.detach().cpu().numpy().astype(np.float32)
            Wk    = L.self_attn.k_proj.weight.detach().cpu().numpy().astype(np.float32)
            Wv    = L.self_attn.v_proj.weight.detach().cpu().numpy().astype(np.float32)
            Wo    = L.self_attn.o_proj.weight.detach().cpu().numpy().astype(np.float32)
            Wgate = L.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
            Wup   = L.mlp.up_proj.weight.detach().cpu().numpy().astype(np.float32)
            Wdown = L.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
            g1    = L.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            g2    = L.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)

            assert Wq.shape    == (D, D),         f"layer {li} Wq {Wq.shape}"
            assert Wk.shape    == (H_KV*HD, D),   f"layer {li} Wk {Wk.shape}"
            assert Wv.shape    == (H_KV*HD, D),   f"layer {li} Wv {Wv.shape}"
            assert Wo.shape    == (D, D),         f"layer {li} Wo {Wo.shape}"
            assert Wgate.shape == (FFN, D),       f"layer {li} Wgate {Wgate.shape}"
            assert Wup.shape   == (FFN, D),       f"layer {li} Wup {Wup.shape}"
            assert Wdown.shape == (D, FFN),       f"layer {li} Wdown {Wdown.shape}"

            Wq_i, sq    = quant_int8_per_row(Wq)
            Wk_i, sk    = quant_int8_per_row(Wk)
            Wv_i, sv    = quant_int8_per_row(Wv)
            Wo_i, so    = quant_int8_per_row(Wo)
            Wg_i, sgate = quant_int8_per_row(Wgate)
            Wu_i, sup   = quant_int8_per_row(Wup)
            Wd_i, sdown = quant_int8_per_row(Wdown)
            gamma1 = quant_q15(g1)
            gamma2 = quant_q15(g2)

            layers.append(dict(
                n_heads=H_Q, n_kv_heads=H_KV, head_dim=HD,
                W_q=Wq_i, sca_q=sq, W_k=Wk_i, sca_k=sk, W_v=Wv_i, sca_v=sv,
                W_o=Wo_i, sca_o=so,
                W_gate=Wg_i, sca_gate=sgate, W_up=Wu_i, sca_up=sup,
                W_down=Wd_i, sca_down=sdown,
                gamma1_q15=gamma1, gamma2_q15=gamma2,
            ))
    print(f"  extracted {len(layers)} layers", file=sys.stderr)

    # ----------------------------------------------------------------------
    # Embed tokens → Q1.15 hidden states.  Then run through all NL layers
    # for positions 0..POS, building per-layer KV cache; snapshot pre-state
    # at position POS (i.e. caches for positions 0..POS-1) and capture
    # the final hidden_out for the test position.
    # ----------------------------------------------------------------------
    kv_caches  = [{'k': np.zeros((MAX_CTX, H_KV, HD), dtype=np.int16),
                   'v': np.zeros((MAX_CTX, H_KV, HD), dtype=np.int16)}
                  for _ in range(NL)]
    pre_caches = [None] * NL
    final_hidden_in  = None
    final_hidden_out = None
    for p in range(POS + 1):
        # Q1.15 embedding for token at position p.  Embed values are
        # typically tiny (|x| << 1) — clip just in case.
        h = quant_q15(embed_w[ids[p]])
        if p == POS:
            for li in range(NL):
                pre_caches[li] = {
                    'k': kv_caches[li]['k'].copy(),
                    'v': kv_caches[li]['v'].copy(),
                }
            final_hidden_in = h.copy()
        for li in range(NL):
            h, trace = forward_layer(h, layers[li], p, kv_caches[li], p)
        if p == POS:
            final_hidden_out = h.copy()

    n_sat = int(np.sum((final_hidden_out == 32767) | (final_hidden_out == -32768)))
    print(f"  hidden_out range [{final_hidden_out.min()},{final_hidden_out.max()}]  "
          f"saturated: {n_sat}/{D}", file=sys.stderr)

    # ----------------------------------------------------------------------
    # Emit files in tm_layer_* format.
    # ----------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = "tm_layer_"

    # Concatenated scales, gammas, KV-cache init.
    for name, key in [("Q","sca_q"),("K","sca_k"),("V","sca_v"),("O","sca_o"),
                      ("GATE","sca_gate"),("UP","sca_up"),("DOWN","sca_down")]:
        with open(os.path.join(OUT_DIR, f"{prefix}SCALE_{name}.hex"), "w") as f:
            for li in range(NL):
                for v in layers[li][key]:
                    f.write(f"{int(v) & 0xFFFF:04x}\n")
    for name, key in [("GAMMA1", "gamma1_q15"), ("GAMMA2", "gamma2_q15")]:
        with open(os.path.join(OUT_DIR, f"{prefix}{name}.hex"), "w") as f:
            for li in range(NL):
                for v in layers[li][key]:
                    f.write(f"{int(v) & 0xFFFF:04x}\n")
    for name, key in [("K_CACHE_INIT", "k"), ("V_CACHE_INIT", "v")]:
        with open(os.path.join(OUT_DIR, f"{prefix}{name}.hex"), "w") as f:
            for li in range(NL):
                for v in pre_caches[li][key].flatten():
                    f.write(f"{int(v) & 0xFFFF:04x}\n")

    # Per-layer weight files (gen_layer_ddr3.py packs them into the DDR3 image).
    for li in range(NL):
        for name, mat in [("Q",layers[li]['W_q']),("K",layers[li]['W_k']),
                          ("V",layers[li]['W_v']),("O",layers[li]['W_o']),
                          ("GATE",layers[li]['W_gate']),
                          ("UP",  layers[li]['W_up']),
                          ("DOWN",layers[li]['W_down'])]:
            write_packed_w(OUT_DIR, f"{prefix}L{li}_W_{name}.hex", mat)

    # hidden_in (Q1.15) — one entry per D lane.
    write_hex(f"{prefix}HIDDEN_IN.hex", final_hidden_in, hex_word)

    # Case-statement hidden_in ROM for the multilayer-tm selftest.
    with open(os.path.join(OUT_DIR, "layer_hidden_in_packed.svh"), "w") as f:
        f.write(f"// AUTO-GENERATED by gen_smollm_real.py — do not edit.\n")
        f.write(f"// Real SmolLM2 embed of token id {ids[POS]} "
                f"({tok.decode([ids[POS]])!r}) at position {POS}.\n\n")
        f.write(f"function automatic logic [15:0] layer_hidden_in_lut(input int unsigned idx);\n")
        f.write(f"  case (idx)\n")
        for i, v in enumerate(final_hidden_in):
            f.write(f"    {i:>4}: layer_hidden_in_lut = 16'h{int(np.uint16(v)):04x};\n")
        f.write(f"    default: layer_hidden_in_lut = 16'hdead;\n")
        f.write(f"  endcase\n")
        f.write(f"endfunction\n")

    # Constants SVH consumed by the testbench.
    with open(os.path.join(OUT_DIR, "tm_layer_data.svh"), "w") as f:
        f.write(f"// AUTO-GENERATED by gen_smollm_real.py — do not edit.\n")
        f.write(f"// SmolLM2-135M real weights, prompt={args.prompt!r}, pos={POS}\n\n")
        f.write(f"localparam int TM_NL  = {NL};\n")
        f.write(f"localparam int TM_POS = {POS};\n")

    # Reference hidden_out for selftest_verify.py.
    with open(os.path.join(OUT_DIR, "tm_layer_expected.txt"), "w") as f:
        f.write(f"# tm-multilayer final hidden_out (real SmolLM2-135M, "
                f"prompt={args.prompt!r}, pos={POS}, NL={NL})\n")
        for v in final_hidden_out:
            f.write(f"{int(v) & 0xFFFF:04x}  {int(v):+7d}\n")

    print(f"\nNL={NL}  D={D}  pos={POS}  REAL SmolLM2 weights",
          file=sys.stderr)
    print(f"wrote {prefix}*.hex (concatenated NL={NL} layers)", file=sys.stderr)
    print(f"\nNext steps:", file=sys.stderr)
    print(f"  1. (one-time) regen rope freq table for theta={rope_theta:.0f}:", file=sys.stderr)
    print(f"     python3 sim/gen_rope_freq_table.py --base {rope_theta:.0f}", file=sys.stderr)
    print(f"  2. python3 host/gen_layer_ddr3.py --mode tm --nlayers {NL}", file=sys.stderr)
    print(f"  3. (regenerate brom_*.sv from new .hex — Makefile pattern rule does this)", file=sys.stderr)
    print(f"  4. make    # rebuilds bitstream with new brom INIT contents", file=sys.stderr)


if __name__ == "__main__":
    main()
