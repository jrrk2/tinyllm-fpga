#!/usr/bin/env python3
"""Test data for `smollm_full.sv` (Phase F.2): embed + NL layers + decode head.

Pipeline:
  in_token → embed[in_token] → NL transformer layers → final-RMSNorm
           → lm_head matvec → argmax → next_token

Outputs in ../generated/, all with `full_` prefix:
  full_EMBED.hex                VOCAB × D × Q1.15 (lookup table)
  full_L<N>_W_*.hex             per-layer packed weights (same layout as
                                multilayer)
  full_L<N>_SCALE_*.hex         per-layer scales
  full_L<N>_GAMMA1/2.hex        per-layer norms
  full_L<N>_K_CACHE_INIT.hex    per-layer KV cache pre-state
  full_L<N>_V_CACHE_INIT.hex
  full_GAMMA.hex                final-norm gamma
  full_W_LM.hex                 lm-head packed weights
  full_SCALE_LM.hex             lm-head per-row scales
  full_data.svh                 NL / D / VOCAB / IN_TOKEN constants
  full_expected.txt             expected next_token (decimal int)
"""
import argparse
import math
import os
import sys
import numpy as np

from gen_layer_test import (
    quant_q15, dequant_q15, rmsnorm_pyref, matvec_int8_pyref,
    quant_int8_per_row, forward_layer,
)

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "generated"))


def hex_byte(v): return f"{int(v) & 0xFF:02x}"
def hex_word(v): return f"{int(v) & 0xFFFF:04x}"


def write_hex(name, values, hexer):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        for v in values:
            f.write(f"{hexer(v)}\n")


def write_packed_w(name, mat):
    out_dim, in_dim = mat.shape
    assert out_dim % 16 == 0
    n_chunks = out_dim // 16
    with open(os.path.join(OUT_DIR, name), "w") as f:
        for c in range(n_chunks):
            for k in range(in_dim):
                packed = 0
                for l in range(16):
                    packed |= (int(mat[c*16 + l, k]) & 0xFF) << (l * 8)
                f.write(f"{packed:032x}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--in-token", type=int, default=37)
    ap.add_argument("--pos",      type=int, default=3)
    ap.add_argument("--nlayers",  type=int, default=3)
    args = ap.parse_args()

    D, H_Q, H_KV, HD, FFN, MAX_CTX = 128, 2, 1, 64, 128, 4
    VOCAB = 128
    NL = args.nlayers

    rng = np.random.default_rng(args.seed)
    w_std = 0.08 * math.sqrt(64.0 / D)

    # 1. Embedding table.  embed[v, e] is the e-th element of token v's hidden.
    embed_real = rng.normal(0, 0.1, (VOCAB, D))
    embed_q15  = quant_q15(embed_real)

    # 2. NL layers' weights / scales / γ
    layers = []
    for li in range(NL):
        Wq    = rng.normal(0, w_std, (D, D))
        Wk    = rng.normal(0, w_std, (H_KV*HD, D))
        Wv    = rng.normal(0, w_std, (H_KV*HD, D))
        Wo    = rng.normal(0, w_std, (D, D))
        Wgate = rng.normal(0, w_std, (FFN, D))
        Wup   = rng.normal(0, w_std, (FFN, D))
        Wdown = rng.normal(0, w_std, (D, FFN))
        Wq_i, sq    = quant_int8_per_row(Wq)
        Wk_i, sk    = quant_int8_per_row(Wk)
        Wv_i, sv    = quant_int8_per_row(Wv)
        Wo_i, so    = quant_int8_per_row(Wo)
        Wg_i, sgate = quant_int8_per_row(Wgate)
        Wu_i, sup   = quant_int8_per_row(Wup)
        Wd_i, sdown = quant_int8_per_row(Wdown)
        gamma1 = quant_q15(rng.normal(0, 0.3, D))
        gamma2 = quant_q15(rng.normal(0, 0.3, D))
        layers.append(dict(
            n_heads=H_Q, n_kv_heads=H_KV, head_dim=HD,
            W_q=Wq_i, sca_q=sq, W_k=Wk_i, sca_k=sk, W_v=Wv_i, sca_v=sv,
            W_o=Wo_i, sca_o=so,
            W_gate=Wg_i, sca_gate=sgate, W_up=Wu_i, sca_up=sup,
            W_down=Wd_i, sca_down=sdown,
            gamma1_q15=gamma1, gamma2_q15=gamma2,
        ))

    # 3. Final-norm gamma + LM head weights
    final_gamma = quant_q15(rng.normal(0, 0.3, D))
    W_lm_real   = rng.normal(0, w_std, (VOCAB, D))
    W_lm_int8, sca_lm = quant_int8_per_row(W_lm_real)

    # 4. Forward pass.  Run preceding positions (0..pos-1) with random tokens
    # to fill the per-layer KV caches; at position `pos`, use --in-token.
    kv_caches = [{'k': np.zeros((MAX_CTX, H_KV, HD), dtype=np.int16),
                  'v': np.zeros((MAX_CTX, H_KV, HD), dtype=np.int16)}
                 for _ in range(NL)]
    pre_caches = [None] * NL

    final_hidden_out = None
    for p in range(args.pos + 1):
        token_p = args.in_token if p == args.pos else int(rng.integers(0, VOCAB))
        h = embed_q15[token_p].copy()
        if p == args.pos:
            for li in range(NL):
                pre_caches[li] = {
                    'k': kv_caches[li]['k'].copy(),
                    'v': kv_caches[li]['v'].copy(),
                }
        for li in range(NL):
            h, _ = forward_layer(h, layers[li], p, kv_caches[li], p)
        if p == args.pos:
            final_hidden_out = h

    # 5. Decode-head: final norm + lm_head matvec + argmax
    normed = rmsnorm_pyref(final_hidden_out, final_gamma)
    logits = matvec_int8_pyref(W_lm_int8, sca_lm, normed)
    expected_token = int(np.argmax(logits))

    print(f"NL={NL} D={D} VOCAB={VOCAB} pos={args.pos} in_token={args.in_token}",
          file=sys.stderr)
    print(f"expected next_token = {expected_token}  "
          f"(top_logit={logits[expected_token]})", file=sys.stderr)
    print(f"top-3 candidates: {np.argsort(logits)[-3:][::-1].tolist()}",
          file=sys.stderr)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ----- Embedding (full_EMBED.hex): one Q1.15 entry per (token, e) -----
    # Layout: full_EMBED.hex line at index (v * D + e) is embed[v, e].
    write_hex("full_EMBED.hex", embed_q15.flatten(), hex_word)

    # ----- Per-layer files with prefix full_L<N>_ -----
    for li, lp in enumerate(layers):
        for name, mat in [("Q",lp['W_q']),("K",lp['W_k']),("V",lp['W_v']),
                          ("O",lp['W_o']),("GATE",lp['W_gate']),
                          ("UP",lp['W_up']),("DOWN",lp['W_down'])]:
            write_packed_w(f"full_L{li}_W_{name}.hex", mat)
        for name, sca in [("Q",lp['sca_q']),("K",lp['sca_k']),("V",lp['sca_v']),
                          ("O",lp['sca_o']),("GATE",lp['sca_gate']),
                          ("UP",lp['sca_up']),("DOWN",lp['sca_down'])]:
            write_hex(f"full_L{li}_SCALE_{name}.hex", sca, hex_word)
        write_hex(f"full_L{li}_GAMMA1.hex", lp['gamma1_q15'], hex_word)
        write_hex(f"full_L{li}_GAMMA2.hex", lp['gamma2_q15'], hex_word)
        write_hex(f"full_L{li}_K_CACHE_INIT.hex",
                  pre_caches[li]['k'].astype(np.int16).flatten(), hex_word)
        write_hex(f"full_L{li}_V_CACHE_INIT.hex",
                  pre_caches[li]['v'].astype(np.int16).flatten(), hex_word)

    # ----- Decode head with prefix full_ -----
    write_hex("full_GAMMA.hex",     final_gamma, hex_word)
    write_packed_w("full_W_LM.hex", W_lm_int8)
    write_hex("full_SCALE_LM.hex",  sca_lm, hex_word)

    with open(os.path.join(OUT_DIR, "full_data.svh"), "w") as f:
        f.write("// AUTO-GENERATED by host/gen_full_test.py — do not edit.\n")
        f.write(f"// NL={NL} D={D} VOCAB={VOCAB} pos={args.pos} in_token={args.in_token}\n\n")
        f.write(f"localparam int FULL_NL       = {NL};\n")
        f.write(f"localparam int FULL_D        = {D};\n")
        f.write(f"localparam int FULL_VOCAB    = {VOCAB};\n")
        f.write(f"localparam int FULL_POS      = {args.pos};\n")
        f.write(f"localparam int FULL_IN_TOKEN = {args.in_token};\n")

    with open(os.path.join(OUT_DIR, "full_expected.txt"), "w") as f:
        f.write(f"# smollm_full expected next-token "
                f"(in_token={args.in_token}, pos={args.pos}, NL={NL})\n")
        f.write(f"{expected_token}\n")

    print(f"wrote full_*.hex / .svh / .txt", file=sys.stderr)


if __name__ == "__main__":
    main()
