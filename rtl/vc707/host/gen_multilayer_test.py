#!/usr/bin/env python3
"""Generate test data for the multi-layer Verilator test.

Wraps gen_layer_test.py's per-layer math: produces NL distinct weight
sets (one per layer), runs the same forward through them all, captures
the FINAL hidden_out as the reference.  Layer 0 takes a real hidden_in;
each subsequent layer takes the previous layer's hidden_out.

Outputs in ../generated/:
  multilayer_L<N>_W_*.hex      INT8 weights, per-layer
  multilayer_L<N>_SCALE_*.hex  Q1.15 scales, per-layer
  multilayer_L<N>_GAMMA*.hex   Q1.15 gammas, per-layer
  multilayer_L<N>_K_CACHE_INIT.hex / V_CACHE_INIT.hex
                                 KV cache pre-state, per-layer (for pos test)
  multilayer_HIDDEN_IN.hex     input to layer 0
  multilayer_data.svh          NL constant + per-layer file paths
  multilayer_expected.txt      final hidden_out (after all NL layers)
"""
import argparse
import math
import os
import sys
import numpy as np

# Reuse the layer math from gen_layer_test.
from gen_layer_test import (
    quant_q15, dequant_q15,
    matvec_int8_pyref, rmsnorm_pyref, rope_pyref,
    softmax_pyref, swiglu_pyref,
    quant_int8_per_row,
    forward_layer,
)


def write_packed_w(out_dir, name, mat):
    """Emit weight matrix in packed 128-bit-per-(chunk,k) layout."""
    out_dim, in_dim = mat.shape
    assert out_dim % 16 == 0
    n_chunks = out_dim // 16
    with open(os.path.join(out_dir, name), "w") as f:
        for c in range(n_chunks):
            for k in range(in_dim):
                packed = 0
                for l in range(16):
                    packed |= (int(mat[c*16 + l, k]) & 0xFF) << (l * 8)
                f.write(f"{packed:032x}\n")

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "generated"))


def hex_byte(v): return f"{int(v) & 0xFF:02x}"
def hex_word(v): return f"{int(v) & 0xFFFF:04x}"
def write_hex(name, values, hexer):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        for v in values:
            f.write(f"{hexer(v)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--pos",     type=int, default=3)
    ap.add_argument("--nlayers", type=int, default=3)
    ap.add_argument("--mode",    choices=["per-layer", "tm"], default="per-layer",
                    help="per-layer = NL separate hex sets (multilayer_L<n>_*.hex); "
                         "tm = concatenated single-set (tm_layer_*.hex with NL layers' "
                         "data appended in order, indexed by layer_idx in smollm_layer)")
    ap.add_argument("--config", choices=["small", "smollm"], default="small",
                    help="small = D=128 H_Q=2 H_KV=1 FFN=128 (Verilator regression). "
                         "smollm = D=576 H_Q=9 H_KV=3 FFN=1536 (SmolLM2-135M dims).")
    args = ap.parse_args()

    if args.config == "smollm":
        D, H_Q, H_KV, HD, FFN, MAX_CTX = 576, 9, 3, 64, 1536, 4
    else:
        D, H_Q, H_KV, HD, FFN, MAX_CTX = 128, 2, 1, 64, 128, 4
    NL = args.nlayers
    assert D == H_Q * HD

    rng = np.random.default_rng(args.seed)

    # Per-layer parameter dicts
    w_std = 0.08 * math.sqrt(64.0 / D)
    layers = []
    for li in range(NL):
        Wq = rng.normal(0, w_std, (D, D))
        Wk = rng.normal(0, w_std, (H_KV*HD, D))
        Wv = rng.normal(0, w_std, (H_KV*HD, D))
        Wo = rng.normal(0, w_std, (D, D))
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

    hidden_in = quant_q15(rng.normal(0, 0.1, D))

    # Per-layer KV caches.  Run preceding positions to fill them, then
    # snapshot pre-state for each layer.
    kv_caches = [{'k': np.zeros((MAX_CTX, H_KV, HD), dtype=np.int16),
                  'v': np.zeros((MAX_CTX, H_KV, HD), dtype=np.int16)}
                 for _ in range(NL)]
    pre_caches = [None] * NL

    # Run all positions up to args.pos — at each position, forward through
    # all NL layers (residual stream chaining).  Snapshot per-layer KV cache
    # AT the final position before that position writes to it.
    final_trace = None
    for p in range(args.pos + 1):
        h = hidden_in if p == args.pos else quant_q15(rng.normal(0, 0.1, D))
        if p == args.pos:
            for li in range(NL):
                pre_caches[li] = {
                    'k': kv_caches[li]['k'].copy(),
                    'v': kv_caches[li]['v'].copy(),
                }
        for li in range(NL):
            h, trace = forward_layer(h, layers[li], p, kv_caches[li], p)
            if p == args.pos and li == NL-1:
                final_trace = trace
        # h is now layer NL-1's output for this position.
    final_hidden_out = final_trace['hidden_out']

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.mode == "tm":
        # Time-multiplex layout: a single set of concatenated hex files
        # that the smollm_layer's NL-sized 2D ROMs consume.  Layer N's
        # data lives at offset N × per_layer_count in each file.
        prefix = "tm_layer_"
        # Concatenated scales: tm_layer_SCALE_<X>.hex with NL × <size> entries
        for name, key in [("Q","sca_q"),("K","sca_k"),("V","sca_v"),("O","sca_o"),
                          ("GATE","sca_gate"),("UP","sca_up"),("DOWN","sca_down")]:
            with open(os.path.join(OUT_DIR, f"{prefix}SCALE_{name}.hex"), "w") as f:
                for li in range(NL):
                    for v in layers[li][key]:
                        f.write(f"{int(v) & 0xFFFF:04x}\n")
        # Concatenated gammas
        for name in ("GAMMA1", "GAMMA2"):
            key = "gamma1_q15" if name == "GAMMA1" else "gamma2_q15"
            with open(os.path.join(OUT_DIR, f"{prefix}{name}.hex"), "w") as f:
                for li in range(NL):
                    for v in layers[li][key]:
                        f.write(f"{int(v) & 0xFFFF:04x}\n")
        # Concatenated KV cache pre-state
        for name, key in [("K_CACHE_INIT", "k"), ("V_CACHE_INIT", "v")]:
            with open(os.path.join(OUT_DIR, f"{prefix}{name}.hex"), "w") as f:
                for li in range(NL):
                    for v in pre_caches[li][key].flatten():
                        f.write(f"{int(v) & 0xFFFF:04x}\n")
        # Per-layer weights still emitted as separate L<n>_W_<X>.hex —
        # the DDR3 image generator (gen_multilayer_tm_ddr3.py, future)
        # walks all NL layers and packs them into one big DDR3 image.
        for li, lp in enumerate(layers):
            for name, mat in [("Q",lp['W_q']),("K",lp['W_k']),("V",lp['W_v']),
                              ("O",lp['W_o']),("GATE",lp['W_gate']),
                              ("UP",lp['W_up']),("DOWN",lp['W_down'])]:
                write_packed_w(OUT_DIR, f"{prefix}L{li}_W_{name}.hex", mat)
        write_hex(f"{prefix}HIDDEN_IN.hex", hidden_in, hex_word)

        # Case-statement hidden_in ROM for smollm_multilayer_tm_selftest.sv
        # (which `\`include`s layer_hidden_in_packed.svh).  Same pattern as
        # gen_layer_test.py — the only ROM form Vivado reliably synthesizes
        # from $readmem / packed-localparam alternatives.
        with open(os.path.join(OUT_DIR, "layer_hidden_in_packed.svh"), "w") as f:
            f.write(f"// AUTO-GENERATED by gen_multilayer_test.py --mode tm — do not edit.\n")
            f.write(f"// hidden_in[0..{D-1}] for the multilayer-tm selftest (D={D}).\n\n")
            f.write(f"function automatic logic [15:0] layer_hidden_in_lut(input int unsigned idx);\n")
            f.write(f"  case (idx)\n")
            for i, v in enumerate(hidden_in):
                f.write(f"    {i:>4}: layer_hidden_in_lut = 16'h{int(np.uint16(v)):04x};\n")
            f.write(f"    default: layer_hidden_in_lut = 16'hdead;\n")
            f.write(f"  endcase\n")
            f.write(f"endfunction\n")

        # SVH + reference output
        with open(os.path.join(OUT_DIR, "tm_layer_data.svh"), "w") as f:
            f.write(f"// AUTO-GENERATED by gen_multilayer_test.py --mode tm — do not edit.\n")
            f.write(f"// NL={NL} D={D} H_Q={H_Q} H_KV={H_KV} HD={HD} FFN={FFN} "
                    f"MAX_CTX={MAX_CTX} pos={args.pos}\n\n")
            f.write(f"localparam int TM_NL  = {NL};\n")
            f.write(f"localparam int TM_POS = {args.pos};\n")
        with open(os.path.join(OUT_DIR, "tm_layer_expected.txt"), "w") as f:
            f.write(f"# tm-multilayer final hidden_out (after {NL} layers, Q1.15)\n")
            for v in final_hidden_out:
                f.write(f"{int(v) & 0xFFFF:04x}  {int(v):+7d}\n")
        n_sat = int(np.sum((final_hidden_out == 32767) | (final_hidden_out == -32768)))
        print(f"NL={NL}  D={D}  pos={args.pos}  TM mode  hidden_out range "
              f"[{final_hidden_out.min()},{final_hidden_out.max()}]  "
              f"saturated: {n_sat}/{D}")
        print(f"wrote {prefix}*.hex (concatenated NL={NL} layers)")
        return

    # Emit per-layer hex files (Phase D packed-128b layout for weights)
    for li, lp in enumerate(layers):
        for name, mat in [("Q",lp['W_q']),("K",lp['W_k']),("V",lp['W_v']),
                          ("O",lp['W_o']),("GATE",lp['W_gate']),
                          ("UP",lp['W_up']),("DOWN",lp['W_down'])]:
            write_packed_w(OUT_DIR, f"multilayer_L{li}_W_{name}.hex", mat)
        for name, sca in [("Q",lp['sca_q']),("K",lp['sca_k']),("V",lp['sca_v']),
                          ("O",lp['sca_o']),("GATE",lp['sca_gate']),
                          ("UP",lp['sca_up']),("DOWN",lp['sca_down'])]:
            write_hex(f"multilayer_L{li}_SCALE_{name}.hex", sca, hex_word)
        write_hex(f"multilayer_L{li}_GAMMA1.hex", lp['gamma1_q15'], hex_word)
        write_hex(f"multilayer_L{li}_GAMMA2.hex", lp['gamma2_q15'], hex_word)
        # KV cache pre-state
        write_hex(f"multilayer_L{li}_K_CACHE_INIT.hex",
                  pre_caches[li]['k'].flatten(), hex_word)
        write_hex(f"multilayer_L{li}_V_CACHE_INIT.hex",
                  pre_caches[li]['v'].flatten(), hex_word)
    write_hex("multilayer_HIDDEN_IN.hex", hidden_in, hex_word)

    # SVH with constants
    svh = os.path.join(OUT_DIR, "multilayer_data.svh")
    with open(svh, "w") as f:
        f.write(f"// AUTO-GENERATED by host/gen_multilayer_test.py — do not edit.\n")
        f.write(f"// NL={NL} D={D} H_Q={H_Q} H_KV={H_KV} HD={HD} FFN={FFN} "
                f"MAX_CTX={MAX_CTX} pos={args.pos}\n\n")
        f.write(f"localparam int MULTILAYER_NL  = {NL};\n")
        f.write(f"localparam int MULTILAYER_POS = {args.pos};\n")

    txt = os.path.join(OUT_DIR, "multilayer_expected.txt")
    with open(txt, "w") as f:
        f.write(f"# multilayer final hidden_out (after {NL} layers, Q1.15)\n")
        for v in final_hidden_out:
            f.write(f"{int(v) & 0xFFFF:04x}  {int(v):+7d}\n")

    n_sat = int(np.sum((final_hidden_out == 32767) | (final_hidden_out == -32768)))
    print(f"NL={NL}  D={D}  pos={args.pos}  hidden_out range "
          f"[{final_hidden_out.min()},{final_hidden_out.max()}]  "
          f"saturated: {n_sat}/{D}", file=sys.stderr)
    print(f"wrote {NL} × 7 weight matrices + scales + γ + KV pre-state "
          f"to {OUT_DIR}/multilayer_*", file=sys.stderr)


if __name__ == "__main__":
    main()
