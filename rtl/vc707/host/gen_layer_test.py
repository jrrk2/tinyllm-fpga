#!/usr/bin/env python3
"""Reference + test-data generator for smollm_layer.sv.

Implements one SmolLM2-style transformer layer end-to-end in numpy with
the exact integer arithmetic the FPGA performs (INT8 weights + per-row
scale, Q1.15 activations + per-bus scale, NR-1/sqrt for RMSNorm, exp-LUT
softmax, CORDIC sin/cos for RoPE, SiLU LUT for SwiGLU).

Emits two files into ../generated/:
  layer_test_data.svh        — packed-localparam test inputs (hidden_in,
                              all weight matrices, all per-row scales,
                              gamma vectors, RoPE pos, etc.)
  layer_test_expected.txt    — golden output values:
                                 q[0..D-1], k[0..H_KV*HD-1], v[..],
                                 q_rot, k_rot, attn_out, hidden1,
                                 mlp_out, hidden2 (final)
                              Each section labelled so the testbench
                              can selectively check intermediate signals.

Tiny config matching smollm_layer.sv defaults:
  D=16  N_HEADS=1  N_KV_HEADS=1  HEAD_DIM=16  FFN_DIM=32  MAX_CTX=4
"""
import argparse
import os
import sys
import numpy as np

# ---------------------- bit-exact mimics of FPGA leaf ops ----------------------

def quant_q15(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)

def dequant_q15(q: np.ndarray) -> np.ndarray:
    return q.astype(np.float64) / 32768.0

def matvec_int8_pyref(w_int8: np.ndarray, scale_q15: np.ndarray, x_q15: np.ndarray) -> np.ndarray:
    """Bit-exact mimic of matvec_int8_engine.sv."""
    L, IN = w_int8.shape
    acc = np.zeros(L, dtype=np.int64)
    for k in range(IN):
        acc += x_q15[k].astype(np.int64) * w_int8[:, k].astype(np.int64)
    out = np.zeros(L, dtype=np.int16)
    for l in range(L):
        s  = int(acc[l]) * int(scale_q15[l])
        sh = s >> 15
        out[l] = max(-32768, min(32767, sh))
    return out

def rmsnorm_pyref(x_q15: np.ndarray, gamma_q15: np.ndarray, eps_real: float = 1e-5) -> np.ndarray:
    """Float-equivalent reference.  Within ±1 LSB of the SV NR-1/sqrt impl."""
    x  = dequant_q15(x_q15)
    g  = dequant_q15(gamma_q15)
    ms = float(np.mean(x * x))
    inv_rms = 1.0 / np.sqrt(ms + eps_real)
    return quant_q15(x * g * inv_rms)

def rope_pyref(x_q15: np.ndarray, pos: int, base: float = 10000.0) -> np.ndarray:
    """LLaMA convention RoPE."""
    H = x_q15.shape[0]
    H2 = H // 2
    j = np.arange(H2)
    freq = 1.0 / (base ** (2.0 * j / H))
    angle = pos * freq
    c = np.cos(angle); s = np.sin(angle)
    xd = dequant_q15(x_q15)
    lo = xd[:H2]; hi = xd[H2:]
    y_lo =  lo * c - hi * s
    y_hi =  hi * c + lo * s
    return quant_q15(np.concatenate([y_lo, y_hi]))

def softmax_pyref(x_q15: np.ndarray) -> np.ndarray:
    x = dequant_q15(x_q15)
    m = x.max()
    e = np.exp(x - m)
    return quant_q15(e / e.sum())

def swiglu_pyref(gate_q15: np.ndarray, up_q15: np.ndarray) -> np.ndarray:
    g = dequant_q15(gate_q15); u = dequant_q15(up_q15)
    silu = g * (1.0 / (1.0 + np.exp(-g)))
    return quant_q15(silu * u)


# ---------------------- weight quantisation ----------------------

def quant_int8_per_row(w_real: np.ndarray):
    """Per-row symmetric INT8 + Q1.15 scale_q15 packed for the matvec engine.
    Returns (w_int8 [out, in], scale_q15 [out]).
    Interpretation: real_y[lane] = (sum of x_q15 * w_int8[lane]) * scale_q15[lane] / 2^15."""
    amax = np.maximum(np.abs(w_real).max(axis=1), 1e-8)
    weight_scale_real = amax / 127.0
    w_int8 = np.round(w_real / weight_scale_real[:, None]).clip(-128, 127).astype(np.int8)
    # For x in Q1.15 (full-scale = 1.0), output Q1.15 wants:
    #   y_q15  = round(real_y * 32768)
    #          = round((sum(x_q15 * w_int8) * weight_scale_real / 32768) * 32768)
    #          = sum(x_q15 * w_int8) * weight_scale_real
    # Engine produces (acc * scale_q15) >> 15.  Equate:
    #   scale_q15 = round(weight_scale_real * 32768)
    scale_q15 = np.round(weight_scale_real * 32768.0).clip(-32768, 32767).astype(np.int16)
    return w_int8, scale_q15


# ---------------------- one transformer layer ----------------------

def forward_layer(hidden_q15, params, pos, kv_cache, kv_pos):
    """
    params: dict with W_q/W_k/W_v/W_o/W_gate/W_up/W_down (int8) + corresponding
            sca_* (int16 Q1.15) + gamma1_q15 + gamma2_q15 (Q1.15).
    pos:    current token position (int)
    kv_cache: dict {'k':[MAX_CTX, H_KV, HD], 'v':[MAX_CTX, H_KV, HD]} of int16
    kv_pos: how many positions are filled in the cache (this token writes index kv_pos).
    Returns: hidden_out_q15, dict of intermediate signals.
    """
    D = hidden_q15.shape[0]
    H_KV = params['n_kv_heads']
    HD   = params['head_dim']
    H_Q  = params['n_heads']

    trace = {}

    # --- Pre-attn norm + Q/K/V projections ---
    n1 = rmsnorm_pyref(hidden_q15, params['gamma1_q15'])
    trace['norm1'] = n1.copy()

    q = matvec_int8_pyref(params['W_q'], params['sca_q'], n1)
    k = matvec_int8_pyref(params['W_k'], params['sca_k'], n1)
    v = matvec_int8_pyref(params['W_v'], params['sca_v'], n1)
    trace['q'] = q.copy(); trace['k'] = k.copy(); trace['v'] = v.copy()

    # --- RoPE on Q and K (per head) ---
    # Q: H_Q heads × HD.  K: H_KV × HD.
    q_rot = q.copy()
    k_rot = k.copy()
    for h in range(H_Q):
        q_rot[h*HD:(h+1)*HD] = rope_pyref(q[h*HD:(h+1)*HD], pos)
    for h in range(H_KV):
        k_rot[h*HD:(h+1)*HD] = rope_pyref(k[h*HD:(h+1)*HD], pos)
    trace['q_rot'] = q_rot.copy(); trace['k_rot'] = k_rot.copy()

    # --- Write to KV cache at index kv_pos ---
    for h in range(H_KV):
        kv_cache['k'][kv_pos, h] = k_rot[h*HD:(h+1)*HD]
        kv_cache['v'][kv_pos, h] = v    [h*HD:(h+1)*HD]

    # --- Attention Q@K^T scaled, softmax, @V (per Q head) ---
    grp = H_Q // H_KV     # GQA grouping factor
    attn_out = np.zeros(D, dtype=np.int16)
    for h in range(H_Q):
        kv_h = h // grp
        q_h  = q_rot[h*HD:(h+1)*HD]                           # int16 Q1.15
        # Scores over [0..kv_pos] cached positions
        T = kv_pos + 1
        scores = np.zeros(T, dtype=np.int16)
        # 1/sqrt(HD) — implemented as right-shift.  HD=64 → 1/8 → shift 3.
        # HD=16 → 1/4 → shift 2.  Generalise: shift = int(log2(HD))//2 for
        # power-of-two HD with even log2.
        import math
        hd_shift = int(math.log2(HD)) // 2
        q_h_scaled = (q_h.astype(np.int32) >> hd_shift).astype(np.int16)
        for t in range(T):
            k_t = kv_cache['k'][t, kv_h]
            # Dot product as Q2.30 acc → saturate to Q1.15.
            acc = int((q_h_scaled.astype(np.int64) * k_t.astype(np.int64)).sum())
            shifted = acc >> 15
            scores[t] = max(-32768, min(32767, shifted))
        # Softmax over T scores
        sm = softmax_pyref(scores)
        # Weighted V: sum_t sm[t] * v[t, kv_h]
        out_h = np.zeros(HD, dtype=np.int64)
        for t in range(T):
            v_t = kv_cache['v'][t, kv_h]
            out_h += sm[t].astype(np.int64) * v_t.astype(np.int64)
        # Q1.15 * Q1.15 = Q2.30; >> 15 → Q1.15
        out_h_q15 = np.clip(out_h >> 15, -32768, 32767).astype(np.int16)
        attn_out[h*HD:(h+1)*HD] = out_h_q15
    trace['attn'] = attn_out.copy()

    # --- O projection + residual ---
    o = matvec_int8_pyref(params['W_o'], params['sca_o'], attn_out)
    h1 = np.clip(hidden_q15.astype(np.int32) + o.astype(np.int32),
                 -32768, 32767).astype(np.int16)
    trace['hidden1'] = h1.copy()

    # --- Post-attn norm + MLP ---
    n2 = rmsnorm_pyref(h1, params['gamma2_q15'])
    trace['norm2'] = n2.copy()

    gate = matvec_int8_pyref(params['W_gate'], params['sca_gate'], n2)
    up   = matvec_int8_pyref(params['W_up'],   params['sca_up'],   n2)
    sg   = swiglu_pyref(gate, up)
    trace['mlp_inter'] = sg.copy()

    down = matvec_int8_pyref(params['W_down'], params['sca_down'], sg)
    h2   = np.clip(h1.astype(np.int32) + down.astype(np.int32),
                   -32768, 32767).astype(np.int16)
    trace['hidden_out'] = h2.copy()
    return h2, trace


# ---------------------- emitter ----------------------

def pack_lane0_low(values: np.ndarray, width_bits: int) -> str:
    """Pack `values` little-endian (index 0 in low bits) into a hex string."""
    fmt = f"{{:0{width_bits//4}x}}"
    mask = (1 << width_bits) - 1
    return "".join(fmt.format(int(v) & mask) for v in reversed(values))

def emit_localparam(f, name, values, elt_bits):
    total = len(values) * elt_bits
    h = pack_lane0_low(values, elt_bits)
    f.write(f"localparam logic [{total-1}:0] {name} = {total}'h{h};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--pos",      type=int, default=3)   # = MAX_CTX-1 → no softmax padding needed
    ap.add_argument("--config",   choices=["smollm", "small"], default="smollm",
                    help="smollm = real SmolLM2 dims (D=576), small = D=64 selftest config")
    args = ap.parse_args()

    if args.config == "smollm":
        # Real SmolLM2-135M layer dimensions.
        D, H_Q, H_KV, HD, FFN, MAX_CTX = 576, 9, 3, 64, 1536, 4
    else:
        # Tiny selftest config that fits VC707 LUTs without BRAM-friendly
        # buffer refactoring.
        D, H_Q, H_KV, HD, FFN, MAX_CTX = 64, 1, 1, 64, 128, 4
    assert D == H_Q * HD
    assert D == H_Q * HD
    assert (H_KV * HD) <= D
    OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "generated"))
    os.makedirs(OUT_DIR, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Small-magnitude weights — std scales as 1/sqrt(D) to keep matmul
    # outputs within Q1.15 range.  At D=576 that's ~0.04.
    import math
    w_std = 0.08 * math.sqrt(64.0 / D)
    def w(shape): return rng.normal(0, w_std, shape).astype(np.float64)
    W_q    = w((D,         D))
    W_k    = w((H_KV * HD, D))
    W_v    = w((H_KV * HD, D))
    W_o    = w((D,         D))
    W_gate = w((FFN,       D))
    W_up   = w((FFN,       D))
    W_down = w((D,         FFN))

    Wq_i,  sq    = quant_int8_per_row(W_q)
    Wk_i,  sk    = quant_int8_per_row(W_k)
    Wv_i,  sv    = quant_int8_per_row(W_v)
    Wo_i,  so    = quant_int8_per_row(W_o)
    Wg_i,  sgate = quant_int8_per_row(W_gate)
    Wu_i,  sup   = quant_int8_per_row(W_up)
    Wd_i,  sdown = quant_int8_per_row(W_down)

    gamma1 = quant_q15(rng.normal(0, 0.3, D))
    gamma2 = quant_q15(rng.normal(0, 0.3, D))

    # hidden_in stddev fixed at 0.1 (does NOT scale with D).  Smaller values
    # make mean(x²) too small and the rmsnorm's inv_rms overflows Q5.12 (max 16).
    h_std = 0.1
    hidden_in = quant_q15(rng.normal(0, h_std, D))

    # Empty KV cache; advance position-by-position to args.pos so the
    # earlier positions actually get cached and the attention is over
    # multiple keys.
    kv_cache = {'k': np.zeros((MAX_CTX, H_KV, HD), dtype=np.int16),
                'v': np.zeros((MAX_CTX, H_KV, HD), dtype=np.int16)}

    params = dict(
        n_heads=H_Q, n_kv_heads=H_KV, head_dim=HD,
        W_q=Wq_i, sca_q=sq,  W_k=Wk_i, sca_k=sk,  W_v=Wv_i, sca_v=sv,
        W_o=Wo_i, sca_o=so,
        W_gate=Wg_i, sca_gate=sgate, W_up=Wu_i, sca_up=sup,
        W_down=Wd_i, sca_down=sdown,
        gamma1_q15=gamma1, gamma2_q15=gamma2,
    )

    # Run all positions up to args.pos, each with its own random hidden,
    # so the KV cache is realistic.  We'll capture the trace of the LAST one.
    # We snapshot the kv_cache *before* the final position runs so the SV
    # can initialise its own cache from the pre-final state, then run only
    # the final position itself.
    last_trace = None
    kv_cache_pre = None
    for p in range(args.pos + 1):
        h_in_p = hidden_in if p == args.pos else quant_q15(rng.normal(0, h_std, D))
        if p == args.pos:
            # Snapshot k_cache, v_cache as they are *before* this final write.
            kv_cache_pre = {
                'k': kv_cache['k'].copy(),
                'v': kv_cache['v'].copy(),
            }
        h_out_p, last_trace = forward_layer(h_in_p, params, p, kv_cache, p)
    final_hidden_out = last_trace['hidden_out']

    # ---- Emit hex files for $readmemh of unpacked arrays ----
    # Phase D layout: weights are packed 128-bit per entry (16 lanes ×
    # 8-bit), one entry per (chunk, in_idx).  This means smollm_layer
    # does one read per matvec drive cycle (vs 16 separate byte reads
    # in the original layout) — a single BRAM per matrix instead of
    # ~16× replicated ROMs.  Cleanly DDR3-streamable later: one tile
    # entry per cycle.
    #   For W [out_dim, in_dim], file has (out_dim/16) × in_dim lines.
    #   Line at index (chunk * in_dim + k) contains 16 lane bytes packed
    #   with lane 0 in the LOW byte.
    hex_byte = lambda v: f"{int(v) & 0xFF:02x}"
    hex_word = lambda v: f"{int(v) & 0xFFFF:04x}"
    def write_hex(name, values, hexer):
        with open(os.path.join(OUT_DIR, name), "w") as f:
            for v in values:
                f.write(f"{hexer(v)}\n")
    def write_packed_w(name, mat):
        """Emit weight matrix in packed 128-bit-per-(chunk,k) layout."""
        out_dim, in_dim = mat.shape
        assert out_dim % 16 == 0, f"out_dim={out_dim} must be /16"
        n_chunks = out_dim // 16
        with open(os.path.join(OUT_DIR, name), "w") as f:
            for c in range(n_chunks):
                for k in range(in_dim):
                    # Pack lanes 0..15 with lane 0 in the LOW byte.
                    packed = 0
                    for l in range(16):
                        packed |= (int(mat[c*16 + l, k]) & 0xFF) << (l * 8)
                    f.write(f"{packed:032x}\n")
    for name, mat in [("Q",Wq_i),("K",Wk_i),("V",Wv_i),("O",Wo_i),
                      ("GATE",Wg_i),("UP",Wu_i),("DOWN",Wd_i)]:
        write_packed_w(f"layer_W_{name}.hex", mat)
    for name, sca in [("Q",sq),("K",sk),("V",sv),("O",so),
                      ("GATE",sgate),("UP",sup),("DOWN",sdown)]:
        write_hex(f"layer_SCALE_{name}.hex", sca, hex_word)
    write_hex("layer_GAMMA1.hex",    gamma1,    hex_word)
    write_hex("layer_GAMMA2.hex",    gamma2,    hex_word)
    write_hex("layer_HIDDEN_IN.hex", hidden_in, hex_word)

    # ---- Emit small scalar SVH (just LAYER_POS) ----
    svh = os.path.join(OUT_DIR, "layer_test_data.svh")
    with open(svh, "w") as f:
        f.write(f"// AUTO-GENERATED by host/gen_layer_test.py — do not edit.\n")
        f.write(f"// D={D} H_Q={H_Q} H_KV={H_KV} HD={HD} FFN={FFN} MAX_CTX={MAX_CTX} pos={args.pos}\n")
        f.write(f"// Weights/scales/γ/hidden_in/kv_cache live in matching .hex files\n")
        f.write(f"// ($readmemh into unpacked arrays — avoids Vivado segfault on huge packed selects).\n\n")
        f.write(f"localparam int LAYER_POS = {args.pos};\n")

    # ---- Emit hidden_in as a case-statement function SVH ----
    # smollm_layer_selftest used to do $readmemh into an array that's
    # read combinationally via a generate-loop continuous assign — which
    # Vivado silently drops ("invalid memory name").  Case statement
    # bakes values as compile-time constants and elaboration fails
    # loudly if the SVH is missing or short.
    hi_svh = os.path.join(OUT_DIR, "layer_hidden_in_packed.svh")
    with open(hi_svh, "w") as f:
        f.write(f"// AUTO-GENERATED by host/gen_layer_test.py — do not edit.\n")
        f.write(f"// hidden_in[0..{D-1}] as a case-statement ROM (D={D}).\n\n")
        f.write(f"function automatic logic [15:0] layer_hidden_in_lut(input int unsigned idx);\n")
        f.write(f"  case (idx)\n")
        for i, v in enumerate(hidden_in):
            f.write(f"    {i:>4}: layer_hidden_in_lut = 16'h{int(np.uint16(v)):04x};\n")
        f.write(f"    default: layer_hidden_in_lut = 16'hdead; // out of range — visible\n")
        f.write(f"  endcase\n")
        f.write(f"endfunction\n")

    # KV cache pre-state — positions 0..args.pos-1 already filled.
    # Flat layout: position-major, then (h_kv, e).  256 entries × int16.
    k_flat = kv_cache_pre['k'].astype(np.int16).flatten()
    v_flat = kv_cache_pre['v'].astype(np.int16).flatten()
    write_hex("layer_K_CACHE_INIT.hex", k_flat, hex_word)
    write_hex("layer_V_CACHE_INIT.hex", v_flat, hex_word)

    # ---- Emit expected.txt ----
    txt = os.path.join(OUT_DIR, "layer_test_expected.txt")
    with open(txt, "w") as f:
        f.write(f"# layer_test golden trace.  Run: gen_layer_test.py --seed {args.seed} --pos {args.pos}\n")
        for name in ["norm1", "q", "k", "v", "q_rot", "k_rot",
                     "attn", "hidden1", "norm2", "mlp_inter", "hidden_out"]:
            arr = last_trace[name]
            f.write(f"# {name}  ({arr.shape[0]} values)\n")
            for v in arr:
                f.write(f"{int(v) & 0xFFFF:04x}  {int(v):+7d}\n")

    # Convenience: emit just the hidden_out values into a flat file the
    # FPGA selftest verifier can read directly.
    txt2 = os.path.join(OUT_DIR, "smollm_layer_selftest_expected.txt")
    with open(txt2, "w") as f:
        f.write(f"# smollm_layer hidden_out (Q1.15, lane 0 first)\n")
        for v in last_trace['hidden_out']:
            f.write(f"{int(v) & 0xFFFF:04x}  {int(v):+7d}\n")

    n_sat = int(np.sum((final_hidden_out == 32767) | (final_hidden_out == -32768)))
    print(f"D={D}  pos={args.pos}  hidden_out range "
          f"[{final_hidden_out.min()},{final_hidden_out.max()}]  saturated: {n_sat}/{D}",
          file=sys.stderr)
    print(f"wrote {svh}", file=sys.stderr)
    print(f"wrote {txt}", file=sys.stderr)


if __name__ == "__main__":
    main()
