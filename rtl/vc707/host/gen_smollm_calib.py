#!/usr/bin/env python3
"""Generate FPGA test data from REAL SmolLM2-135M weights, with per-tensor
calibrated scales folded into existing per-row weight scales/gammas, and
24-bit (Q15.9) residual stream.

Replaces gen_smollm_real.py for the real-text-generation flow.  The
quantization scheme is the one validated end-to-end by host/sim_fpga_arith.py
(produces "Once upon a time, there was a little girl named Lily…").

Files emitted (same filenames as gen_multilayer_test.py --mode tm so the
existing brom + DDR3 + Vivado flows pick them up):
  tm_layer_SCALE_<X>.hex      (folded with per-tensor s_in / s_out)
  tm_layer_GAMMA{1,2}.hex     (folded with per-tensor s_in / s_out)
  tm_layer_K_CACHE_INIT.hex   (Q1.15 at the layer's KV scale)
  tm_layer_V_CACHE_INIT.hex
  tm_layer_L<N>_W_{Q,K,V,O,GATE,UP,DOWN}.hex
  tm_layer_HIDDEN_IN.hex      *** NEW FORMAT: 24-bit Q15.9 (6 hex chars/line) ***
  tm_layer_expected.txt       *** NEW FORMAT: 24-bit Q15.9 (6 hex chars + signed dec) ***
  tm_layer_RESCALE.hex        *** NEW: per-layer (resid1_factor, resid2_factor,
                                  hidden_shift_in1, hidden_shift_in2) — 16-bit each ***
  tm_layer_data.svh           constants
  layer_hidden_in_packed.svh  case-statement ROM at 24-bit width
"""
import argparse, os, sys, math
import numpy as np

# ----------------------------------------------------------------------
# Hidden-state format: fixed Q15.9 in 24-bit signed.
#   Range:  ±32768                LSB: 1/512  ≈  0.00195
#   Stored: real_value * Q15_9_ONE
# ----------------------------------------------------------------------
HIDDEN_BITS  = 24
Q15_9_ONE    = 1 << 9
HIDDEN_MAX   = (1 << (HIDDEN_BITS - 1)) - 1
HIDDEN_MIN   = -(1 << (HIDDEN_BITS - 1))


def quantize_hidden_q159(x_real):
    """Real → 24-bit signed Q15.9, saturated."""
    q = np.round(x_real * Q15_9_ONE)
    return np.clip(q, HIDDEN_MIN, HIDDEN_MAX).astype(np.int32)


def dequantize_hidden_q159(x_int):
    return x_int.astype(np.float64) / Q15_9_ONE


def quantize_q15_per_tensor(x_real, scale):
    """Real → 16-bit signed Q1.15 with full-scale = scale."""
    if scale <= 0: scale = 1e-6
    q = np.round(np.clip(x_real, -scale, scale - scale/32768) * (32768.0 / scale))
    return np.clip(q, -32768, 32767).astype(np.int16)


def dequantize_q15_per_tensor(x_int, scale):
    return x_int.astype(np.float64) * (scale / 32768.0)


def quantize_int8_per_row(W):
    """Per-row symmetric INT8 quantization.  Returns (Wq[int8], row_scale[float])."""
    amax = np.maximum(np.abs(W).max(axis=-1, keepdims=True), 1e-8)
    row_scale = amax / 127.0
    Wq = np.round(W / row_scale).clip(-128, 127).astype(np.int8)
    return Wq, row_scale.squeeze(-1)


# ----------------------------------------------------------------------
# Per-FPGA arithmetic forward (mirrors what RTL will implement)
# ----------------------------------------------------------------------

def matvec_int8_q(x_int16, x_scale, W_q8, W_rsc, out_scale):
    """Emulate the FPGA matvec engine: int16 act × int8 weight → narrow at out_scale.
       Folds (x_scale * W_rsc / out_scale) into a single scale_q15 per row,
       which is what we will store on the FPGA."""
    acc = (x_int16.astype(np.int64) @ W_q8.T.astype(np.int64))
    y_real = acc.astype(np.float64) * x_scale * W_rsc / 32768.0
    return quantize_q15_per_tensor(y_real, out_scale)


def rmsnorm_q(x_real, gamma_real, eps=1e-5):
    """RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps).  Scale-invariant on input."""
    rms = np.sqrt(np.mean(x_real * x_real) + eps)
    return (x_real / rms) * gamma_real


def rope_q(x_real, pos, base=10000.0):
    HD = x_real.shape[0]; H2 = HD // 2
    out = x_real.copy()
    for j in range(H2):
        freq = 1.0 / (base ** (2.0 * j / HD))
        ang  = pos * freq
        c, s = math.cos(ang), math.sin(ang)
        x_lo = x_real[j];     x_hi = x_real[j + H2]
        out[j]      = x_lo * c - x_hi * s
        out[j + H2] = x_hi * c + x_lo * s
    return out


def silu(x): return x / (1.0 + np.exp(-x))

def softmax_q(x_real):
    x = x_real - x_real.max()
    e = np.exp(x)
    return e / e.sum()


def forward_layer_fpga(hidden_in_int24, layer_w, layer_sc, pos, kv, kv_pos, cfg):
    """Per-layer forward emulating the FPGA's exact integer arithmetic.
       hidden_in_int24: 24-bit Q15.9 stored array [D]
       Returns hidden_out_int24 [D]."""
    D    = cfg["D"]
    H_Q  = cfg["H_Q"];  H_KV = cfg["H_KV"];  HD = cfg["HD"]
    grp  = H_Q // H_KV

    h_real = dequantize_hidden_q159(hidden_in_int24)

    # RMSNorm 1 (scale-invariant input — feed real or shifted, math equivalent)
    n1_real = rmsnorm_q(h_real, layer_w["g1"])
    n1_int  = quantize_q15_per_tensor(n1_real, layer_sc["norm1"])

    # Q, K, V matvecs
    q = matvec_int8_q(n1_int, layer_sc["norm1"], layer_w["Wq_q8"], layer_w["Wq_rsc"], layer_sc["q"])
    k = matvec_int8_q(n1_int, layer_sc["norm1"], layer_w["Wk_q8"], layer_w["Wk_rsc"], layer_sc["k"])
    v = matvec_int8_q(n1_int, layer_sc["norm1"], layer_w["Wv_q8"], layer_w["Wv_rsc"], layer_sc["v"])

    # RoPE on Q, K (per head); V untouched
    q_real = dequantize_q15_per_tensor(q, layer_sc["q"])
    k_real = dequantize_q15_per_tensor(k, layer_sc["k"])
    for h in range(H_Q):
        q_real[h*HD:(h+1)*HD] = rope_q(q_real[h*HD:(h+1)*HD], pos)
    for h in range(H_KV):
        k_real[h*HD:(h+1)*HD] = rope_q(k_real[h*HD:(h+1)*HD], pos)

    # Write KV cache (Q1.15 at bus scale)
    for h in range(H_KV):
        kv["k_int"][kv_pos, h] = quantize_q15_per_tensor(k_real[h*HD:(h+1)*HD], layer_sc["k"])
        kv["v_int"][kv_pos, h] = quantize_q15_per_tensor(
            dequantize_q15_per_tensor(v[h*HD:(h+1)*HD], layer_sc["v"]),
            layer_sc["v"])

    # Attention: Q@K^T / sqrt(HD), softmax, @V (per Q head, GQA grouping)
    attn_real = np.zeros(D, dtype=np.float64)
    for h in range(H_Q):
        kv_h = h // grp
        q_h  = q_real[h*HD:(h+1)*HD]
        scores = np.zeros(kv_pos + 1, dtype=np.float64)
        for t in range(kv_pos + 1):
            k_th = dequantize_q15_per_tensor(kv["k_int"][t, kv_h], layer_sc["k"])
            scores[t] = float(np.dot(q_h, k_th)) / math.sqrt(HD)
        sm = softmax_q(scores)
        for t in range(kv_pos + 1):
            v_th = dequantize_q15_per_tensor(kv["v_int"][t, kv_h], layer_sc["v"])
            attn_real[h*HD:(h+1)*HD] += sm[t] * v_th

    attn_int = quantize_q15_per_tensor(attn_real, layer_sc["attn"])

    # O projection
    o_int  = matvec_int8_q(attn_int, layer_sc["attn"], layer_w["Wo_q8"], layer_w["Wo_rsc"], layer_sc["attn"])
    o_real = dequantize_q15_per_tensor(o_int, layer_sc["attn"])

    # Residual 1 (WIDE, Q15.9)
    h1_real = dequantize_hidden_q159(hidden_in_int24) + o_real
    h1_int  = quantize_hidden_q159(h1_real)

    # RMSNorm 2
    n2_real = rmsnorm_q(dequantize_hidden_q159(h1_int), layer_w["g2"])
    n2_int  = quantize_q15_per_tensor(n2_real, layer_sc["norm2"])

    # Gate, Up
    gate_int = matvec_int8_q(n2_int, layer_sc["norm2"], layer_w["Wgate_q8"], layer_w["Wgate_rsc"], layer_sc["gate"])
    up_int   = matvec_int8_q(n2_int, layer_sc["norm2"], layer_w["Wup_q8"],   layer_w["Wup_rsc"],   layer_sc["up"])
    gate_real = dequantize_q15_per_tensor(gate_int, layer_sc["gate"])
    up_real   = dequantize_q15_per_tensor(up_int,   layer_sc["up"])

    # SwiGLU
    mlp_real = silu(gate_real) * up_real
    mlp_int  = quantize_q15_per_tensor(mlp_real, layer_sc["mlp"])
    mlp_q    = dequantize_q15_per_tensor(mlp_int, layer_sc["mlp"])
    mlp_re   = quantize_q15_per_tensor(mlp_q, layer_sc["mlp"])  # round-trip stable

    # Down
    down_int  = matvec_int8_q(mlp_re, layer_sc["mlp"], layer_w["Wdown_q8"], layer_w["Wdown_rsc"], layer_sc["down"])
    down_real = dequantize_q15_per_tensor(down_int, layer_sc["down"])

    # Residual 2 (WIDE)
    ho_real = dequantize_hidden_q159(h1_int) + down_real
    return quantize_hidden_q159(ho_real)


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

def calibrate(model, tok, prompt: str, margin: float = 1.5):
    NL = model.config.num_hidden_layers
    stats = {li: {} for li in range(NL)}
    handles = []
    def hf(li, key):
        def h(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            v = float(t.abs().max())
            stats[li][key] = max(stats[li].get(key, 0.0), v)
        return h
    for li, L in enumerate(model.model.layers):
        handles.append(L.input_layernorm.register_forward_hook(hf(li, "norm1")))
        handles.append(L.self_attn.q_proj.register_forward_hook(hf(li, "q")))
        handles.append(L.self_attn.k_proj.register_forward_hook(hf(li, "k")))
        handles.append(L.self_attn.v_proj.register_forward_hook(hf(li, "v")))
        handles.append(L.self_attn.o_proj.register_forward_hook(hf(li, "attn")))
        handles.append(L.post_attention_layernorm.register_forward_hook(hf(li, "norm2")))
        handles.append(L.mlp.gate_proj.register_forward_hook(hf(li, "gate")))
        handles.append(L.mlp.up_proj.register_forward_hook(hf(li, "up")))
        handles.append(L.mlp.down_proj.register_forward_hook(hf(li, "down")))
        # Layer wrapper hook captures hidden_in (input) and hidden_out (out)
        def make_layer_hook(li):
            def lh(_m, inp, out):
                hi = inp[0] if isinstance(inp, tuple) else inp
                ho = out[0] if isinstance(out, tuple) else out
                stats[li]["hidden_in"]  = max(stats[li].get("hidden_in",  0.0), float(hi.abs().max()))
                stats[li]["hidden_out"] = max(stats[li].get("hidden_out", 0.0), float(ho.abs().max()))
            return lh
        handles.append(L.register_forward_hook(make_layer_hook(li)))

    import torch
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        _ = model(ids)
    for h in handles: h.remove()
    scales = {li: {k: max(v * margin, 1e-3) for k, v in stats[li].items()} for li in range(NL)}
    # mlp not directly hooked; conservatively bound by gate * up worst case (silu cap≈1)
    for li in range(NL):
        if "mlp" not in scales[li]:
            scales[li]["mlp"] = max(scales[li]["gate"] * scales[li]["up"] * 0.5,
                                    scales[li]["down"], 1e-3)
    return scales


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",     default="Once upon a time")
    ap.add_argument("--cal-prompt", default="Once upon a time there was a princess "
                                            "who lived in a castle by the sea.")
    ap.add_argument("--pos",        type=int, default=3)
    ap.add_argument("--max-ctx",    type=int, default=4)
    ap.add_argument("--margin",     type=float, default=1.5)
    args = ap.parse_args()

    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "generated")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"loading SmolLM2-135M ...", file=sys.stderr)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok   = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float32).eval()
    cfg = dict(
        D       = model.config.hidden_size,
        H_Q     = model.config.num_attention_heads,
        H_KV    = model.config.num_key_value_heads,
        HD      = model.config.hidden_size // model.config.num_attention_heads,
        FFN     = model.config.intermediate_size,
        NL      = model.config.num_hidden_layers,
        MAX_CTX = args.max_ctx,
    )
    NL  = cfg["NL"]
    D   = cfg["D"]
    POS = args.pos
    print(f"  D={D} NL={NL} H_Q={cfg['H_Q']} H_KV={cfg['H_KV']} FFN={cfg['FFN']} MAX_CTX={cfg['MAX_CTX']}",
          file=sys.stderr)

    print(f"calibrating on {args.cal_prompt!r} ...", file=sys.stderr)
    scales = calibrate(model, tok, args.cal_prompt, margin=args.margin)

    print("extracting + quantizing layer weights ...", file=sys.stderr)
    layers = []
    with torch.no_grad():
        for li in range(NL):
            L = model.model.layers[li]
            d = {}
            for nm, mod_attr in [
                ("Wq",   "self_attn.q_proj"),
                ("Wk",   "self_attn.k_proj"),
                ("Wv",   "self_attn.v_proj"),
                ("Wo",   "self_attn.o_proj"),
                ("Wgate","mlp.gate_proj"),
                ("Wup",  "mlp.up_proj"),
                ("Wdown","mlp.down_proj"),
            ]:
                m = L
                for p in mod_attr.split("."):
                    m = getattr(m, p)
                W  = m.weight.detach().cpu().numpy().astype(np.float32)
                Wq, sc = quantize_int8_per_row(W)
                d[nm + "_q8"]  = Wq
                d[nm + "_rsc"] = sc
            d["g1"] = L.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            d["g2"] = L.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            layers.append(d)
        embed = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)

    # ----------------------------------------------------------------------
    # Compute folded scale_q15 + effective bus scales (post cap) FIRST,
    # then run the reference forward with the SAME effective scales the
    # FPGA will see.  Otherwise reference and hardware diverge.
    # ----------------------------------------------------------------------
    # NB: definitions of fold_matvec_scales, fold_gamma, MIN_SCALE_Q15
    # appear later in the file — reorder by precomputing here.
    MIN_SCALE_Q15 = 32
    def _fold_matvec(W_rsc, s_in, s_out_in):
        wmax = float(np.max(W_rsc))
        s_out_cap = (s_in * wmax / MIN_SCALE_Q15) * 32768.0
        s_out_eff = min(s_out_in, s_out_cap)
        v = np.round(s_in * W_rsc / s_out_eff * 32768.0)
        return np.clip(v, -32768, 32767).astype(np.int16), s_out_eff

    eff_scales = []
    folded_pre = []   # to use in file emission below (avoid recomputing)
    for li in range(NL):
        d = layers[li]; sc = scales[li]
        sca_q,    eff_q    = _fold_matvec(d["Wq_rsc"],    sc["norm1"], sc["q"])
        sca_k,    eff_k    = _fold_matvec(d["Wk_rsc"],    sc["norm1"], sc["k"])
        sca_v,    eff_v    = _fold_matvec(d["Wv_rsc"],    sc["norm1"], sc["v"])
        sca_o,    eff_o    = _fold_matvec(d["Wo_rsc"],    sc["attn"],  sc["attn"])
        sca_gate, eff_gate = _fold_matvec(d["Wgate_rsc"], sc["norm2"], sc["gate"])
        sca_up,   eff_up   = _fold_matvec(d["Wup_rsc"],   sc["norm2"], sc["up"])
        sca_down, eff_down = _fold_matvec(d["Wdown_rsc"], sc["mlp"],   sc["down"])
        eff_scales.append({
            "norm1": sc["norm1"], "norm2": sc["norm2"], "mlp": sc["mlp"],
            "q": eff_q, "k": eff_k, "v": eff_v, "attn": eff_o,
            "gate": eff_gate, "up": eff_up, "down": eff_down,
        })
        folded_pre.append({
            "sca_q": sca_q, "sca_k": sca_k, "sca_v": sca_v, "sca_o": sca_o,
            "sca_gate": sca_gate, "sca_up": sca_up, "sca_down": sca_down,
        })

    print(f"  s_out caps applied (most aggressive): " +
          ", ".join(f"{b}=[{min(eff_scales[li][b] for li in range(NL)):.1f}.."
                       f"{max(eff_scales[li][b] for li in range(NL)):.1f}]"
                    for b in ("attn","down","gate","up")), file=sys.stderr)

    # ----------------------------------------------------------------------
    # Tokenize prompt and build cascaded KV state up to position POS.
    # ----------------------------------------------------------------------
    ids = tok(args.prompt, add_special_tokens=False).input_ids
    while len(ids) < cfg["MAX_CTX"]:
        ids.append(0)
    ids = ids[:cfg["MAX_CTX"]]
    print(f"  prompt={args.prompt!r} → ids[:{cfg['MAX_CTX']}]={ids}", file=sys.stderr)
    print(f"  test position {POS} = id {ids[POS]} = {tok.decode([ids[POS]])!r}",
          file=sys.stderr)

    kv_caches = [{
        "k_int": np.zeros((cfg["MAX_CTX"], cfg["H_KV"], cfg["HD"]), dtype=np.int16),
        "v_int": np.zeros((cfg["MAX_CTX"], cfg["H_KV"], cfg["HD"]), dtype=np.int16),
    } for _ in range(NL)]
    pre_caches = [None] * NL
    final_hidden_in_int24 = None
    final_hidden_out_int24 = None
    for p in range(POS + 1):
        h_int = quantize_hidden_q159(embed[ids[p]].astype(np.float32))
        if p == POS:
            for li in range(NL):
                pre_caches[li] = {
                    "k": kv_caches[li]["k_int"].copy(),
                    "v": kv_caches[li]["v_int"].copy(),
                }
            final_hidden_in_int24 = h_int.copy()
        for li in range(NL):
            h_int = forward_layer_fpga(h_int, layers[li], eff_scales[li], p, kv_caches[li], p, cfg)
        if p == POS:
            final_hidden_out_int24 = h_int.copy()

    n_sat = int(np.sum((final_hidden_out_int24 == HIDDEN_MAX) |
                       (final_hidden_out_int24 == HIDDEN_MIN)))
    print(f"  hidden_out (Q15.9) range [{final_hidden_out_int24.min()},{final_hidden_out_int24.max()}]"
          f"  saturated: {n_sat}/{D}", file=sys.stderr)

    # ----------------------------------------------------------------------
    # Fold per-tensor scales into matvec scale_q15 and gammas.
    # ----------------------------------------------------------------------
    # For each layer, for each matvec, compute new per-row scale_q15:
    #   scale_q15[row] = round(s_in * row_scale_real / s_out * 32768)
    # For gammas: gamma_q15 = round(real_gamma * s_in / s_out * 32768)
    # For RMSNorm 1: input is RMSNORM-shifted hidden (scale-invariant in real),
    #   but treated as Q1.15 representing [-1, 1).  The shift_amt sets the
    #   effective input scale.  Since output of RMSNorm is gamma*x/rms which
    #   is scale-invariant, gamma must be calibrated for the desired output
    #   bus scale (s_out_norm).  s_in_norm cancels; gamma_q15 = real_g * s_in_norm_eff / s_out_norm * 32768.
    #   We pick s_in_norm_eff = 1 (treat the post-shift hidden as Q1.15).
    # ----------------------------------------------------------------------
    # Minimum scale_q15 we'll allow per row before capping s_out.  The
    # matvec engine computes (acc * scale_q15) >> 15; scale_q15 < ~16
    # collapses precision to <6%, which compounds catastrophically over
    # 30 layers.  Capping s_out shrinks the bus range — narrow values
    # saturate but the wide residual stream still accumulates.
    MIN_SCALE_Q15 = 32

    def fold_matvec_scales(W_rsc, s_in, s_out_in):
        """Return (scale_q15 array per row, s_out_eff used after capping)."""
        # Smallest s_out that yields scale_q15 ≥ MIN for ALL rows.
        wmax = float(np.max(W_rsc))
        s_out_cap = (s_in * wmax / MIN_SCALE_Q15) * 32768.0
        s_out_eff = min(s_out_in, s_out_cap)
        v = np.round(s_in * W_rsc / s_out_eff * 32768.0)
        return np.clip(v, -32768, 32767).astype(np.int16), s_out_eff

    def fold_gamma(g_real, s_in_eff, s_out):
        v = np.round(g_real * s_in_eff / s_out * 32768.0)
        return np.clip(v, -32768, 32767).astype(np.int16)

    # ----------------------------------------------------------------------
    # Compute residual rescale factors and hidden→RMSNorm shifts.
    #
    # Residual:  h24_new = h24 + delta_int16 * factor_int16
    #   real_h_new = real_h + real_delta
    #   real_delta = delta_int16 / 32768 * s_delta
    #   stored_int24 = real * Q15_9_ONE = real_h*Q15_9_ONE + delta_int16*s_delta*Q15_9_ONE/32768
    #   factor_int16 = round(s_delta * Q15_9_ONE / 32768)
    #
    # hidden→RMSNorm shift: pick smallest right-shift of the 24-bit hidden
    # that fits into 16-bit signed.  Per layer.
    # ----------------------------------------------------------------------
    # Residual rescale factors stored in Q16.8 — 8 fractional bits give
    # us precision down to 1/256 ≈ 0.004, enough for very small bus
    # scales (e.g. layer-0 attn ≈ 3 → factor ≈ 0.047 < 1).  RTL applies:
    #   contribution = (delta_int16 * factor_q24) >> 8
    RESID_Q = 8

    # rms*_shift is per-layer: chosen so that the LARGEST hidden value
    # likely to appear at this layer fits in 16-bit signed after the shift.
    # Smaller shift = more precision; too-small shift = saturation.
    # We use the FP-calibrated max as the upper bound.
    def shift_for_max(max_real):
        # Q15.9-stored max = max_real * 512.  Want shifted ≤ 32767.
        # shift = ceil(log2((max_real * 512) / 32767))
        if max_real <= 0: return 0
        v = max_real * 512.0
        s = 0
        while v > 32767 and s < 15:
            v *= 0.5; s += 1
        return s

    rescale_data = []
    for li in range(NL):
        f1 = int(round(eff_scales[li]["attn"] * Q15_9_ONE / 32768.0 * (1 << RESID_Q)))
        f1 = max(1, min(f1, (1 << 24) - 1))
        f2 = int(round(eff_scales[li]["down"] * Q15_9_ONE / 32768.0 * (1 << RESID_Q)))
        f2 = max(1, min(f2, (1 << 24) - 1))
        # rms1_shift: rms1 reads hidden_in (= previous layer's hidden_out for
        # li > 0; embedding for li == 0).  Use FP calibration's hidden_in max.
        # rms2_shift: rms2 reads hidden1 (after first residual).  Approximate
        # by max of hidden_in and attn (worst-case sum).
        h_in_max = scales[li].get("hidden_in", 1.0) if li > 0 else 1.0
        h1_max   = h_in_max + scales[li].get("attn", 1.0)
        rescale_data.append((f1, f2, shift_for_max(h_in_max), shift_for_max(h1_max)))

    # ----------------------------------------------------------------------
    # Recompute the scale/gamma .hex contents with calibrated folding.
    # ----------------------------------------------------------------------
    # Matvec input scales:
    #   - Q/K/V matvecs take norm1 (s_in = scales[li]["norm1"])
    #   - O matvec takes attn (s_in = scales[li]["attn"])
    #   - Gate/Up matvecs take norm2 (s_in = scales[li]["norm2"])
    #   - Down matvec takes mlp  (s_in = scales[li]["mlp"])
    # Gammas:
    #   - g1 input is post-shift hidden treated as Q1.15 (s_in_eff = 1)
    #   - g2 input is post-shift hidden treated as Q1.15 (s_in_eff = 1)
    #   - both feed RMSNorm whose output bus scale is scales[li]["normN"]
    #
    # NOTE: forward_layer_fpga above used REAL gammas / scales — that's the
    # "high-fidelity" reference.  The FPGA stores int16 versions, with all
    # the per-tensor scale folding baked in.
    folded = []
    for li in range(NL):
        d  = layers[li]
        fp = folded_pre[li]
        folded.append({
            **fp,
            "g1_q15":   fold_gamma(d["g1"], 1.0, scales[li]["norm1"]),
            "g2_q15":   fold_gamma(d["g2"], 1.0, scales[li]["norm2"]),
        })

    # ----------------------------------------------------------------------
    # Emit files
    # ----------------------------------------------------------------------
    prefix = "tm_layer_"

    for name, key in [("Q","sca_q"),("K","sca_k"),("V","sca_v"),("O","sca_o"),
                      ("GATE","sca_gate"),("UP","sca_up"),("DOWN","sca_down")]:
        with open(os.path.join(OUT_DIR, f"{prefix}SCALE_{name}.hex"), "w") as f:
            for li in range(NL):
                for v in folded[li][key]:
                    f.write(f"{int(v) & 0xFFFF:04x}\n")
    for name, key in [("GAMMA1","g1_q15"), ("GAMMA2","g2_q15")]:
        with open(os.path.join(OUT_DIR, f"{prefix}{name}.hex"), "w") as f:
            for li in range(NL):
                for v in folded[li][key]:
                    f.write(f"{int(v) & 0xFFFF:04x}\n")
    for name, key in [("K_CACHE_INIT","k"), ("V_CACHE_INIT","v")]:
        with open(os.path.join(OUT_DIR, f"{prefix}{name}.hex"), "w") as f:
            for li in range(NL):
                for v in pre_caches[li][key].flatten():
                    f.write(f"{int(v) & 0xFFFF:04x}\n")

    # Per-layer weight .hex (unchanged format from gen_multilayer_test.py)
    for li in range(NL):
        for name, mat in [("Q",   layers[li]["Wq_q8"]), ("K",   layers[li]["Wk_q8"]),
                          ("V",   layers[li]["Wv_q8"]), ("O",   layers[li]["Wo_q8"]),
                          ("GATE",layers[li]["Wgate_q8"]),
                          ("UP",  layers[li]["Wup_q8"]),
                          ("DOWN",layers[li]["Wdown_q8"])]:
            out_dim, in_dim = mat.shape
            assert out_dim % 16 == 0
            with open(os.path.join(OUT_DIR, f"{prefix}L{li}_W_{name}.hex"), "w") as f:
                for chunk in range(out_dim // 16):
                    for k in range(in_dim):
                        line = 0
                        for lane in range(16):
                            b = int(mat[chunk*16 + lane, k]) & 0xFF
                            line |= b << (lane * 8)
                        f.write(f"{line:032x}\n")

    # 24-bit hidden_in (Q15.9, 6 hex chars per line)
    with open(os.path.join(OUT_DIR, f"{prefix}HIDDEN_IN.hex"), "w") as f:
        for v in final_hidden_in_int24:
            f.write(f"{int(v) & 0xFFFFFF:06x}\n")

    # Reference 24-bit hidden_out
    with open(os.path.join(OUT_DIR, f"{prefix}expected.txt"), "w") as f:
        f.write(f"# tm-multilayer final hidden_out (24-bit Q15.9)\n")
        f.write(f"# real SmolLM2-135M, prompt={args.prompt!r}, pos={POS}, NL={NL}\n")
        for v in final_hidden_out_int24:
            f.write(f"{int(v) & 0xFFFFFF:06x}  {int(v):+9d}\n")

    # Per-layer rescale brom data (24-bit r1, 24-bit r2, 4-bit s1, 4-bit s2)
    with open(os.path.join(OUT_DIR, f"{prefix}RESCALE.hex"), "w") as f:
        f.write(f"# Per-layer rescale (resid1_q16.8, resid2_q16.8, rms1_shift, rms2_shift)\n")
        for li in range(NL):
            r1, r2, s1, s2 = rescale_data[li]
            f.write(f"{r1 & 0xFFFFFF:06x} {r2 & 0xFFFFFF:06x} {s1 & 0xF:01x} {s2 & 0xF:01x}\n")

    # case-statement hidden_in ROM (24-bit per entry)
    with open(os.path.join(OUT_DIR, "layer_hidden_in_packed.svh"), "w") as f:
        f.write(f"// AUTO-GENERATED by gen_smollm_calib.py — do not edit.\n")
        f.write(f"// Real SmolLM2 embed of token id {ids[POS]} ({tok.decode([ids[POS]])!r}) "
                f"at pos {POS}.  Q15.9 (24-bit signed).\n\n")
        f.write(f"function automatic logic [23:0] layer_hidden_in_lut(input int unsigned idx);\n")
        f.write(f"  case (idx)\n")
        for i, v in enumerate(final_hidden_in_int24):
            f.write(f"    {i:>4}: layer_hidden_in_lut = 24'h{int(v) & 0xFFFFFF:06x};\n")
        f.write(f"    default: layer_hidden_in_lut = 24'hdeadbe;\n")
        f.write(f"  endcase\n")
        f.write(f"endfunction\n")

    # SVH constants
    with open(os.path.join(OUT_DIR, f"{prefix}data.svh"), "w") as f:
        f.write(f"// AUTO-GENERATED by gen_smollm_calib.py — do not edit.\n")
        f.write(f"// SmolLM2-135M, prompt={args.prompt!r}, pos={POS}\n\n")
        f.write(f"localparam int TM_NL  = {NL};\n")
        f.write(f"localparam int TM_POS = {POS};\n")
        f.write(f"// Rescale factors for the residual adders (per layer):\n")
        f.write(f"//   bits[23:0]   = resid1_factor_q16.8 (× attn_int16, then >>8 → Q15.9)\n")
        f.write(f"//   bits[47:24]  = resid2_factor_q16.8 (× down_int16, then >>8 → Q15.9)\n")
        f.write(f"//   bits[51:48]  = rms1_shift  (right-shift of hidden→rmsnorm input)\n")
        f.write(f"//   bits[55:52]  = rms2_shift\n")
        f.write(f"localparam logic [63:0] TM_RESCALE [0:{NL-1}] = '{{\n")
        for li in range(NL):
            r1, r2, s1, s2 = rescale_data[li]
            v = ((r1 & 0xFFFFFF))           \
              | ((r2 & 0xFFFFFF) << 24)     \
              | ((s1 & 0xF)      << 48)     \
              | ((s2 & 0xF)      << 52)
            comma = "," if li < NL - 1 else " "
            f.write(f"  64'h{v:016x}{comma}  // L{li}: r1={r1} r2={r2} s1={s1} s2={s2}\n")
        f.write(f"}};\n")

    # ----------------------------------------------------------------------
    # Sanity: project the int24 hidden_out through lm_head and decode the
    # predicted next token.  Should match what sim_fpga_arith.py produced.
    # ----------------------------------------------------------------------
    with torch.no_grad():
        norm_final = model.model.norm.weight.detach().cpu().numpy().astype(np.float32)
        lm_head_w  = embed                              # SmolLM2 ties embed/lm_head
    h_real = dequantize_hidden_q159(final_hidden_out_int24).astype(np.float32)
    h_normed = (h_real / np.sqrt(np.mean(h_real*h_real) + 1e-5)) * norm_final
    logits = h_normed @ lm_head_w.T
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"\n  predicted next token (top-5 from FPGA-equivalent reference):",
          file=sys.stderr)
    for t in top5:
        print(f"    {tok.decode([int(t)])!r}  (logit {logits[int(t)]:.2f})",
              file=sys.stderr)

    print(f"\nNL={NL}  D={D}  pos={POS}  REAL SmolLM2 weights, calibrated", file=sys.stderr)
    print(f"  resid factors: r1=[{min(r[0] for r in rescale_data)}..{max(r[0] for r in rescale_data)}]  "
          f"r2=[{min(r[1] for r in rescale_data)}..{max(r[1] for r in rescale_data)}]", file=sys.stderr)
    print(f"  hidden→RMSNorm shifts: {rescale_data[0][2]} (uniform across layers)", file=sys.stderr)
    print(f"\nNext: python3 host/gen_layer_ddr3.py --mode tm --nlayers {NL}", file=sys.stderr)


if __name__ == "__main__":
    main()
