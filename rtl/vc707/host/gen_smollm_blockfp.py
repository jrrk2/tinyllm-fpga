#!/usr/bin/env python3
"""Block-FP variant of gen_smollm_calib.py.

Hidden state: 16-bit Q1.15 with PER-LAYER scale = 2^p_hid.  Each layer
has h_in_p2, h1_p2, h_out_p2 (4-bit each).  Wrapper rescales hidden
state at inter-layer cascade.  Within a layer, two residual sums apply
signed shifts to align operands to the residual's target scale.

Files emitted (FPGA-compatible naming):
  tm_layer_HIDDEN_IN.hex           — 16-bit Q1.15 (one entry per line, 4 hex chars)
  tm_layer_expected.txt            — 16-bit Q1.15 at LAST layer's h_out_p2 scale
  layer_hidden_in_packed.svh       — case-statement, 16-bit entries
  tm_layer_data.svh                — TM_RESCALE[NL] : 64 bits/layer:
       [23:0]   resid1_factor_q24    (Q16.8 — multiplied by o_buf, >>8 to get Q1.15)
       [47:24]  resid2_factor_q24    (Q16.8 — multiplied by down_buf)
       [51:48]  h_in_p2              (this layer's hidden_in scale exponent)
       [55:52]  h_out_p2             (this layer's hidden_out scale exponent)
       [59:56]  sh_h_in_to_h1        (signed: h1_p2 - h_in_p2;  >0 = right shift)
       [63:60]  sh_h1_to_h_out       (signed: h_out_p2 - h1_p2; >0 = right shift)
  tm_layer_{GAMMA,SCALE_*}.hex     — calibrated, same format as before
  tm_layer_L<N>_W_*.hex            — per-layer weights (unchanged)
  tm_layer_RESCALE.hex             — diagnostics

Scales are powers of 2; sim_blockfp.py validates this produces coherent text.
"""
import argparse, os, sys, math
import numpy as np


def quantize_q15(x_real, scale):
    if scale <= 0: scale = 1e-6
    q = np.round(np.clip(x_real, -scale, scale - scale/32768) * (32768.0 / scale))
    return np.clip(q, -32768, 32767).astype(np.int16)

def dequantize_q15(x_int, scale): return x_int.astype(np.float64) * (scale / 32768.0)

def quantize_int8_per_row(W):
    amax = np.maximum(np.abs(W).max(axis=-1, keepdims=True), 1e-8)
    rsc = amax / 127.0
    return np.round(W / rsc).clip(-128, 127).astype(np.int8), rsc.squeeze(-1)


def matvec_int8(x_int16, x_scale, W_q8, W_rsc, out_scale):
    acc = (x_int16.astype(np.int64) @ W_q8.T.astype(np.int64))
    y_real = acc.astype(np.float64) * x_scale * W_rsc / 32768.0
    return quantize_q15(y_real, out_scale)


def rmsnorm(x_real, gamma, eps=1e-5):
    rms = np.sqrt(np.mean(x_real * x_real) + eps)
    return (x_real / rms) * gamma


def rope(x, pos, base=10000.0):
    HD = x.shape[0]; H2 = HD // 2; out = x.copy()
    for j in range(H2):
        f = 1.0/(base**(2.0*j/HD)); a = pos*f
        c, s = math.cos(a), math.sin(a)
        l, h = x[j], x[j+H2]
        out[j] = l*c - h*s; out[j+H2] = h*c + l*s
    return out


def silu(x): return x/(1.0 + np.exp(-x))
def softmax(x): e = np.exp(x - x.max()); return e / e.sum()


def pow2(x):
    if x <= 1: return 0
    return int(math.ceil(math.log2(x)))


def calibrate(model, tok, prompt, margin=1.5):
    NL = model.config.num_hidden_layers
    stats = {li: {} for li in range(NL)}
    handles = []
    def hf(li, key):
        def h(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            stats[li][key] = max(stats[li].get(key, 0.0), float(t.abs().max()))
        return h
    for li, L in enumerate(model.model.layers):
        for nm, sub in [("norm1", L.input_layernorm), ("q", L.self_attn.q_proj),
                        ("k", L.self_attn.k_proj), ("v", L.self_attn.v_proj),
                        ("attn", L.self_attn.o_proj), ("norm2", L.post_attention_layernorm),
                        ("gate", L.mlp.gate_proj), ("up", L.mlp.up_proj),
                        ("down", L.mlp.down_proj)]:
            handles.append(sub.register_forward_hook(hf(li, nm)))
        def make_lh(li):
            def h(_m, inp, out):
                hi = inp[0] if isinstance(inp, tuple) else inp
                ho = out[0] if isinstance(out, tuple) else out
                stats[li]["hidden_in"]  = max(stats[li].get("hidden_in",  0.0), float(hi.abs().max()))
                stats[li]["hidden_out"] = max(stats[li].get("hidden_out", 0.0), float(ho.abs().max()))
            return h
        handles.append(L.register_forward_hook(make_lh(li)))
    import torch
    with torch.no_grad():
        _ = model(tok(prompt, return_tensors="pt").input_ids)
    for h in handles: h.remove()
    sc = {li: {k: max(v*margin, 1e-3) for k, v in stats[li].items()} for li in range(NL)}
    for li in range(NL):
        sc[li].setdefault("mlp", max(sc[li]["gate"]*sc[li]["up"]*0.5, sc[li]["down"]))
    return sc


def fwd_blockfp(h_int16, lw, lsc, h_in_p2, h1_p2, h_out_p2, pos, kv, kv_pos, cfg):
    """Block-FP forward (matches RTL semantics).  h_int16 is at scale 2^h_in_p2."""
    D = cfg["D"]; H_Q=cfg["H_Q"]; H_KV=cfg["H_KV"]; HD=cfg["HD"]; grp=H_Q//H_KV
    s_in   = float(1 << h_in_p2)
    s_h1   = float(1 << h1_p2)
    s_out  = float(1 << h_out_p2)

    h_real = dequantize_q15(h_int16, s_in)
    n1 = quantize_q15(rmsnorm(h_real, lw["g1"]), lsc["norm1"])
    q  = matvec_int8(n1, lsc["norm1"], lw["Wq"], lw["Wq_r"], lsc["q"])
    k  = matvec_int8(n1, lsc["norm1"], lw["Wk"], lw["Wk_r"], lsc["k"])
    v  = matvec_int8(n1, lsc["norm1"], lw["Wv"], lw["Wv_r"], lsc["v"])
    qr = dequantize_q15(q, lsc["q"]); kr = dequantize_q15(k, lsc["k"]); vr = dequantize_q15(v, lsc["v"])
    for h in range(H_Q):  qr[h*HD:(h+1)*HD] = rope(qr[h*HD:(h+1)*HD], pos)
    for h in range(H_KV): kr[h*HD:(h+1)*HD] = rope(kr[h*HD:(h+1)*HD], pos)
    for h in range(H_KV):
        kv["k_int"][kv_pos, h] = quantize_q15(kr[h*HD:(h+1)*HD], lsc["k"])
        kv["v_int"][kv_pos, h] = quantize_q15(vr[h*HD:(h+1)*HD], lsc["v"])

    attn = np.zeros(D, dtype=np.float64)
    for h in range(H_Q):
        kv_h = h // grp
        scores = np.zeros(kv_pos+1)
        for t in range(kv_pos+1):
            scores[t] = float(np.dot(qr[h*HD:(h+1)*HD],
                                     dequantize_q15(kv["k_int"][t, kv_h], lsc["k"]))) / math.sqrt(HD)
        sm = softmax(scores)
        for t in range(kv_pos+1):
            attn[h*HD:(h+1)*HD] += sm[t] * dequantize_q15(kv["v_int"][t, kv_h], lsc["v"])
    attn_int = quantize_q15(attn, lsc["attn"])
    o_int    = matvec_int8(attn_int, lsc["attn"], lw["Wo"], lw["Wo_r"], lsc["attn"])

    # Residual 1 — both operands at s_h1; sat16
    h1_real = dequantize_q15(h_int16, s_in) + dequantize_q15(o_int, lsc["attn"])
    h1_int  = quantize_q15(h1_real, s_h1)

    n2 = quantize_q15(rmsnorm(dequantize_q15(h1_int, s_h1), lw["g2"]), lsc["norm2"])
    g  = matvec_int8(n2, lsc["norm2"], lw["Wg"], lw["Wg_r"], lsc["gate"])
    u  = matvec_int8(n2, lsc["norm2"], lw["Wu"], lw["Wu_r"], lsc["up"])
    mlp_real = silu(dequantize_q15(g, lsc["gate"])) * dequantize_q15(u, lsc["up"])
    mlp_int  = quantize_q15(mlp_real, lsc["mlp"])
    d_int    = matvec_int8(mlp_int, lsc["mlp"], lw["Wd"], lw["Wd_r"], lsc["down"])

    out_real = dequantize_q15(h1_int, s_h1) + dequantize_q15(d_int, lsc["down"])
    return quantize_q15(out_real, s_out)


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
    NL  = cfg["NL"];  D = cfg["D"];  POS = args.pos

    print(f"  D={D} NL={NL} H_Q={cfg['H_Q']} H_KV={cfg['H_KV']} FFN={cfg['FFN']}",
          file=sys.stderr)
    print(f"calibrating on {args.cal_prompt!r} ...", file=sys.stderr)
    sc = calibrate(model, tok, args.cal_prompt, margin=args.margin)

    # Per-layer block-FP shifts.  h1_p2 = max(h_in, h_in+attn).
    h_p2 = []
    for li in range(NL):
        h_in_p2  = pow2(sc[li].get("hidden_in",  1.0))
        h_out_p2 = pow2(sc[li].get("hidden_out", 1.0))
        h1_p2    = pow2(max(sc[li].get("hidden_in", 1.0) + sc[li]["attn"], 1.0))
        h_p2.append((h_in_p2, h1_p2, h_out_p2))
    print(f"  h-scales (p2) L0..L{NL-1}: " +
          ", ".join(f"({a},{b},{c})" for (a,b,c) in h_p2[:5]) + " ... " +
          ", ".join(f"({a},{b},{c})" for (a,b,c) in h_p2[-3:]), file=sys.stderr)

    # Extract + quantize weights
    print("extracting weights ...", file=sys.stderr)
    layers = []
    with torch.no_grad():
        for li in range(NL):
            L = model.model.layers[li]; d = {}
            for nm, sub in [("Wq", L.self_attn.q_proj), ("Wk", L.self_attn.k_proj),
                            ("Wv", L.self_attn.v_proj), ("Wo", L.self_attn.o_proj),
                            ("Wg", L.mlp.gate_proj),    ("Wu", L.mlp.up_proj),
                            ("Wd", L.mlp.down_proj)]:
                W = sub.weight.detach().cpu().numpy().astype(np.float32)
                Wq, rsc = quantize_int8_per_row(W)
                d[nm] = Wq; d[nm + "_r"] = rsc
            d["g1"] = L.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            d["g2"] = L.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            layers.append(d)
        embed     = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        norm_w    = model.model.norm.weight.detach().cpu().numpy().astype(np.float32)
        lm_head_w = embed   # tied

    # Cap matvec output scales so scale_q15 stays representable in 16-bit.
    # MIN_SCALE_Q15=4 trades some scale_q15 rounding noise for less
    # output saturation on the high-scale buses (down at layers 11+).
    MIN_SCALE_Q15 = 4
    def fold_matvec(W_rsc, s_in, s_out_in):
        wmax = float(np.max(W_rsc))
        cap  = (s_in * wmax / MIN_SCALE_Q15) * 32768.0
        s_out_eff = min(s_out_in, cap)
        v = np.round(s_in * W_rsc / s_out_eff * 32768.0)
        return np.clip(v, -32768, 32767).astype(np.int16), s_out_eff

    def fold_gamma(g, s_in_eff, s_out):
        v = np.round(g * s_in_eff / s_out * 32768.0)
        return np.clip(v, -32768, 32767).astype(np.int16)

    folded = []; eff_sc = []
    for li in range(NL):
        d = layers[li]; cs = sc[li]
        sca_q,    eq    = fold_matvec(d["Wq_r"], cs["norm1"], cs["q"])
        sca_k,    ek    = fold_matvec(d["Wk_r"], cs["norm1"], cs["k"])
        sca_v,    ev    = fold_matvec(d["Wv_r"], cs["norm1"], cs["v"])
        sca_o,    eo    = fold_matvec(d["Wo_r"], cs["attn"],  cs["attn"])
        sca_g,    egate = fold_matvec(d["Wg_r"], cs["norm2"], cs["gate"])
        sca_u,    eup   = fold_matvec(d["Wu_r"], cs["norm2"], cs["up"])
        sca_d,    edown = fold_matvec(d["Wd_r"], cs["mlp"],   cs["down"])
        folded.append({
            "sca_q":sca_q, "sca_k":sca_k, "sca_v":sca_v, "sca_o":sca_o,
            "sca_g":sca_g, "sca_u":sca_u, "sca_d":sca_d,
            "g1": fold_gamma(d["g1"], 1.0, cs["norm1"]),
            "g2": fold_gamma(d["g2"], 1.0, cs["norm2"]),
        })
        eff_sc.append({"norm1":cs["norm1"], "norm2":cs["norm2"], "mlp":cs["mlp"],
                       "q":eq, "k":ek, "v":ev, "attn":eo,
                       "gate":egate, "up":eup, "down":edown})

    # Tokenize + reference forward with eff_sc
    ids = tok(args.prompt, add_special_tokens=False).input_ids
    while len(ids) < cfg["MAX_CTX"]: ids.append(0)
    ids = ids[:cfg["MAX_CTX"]]
    print(f"  prompt={args.prompt!r} → ids={ids}", file=sys.stderr)
    print(f"  test pos {POS} = id {ids[POS]} = {tok.decode([ids[POS]])!r}", file=sys.stderr)

    kv = [{"k_int": np.zeros((cfg["MAX_CTX"], cfg["H_KV"], cfg["HD"]), dtype=np.int16),
           "v_int": np.zeros((cfg["MAX_CTX"], cfg["H_KV"], cfg["HD"]), dtype=np.int16)}
          for _ in range(NL)]
    pre_kv = [None]*NL
    final_h_in_int = None;  final_h_out_int = None
    for p in range(POS+1):
        # First-layer hidden_in = embedding quantized at layer 0's h_in_p2
        h = quantize_q15(embed[ids[p]].astype(np.float32), float(1 << h_p2[0][0]))
        if p == POS:
            for li in range(NL):
                pre_kv[li] = {"k": kv[li]["k_int"].copy(), "v": kv[li]["v_int"].copy()}
            final_h_in_int = h.copy()
        for li in range(NL):
            h_in_p2, h1_p2, h_out_p2 = h_p2[li]
            h = fwd_blockfp(h, layers[li], eff_sc[li], h_in_p2, h1_p2, h_out_p2,
                            p, kv[li], p, cfg)
            # Cascade: rescale h to next layer's h_in_p2
            if li < NL - 1:
                next_in_p2 = h_p2[li+1][0]
                shift = next_in_p2 - h_out_p2
                if shift > 0:    h = np.clip(h.astype(np.int32) >> shift, -32768, 32767).astype(np.int16)
                elif shift < 0:  h = np.clip(h.astype(np.int32) << (-shift), -32768, 32767).astype(np.int16)
        if p == POS:
            final_h_out_int = h.copy()

    n_sat = int(np.sum((final_h_out_int == 32767) | (final_h_out_int == -32768)))
    print(f"  final h_out range [{final_h_out_int.min()},{final_h_out_int.max()}]  saturated: {n_sat}/{D}",
          file=sys.stderr)

    # Quick decode
    last_h_out_p2 = h_p2[-1][2]
    h_real = dequantize_q15(final_h_out_int, float(1 << last_h_out_p2)).astype(np.float32)
    h_normed = (h_real / np.sqrt(np.mean(h_real*h_real) + 1e-5)) * norm_w
    logits = h_normed @ lm_head_w.T
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"\n  predicted top-5: " +
          ", ".join(f"{tok.decode([int(t)])!r}" for t in top5), file=sys.stderr)

    # ----------------------------------------------------------------------
    # Emit files
    # ----------------------------------------------------------------------
    prefix = "tm_layer_"
    for nm, key in [("Q","sca_q"),("K","sca_k"),("V","sca_v"),("O","sca_o"),
                    ("GATE","sca_g"),("UP","sca_u"),("DOWN","sca_d")]:
        with open(os.path.join(OUT_DIR, f"{prefix}SCALE_{nm}.hex"), "w") as f:
            for li in range(NL):
                for v in folded[li][key]:
                    f.write(f"{int(v) & 0xFFFF:04x}\n")
    for nm, key in [("GAMMA1","g1"),("GAMMA2","g2")]:
        with open(os.path.join(OUT_DIR, f"{prefix}{nm}.hex"), "w") as f:
            for li in range(NL):
                for v in folded[li][key]:
                    f.write(f"{int(v) & 0xFFFF:04x}\n")
    for nm, key in [("K_CACHE_INIT","k"),("V_CACHE_INIT","v")]:
        with open(os.path.join(OUT_DIR, f"{prefix}{nm}.hex"), "w") as f:
            for li in range(NL):
                for v in pre_kv[li][key].flatten():
                    f.write(f"{int(v) & 0xFFFF:04x}\n")
    for li in range(NL):
        for nm, mat in [("Q",  layers[li]["Wq"]),  ("K",   layers[li]["Wk"]),
                        ("V",  layers[li]["Wv"]),  ("O",   layers[li]["Wo"]),
                        ("GATE",layers[li]["Wg"]), ("UP",  layers[li]["Wu"]),
                        ("DOWN",layers[li]["Wd"])]:
            out_dim, in_dim = mat.shape
            with open(os.path.join(OUT_DIR, f"{prefix}L{li}_W_{nm}.hex"), "w") as f:
                for chunk in range(out_dim // 16):
                    for k in range(in_dim):
                        line = 0
                        for lane in range(16):
                            b = int(mat[chunk*16 + lane, k]) & 0xFF
                            line |= b << (lane * 8)
                        f.write(f"{line:032x}\n")

    # 16-bit hidden_in at layer 0's h_in_p2 scale
    with open(os.path.join(OUT_DIR, f"{prefix}HIDDEN_IN.hex"), "w") as f:
        for v in final_h_in_int:
            f.write(f"{int(v) & 0xFFFF:04x}\n")

    # Reference at last layer's h_out_p2 scale
    with open(os.path.join(OUT_DIR, f"{prefix}expected.txt"), "w") as f:
        f.write(f"# multilayer hidden_out (16-bit Q1.15, h_out_p2={last_h_out_p2})\n")
        f.write(f"# real SmolLM2-135M, prompt={args.prompt!r}, pos={POS}\n")
        for v in final_h_out_int:
            f.write(f"{int(v) & 0xFFFF:04x}  {int(v):+7d}\n")

    # case-statement hidden_in ROM (16-bit)
    with open(os.path.join(OUT_DIR, "layer_hidden_in_packed.svh"), "w") as f:
        f.write(f"// AUTO-GENERATED by gen_smollm_blockfp.py — do not edit.\n")
        f.write(f"// Real SmolLM2 embed of token id {ids[POS]} ({tok.decode([ids[POS]])!r}) "
                f"at pos {POS}.  16-bit Q1.15 at h_in_p2={h_p2[0][0]}.\n\n")
        f.write(f"function automatic logic [15:0] layer_hidden_in_lut(input int unsigned idx);\n")
        f.write(f"  case (idx)\n")
        for i, v in enumerate(final_h_in_int):
            f.write(f"    {i:>4}: layer_hidden_in_lut = 16'h{int(np.uint16(v)):04x};\n")
        f.write(f"    default: layer_hidden_in_lut = 16'hdead;\n")
        f.write(f"  endcase\n")
        f.write(f"endfunction\n")

    # Per-layer rescale data.  RTL applies: contribution = (delta_int16 * f) >> 8.
    # delta_int16 is at the bus's own scale s_bus; we want it at h_out's scale 2^h_p2.
    #   real_delta = delta_int16 * s_bus / 32768
    #   stored at h_out scale = real_delta / 2^h_p2 * 32768 = delta_int16 * s_bus / 2^h_p2
    # So f / 256 = s_bus / 2^h_p2  →  f = round(s_bus / 2^h_p2 * 256)
    RESID_Q = 8
    rescale_rows = []
    for li in range(NL):
        h_in_p2, h1_p2, h_out_p2 = h_p2[li]
        f1 = int(round(eff_sc[li]["attn"] / float(1 << h1_p2)    * (1 << RESID_Q)))
        f2 = int(round(eff_sc[li]["down"] / float(1 << h_out_p2) * (1 << RESID_Q)))
        f1 = max(1, min(f1, (1 << 24) - 1))
        f2 = max(1, min(f2, (1 << 24) - 1))
        sh1 = h1_p2 - h_in_p2
        sh2 = h_out_p2 - h1_p2
        rescale_rows.append((f1, f2, h_in_p2, h_out_p2, sh1, sh2))

    # ----------------------------------------------------------------------
    # Per-layer scale-aware SwiGLU + Attn-AV factors.
    # SwiGLU: gate, up, mlp live at different per-tensor scales; the SiLU LUT
    # is at SILU_LUT_SCALE (32).  RTL needs three Q-fixed factors per layer:
    #   gate_in_factor  = round(lsc[gate] / SILU_LUT_SCALE * 32768)   Q1.15
    #   up_in_factor    = round(lsc[up]   / SILU_LUT_SCALE * 32768)   Q1.15
    #   mlp_out_factor  = round(SILU_LUT_SCALE^2 / lsc[mlp] * 256)    Q16.8
    # Attn-AV: V is at lsc[v], result wants to be at lsc[attn]:
    #   attn_factor     = round(lsc[v] / lsc[attn] * 256)             Q16.8
    # ----------------------------------------------------------------------
    SILU_LUT_SCALE = 32.0
    swiglu_attn_rows = []
    for li in range(NL):
        e = eff_sc[li]
        gate_in_factor = int(round(e["gate"] / SILU_LUT_SCALE * 32768.0))
        up_in_factor   = int(round(e["up"]   / SILU_LUT_SCALE * 32768.0))
        mlp_out_factor = int(round(SILU_LUT_SCALE**2 / e["mlp"] * 256.0))
        attn_factor    = int(round(e["v"] / e["attn"] * 256.0))
        # Clip to representable range.  Q1.15 unsigned ≤ 32767; assume lsc[gate],
        # lsc[up] never exceed SILU_LUT_SCALE for SmolLM2 (calibrated max ~21).
        gate_in_factor = max(0, min(gate_in_factor, 32767))
        up_in_factor   = max(0, min(up_in_factor,   32767))
        mlp_out_factor = max(1, min(mlp_out_factor, (1 << 24) - 1))
        attn_factor    = max(1, min(attn_factor,    (1 << 24) - 1))
        swiglu_attn_rows.append((gate_in_factor, up_in_factor, mlp_out_factor, attn_factor))

    with open(os.path.join(OUT_DIR, f"{prefix}swiglu_attn.svh"), "w") as f:
        f.write(f"// AUTO-GENERATED by gen_smollm_blockfp.py — do not edit.\n")
        f.write(f"// SwiGLU + Attn-AV scale-aware factors, real SmolLM2-135M.\n")
        f.write(f"// SILU_LUT_SCALE = {SILU_LUT_SCALE} (also in gen_swiglu_lut.py, swiglu.sv).\n\n")
        f.write(f"// 64 bits/layer:\n")
        f.write(f"//   [15:0]   gate_in_factor   (Q1.15)\n")
        f.write(f"//   [31:16]  up_in_factor     (Q1.15)\n")
        f.write(f"//   [55:32]  mlp_out_factor   (Q16.8)\n")
        f.write(f"//   [63:56]  reserved\n")
        f.write(f"localparam logic [63:0] TM_SWIGLU_SCALES [0:{NL-1}] = '{{\n")
        for li, (gif, uif, mof, _) in enumerate(swiglu_attn_rows):
            v = (gif & 0xFFFF) | ((uif & 0xFFFF) << 16) | ((mof & 0xFFFFFF) << 32)
            comma = "," if li < NL - 1 else " "
            f.write(f"  64'h{v:016x}{comma}  // L{li}: gate_f={gif} up_f={uif} mlp_f={mof}\n")
        f.write(f"}};\n\n")
        f.write(f"localparam logic [23:0] TM_ATTN_FACTOR [0:{NL-1}] = '{{\n")
        for li, (_, _, _, af) in enumerate(swiglu_attn_rows):
            comma = "," if li < NL - 1 else " "
            f.write(f"  24'h{af:06x}{comma}  // L{li}: attn_f={af} (= round(lsc[v]/lsc[attn]*256))\n")
        f.write(f"}};\n")

    with open(os.path.join(OUT_DIR, f"{prefix}data.svh"), "w") as f:
        f.write(f"// AUTO-GENERATED by gen_smollm_blockfp.py — do not edit.\n")
        f.write(f"// Block-FP 16-bit hidden, real SmolLM2-135M, prompt={args.prompt!r}, pos={POS}\n\n")
        f.write(f"localparam int TM_NL  = {NL};\n")
        f.write(f"localparam int TM_POS = {POS};\n")
        f.write(f"// TM_RESCALE per layer (64 bits):\n")
        f.write(f"//   [23:0]   r1_factor (Q16.8)\n")
        f.write(f"//   [47:24]  r2_factor (Q16.8)\n")
        f.write(f"//   [51:48]  h_in_p2    (4-bit unsigned)\n")
        f.write(f"//   [55:52]  h_out_p2   (4-bit unsigned)\n")
        f.write(f"//   [59:56]  sh_h_in_to_h1   (signed 4-bit, two's complement)\n")
        f.write(f"//   [63:60]  sh_h1_to_h_out  (signed 4-bit, two's complement)\n")
        f.write(f"localparam logic [63:0] TM_RESCALE [0:{NL-1}] = '{{\n")
        for li, (r1, r2, hin, hout, s1, s2) in enumerate(rescale_rows):
            v = (r1 & 0xFFFFFF) | ((r2 & 0xFFFFFF) << 24) \
              | ((hin & 0xF) << 48) | ((hout & 0xF) << 52) \
              | ((s1 & 0xF)  << 56) | ((s2 & 0xF)  << 60)
            comma = "," if li < NL - 1 else " "
            f.write(f"  64'h{v:016x}{comma}  // L{li}: r1={r1} r2={r2} hin_p2={hin} hout_p2={hout} sh1={s1} sh2={s2}\n")
        f.write(f"}};\n")

    print(f"\nNL={NL} D={D} pos={POS}  block-FP 16-bit hidden", file=sys.stderr)
    print(f"  resid factors: r1=[{min(r[0] for r in rescale_rows)}..{max(r[0] for r in rescale_rows)}]  "
          f"r2=[{min(r[1] for r in rescale_rows)}..{max(r[1] for r in rescale_rows)}]", file=sys.stderr)
    print(f"\nNext: python3 host/gen_layer_ddr3.py --mode tm --nlayers {NL}", file=sys.stderr)


if __name__ == "__main__":
    main()
