#!/usr/bin/env python3
"""sim_blockfp.py with INTEGER RMSNorm emulating rmsnorm.sv exactly.
If this produces coherent text, the FPGA's gibberish is a separate bug
(not the integer-RMSNorm precision).  If it produces gibberish, we know
RMSNorm widening is the actual blocker."""
import sys, math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/jonathan/TALOS-V2/rtl/vc707/host")
from sim_blockfp import (
    quantize_q15, dequantize_q15, quantize_int8_row, matvec_int8,
    rope, silu, softmax, calibrate, pow2, MODEL,
)


def rmsnorm_int_emul(x_int16, gamma_q15, D, eps_real=1e-5):
    """Integer RMSNorm matching rmsnorm.sv exactly.
    Inputs: int16 x and int16 gamma_q15.  Output int16 y_q15."""
    x = x_int16.astype(np.int64)
    sum_sq = int((x * x).sum())                      # int64 (always non-negative)
    INV_D  = (1 << 32) // D
    mean_sq_q30 = (sum_sq * INV_D) >> 32             # Q2.30
    # Add eps in Q2.30: eps_real = 1e-5 → Q2.30 = round(1e-5 * 2^30) = 10737
    EPS_Q30 = int(round(eps_real * (1 << 30)))
    v_q30 = mean_sq_q30 + EPS_Q30
    if v_q30 <= 0: v_q30 = 1
    # inv_rms in Q5.12 unsigned = round(2^12 / sqrt(v_real))
    # v_real = v_q30 / 2^30
    v_real = v_q30 / (1 << 30)
    inv_rms_real = 1.0 / math.sqrt(v_real)
    inv_rms = int(round(inv_rms_real * (1 << 12)))
    inv_rms = max(0, min(inv_rms, 65535))
    # y = (x * gamma * inv_rms) >> 27, sat int16
    y = np.zeros(D, dtype=np.int64)
    for i in range(D):
        xg   = int(x_int16[i]) * int(gamma_q15[i])     # signed 32-bit
        xgir = xg * inv_rms                            # signed 48-bit
        sh   = xgir >> 27                              # arithmetic right shift
        y[i] = max(-32768, min(32767, sh))
    return y.astype(np.int16)


def fwd_blockfp_intnorm(h_int16, lw, lsc, h_in_p2, h1_p2, h_out_p2, pos, kv, kv_pos, cfg):
    D=cfg["D"]; H_Q=cfg["H_Q"]; H_KV=cfg["H_KV"]; HD=cfg["HD"]; grp=H_Q//H_KV
    s_in   = float(1 << h_in_p2)
    s_h1   = float(1 << h1_p2)
    s_out  = float(1 << h_out_p2)

    # Pre-fold gamma for RMSNorm 1: gamma_real * s_in_eff/s_out_norm * 32768
    # We treat the int16 hidden as if it's Q1.15 of full-scale 1 (== ignore the
    # block-FP scale, since RMSNorm is scale-invariant); gamma is calibrated
    # for s_norm output.
    g1_q15 = np.round(lw["g1"] * 1.0 / lsc["norm1"] * 32768).clip(-32768, 32767).astype(np.int16)
    n1     = rmsnorm_int_emul(h_int16, g1_q15, D)

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

    # Residual 1 (block-FP) — same as before (FP-summed then quantised at h1 scale)
    h1_real = dequantize_q15(h_int16, s_in) + dequantize_q15(o_int, lsc["attn"])
    h1_int  = quantize_q15(h1_real, s_h1)

    g2_q15 = np.round(lw["g2"] * 1.0 / lsc["norm2"] * 32768).clip(-32768, 32767).astype(np.int16)
    n2     = rmsnorm_int_emul(h1_int, g2_q15, D)

    g  = matvec_int8(n2, lsc["norm2"], lw["Wg"], lw["Wg_r"], lsc["gate"])
    u  = matvec_int8(n2, lsc["norm2"], lw["Wu"], lw["Wu_r"], lsc["up"])
    mlp_real = silu(dequantize_q15(g, lsc["gate"])) * dequantize_q15(u, lsc["up"])
    mlp_int  = quantize_q15(mlp_real, lsc["mlp"])
    d_int    = matvec_int8(mlp_int, lsc["mlp"], lw["Wd"], lw["Wd_r"], lsc["down"])

    out_real = dequantize_q15(h1_int, s_h1) + dequantize_q15(d_int, lsc["down"])
    return quantize_q15(out_real, s_out)


def main():
    print("loading model ...", file=sys.stderr)
    tok   = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    cfg = dict(D=model.config.hidden_size, H_Q=model.config.num_attention_heads,
               H_KV=model.config.num_key_value_heads,
               HD=model.config.hidden_size//model.config.num_attention_heads,
               FFN=model.config.intermediate_size, NL=model.config.num_hidden_layers,
               MAX_CTX=64)
    NL = cfg["NL"]; D = cfg["D"]
    print(f"  D={D} NL={NL}  INTEGER-RMSNorm emulation", file=sys.stderr)

    sc = calibrate(model, tok, "Once upon a time there was a princess.", margin=1.5)

    h_p2 = []
    for li in range(NL):
        h_p2.append((pow2(sc[li].get("hidden_in", 1.0)),
                     pow2(max(sc[li].get("hidden_in", 1.0) + sc[li]["attn"], 1.0)),
                     pow2(sc[li].get("hidden_out", 1.0))))

    layers = []
    with torch.no_grad():
        for li in range(NL):
            L = model.model.layers[li]; d = {}
            for nm, sub in [("Wq", L.self_attn.q_proj),  ("Wk", L.self_attn.k_proj),
                            ("Wv", L.self_attn.v_proj),  ("Wo", L.self_attn.o_proj),
                            ("Wg", L.mlp.gate_proj),     ("Wu", L.mlp.up_proj),
                            ("Wd", L.mlp.down_proj)]:
                W = sub.weight.detach().cpu().numpy().astype(np.float32)
                Wq, rsc = quantize_int8_row(W)
                d[nm] = Wq; d[nm + "_r"] = rsc
            d["g1"] = L.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            d["g2"] = L.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            layers.append(d)
        embed     = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        norm_w    = model.model.norm.weight.detach().cpu().numpy().astype(np.float32)
        lm_head_w = embed

    ids = tok("Once upon a time", return_tensors="pt").input_ids[0].tolist()
    print(f"\nprompt → {ids}", file=sys.stderr)
    kv = [{"k_int": np.zeros((cfg["MAX_CTX"], cfg["H_KV"], cfg["HD"]), dtype=np.int16),
           "v_int": np.zeros((cfg["MAX_CTX"], cfg["H_KV"], cfg["HD"]), dtype=np.int16)}
          for _ in range(NL)]

    generated = list(ids)
    for step in range(len(ids) + 12):
        tid = ids[step] if step < len(ids) else generated[-1]
        h = quantize_q15(embed[tid].astype(np.float32), float(1 << h_p2[0][0]))
        for li in range(NL):
            h_in_p2, h1_p2, h_out_p2 = h_p2[li]
            h = fwd_blockfp_intnorm(h, layers[li], sc[li], h_in_p2, h1_p2, h_out_p2,
                                     step, kv[li], step, cfg)
            if li < NL - 1:
                shift = h_p2[li+1][0] - h_out_p2
                if shift > 0:    h = np.clip(h.astype(np.int32) >> shift, -32768, 32767).astype(np.int16)
                elif shift < 0:  h = np.clip(h.astype(np.int32) << (-shift), -32768, 32767).astype(np.int16)
        h_real = dequantize_q15(h, float(1 << h_p2[-1][2])).astype(np.float32)
        h_normed = (h_real / np.sqrt(np.mean(h_real*h_real) + 1e-5)) * norm_w
        logits = h_normed @ lm_head_w.T
        nid = int(np.argmax(logits))
        if step >= len(ids) - 1:
            print(tok.decode([nid]), end="", flush=True)
            generated.append(nid)
        if step == len(ids) - 1:
            top5 = np.argsort(logits)[-5:][::-1]
            print(f"\n[top-5: " + ", ".join(f"{tok.decode([int(i)])!r}" for i in top5) + "]\n",
                  file=sys.stderr)

    print(f"\n\ngenerated: {tok.decode(generated)!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
