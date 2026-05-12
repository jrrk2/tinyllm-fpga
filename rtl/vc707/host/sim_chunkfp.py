#!/usr/bin/env python3
"""sim_blockfp.py variant — Format B (chunk-FP).

Hidden state: 16-bit Q1.15 mantissas + 8-bit signed exponent per
CHUNK of 16 elements (D/16 = 36 chunks at SmolLM2 dims).

Per-element real value:  mant[i] * 2^exp[i//16] / 32768

Operations:
  - matvec output: per-chunk exp = ceil(log2(max_abs * 2^15)) so each
    chunk's mantissas use the full int16 range
  - residual sum: align two operands' chunk exps; sum mantissas;
    renormalise per chunk
  - RMSNorm input: shift mantissas by (max_chunk_exp - this_exp) to
    align all chunks to a common scale

If this produces coherent text the RTL is justified.
"""
import argparse, sys, math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"
CHUNK = 16


def quantize_q15(x_real, scale):
    if scale <= 0: scale = 1e-6
    q = np.round(np.clip(x_real, -scale, scale - scale/32768) * (32768.0 / scale))
    return np.clip(q, -32768, 32767).astype(np.int16)

def dequantize_q15(x_int, scale): return x_int.astype(np.float64) * (scale / 32768.0)

def quantize_int8_row(W):
    amax = np.maximum(np.abs(W).max(axis=-1, keepdims=True), 1e-8)
    rsc = amax / 127.0
    return np.round(W / rsc).clip(-128, 127).astype(np.int8), rsc.squeeze(-1)

def matvec_int8(x_int16, x_scale, W_q8, W_rsc, out_scale):
    acc = (x_int16.astype(np.int64) @ W_q8.T.astype(np.int64))
    y_real = acc.astype(np.float64) * x_scale * W_rsc / 32768.0
    return quantize_q15(y_real, out_scale)


# ----------------------------------------------------------------------
# Chunk-FP helpers
# ----------------------------------------------------------------------

def real_to_chunkfp(x_real, D):
    """Real array → (mant[D] int16, exp[D/CHUNK] int8).  Per-chunk exp
    chosen so max element fills int16."""
    n_chunks = D // CHUNK
    mant = np.zeros(D, dtype=np.int16)
    exps = np.zeros(n_chunks, dtype=np.int8)
    for c in range(n_chunks):
        chunk = x_real[c*CHUNK:(c+1)*CHUNK]
        amax = float(np.max(np.abs(chunk)))
        if amax <= 1e-12:
            exps[c] = -127
            mant[c*CHUNK:(c+1)*CHUNK] = 0
        else:
            # We want round(chunk * 2^15 / 2^exp) to fit in [-32768, 32767]
            # → 2^exp >= amax → exp = ceil(log2(amax))
            exp = int(math.ceil(math.log2(amax))) if amax > 0 else -127
            exp = max(-127, min(127, exp))
            exps[c] = exp
            scale = (1 << 15) / (2.0 ** exp)
            mant[c*CHUNK:(c+1)*CHUNK] = np.clip(np.round(chunk * scale), -32768, 32767).astype(np.int16)
    return mant, exps


def chunkfp_to_real(mant, exps, D):
    n_chunks = D // CHUNK
    out = np.zeros(D, dtype=np.float64)
    for c in range(n_chunks):
        scale = (2.0 ** int(exps[c])) / (1 << 15)
        out[c*CHUNK:(c+1)*CHUNK] = mant[c*CHUNK:(c+1)*CHUNK].astype(np.float64) * scale
    return out


def chunkfp_for_rmsnorm(mant, exps, D):
    """Dequantise via per-chunk exp.  RMSNorm needs the whole D vector at
    a common scale; pick max exp and right-shift small-exp chunks
    (those values get smaller in mantissa, but their tiny magnitudes
    don't move the rms anyway)."""
    return chunkfp_to_real(mant, exps, D).astype(np.float32)


# ----------------------------------------------------------------------
# Standard ops (FP)
# ----------------------------------------------------------------------

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
    with torch.no_grad():
        _ = model(tok(prompt, return_tensors="pt").input_ids)
    for h in handles: h.remove()
    sc = {li: {k: max(v*margin, 1e-3) for k, v in stats[li].items()} for li in range(NL)}
    for li in range(NL):
        sc[li].setdefault("mlp", max(sc[li]["gate"]*sc[li]["up"]*0.5, sc[li]["down"]))
    return sc


# ----------------------------------------------------------------------
# Per-layer forward.  Hidden state is chunk-FP (mant + per-chunk exp).
# ----------------------------------------------------------------------

def fwd_chunkfp(h_mant, h_exp, lw, lsc, pos, kv, kv_pos, cfg):
    D=cfg["D"]; H_Q=cfg["H_Q"]; H_KV=cfg["H_KV"]; HD=cfg["HD"]; grp=H_Q//H_KV

    h_real = chunkfp_to_real(h_mant, h_exp, D).astype(np.float32)

    # RMSNorm 1 (FP, scale-invariant)
    n1_real = rmsnorm(h_real, lw["g1"])
    n1_int  = quantize_q15(n1_real, lsc["norm1"])

    q = matvec_int8(n1_int, lsc["norm1"], lw["Wq"], lw["Wq_r"], lsc["q"])
    k = matvec_int8(n1_int, lsc["norm1"], lw["Wk"], lw["Wk_r"], lsc["k"])
    v = matvec_int8(n1_int, lsc["norm1"], lw["Wv"], lw["Wv_r"], lsc["v"])
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

    # Residual 1 — chunk-FP add
    h1_real_full = h_real + dequantize_q15(o_int, lsc["attn"])
    h1_mant, h1_exp = real_to_chunkfp(h1_real_full, D)

    # RMSNorm 2
    h1_real_back = chunkfp_to_real(h1_mant, h1_exp, D).astype(np.float32)
    n2_real = rmsnorm(h1_real_back, lw["g2"])
    n2_int  = quantize_q15(n2_real, lsc["norm2"])

    g  = matvec_int8(n2_int, lsc["norm2"], lw["Wg"], lw["Wg_r"], lsc["gate"])
    u  = matvec_int8(n2_int, lsc["norm2"], lw["Wu"], lw["Wu_r"], lsc["up"])
    mlp_real = silu(dequantize_q15(g, lsc["gate"])) * dequantize_q15(u, lsc["up"])
    mlp_int  = quantize_q15(mlp_real, lsc["mlp"])
    d_int    = matvec_int8(mlp_int, lsc["mlp"], lw["Wd"], lw["Wd_r"], lsc["down"])

    # Residual 2 — chunk-FP add
    out_real_full = h1_real_back + dequantize_q15(d_int, lsc["down"])
    out_mant, out_exp = real_to_chunkfp(out_real_full, D)
    return out_mant, out_exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--cal-prompt", default="Once upon a time there was a princess.")
    args = ap.parse_args()

    print("loading model ...", file=sys.stderr)
    tok   = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    cfg = dict(D=model.config.hidden_size, H_Q=model.config.num_attention_heads,
               H_KV=model.config.num_key_value_heads,
               HD=model.config.hidden_size//model.config.num_attention_heads,
               FFN=model.config.intermediate_size, NL=model.config.num_hidden_layers,
               MAX_CTX=64)
    NL = cfg["NL"]; D = cfg["D"]
    print(f"  D={D} NL={NL}  CHUNK={CHUNK}  N_CHUNKS={D//CHUNK}", file=sys.stderr)

    print("calibrating ...", file=sys.stderr)
    sc = calibrate(model, tok, args.cal_prompt, margin=1.5)

    print("extracting weights ...", file=sys.stderr)
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
        lm_head_w = embed   # tied

    ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()
    print(f"\nprompt {args.prompt!r} → {ids}", file=sys.stderr)
    kv = [{"k_int": np.zeros((cfg["MAX_CTX"], cfg["H_KV"], cfg["HD"]), dtype=np.int16),
           "v_int": np.zeros((cfg["MAX_CTX"], cfg["H_KV"], cfg["HD"]), dtype=np.int16)}
          for _ in range(NL)]

    generated = list(ids)
    for step in range(len(ids) + args.tokens):
        tid = ids[step] if step < len(ids) else generated[-1]
        e_real = embed[tid].astype(np.float32)
        h_mant, h_exp = real_to_chunkfp(e_real, D)
        for li in range(NL):
            h_mant, h_exp = fwd_chunkfp(h_mant, h_exp, layers[li], sc[li],
                                         pos=step, kv=kv[li], kv_pos=step, cfg=cfg)
        h_real = chunkfp_to_real(h_mant, h_exp, D).astype(np.float32)
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
