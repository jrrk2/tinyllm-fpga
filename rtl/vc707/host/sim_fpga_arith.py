#!/usr/bin/env python3
"""Bit-accurate simulation of the proposed FPGA quantization scheme.

Models per-tensor calibrated scales on the NARROW buses (norm1, q, k, v,
attn, norm2, gate, up, mlp, down) and an infinitely precise (= 24-bit
widened) RESIDUAL stream (hidden_in / hidden1 / hidden_out).

Goal: confirm that with this scheme, real SmolLM2-135M can predict
plausible next tokens — if yes, the RTL widening to 24-bit residual
will be worth implementing.

Quantization at each narrow bus:
    x_int16 = round(clip(real_x, -scale, scale-eps) * 32768 / scale)
    real_x' = x_int16 / 32768 * scale

Per-row INT8 weight quantization (matches FPGA matvec_int8_engine):
    w_int8 = round(real_w / row_scale).clip(-128, 127)
    row_scale = max(|row|) / 127

Run:
    python3 host/sim_fpga_arith.py --prompt "Once upon a time"
    python3 host/sim_fpga_arith.py --prompt "Once upon a time" --tokens 20
"""

import argparse, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


# ----------------------------------------------------------------------
# Quantization primitives
# ----------------------------------------------------------------------

def quantize_narrow(x_real: np.ndarray, scale: float) -> np.ndarray:
    """Quantize a real array to int16 representing values in [-scale, +scale]."""
    if scale <= 0: scale = 1e-6
    q = np.round(np.clip(x_real, -scale, scale - scale/32768) * (32768.0 / scale))
    return q.clip(-32768, 32767).astype(np.int16)

def dequantize_narrow(x_int: np.ndarray, scale: float) -> np.ndarray:
    return x_int.astype(np.float32) * (scale / 32768.0)

def quantize_int8_row(W: np.ndarray):
    amax = np.maximum(np.abs(W).max(axis=-1, keepdims=True), 1e-8)
    row_scale = amax / 127.0
    Wq = np.round(W / row_scale).clip(-128, 127).astype(np.int8)
    return Wq, row_scale.squeeze(-1)


# ----------------------------------------------------------------------
# Calibration: capture per-bus max-abs across a calibration prompt.
# Returns a dict of dicts indexed by [layer_idx][bus_name] -> max-abs.
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

    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        _ = model(ids)
    for h in handles: h.remove()

    # Apply margin and floor; return scales (full-scale magnitude per bus per layer).
    scales = {li: {} for li in range(NL)}
    for li in range(NL):
        for k, v in stats[li].items():
            scales[li][k] = max(v * margin, 1e-3)
    return scales


# ----------------------------------------------------------------------
# Pre-extract layer weights as quantized INT8 + per-row scale (FP).
# ----------------------------------------------------------------------

def extract_layer_weights(model, dtype=np.float32):
    NL = model.config.num_hidden_layers
    layers = []
    with torch.no_grad():
        for li in range(NL):
            L = model.model.layers[li]
            d = {}
            for nm, mod_attr, attr in [
                ("Wq",   "self_attn.q_proj",    "weight"),
                ("Wk",   "self_attn.k_proj",    "weight"),
                ("Wv",   "self_attn.v_proj",    "weight"),
                ("Wo",   "self_attn.o_proj",    "weight"),
                ("Wgate","mlp.gate_proj",       "weight"),
                ("Wup",  "mlp.up_proj",         "weight"),
                ("Wdown","mlp.down_proj",       "weight"),
            ]:
                m = L
                for p in mod_attr.split("."):
                    m = getattr(m, p)
                W = getattr(m, attr).detach().cpu().numpy().astype(dtype)
                Wq, sc = quantize_int8_row(W)
                d[nm + "_q8"]   = Wq
                d[nm + "_rsc"]  = sc
            d["g1"] = L.input_layernorm.weight.detach().cpu().numpy().astype(dtype)
            d["g2"] = L.post_attention_layernorm.weight.detach().cpu().numpy().astype(dtype)
            layers.append(d)
    return layers


# ----------------------------------------------------------------------
# Per-layer forward — emulates FPGA arithmetic exactly.
#   hidden_in  : np.float32 array [D]   (residual stream — wide)
#   layer_w    : dict of weights from extract_layer_weights
#   layer_sc   : dict of bus scales from calibrate()  (per-bus full-scale)
#   pos        : current position
#   kv_cache   : dict {'k': [MAX_CTX, H_KV, HD], 'v': [MAX_CTX, H_KV, HD]}
#                of int16 with corresponding scales
# Returns hidden_out: np.float32 [D]
# ----------------------------------------------------------------------

def matvec_int8_q(x_int16, x_scale, W_q8, W_rsc, out_scale, out_dim):
    """Emulate the FPGA matvec: in_q15 × w_int8 → acc → narrow out at out_scale."""
    # acc[k] = sum_j x_int16[j] * W_q8[k, j]    (per row)
    acc = (x_int16.astype(np.int64) @ W_q8.T.astype(np.int64))
    # real_y = acc * x_scale * W_rsc / 32768
    y_real = acc.astype(np.float64) * x_scale * W_rsc / 32768.0
    return quantize_narrow(y_real.astype(np.float32), out_scale)


def rmsnorm_q(x_real, gamma, eps=1e-5):
    """RMSNorm done in FP (since numerically critical) with FP gamma."""
    rms = np.sqrt(np.mean(x_real * x_real) + eps)
    return (x_real / rms) * gamma


def rope_q(x_real, pos, base=10000.0):
    HD = x_real.shape[0]
    H2 = HD // 2
    out = x_real.copy()
    for j in range(H2):
        freq = 1.0 / (base ** (2.0 * j / HD))
        ang  = pos * freq
        c, s = np.cos(ang), np.sin(ang)
        x_lo = x_real[j]
        x_hi = x_real[j + H2]
        out[j]      = x_lo * c - x_hi * s
        out[j + H2] = x_hi * c + x_lo * s
    return out


def silu(x): return x / (1.0 + np.exp(-x))


def softmax_q(x_real):
    x = x_real - x_real.max()
    e = np.exp(x)
    return e / e.sum()


def forward_layer_fpga(hidden_in_real, lw, lsc, pos, kv, kv_pos, cfg):
    D    = cfg["D"]
    H_Q  = cfg["H_Q"]
    H_KV = cfg["H_KV"]
    HD   = cfg["HD"]
    grp  = H_Q // H_KV

    # Pre-norm 1
    n1_real = rmsnorm_q(hidden_in_real, lw["g1"])
    n1_int  = quantize_narrow(n1_real, lsc["norm1"])

    # Q, K, V projections (matvec INT8)
    q = matvec_int8_q(n1_int, lsc["norm1"], lw["Wq_q8"], lw["Wq_rsc"], lsc["q"], D)
    k = matvec_int8_q(n1_int, lsc["norm1"], lw["Wk_q8"], lw["Wk_rsc"], lsc["k"], H_KV*HD)
    v = matvec_int8_q(n1_int, lsc["norm1"], lw["Wv_q8"], lw["Wv_rsc"], lsc["v"], H_KV*HD)

    # RoPE (per head) — keep at narrow precision but recompute scale
    q_real = dequantize_narrow(q, lsc["q"])
    k_real = dequantize_narrow(k, lsc["k"])
    v_real = dequantize_narrow(v, lsc["v"])
    q_rot_real = q_real.copy()
    k_rot_real = k_real.copy()
    for h in range(H_Q):
        q_rot_real[h*HD:(h+1)*HD] = rope_q(q_real[h*HD:(h+1)*HD], pos)
    for h in range(H_KV):
        k_rot_real[h*HD:(h+1)*HD] = rope_q(k_real[h*HD:(h+1)*HD], pos)

    # Write KV cache (rotated) at this position; store as int16 with bus scale.
    for h in range(H_KV):
        kv["k_int"][kv_pos, h] = quantize_narrow(k_rot_real[h*HD:(h+1)*HD], lsc["k"])
        kv["v_int"][kv_pos, h] = quantize_narrow(v_real[h*HD:(h+1)*HD],     lsc["v"])

    # Attention
    attn_out_real = np.zeros(D, dtype=np.float32)
    for h in range(H_Q):
        kv_h = h // grp
        q_h  = q_rot_real[h*HD:(h+1)*HD]
        scores = np.zeros(kv_pos + 1, dtype=np.float32)
        for t in range(kv_pos + 1):
            k_th = dequantize_narrow(kv["k_int"][t, kv_h], lsc["k"])
            scores[t] = float(np.dot(q_h, k_th)) / math.sqrt(HD)
        sm = softmax_q(scores)
        for t in range(kv_pos + 1):
            v_th = dequantize_narrow(kv["v_int"][t, kv_h], lsc["v"])
            attn_out_real[h*HD:(h+1)*HD] += sm[t] * v_th

    attn_int = quantize_narrow(attn_out_real, lsc["attn"])

    # O projection (matvec INT8)
    o_int = matvec_int8_q(attn_int, lsc["attn"], lw["Wo_q8"], lw["Wo_rsc"],
                          lsc["attn"], D)
    o_real = dequantize_narrow(o_int, lsc["attn"])

    # Residual 1 (WIDE — kept in FP to model 24-bit storage)
    hidden1_real = hidden_in_real + o_real

    # Pre-norm 2
    n2_real = rmsnorm_q(hidden1_real, lw["g2"])
    n2_int  = quantize_narrow(n2_real, lsc["norm2"])

    # Gate, Up
    gate_int = matvec_int8_q(n2_int, lsc["norm2"], lw["Wgate_q8"], lw["Wgate_rsc"],
                             lsc["gate"], cfg["FFN"])
    up_int   = matvec_int8_q(n2_int, lsc["norm2"], lw["Wup_q8"],   lw["Wup_rsc"],
                             lsc["up"],   cfg["FFN"])
    gate_real = dequantize_narrow(gate_int, lsc["gate"])
    up_real   = dequantize_narrow(up_int,   lsc["up"])

    # SwiGLU
    mlp_real = silu(gate_real) * up_real
    mlp_scale = max(float(np.abs(mlp_real).max()) * 1.5, 1e-3)
    # Use observed scale here (lsc has 'mlp' at this layer if calibrated).
    mlp_int = quantize_narrow(mlp_real, lsc.get("mlp", mlp_scale))
    mlp_q   = dequantize_narrow(mlp_int, lsc.get("mlp", mlp_scale))

    # Down projection
    down_int = matvec_int8_q(quantize_narrow(mlp_q, lsc.get("mlp", mlp_scale)),
                             lsc.get("mlp", mlp_scale),
                             lw["Wdown_q8"], lw["Wdown_rsc"],
                             lsc["down"], D)
    down_real = dequantize_narrow(down_int, lsc["down"])

    # Residual 2 (WIDE)
    hidden_out_real = hidden1_real + down_real
    return hidden_out_real


# ----------------------------------------------------------------------
# Generation loop
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--cal-prompt", default="Once upon a time there was a princess "
                                            "who lived in a castle by the sea.")
    ap.add_argument("--margin", type=float, default=1.5)
    args = ap.parse_args()

    print(f"loading {MODEL_NAME} ...", file=sys.stderr)
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32).eval()

    cfg = dict(
        D       = model.config.hidden_size,
        H_Q     = model.config.num_attention_heads,
        H_KV    = model.config.num_key_value_heads,
        HD      = model.config.hidden_size // model.config.num_attention_heads,
        FFN     = model.config.intermediate_size,
        NL      = model.config.num_hidden_layers,
        MAX_CTX = 64,                    # plenty for short generations
    )
    print(f"  D={cfg['D']} NL={cfg['NL']} H_Q={cfg['H_Q']} H_KV={cfg['H_KV']} "
          f"FFN={cfg['FFN']}", file=sys.stderr)

    print(f"calibrating on {args.cal_prompt!r} ...", file=sys.stderr)
    scales = calibrate(model, tok, args.cal_prompt, margin=args.margin)
    # mlp bus isn't directly hooked — back-fill from up*gate worst case
    for li in range(cfg["NL"]):
        if "mlp" not in scales[li]:
            # Heuristic: silu(gate)*up max ≤ |gate_max| * |up_max| but usually less.
            scales[li]["mlp"] = max(scales[li]["gate"] * scales[li]["up"] * 0.5,
                                    scales[li]["down"])

    print("extracting + quantizing layer weights ...", file=sys.stderr)
    layer_weights = extract_layer_weights(model)

    # Embedding + lm_head (FP, since we don't quantize them in this experiment)
    with torch.no_grad():
        embed = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        # SmolLM2 ties embeddings with lm_head
        lm_head_w = embed   # tied
        norm_final = model.model.norm.weight.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Run generation
    # ------------------------------------------------------------------
    ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()
    print(f"\nprompt: {args.prompt!r}  →  ids {ids}", file=sys.stderr)

    # Per-layer KV cache.
    kv_caches = [{
        "k_int": np.zeros((cfg["MAX_CTX"], cfg["H_KV"], cfg["HD"]), dtype=np.int16),
        "v_int": np.zeros((cfg["MAX_CTX"], cfg["H_KV"], cfg["HD"]), dtype=np.int16),
    } for _ in range(cfg["NL"])]

    generated = list(ids)
    for step in range(len(ids) + args.tokens):
        if step < len(ids):
            tok_id = ids[step]
        else:
            # use last predicted token
            tok_id = generated[-1]
        h = embed[tok_id].astype(np.float32)         # FP residual stream
        for li in range(cfg["NL"]):
            h = forward_layer_fpga(
                h, layer_weights[li], scales[li],
                pos=step, kv=kv_caches[li], kv_pos=step, cfg=cfg)
        # final norm + lm_head
        h_normed = (h / np.sqrt(np.mean(h*h) + 1e-5)) * norm_final
        logits = h_normed @ lm_head_w.T              # [vocab]
        next_id = int(np.argmax(logits))
        if step >= len(ids) - 1:
            piece = tok.decode([next_id])
            print(piece, end="", flush=True)
            generated.append(next_id)
        # Show top-5 predictions for the prompt-final position too
        if step == len(ids) - 1:
            top5 = np.argsort(logits)[-5:][::-1]
            print(f"\n[top-5 after prompt: " +
                  ", ".join(f"{tok.decode([i])!r}" for i in top5) + "]\n",
                  file=sys.stderr)

    print(f"\n\ngenerated: {tok.decode(generated)!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
