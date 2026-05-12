#!/usr/bin/env python3
"""sim_fpga_arith.py variant — Option 2 architecture: block-FP 16-bit hidden.
Hidden state stored as int16 with a per-layer power-of-2 scale; residual
sums combine operands via right-shifts to a common scale, sat16.
"""
import argparse, sys, math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"


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
        def make_lh(li):
            def h(_m, inp, out):
                hi = inp[0] if isinstance(inp, tuple) else inp
                ho = out[0] if isinstance(out, tuple) else out
                stats[li]["hidden_in"]  = max(stats[li].get("hidden_in",  0.0), float(hi.abs().max()))
                stats[li]["hidden_out"] = max(stats[li].get("hidden_out", 0.0), float(ho.abs().max()))
            return h
        handles.append(L.register_forward_hook(make_lh(li)))
    with torch.no_grad():
        _ = model(tok(prompt, return_tensors="pt").input_ids)
    for h in handles: h.remove()
    sc = {li: {k: max(v*margin, 1e-3) for k, v in stats[li].items()} for li in range(NL)}
    for li in range(NL):
        sc[li].setdefault("mlp", max(sc[li]["gate"]*sc[li]["up"]*0.5, sc[li]["down"]))
    return sc


def pow2(x):
    """Smallest power of 2 ≥ x.  Returns the power (so result = 2**pow)."""
    if x <= 1: return 0
    return int(math.ceil(math.log2(x)))


def fwd_blockfp(h_int16, lw, lsc, hsc_in_p2, hsc_h1_p2, hsc_out_p2, pos, kv, kv_pos, cfg, dump_li=None):
    """Block-FP 16-bit hidden.  hsc_*_p2 = log2 of hidden's full-scale at each
    of the three points in the layer (in / mid / out).
    dump_li: if not None, write each intermediate buffer to py_l{li}_<name>.txt
    in Q1.15 signed-int decimal at its captured scale, for comparison to FPGA."""
    D = cfg["D"]; H_Q=cfg["H_Q"]; H_KV=cfg["H_KV"]; HD=cfg["HD"]; grp=H_Q//H_KV
    s_in   = float(1 << hsc_in_p2)
    s_h1   = float(1 << hsc_h1_p2)
    s_out  = float(1 << hsc_out_p2)
    def _dump(name, vec):
        if dump_li is None: return
        with open(f"py_l{dump_li:02d}_{name}.txt", "w") as f:
            f.write(f"# layer {dump_li} {name}: {len(vec)} lanes Q1.15 signed dec\n")
            for v in np.asarray(vec).ravel(): f.write(f"{int(v)}\n")

    h_real = dequantize_q15(h_int16, s_in)

    n1 = quantize_q15(rmsnorm(h_real, lw["g1"]), lsc["norm1"])
    _dump("norm1", n1)
    q  = matvec_int8(n1, lsc["norm1"], lw["Wq"], lw["Wq_r"], lsc["q"])
    k  = matvec_int8(n1, lsc["norm1"], lw["Wk"], lw["Wk_r"], lsc["k"])
    v  = matvec_int8(n1, lsc["norm1"], lw["Wv"], lw["Wv_r"], lsc["v"])
    _dump("q", q); _dump("k", k); _dump("v", v)
    qr = dequantize_q15(q, lsc["q"]); kr = dequantize_q15(k, lsc["k"]); vr = dequantize_q15(v, lsc["v"])
    for h in range(H_Q):  qr[h*HD:(h+1)*HD] = rope(qr[h*HD:(h+1)*HD], pos)
    for h in range(H_KV): kr[h*HD:(h+1)*HD] = rope(kr[h*HD:(h+1)*HD], pos)
    _dump("q_rot", quantize_q15(qr, lsc["q"]))
    _dump("k_rot", quantize_q15(kr, lsc["k"]))
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
    _dump("attn", attn_int)
    o_int    = matvec_int8(attn_int, lsc["attn"], lw["Wo"], lw["Wo_r"], lsc["attn"])
    _dump("o", o_int)
    o_real   = dequantize_q15(o_int, lsc["attn"])

    # Residual 1 — block-FP add: rescale operands to s_h1, sat16
    h1_real = dequantize_q15(h_int16, s_in) + o_real
    h1_int  = quantize_q15(h1_real, s_h1)
    _dump("hidden1", h1_int)

    n2 = quantize_q15(rmsnorm(dequantize_q15(h1_int, s_h1), lw["g2"]), lsc["norm2"])
    _dump("norm2", n2)
    g  = matvec_int8(n2, lsc["norm2"], lw["Wg"], lw["Wg_r"], lsc["gate"])
    u  = matvec_int8(n2, lsc["norm2"], lw["Wu"], lw["Wu_r"], lsc["up"])
    _dump("gate", g); _dump("up", u)
    mlp_real = silu(dequantize_q15(g, lsc["gate"])) * dequantize_q15(u, lsc["up"])
    mlp_int  = quantize_q15(mlp_real, lsc["mlp"])
    _dump("swiglu", mlp_int)
    d_int    = matvec_int8(mlp_int, lsc["mlp"], lw["Wd"], lw["Wd_r"], lsc["down"])
    _dump("down", d_int)
    d_real   = dequantize_q15(d_int, lsc["down"])

    out_real = dequantize_q15(h1_int, s_h1) + d_real
    out_int  = quantize_q15(out_real, s_out)
    _dump("hidden_out", out_int)
    return out_int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=10)
    ap.add_argument("--cal-prompt", default="Once upon a time there was a princess.")
    ap.add_argument("--dump-layers", action="store_true",
                    help="emit py_layer_NN.txt at the position FPGA captures snapshots "
                         "(after each layer's compute, at that layer's h_out_p2 scale).  "
                         "One file per layer, 576 lanes Q1.15 signed dec — same format "
                         "as host/fpga_per_layer_dump.py output.")
    ap.add_argument("--dump-stages", type=int, default=-1, metavar="L",
                    help="dump every intermediate buffer (norm1/q/k/v/attn/o/hidden1/"
                         "norm2/gate/up/swiglu/down/hidden_out) for layer L on the "
                         "final forward pass.  Filenames: py_l{L:02d}_<name>.txt")
    args = ap.parse_args()

    print("loading model ...", file=sys.stderr)
    tok   = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    cfg = dict(D=model.config.hidden_size, H_Q=model.config.num_attention_heads,
               H_KV=model.config.num_key_value_heads,
               HD=model.config.hidden_size//model.config.num_attention_heads,
               FFN=model.config.intermediate_size, NL=model.config.num_hidden_layers,
               MAX_CTX=64)
    NL = cfg["NL"]
    print(f"  D={cfg['D']} NL={NL}", file=sys.stderr)

    print("calibrating ...", file=sys.stderr)
    sc = calibrate(model, tok, args.cal_prompt, margin=1.5)

    # Per-layer hidden block-FP shifts
    h_p2 = []
    for li in range(NL):
        h_in_p2  = pow2(sc[li].get("hidden_in",  1.0))
        h_out_p2 = pow2(sc[li].get("hidden_out", 1.0))
        # h1 = max(h_in, attn) safe upper bound
        h1_p2    = pow2(max(sc[li].get("hidden_in", 1.0) + sc[li]["attn"], 1.0))
        h_p2.append((h_in_p2, h1_p2, h_out_p2))

    print(f"  hidden block-FP scales (log2): " +
          ", ".join(f"L{li}=({a},{b},{c})" for li,(a,b,c) in
                    list(enumerate(h_p2))[::5]), file=sys.stderr)

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
        # Embed + quantize at first layer's hidden_in scale
        e_real = embed[tid].astype(np.float32)
        h = quantize_q15(e_real, float(1 << h_p2[0][0]))
        for li in range(NL):
            h_in_p2, h1_p2, h_out_p2 = h_p2[li]
            dump_li = li if (args.dump_stages == li and step == len(ids) - 1) else None
            h = fwd_blockfp(h, layers[li], sc[li], h_in_p2, h1_p2, h_out_p2,
                            pos=step, kv=kv[li], kv_pos=step, cfg=cfg,
                            dump_li=dump_li)
            # FPGA snapshot capture point: after layer li's compute, at
            # h_out_p2[li] scale.  Only dump the FINAL forward pass (the
            # one that decodes to the next predicted token, matching
            # fpga_per_layer_dump.py which captures once at done).
            if args.dump_layers and step == len(ids) - 1:
                with open(f"py_layer_{li:02d}.txt", "w") as f:
                    f.write(f"# python sim hidden after layer {li}, "
                            f"576 lanes Q1.15 signed dec (h_out_p2={h_out_p2})\n")
                    for v in h:
                        f.write(f"{int(v)}\n")
            # Cascade: convert this layer's out to next layer's in scale
            if li < NL - 1:
                h_real = dequantize_q15(h, float(1 << h_out_p2))
                h = quantize_q15(h_real, float(1 << h_p2[li+1][0]))
        # Final norm + lm_head
        h_real = dequantize_q15(h, float(1 << h_p2[-1][2]))
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
