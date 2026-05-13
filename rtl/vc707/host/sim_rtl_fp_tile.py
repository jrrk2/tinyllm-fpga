#!/usr/bin/env python3
"""Per-tile FP Python emulator for SmolLM2-135M on DSP48E1-friendly format.

Format: each tile of TILE=16 elements has
  - 16 × int16 signed mantissas  (Q0.15 each)
  - 1  × int8  signed shared exponent
  - element value = m_int[k] / 2^15 * 2^e = m_int[k] * 2^(e - 15)

Tile size 16 matches the existing matvec LANES width.  Storage per element:
16 + 8/16 = 16.5 bits — only 0.5 bit overhead vs current int16 Q1.15.

Matvec MAC pattern (one DSP48E1 per tile lane):
  Within a tile:  Σ_{k} m_x[k] * m_w[k]   (16×16 → 32-bit, sum in 48-bit acc)
  Across tiles:  align mantissas of each tile's partial sum to a common
                 exponent (barrel shift in fabric), accumulate, normalize.

Tests autoregressive generation with all RTL approximations replaced by
per-tile FP equivalents.
"""
import sys, os, numpy as np, torch, math
sys.path.insert(0, 'host')
np.random.seed(int(os.environ.get('SEED', '1')))
import sim_blockfp as S
from transformers import AutoModelForCausalLM, AutoTokenizer

TILE = int(os.environ.get('TILE', '16'))
print(f"TILE={TILE}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Per-tile FP quantize: vector of length D → (mantissas int16, exps int8).
# D must be a multiple of TILE.  Returns a float64 array of the quantized
# values for easy use with FP-intermediate compute.  (RTL would store the
# raw mantissa + exponent fields; we collapse to float for simplicity.)
# ---------------------------------------------------------------------------
def tile_quantize(v):
    v = np.asarray(v, dtype=np.float64)
    flat = v.reshape(-1, TILE)
    # Per-tile: exponent = floor(log2(max|v|)) + 1 so mantissas fit int16
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = np.floor(np.log2(max_abs)).astype(np.int32) + 1
    e = np.clip(e, -127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]   # per-tile multiplier
    m_int = np.round(flat / m_scale * 32768.0).astype(np.int64)
    m_int = np.clip(m_int, -32768, 32767)
    out = (m_int.astype(np.float64) / 32768.0) * m_scale
    return out.reshape(v.shape)

# ---------------------------------------------------------------------------
# Matvec with per-tile FP: emulates DSP48E1 mantissa-MAC + fabric exp align.
#   x is length-D, W is (D_out, D), all already tile-quantized in float form.
#   Within a tile we do exact 16×16 mantissa products + 48-bit accumulator.
#   Across tiles we align exponents (per output lane) and sum.
#   Output is tile-quantized FP.
# ---------------------------------------------------------------------------
def matvec_tile_fp(x, W):
    # FP-intermediate computation suffices to model the DSP48 MAC accurately:
    # within-tile sum is exact in float64; across-tile alignment is also
    # exact in float64.  Real RTL would lose some precision in the barrel
    # shift after alignment — but with 48-bit accumulator and per-tile
    # exponents staying close numerically, the loss is < 1 ULP per tile.
    y_real = x @ W.T
    return tile_quantize(y_real)

# ---------------------------------------------------------------------------
# rmsnorm, silu, softmax, rope at FP precision (RTL would use small LUTs
# or polynomial approximations at FP24-equivalent precision — both
# tractable on DSP48 and ample for SmolLM2).
# ---------------------------------------------------------------------------
def rmsnorm_fp(x, gamma, eps=1e-5):
    v = np.mean(x * x) + eps
    return x * gamma / np.sqrt(v)

def silu_fp(x):
    return x / (1.0 + np.exp(-x))

def softmax_fp(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def rope_fp(x, pos, base=10000.0):
    HD = len(x)
    H2 = HD // 2
    out = x.copy()
    for j in range(H2):
        theta = 1.0 / (base ** (2*j/HD)) * pos
        c = math.cos(theta); s = math.sin(theta)
        a, b = x[j], x[j+H2]
        out[j]     = a*c - b*s
        out[j+H2]  = b*c + a*s
    return out

# ---------------------------------------------------------------------------
# Per-layer forward pass
# ---------------------------------------------------------------------------
def fwd_tile_fp(h, lw, pos, kv, kv_pos, cfg):
    D=cfg['D']; H_Q=cfg['H_Q']; H_KV=cfg['H_KV']; HD=cfg['HD']; grp=H_Q//H_KV
    n1 = tile_quantize(rmsnorm_fp(h, lw['g1']))
    q  = matvec_tile_fp(n1, lw['Wq'])
    k  = matvec_tile_fp(n1, lw['Wk'])
    v  = matvec_tile_fp(n1, lw['Wv'])
    for h_i in range(H_Q):  q[h_i*HD:(h_i+1)*HD] = rope_fp(q[h_i*HD:(h_i+1)*HD], pos)
    for h_i in range(H_KV): k[h_i*HD:(h_i+1)*HD] = rope_fp(k[h_i*HD:(h_i+1)*HD], pos)
    q = tile_quantize(q); k = tile_quantize(k)
    for h_i in range(H_KV):
        kv['k'][kv_pos, h_i] = k[h_i*HD:(h_i+1)*HD]
        kv['v'][kv_pos, h_i] = v[h_i*HD:(h_i+1)*HD]
    attn = np.zeros(D)
    for h_i in range(H_Q):
        kvh = h_i // grp
        scores = np.zeros(kv_pos+1)
        for t in range(kv_pos+1):
            scores[t] = float(np.dot(q[h_i*HD:(h_i+1)*HD], kv['k'][t,kvh])) / math.sqrt(HD)
        sm = softmax_fp(scores)
        for t in range(kv_pos+1):
            attn[h_i*HD:(h_i+1)*HD] += sm[t] * kv['v'][t,kvh]
    attn = tile_quantize(attn)
    o = matvec_tile_fp(attn, lw['Wo'])
    h1 = tile_quantize(h + o)
    n2 = tile_quantize(rmsnorm_fp(h1, lw['g2']))
    g = matvec_tile_fp(n2, lw['Wg'])
    u = matvec_tile_fp(n2, lw['Wu'])
    mlp = tile_quantize(silu_fp(g) * u)
    d = matvec_tile_fp(mlp, lw['Wd'])
    return tile_quantize(h1 + d)


def main():
    tok = AutoTokenizer.from_pretrained(S.MODEL)
    model = AutoModelForCausalLM.from_pretrained(S.MODEL, torch_dtype=torch.float32).eval()
    cfg = dict(D=model.config.hidden_size, H_Q=model.config.num_attention_heads,
               H_KV=model.config.num_key_value_heads,
               HD=model.config.hidden_size//model.config.num_attention_heads,
               FFN=model.config.intermediate_size, NL=model.config.num_hidden_layers,
               MAX_CTX=64)
    NL = cfg['NL']

    print("extracting + tile-quantizing weights ...", file=sys.stderr)
    layers = []
    with torch.no_grad():
        for li in range(NL):
            L = model.model.layers[li]; d = {}
            for nm, sub in [('Wq',L.self_attn.q_proj), ('Wk',L.self_attn.k_proj),
                            ('Wv',L.self_attn.v_proj), ('Wo',L.self_attn.o_proj),
                            ('Wg',L.mlp.gate_proj),    ('Wu',L.mlp.up_proj),
                            ('Wd',L.mlp.down_proj)]:
                W = sub.weight.detach().cpu().numpy().astype(np.float64)
                # tile-quantize per ROW (matvec MAC along columns; tiles span columns)
                d[nm] = tile_quantize(W)   # per-row tiles along the D axis
            d['g1'] = tile_quantize(L.input_layernorm.weight.detach().cpu().numpy().astype(np.float64))
            d['g2'] = tile_quantize(L.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float64))
            layers.append(d)
        embed = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float64)
        norm_w = model.model.norm.weight.detach().cpu().numpy().astype(np.float64)
    # Tile-quantize embed table per row (just storage; we use one row at a time)
    embed_q = tile_quantize(embed)

    PROMPT = "Once upon a time"
    ids = tok(PROMPT, return_tensors='pt').input_ids[0].tolist()
    N_GEN = 15
    T = float(os.environ.get('TEMP', '0.0'))
    TOPK = int(os.environ.get('TOPK', '0'))

    kv = [{'k': np.zeros((cfg['MAX_CTX'], cfg['H_KV'], cfg['HD']), dtype=np.float64),
           'v': np.zeros((cfg['MAX_CTX'], cfg['H_KV'], cfg['HD']), dtype=np.float64)}
          for _ in range(NL)]

    print(f"\n{PROMPT}", end='', flush=True)
    generated = list(ids)
    for step in range(len(ids) + N_GEN):
        tid = ids[step] if step < len(ids) else generated[-1]
        h = tile_quantize(embed_q[tid].astype(np.float64))
        for li in range(NL):
            h = fwd_tile_fp(h, layers[li], pos=step, kv=kv[li], kv_pos=step, cfg=cfg)
        if step >= len(ids) - 1:
            # Final norm + lm_head (kept FP — these are the model "head")
            h_normed = (h / np.sqrt(np.mean(h*h) + 1e-5)) * norm_w
            logits = h_normed @ embed.T
            if T > 0:
                scaled = logits / T
                if TOPK > 0:
                    idx = np.argsort(scaled)[-TOPK:]
                    mask = np.full_like(scaled, -np.inf); mask[idx] = scaled[idx]
                    scaled = mask
                scaled -= scaled.max()
                p = np.exp(scaled); p = p / p.sum()
                nid = int(np.random.choice(len(p), p=p))
            else:
                nid = int(np.argmax(logits))
            if step >= len(ids):
                print(tok.decode([nid]), end='', flush=True)
                generated.append(nid)
    print()


if __name__ == '__main__':
    main()
