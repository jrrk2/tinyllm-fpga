#!/usr/bin/env python3
"""Step 2: hardware-faithful per-tile FP emulator.

Models DSP48E1-mapped operations precisely:
  - 16×16 → 32-bit signed mantissa multiply (one DSP per 16-lane tile)
  - 48-bit signed accumulator (DSP48's P register)
  - 8-bit signed tile exponent, sum in fabric LUTs
  - Cross-tile alignment via fabric barrel shifter (max 31-bit shift)
  - Normalize to FP24 output (priority encoder + shifter)
  - LUT silu / exp / sin / cos at FP24-input precision

Verifies the finite-width hardware MAC still produces coherent
autoregressive output.  If yes, RTL design can proceed.
"""
import sys, os, numpy as np, torch, math
sys.path.insert(0, 'host')
np.random.seed(int(os.environ.get('SEED', '1')))
import sim_blockfp as S
from transformers import AutoModelForCausalLM, AutoTokenizer

TILE = int(os.environ.get('TILE', '16'))
ACC_W = 48     # DSP48E1 P register
ALIGN_MAX = 47 # max barrel shift before zeroing out a contribution

# ---------------------------------------------------------------------------
# FP24 quantize (returns mantissa array + exponent array, both int)
# ---------------------------------------------------------------------------
def tile_quantize_raw(v):
    v = np.asarray(v, dtype=np.float64)
    flat = v.reshape(-1, TILE)
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = (np.floor(np.log2(max_abs)).astype(np.int32) + 1).clip(-127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    m_int = np.clip(np.round(flat / m_scale * 32768.0).astype(np.int64),
                    -32768, 32767)
    return m_int, e   # shape (D/TILE, TILE), (D/TILE,)

def tile_decode(m_int, e):
    """Inverse: int16 mantissas + int8 exp → float64 array."""
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    return ((m_int.astype(np.float64) / 32768.0) * m_scale).flatten()

def fp_quantize(v):
    """Convenience: quantize float vec then return float vec at FP-points."""
    m, e = tile_quantize_raw(v)
    return tile_decode(m, e)

# ---------------------------------------------------------------------------
# Hardware-faithful matvec MAC.
#   x_m, x_e: x quantized per-tile, shape (D/TILE, TILE), (D/TILE,)
#   W_m, W_e: weight matrix, per-row-then-per-tile,
#             shape (D_out, D/TILE, TILE), (D_out, D/TILE)
#   Returns FP24-quantized output of length D_out.
# ---------------------------------------------------------------------------
def matvec_hw(x_m, x_e, W_m, W_e):
    """Vectorized hardware-faithful MAC.
    x_m  : (nT, TILE) int64    x_e  : (nT,) int32
    W_m  : (D_out, nT, TILE) int64  W_e  : (D_out, nT) int32
    Returns FP-quantized vector of length D_out.
    """
    D_out, nT = W_e.shape
    # Per-tile mantissa MAC: 16×16→32-bit each, sum 16 = 36-bit (fits int64).
    # tile_sums shape (D_out, nT).
    prods = x_m[None, :, :].astype(np.int64) * W_m.astype(np.int64)
    tile_sums = prods.sum(axis=2)              # (D_out, nT) int64
    tile_exps = x_e[None, :].astype(np.int64) + W_e.astype(np.int64)   # (D_out, nT)
    # Cross-tile align: per-row max exp, then shift each tile's sum down
    max_e = tile_exps.max(axis=1, keepdims=True)   # (D_out, 1)
    shift = (max_e - tile_exps)                    # (D_out, nT) non-negative
    # Saturate any shift > ALIGN_MAX to zero out the term
    mask = shift <= ALIGN_MAX
    shift_capped = np.clip(shift, 0, 63)
    aligned = np.right_shift(tile_sums, shift_capped)   # ASR on int64 ✓
    aligned = np.where(mask, aligned, 0)
    acc = aligned.sum(axis=1)                       # (D_out,) int64
    # Clip to 48-bit signed
    SAT_HI = (1 << 47) - 1
    SAT_LO = -(1 << 47)
    acc = np.clip(acc, SAT_LO, SAT_HI)
    # Final real value: acc * 2^(max_e - 30)
    out_real = acc.astype(np.float64) * np.power(2.0, max_e[:, 0].astype(np.float64) - 30)
    return fp_quantize(out_real)

# ---------------------------------------------------------------------------
# Other ops at FP precision (RTL would use LUT/poly — assume those are
# adequate at FP24 input precision; refine in step 2b if needed).
# ---------------------------------------------------------------------------
def rmsnorm_fp(x, gamma, eps=1e-5):
    v = np.mean(x*x) + eps
    return fp_quantize(x * gamma / np.sqrt(v))

def silu_fp(x):
    return fp_quantize(x / (1.0 + np.exp(-x)))

def softmax_fp(x):
    e = np.exp(x - x.max())
    p = e / e.sum()
    return p   # not tile-quantized; softmax probs are <1 with high dynamic range

def rope_fp(x, pos, base=10000.0):
    HD = len(x); H2 = HD // 2
    out = np.array(x, dtype=np.float64)
    for j in range(H2):
        theta = 1.0 / (base ** (2*j/HD)) * pos
        c = math.cos(theta); s = math.sin(theta)
        a, b = x[j], x[j+H2]
        out[j]    = a*c - b*s
        out[j+H2] = b*c + a*s
    return fp_quantize(out)


def fwd_hw(h, lw, pos, kv, kv_pos, cfg):
    D=cfg['D']; H_Q=cfg['H_Q']; H_KV=cfg['H_KV']; HD=cfg['HD']; grp=H_Q//H_KV
    h_m, h_e = tile_quantize_raw(h)
    n1 = rmsnorm_fp(h, lw['g1'])
    n1_m, n1_e = tile_quantize_raw(n1)
    q = matvec_hw(n1_m, n1_e, *lw['Wq_me'])
    k = matvec_hw(n1_m, n1_e, *lw['Wk_me'])
    v = matvec_hw(n1_m, n1_e, *lw['Wv_me'])
    for h_i in range(H_Q):  q[h_i*HD:(h_i+1)*HD] = rope_fp(q[h_i*HD:(h_i+1)*HD], pos)
    for h_i in range(H_KV): k[h_i*HD:(h_i+1)*HD] = rope_fp(k[h_i*HD:(h_i+1)*HD], pos)
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
    attn = fp_quantize(attn)
    a_m, a_e = tile_quantize_raw(attn)
    o = matvec_hw(a_m, a_e, *lw['Wo_me'])
    h1 = fp_quantize(h + o)
    n2 = rmsnorm_fp(h1, lw['g2'])
    n2_m, n2_e = tile_quantize_raw(n2)
    g = matvec_hw(n2_m, n2_e, *lw['Wg_me'])
    u = matvec_hw(n2_m, n2_e, *lw['Wu_me'])
    mlp = fp_quantize(silu_fp(g) * u)
    mlp_m, mlp_e = tile_quantize_raw(mlp)
    d = matvec_hw(mlp_m, mlp_e, *lw['Wd_me'])
    return fp_quantize(h1 + d)


def main():
    tok = AutoTokenizer.from_pretrained(S.MODEL)
    model = AutoModelForCausalLM.from_pretrained(S.MODEL, torch_dtype=torch.float32).eval()
    cfg = dict(D=576, H_Q=9, H_KV=3, HD=64, FFN=1536, NL=30, MAX_CTX=64)
    NL = cfg['NL']

    print("extracting weights and tile-quantizing ...", file=sys.stderr)
    def quantize_weight_matrix(W):
        # W shape (D_out, D_in). Tile along D_in.  Returns (m, e) shapes
        # (D_out, D_in/TILE, TILE), (D_out, D_in/TILE).
        D_out, D_in = W.shape
        assert D_in % TILE == 0
        nT = D_in // TILE
        m = np.zeros((D_out, nT, TILE), dtype=np.int64)
        e = np.zeros((D_out, nT), dtype=np.int32)
        for i in range(D_out):
            mi, ei = tile_quantize_raw(W[i])
            m[i] = mi; e[i] = ei
        return m, e

    layers = []
    with torch.no_grad():
        for li in range(NL):
            L = model.model.layers[li]; d = {}
            for nm, sub in [('Wq',L.self_attn.q_proj), ('Wk',L.self_attn.k_proj),
                            ('Wv',L.self_attn.v_proj), ('Wo',L.self_attn.o_proj),
                            ('Wg',L.mlp.gate_proj),    ('Wu',L.mlp.up_proj),
                            ('Wd',L.mlp.down_proj)]:
                W = sub.weight.detach().cpu().numpy().astype(np.float64)
                d[nm + '_me'] = quantize_weight_matrix(W)
            d['g1'] = fp_quantize(L.input_layernorm.weight.detach().cpu().numpy().astype(np.float64))
            d['g2'] = fp_quantize(L.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float64))
            layers.append(d)
            print(f"  L{li}", end='\r', file=sys.stderr)
        embed = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float64)
        norm_w = model.model.norm.weight.detach().cpu().numpy().astype(np.float64)
    print("\nweights done", file=sys.stderr)

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
        h = fp_quantize(embed[tid].astype(np.float64))
        for li in range(NL):
            h = fwd_hw(h, layers[li], pos=step, kv=kv[li], kv_pos=step, cfg=cfg)
        if step >= len(ids) - 1:
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
