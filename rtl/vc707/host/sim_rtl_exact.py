#!/usr/bin/env python3
"""Bit-accurate Python emulation of the SmolLM2 RTL forward pass.
Goal: produce lane values identical (or very close) to Verilator/FPGA so we
can localize where the actual RTL bug lives.

Cumulative RTL approximations switched in via OP_LEVEL env-var (default = all):
   ops=rmsnorm    : RTL Newton-Raphson rmsnorm with Q5.12 inv_rms + msb LUT seed
                    + gamma pre-scaled by lsc[norm1/2]  (matches gen_smollm_blockfp)
   ops=silu       : 65K-entry LUT at SILU_LUT_SCALE=32 + gate/up factor saturation
   ops=softmax    : 1024-entry exp LUT + NR reciprocal (RTL softmax_q15.sv)
   ops=residual   : floor-quantize + Q16.8 resid1/2 factors + ASR shifts
   ops=swiglu_fac : factor_ram (gate_in/up_in Q1.15, mlp_out Q16.8, saturated)
   ops=all        : all of the above

usage: python3 host/sim_rtl_exact.py [ops]
"""
import sys, os, math
import numpy as np
import torch
sys.path.insert(0, 'host')
import sim_blockfp as S
from transformers import AutoModelForCausalLM, AutoTokenizer

OPS = (sys.argv[1] if len(sys.argv) > 1 else 'all').split(',')
def has(op): return op in OPS or 'all' in OPS

# ---------------------------------------------------------------------------
# Fixed-point quantize (floor, matches RTL arithmetic right shift)
# ---------------------------------------------------------------------------
def qfloor(x_real, scale):
    return np.clip(np.floor(x_real * 32768.0 / float(scale)).astype(np.int64),
                   -32768, 32767).astype(np.int16)

# ---------------------------------------------------------------------------
# RTL matvec_int8_engine bit-accurate emulation.
#   acc_r = Σ x_int * w_int8        (40-bit signed accumulator)
#   scale_q15[i] = round(x_scale * W_rsc[i] / out_scale * 32768)   int16 sat
#   y_int[i] = sat16( (acc_r * scale_q15[i]) >> 15 )       (>>> = ASR floor)
# ---------------------------------------------------------------------------
def matvec_int8_rtl(x_int16, x_scale, W_q8, W_rsc, out_scale):
    # acc: shape (out_dim,) — integer dot product
    acc = (x_int16.astype(np.int64) @ W_q8.astype(np.int64).T)   # exact, 40-bit fits in int64
    # Per-row scale_q15, clipped to int16
    scale_q15 = np.round(x_scale * W_rsc / float(out_scale) * 32768.0)
    scale_q15 = np.clip(scale_q15, -32768, 32767).astype(np.int64)
    # (acc * scale) >> 15 with ASR (floor toward -inf for negative values)
    prod = acc * scale_q15
    shifted = prod >> 15            # Python >> is ASR for signed ints in int64
    return np.clip(shifted, -32768, 32767).astype(np.int16)

# ---------------------------------------------------------------------------
# RTL RMSNorm — Q5.12 inv_rms + msb LUT seed + 3 NR iterations
# Gamma is pre-scaled by 1/lsc[norm] (matches fold_gamma in gen_smollm_blockfp)
# ---------------------------------------------------------------------------
RMS_INV_W = int(os.environ.get('RMS_INV_W', '16'))    # 16 = RTL Q5.12; >16 = widened
RMS_INV_MAX = (1 << RMS_INV_W) - 1
# Extended seed LUT — fills in entries the RTL table currently saturates on.
SEED = {}
for _msb in range(0, 32):
    _v_approx = 1.5 * (2.0 ** (_msb - 30))
    _seed = round((1.0 / math.sqrt(_v_approx)) * 4096)
    SEED[_msb] = min(_seed, RMS_INV_MAX)
# RTL's actual narrow table (covers msb 22..31 only); use for default width = 16.
RTL_SEED = {31:2365, 30:3344, 29:4730, 28:6689, 27:9459, 26:13377,
            25:18919, 24:26755, 23:37837, 22:53510}

def rmsnorm_rtl(x_int, gamma_scaled):
    EPS_Q30 = 10737
    D = len(x_int)
    INV_D_Q32 = (1 << 32) // D
    sum_sq = int(np.sum(x_int.astype(np.int64) ** 2))
    v_q30 = ((sum_sq * INV_D_Q32) >> 32) + EPS_Q30
    v_msb = (int.bit_length(int(v_q30)) - 1) if v_q30 > 0 else 0
    if RMS_INV_W == 16:
        inv_rms = RTL_SEED.get(v_msb, 65535)
    else:
        inv_rms = SEED.get(v_msb, RMS_INV_MAX)
    C_1P5_Q30 = (3 * (1 << 30)) // 2
    for _ in range(3):
        y_sq = inv_rms * inv_rms
        vy2 = v_q30 * y_sq
        vy2_q30 = vy2 >> 24
        corr = 0 if vy2_q30 >= (1 << 32) else max(0, C_1P5_Q30 - (vy2_q30 >> 1))
        inv_rms = min((inv_rms * corr + (1 << 29)) >> 30, RMS_INV_MAX)
    g_int = np.clip(np.floor(gamma_scaled * 32768).astype(np.int64), -32768, 32767)
    return np.clip((x_int.astype(np.int64) * g_int * inv_rms) >> 27,
                   -32768, 32767).astype(np.int16)

# ---------------------------------------------------------------------------
# RTL silu LUT — 65536 entries at SILU_LUT_SCALE=32 (matches swiglu.sv)
# Loaded from generated/silu_lut.hex.
# ---------------------------------------------------------------------------
SILU_LUT = None
def load_silu_lut():
    global SILU_LUT
    if SILU_LUT is not None: return
    vals = []
    with open('generated/silu_lut.hex') as f:
        for line in f:
            s = line.split('//')[0].strip()
            if s:
                v = int(s, 16)
                if v >= 32768: v -= 65536
                vals.append(v)
    SILU_LUT = np.array(vals, dtype=np.int16)

def swiglu_rtl(gate_int16, up_int16, gate_in_factor, up_in_factor, mlp_out_factor):
    """RTL swiglu.sv emulation, all 16-bit Q1.15/Q16.8 fixed point."""
    load_silu_lut()
    # gate_scaled_w = gate * gate_in_factor (signed * unsigned-as-signed 17b)
    gate_scaled = gate_int16.astype(np.int64) * int(gate_in_factor)
    gate_shifted = gate_scaled >> 15
    gate_lut_idx = np.clip(gate_shifted, -32768, 32767).astype(np.int32)
    silu_at_lut = SILU_LUT[(gate_lut_idx & 0xFFFF)]   # wrap as int16 index
    up_scaled = up_int16.astype(np.int64) * int(up_in_factor)
    up_at_lut = np.clip(up_scaled >> 15, -32768, 32767).astype(np.int32)
    prod = silu_at_lut.astype(np.int64) * up_at_lut.astype(np.int64)
    prod_at_lut2 = np.clip(prod >> 15, -32768, 32767).astype(np.int64)
    result_w = prod_at_lut2 * int(mlp_out_factor)
    result_shifted = result_w >> 8
    return np.clip(result_shifted, -32768, 32767).astype(np.int16)

# ---------------------------------------------------------------------------
# RTL softmax — 1024-entry exp LUT + Newton-Raphson reciprocal (softmax_q15.sv)
# ---------------------------------------------------------------------------
EXP_LUT = None
def load_exp_lut():
    global EXP_LUT
    if EXP_LUT is not None: return
    vals = []
    with open('generated/exp_lut.hex') as f:
        for line in f:
            s = line.split('//')[0].strip()
            if s: vals.append(int(s, 16) & 0xFFFF)
    EXP_LUT = np.array(vals, dtype=np.uint16)

def softmax_rtl(x_int16):
    """RTL softmax_q15.sv emulation.  x is N int16 Q1.15."""
    load_exp_lut()
    x = x_int16.astype(np.int64)
    max_r = int(np.max(x))
    # diff17 = x - max_r, then idx = (diff>>8) + 1023, clamp [0, 1023]
    diff = x - max_r
    raw_idx = (diff >> 8) + 1023
    lut_idx = np.clip(raw_idx, 0, 1023).astype(np.int32)
    e = EXP_LUT[lut_idx].astype(np.int64)
    sum_e = int(np.sum(e))
    if sum_e == 0:
        return np.zeros_like(x_int16, dtype=np.int16)
    # NR reciprocal: y_{n+1} = y_n * (2^33 - sum_e*y_n) >> 32
    msb = int.bit_length(sum_e) - 1
    y = 1 << (31 - msb)
    for _ in range(4):
        y = (y * ((1 << 33) - sum_e * y)) >> 32
    inv_sum = y
    # Output = e[i] * inv_sum >>> some shift to Q1.15
    # RTL emits Q1.15 by inv_sum is 1/sum_e * 2^32.  e[i]*inv_sum ≈ p[i]*2^32.
    # Want p[i] as Q1.15 = p[i]*2^15.  Shift = 32-15 = 17.
    out = (e * inv_sum) >> 17
    return np.clip(out, 0, 32767).astype(np.int16)

# ---------------------------------------------------------------------------
# RoPE — use FP sin/cos for now (RTL uses CORDIC; can swap later)
# ---------------------------------------------------------------------------
def rope_rtl(x, pos, base=10000.0):
    return S.rope(x, pos, base)   # TODO: CORDIC-faithful version

# ---------------------------------------------------------------------------
# Per-layer factor_ram (Q1.15 gate/up, Q16.8 mlp_out, Q16.8 attn)
# Matches gen_smollm_blockfp.py:
#   gate_in_factor  = round(lsc[gate] / SILU_LUT_SCALE * 32768)   Q1.15 sat 32767
#   up_in_factor    = round(lsc[up]   / SILU_LUT_SCALE * 32768)   Q1.15 sat 32767
#   mlp_out_factor  = round(SILU_LUT_SCALE^2 / lsc[mlp] * 256)    Q16.8 sat 24-bit
#   attn_factor     = round(lsc[v] / lsc[attn] * 256)             Q16.8 sat 24-bit
# ---------------------------------------------------------------------------
SILU_LUT_SCALE = 32.0
def compute_factors(lsc):
    gif = max(0, min(int(round(lsc['gate'] / SILU_LUT_SCALE * 32768.0)), 32767))
    uif = max(0, min(int(round(lsc['up']   / SILU_LUT_SCALE * 32768.0)), 32767))
    mof = max(1, min(int(round(SILU_LUT_SCALE**2 / lsc['mlp'] * 256.0)), (1<<24)-1))
    af  = max(1, min(int(round(lsc['v']    / lsc['attn'] * 256.0)),     (1<<24)-1))
    return gif, uif, mof, af

# ---------------------------------------------------------------------------
# Forward pass — wire RTL approximations based on OPS
# ---------------------------------------------------------------------------
def fwd(h_int, lw, lsc, h_in_p2, h1_p2, h_out_p2, pos, kv, kv_pos, cfg):
    D=cfg['D']; H_Q=cfg['H_Q']; H_KV=cfg['H_KV']; HD=cfg['HD']; grp=H_Q//H_KV
    s_in=float(1<<h_in_p2); s_h1=float(1<<h1_p2); s_out=float(1<<h_out_p2)

    # --- RMSNorm 1 ---
    if has('rmsnorm'):
        n1 = rmsnorm_rtl(h_int.astype(np.int16), lw['g1'] / lsc['norm1'])
    else:
        h_real = h_int.astype(np.float64) * s_in / 32768.0
        n1 = qfloor(S.rmsnorm(h_real, lw['g1']), lsc['norm1'])

    mv = matvec_int8_rtl if has('matvec') else S.matvec_int8
    q = mv(n1, lsc['norm1'], lw['Wq'], lw['Wq_r'], lsc['q'])
    k = mv(n1, lsc['norm1'], lw['Wk'], lw['Wk_r'], lsc['k'])
    v = mv(n1, lsc['norm1'], lw['Wv'], lw['Wv_r'], lsc['v'])
    qr = S.dequantize_q15(q, lsc['q']); kr = S.dequantize_q15(k, lsc['k']); vr = S.dequantize_q15(v, lsc['v'])
    for h in range(H_Q):  qr[h*HD:(h+1)*HD] = rope_rtl(qr[h*HD:(h+1)*HD], pos)
    for h in range(H_KV): kr[h*HD:(h+1)*HD] = rope_rtl(kr[h*HD:(h+1)*HD], pos)
    for h in range(H_KV):
        kv['k_int'][kv_pos,h] = qfloor(kr[h*HD:(h+1)*HD], lsc['k'])
        kv['v_int'][kv_pos,h] = qfloor(vr[h*HD:(h+1)*HD], lsc['v'])

    # --- Attention with optional LUT softmax ---
    attn = np.zeros(D)
    for h in range(H_Q):
        kvh = h//grp
        scores_real = np.zeros(kv_pos+1)
        for t in range(kv_pos+1):
            scores_real[t] = float(np.dot(qr[h*HD:(h+1)*HD],
                S.dequantize_q15(kv['k_int'][t,kvh], lsc['k']))) / math.sqrt(HD)
        if has('softmax'):
            # Quantize scores to Q1.15 at calibration scale ~ max_score, then LUT-softmax
            score_scale = max(np.abs(scores_real).max(), 1.0) * 1.5
            s_q = qfloor(scores_real, score_scale)
            sm_q = softmax_rtl(s_q)
            sm = sm_q.astype(np.float64) / 32768.0
            sm = sm / sm.sum() if sm.sum() > 0 else np.zeros_like(sm)
        else:
            sm = S.softmax(scores_real)
        for t in range(kv_pos+1):
            attn[h*HD:(h+1)*HD] += sm[t] * S.dequantize_q15(kv['v_int'][t,kvh], lsc['v'])
    attn_int = qfloor(attn, lsc['attn'])

    o_int = mv(attn_int, lsc['attn'], lw['Wo'], lw['Wo_r'], lsc['attn']).astype(np.int64)

    # --- Residual 1 ---
    r1f = int(round(lsc['attn'] / s_h1 * 256.0))
    sh1 = h1_p2 - h_in_p2
    hi_a = (h_int.astype(np.int64) >> sh1) if sh1 >= 0 else (h_int.astype(np.int64) << (-sh1))
    h1_int = np.clip(hi_a + ((o_int * r1f) >> 8), -32768, 32767).astype(np.int16)

    # --- RMSNorm 2 ---
    if has('rmsnorm'):
        n2 = rmsnorm_rtl(h1_int, lw['g2'] / lsc['norm2'])
    else:
        h1_real = h1_int.astype(np.float64) * s_h1 / 32768.0
        n2 = qfloor(S.rmsnorm(h1_real, lw['g2']), lsc['norm2'])

    g = mv(n2, lsc['norm2'], lw['Wg'], lw['Wg_r'], lsc['gate'])
    u = mv(n2, lsc['norm2'], lw['Wu'], lw['Wu_r'], lsc['up'])

    # --- SwiGLU with RTL LUT + saturated factors ---
    if has('silu') or has('swiglu_fac'):
        gif, uif, mof, _ = compute_factors(lsc)
        mlp_int = swiglu_rtl(g, u, gif, uif, mof)
    else:
        mlp_real = S.silu(S.dequantize_q15(g, lsc['gate'])) * S.dequantize_q15(u, lsc['up'])
        mlp_int = qfloor(mlp_real, lsc['mlp'])

    d_int = mv(mlp_int, lsc['mlp'], lw['Wd'], lw['Wd_r'], lsc['down']).astype(np.int64)

    # --- Residual 2 ---
    r2f = int(round(lsc['down'] / s_out * 256.0))
    sh2 = h_out_p2 - h1_p2
    h1_a = (h1_int.astype(np.int64) >> sh2) if sh2 >= 0 else (h1_int.astype(np.int64) << (-sh2))
    return np.clip(h1_a + ((d_int * r2f) >> 8), -32768, 32767).astype(np.int16)


def main():
    tok = AutoTokenizer.from_pretrained(S.MODEL)
    model = AutoModelForCausalLM.from_pretrained(S.MODEL, torch_dtype=torch.float32).eval()
    cfg = dict(D=576, H_Q=9, H_KV=3, HD=64, FFN=1536, NL=30, MAX_CTX=64)
    sc = S.calibrate(model, tok, 'Once upon a time there was a princess.', margin=1.5)

    h_p2 = [(S.pow2(sc[li].get('hidden_in',1.0)),
             S.pow2(max(sc[li].get('hidden_in',1.0) + sc[li]['attn'], 1.0)),
             S.pow2(sc[li].get('hidden_out',1.0))) for li in range(30)]
    # Replicate gen_smollm_blockfp's fold_matvec to compute eff_sc.
    MIN_SCALE_Q15 = 4
    def fold_matvec_eff(W_rsc, s_in, s_out_in):
        wmax = float(np.max(W_rsc))
        cap  = (s_in * wmax / MIN_SCALE_Q15) * 32768.0
        return min(s_out_in, cap)
    eff_sc = []
    layers = []
    with torch.no_grad():
        for li in range(30):
            L = model.model.layers[li]; d = {}
            for nm, sub in [('Wq',L.self_attn.q_proj),('Wk',L.self_attn.k_proj),
                            ('Wv',L.self_attn.v_proj),('Wo',L.self_attn.o_proj),
                            ('Wg',L.mlp.gate_proj),('Wu',L.mlp.up_proj),
                            ('Wd',L.mlp.down_proj)]:
                W = sub.weight.detach().cpu().numpy().astype(np.float32)
                Wq, rsc = S.quantize_int8_row(W); d[nm]=Wq; d[nm+'_r']=rsc
            d['g1'] = L.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            d['g2'] = L.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            layers.append(d)
            cs = sc[li]
            eq = fold_matvec_eff(d['Wq_r'], cs['norm1'], cs['q'])
            ek = fold_matvec_eff(d['Wk_r'], cs['norm1'], cs['k'])
            ev = fold_matvec_eff(d['Wv_r'], cs['norm1'], cs['v'])
            eo = fold_matvec_eff(d['Wo_r'], cs['attn'],  cs['attn'])
            egate = fold_matvec_eff(d['Wg_r'], cs['norm2'], cs['gate'])
            eup   = fold_matvec_eff(d['Wu_r'], cs['norm2'], cs['up'])
            edown = fold_matvec_eff(d['Wd_r'], cs['mlp'],   cs['down'])
            eff_sc.append({'norm1':cs['norm1'], 'norm2':cs['norm2'], 'mlp':cs['mlp'],
                           'q':eq, 'k':ek, 'v':ev, 'attn':eo,
                           'gate':egate, 'up':eup, 'down':edown})
        embed = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)
        norm_w = model.model.norm.weight.detach().cpu().numpy().astype(np.float32)
    sc = eff_sc   # USE eff_sc throughout — matches RTL per-row brom scaling

    ids = tok('Once upon a time', return_tensors='pt').input_ids[0].tolist()
    kv = [{'k_int': np.zeros((cfg['MAX_CTX'], cfg['H_KV'], cfg['HD']), dtype=np.int16),
           'v_int': np.zeros((cfg['MAX_CTX'], cfg['H_KV'], cfg['HD']), dtype=np.int16)} for _ in range(30)]
    for step, tid in enumerate(ids):
        e_real = embed[tid].astype(np.float32)
        h = qfloor(e_real, float(1 << h_p2[0][0]))
        for li in range(30):
            h_in_p2, h1_p2, h_out_p2 = h_p2[li]
            h = fwd(h, layers[li], sc[li], h_in_p2, h1_p2, h_out_p2,
                    pos=step, kv=kv[li], kv_pos=step, cfg=cfg)
            if step == len(ids) - 1:
                with open(f'pyrtl_layer_{li:02d}.txt', 'w') as f:
                    f.write(f'# pyrtl_exact ops={OPS} layer {li} hout_p2={h_out_p2}\n')
                    for v in h: f.write(f'{int(v)}\n')
            if li < 29:
                h_real = h.astype(np.float64) * float(1 << h_out_p2) / 32768.0
                h = qfloor(h_real, float(1 << h_p2[li+1][0]))

    h_real = h.astype(np.float64) * float(1 << h_p2[-1][2]) / 32768.0
    h_normed = (h_real / np.sqrt(np.mean(h_real*h_real) + 1e-5)) * norm_w
    logits = h_normed @ embed.T
    top5 = np.argsort(logits)[-5:][::-1]
    print(f'ops={OPS}: top-5 = ' + ', '.join(f'{tok.decode([int(i)])!r}' for i in top5))


if __name__ == '__main__':
    main()
