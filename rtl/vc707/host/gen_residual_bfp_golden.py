#!/usr/bin/env python3
"""Block-FP residual add golden."""
import os, sys, numpy as np

D = int(os.environ.get('D', '64'))
TILE = 16
NT = D // TILE
np.random.seed(0)
OUT = "generated"
os.makedirs(OUT, exist_ok=True)


def tile_quantize_raw(v):
    flat = np.asarray(v, dtype=np.float64).reshape(-1, TILE)
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = (np.floor(np.log2(max_abs)).astype(np.int32) + 1).clip(-127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    m_int = np.clip(np.round(flat / m_scale * 32768.0).astype(np.int64),
                    -32768, 32767)
    return m_int.astype(np.int16), e.astype(np.int8)


def golden_residual(a_m, a_e, b_m, b_e):
    out_m = np.zeros(D, dtype=np.int16)
    out_e = np.zeros(NT, dtype=np.int8)
    for t in range(NT):
        ea, eb = int(a_e[t]), int(b_e[t])
        emax = max(ea, eb)
        sums = []
        max_lead = 0
        for k in range(TILE):
            sh_a = emax - ea; sh_b = emax - eb
            a_a = (int(a_m[t, k]) >> sh_a) if sh_a < 16 else 0
            b_a = (int(b_m[t, k]) >> sh_b) if sh_b < 16 else 0
            s = a_a + b_a
            sums.append(s)
            if s != 0:
                lp = abs(s).bit_length() - 1
                if lp > max_lead: max_lead = lp
        shift = max_lead - 14
        for k in range(TILE):
            sh = sums[k] >> shift if shift >= 0 else sums[k] << (-shift)
            m = sh & 0xFFFF
            if m & 0x8000: m -= 0x10000
            out_m[t * TILE + k] = m
        oe = emax + shift
        out_e[t] = max(-128, min(127, oe))
    return out_m, out_e


a = np.random.randn(D) * 3.0
b = np.random.randn(D) * 1.5
a_m, a_e = tile_quantize_raw(a)
b_m, b_e = tile_quantize_raw(b)
y_m, y_e = golden_residual(a_m, a_e, b_m, b_e)

a_real = (a_m.astype(np.float64) / 32768.0) * np.power(2.0, a_e.astype(np.float64))[:, None]
a_real = a_real.flatten()
b_real = (b_m.astype(np.float64) / 32768.0) * np.power(2.0, b_e.astype(np.float64))[:, None]
b_real = b_real.flatten()
y_ref = a_real + b_real
y_bfp = np.array([y_m[t*TILE+k]/32768.0 * (2.0**int(y_e[t])) for t in range(NT) for k in range(TILE)])
rel = np.abs(y_ref - y_bfp) / (np.abs(y_ref) + 1e-9)
print(f"D={D} rel err max/mean = {rel.max():.4e} / {rel.mean():.4e}", file=sys.stderr)


def w(p, ls):
    with open(p, 'w') as f:
        for x in ls: f.write(x + "\n")

w(os.path.join(OUT, "res_bfp_am.hex"), [f"{int(a_m[t,k])&0xFFFF:04x}" for t in range(NT) for k in range(TILE)])
w(os.path.join(OUT, "res_bfp_bm.hex"), [f"{int(b_m[t,k])&0xFFFF:04x}" for t in range(NT) for k in range(TILE)])
w(os.path.join(OUT, "res_bfp_ae.hex"), [f"{int(a_e[t])&0xFF:02x}" for t in range(NT)])
w(os.path.join(OUT, "res_bfp_be.hex"), [f"{int(b_e[t])&0xFF:02x}" for t in range(NT)])
w(os.path.join(OUT, "res_bfp_ym.hex"), [f"{int(y_m[i])&0xFFFF:04x}" for i in range(D)])
w(os.path.join(OUT, "res_bfp_ye.hex"), [f"{int(y_e[t])&0xFF:02x}" for t in range(NT)])
with open(os.path.join(OUT, "res_bfp_cfg.svh"), 'w') as f:
    f.write(f"`define RESBFP_D  {D}\n`define RESBFP_NT {NT}\n")
print("wrote generated/res_bfp_*.hex + cfg.svh", file=sys.stderr)
