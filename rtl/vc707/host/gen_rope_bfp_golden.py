#!/usr/bin/env python3
"""Block-FP RoPE golden — matches rope_bfp.sv exactly."""
import os, sys, math, numpy as np

HD   = 64
H2   = HD // 2
TILE = 16
NT_HD = HD // TILE
POS  = int(os.environ.get('POS', '3'))

np.random.seed(0)
OUT = "generated"
os.makedirs(OUT, exist_ok=True)


def tile_quantize_raw(v):
    v = np.asarray(v, dtype=np.float64)
    flat = v.reshape(-1, TILE)
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = (np.floor(np.log2(max_abs)).astype(np.int32) + 1).clip(-127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    m_int = np.clip(np.round(flat / m_scale * 32768.0).astype(np.int64),
                    -32768, 32767)
    return m_int.astype(np.int16), e.astype(np.int8)


# CORDIC sincos in Python (matches existing CORDIC in cordic_sincos.sv)
PI_HALF_Q27 = 210828714
PI_Q27 = 421657428
PI_OVER_8_Q29 = 210828714
K_Q3_27 = 81504109
ATAN_Q3_27 = [105414357, 62229729, 32880480, 16690645, 8377711, 4192939,
              2096981, 1048555, 524285, 262144, 131072, 65536,
              32768, 16384, 8192, 4096]


def cordic_sincos(angle_q3_27):
    a = angle_q3_27
    if a > PI_HALF_Q27:    z = PI_Q27 - a;  cos_neg = True
    elif a < -PI_HALF_Q27: z = -PI_Q27 - a; cos_neg = True
    else:                  z = a;            cos_neg = False
    x = K_Q3_27; y = 0
    for i in range(16):
        x_shift = x >> i
        y_shift = y >> i
        if z >= 0:
            x, y, z = x - y_shift, y + x_shift, z - ATAN_Q3_27[i]
        else:
            x, y, z = x + y_shift, y - x_shift, z + ATAN_Q3_27[i]
    x_round = x + 2048
    y_round = y + 2048
    xs = x_round >> 12
    ys = y_round >> 12
    if   xs >  32767: cos_out = -32767 if cos_neg else  32767
    elif xs < -32768: cos_out =  32767 if cos_neg else -32768
    else:             cos_out = -xs if cos_neg else xs
    if   ys >  32767: sin_out =  32767
    elif ys < -32768: sin_out = -32768
    else:             sin_out = ys
    return cos_out, sin_out


# Load freq_turns from generated/rope_freq_turns.svh
FREQ_TURNS_Q31 = []
import re
text = open(os.path.join(OUT, "rope_freq_turns.svh")).read()
for m in re.finditer(r"32'd\s*(\d+)", text):
    FREQ_TURNS_Q31.append(int(m.group(1)))
assert len(FREQ_TURNS_Q31) >= 32


def golden_rope(x_m, x_e, pos):
    # Compute cos/sin per pair
    cos_b = [0]*H2
    sin_b = [0]*H2
    for j in range(H2):
        ang_t43 = pos * FREQ_TURNS_Q31[j]
        ang_turns = ang_t43 & ((1 << 31) - 1)
        if ang_turns & (1 << 30): ang_turns -= (1 << 31)
        ang_prod = ang_turns * PI_OVER_8_Q29
        cord_angle = ang_prod >> 29
        cord_angle = cord_angle & ((1 << 31) - 1)
        if cord_angle & (1 << 30): cord_angle -= (1 << 31)
        c, s = cordic_sincos(cord_angle)
        cos_b[j] = c; sin_b[j] = s
    # Per-element rotation, store as (prod_int32, e_max)
    y_prod = [0]*HD
    y_e    = [0]*HD
    for j in range(H2):
        jh2 = j + H2
        mj   = int(x_m[j // TILE, j % TILE])
        mjh2 = int(x_m[jh2 // TILE, jh2 % TILE])
        ej   = int(x_e[j // TILE])
        ejh2 = int(x_e[jh2 // TILE])
        emax = max(ej, ejh2)
        sh_j   = emax - ej
        sh_jh2 = emax - ejh2
        mj_a   = (mj  >> sh_j)   if sh_j   < 16 else 0
        mjh2_a = (mjh2>> sh_jh2) if sh_jh2 < 16 else 0
        prod_lo = mj_a * cos_b[j] - mjh2_a * sin_b[j]
        prod_hi = mjh2_a * cos_b[j] + mj_a * sin_b[j]
        y_prod[j]   = prod_lo
        y_prod[jh2] = prod_hi
        y_e[j]   = emax
        y_e[jh2] = emax
    # Per-element re-quantize to Q1.15 (matching RTL S_OUTPUT)
    out_m = []
    out_e = []
    for k in range(HD):
        p = y_prod[k]
        if p == 0: lead = 0
        else:      lead = abs(p).bit_length() - 1
        shift = lead - 14
        if shift >= 0: sh = p >> shift
        else:          sh = p << (-shift)
        m = sh & 0xFFFF
        if m & 0x8000: m -= 0x10000
        e = y_e[k] + shift - 15
        e = max(-128, min(127, e))
        out_m.append(m); out_e.append(e)
    return out_m, out_e


# Random head — 64 values with N(0, 1)
x = np.random.randn(HD).astype(np.float64)
x_m, x_e = tile_quantize_raw(x)

y_m, y_e = golden_rope(x_m, x_e, POS)

# FP reference rotation
x_real = (x_m.astype(np.float64) / 32768.0) * np.power(2.0, x_e.astype(np.float64))[:, None]
x_real = x_real.flatten()
y_ref = x_real.copy()
for j in range(H2):
    th = POS / (10000.0 ** (2*j/HD))
    c = math.cos(th); s = math.sin(th)
    y_ref[j]    = x_real[j]   * c - x_real[j+H2] * s
    y_ref[j+H2] = x_real[j+H2]* c + x_real[j]    * s

y_bfp = np.array([y_m[k]/32768.0 * (2.0 ** y_e[k]) for k in range(HD)])
print(f"POS={POS} HD={HD}", file=sys.stderr)
print(f"  y_ref[0..3] = {y_ref[:4]}", file=sys.stderr)
print(f"  y_bfp[0..3] = {y_bfp[:4]}", file=sys.stderr)
rel = np.abs(y_ref - y_bfp) / (np.abs(y_ref) + 1e-9)
print(f"  rel err max/mean = {rel.max():.4e} / {rel.mean():.4e}", file=sys.stderr)


def write_hex(path, lines):
    with open(path, 'w') as f:
        for ln in lines: f.write(ln + "\n")

write_hex(os.path.join(OUT, "rope_bfp_xm.hex"),
          [f"{int(x_m[t, k]) & 0xFFFF:04x}" for t in range(NT_HD) for k in range(TILE)])
write_hex(os.path.join(OUT, "rope_bfp_xe.hex"),
          [f"{int(x_e[t]) & 0xFF:02x}" for t in range(NT_HD)])
write_hex(os.path.join(OUT, "rope_bfp_ym.hex"),
          [f"{int(y_m[k]) & 0xFFFF:04x}" for k in range(HD)])
write_hex(os.path.join(OUT, "rope_bfp_ye.hex"),
          [f"{int(y_e[k]) & 0xFF:02x}" for k in range(HD)])

with open(os.path.join(OUT, "rope_bfp_cfg.svh"), "w") as f:
    f.write(f"`define ROPEBFP_HD  {HD}\n")
    f.write(f"`define ROPEBFP_POS {POS}\n")

print("  wrote generated/rope_bfp_*.hex + cfg.svh", file=sys.stderr)
