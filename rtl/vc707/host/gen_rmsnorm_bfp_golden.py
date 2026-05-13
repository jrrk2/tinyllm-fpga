#!/usr/bin/env python3
"""Generate a block-FP rmsnorm golden vector matching rmsnorm_bfp.sv
exactly: tile-grouped (m, e) input/output, sum-of-squares via tile-MAC +
cross-tile align, msb-LUT-seed Newton-Raphson 1/sqrt with Q12.12 inv_rms,
per-element y = x*g*inv_rms with per-tile mantissa re-normalize.
"""
import os, sys, math, numpy as np

D     = int(os.environ.get('D', '64'))    # default tiny so debug is fast
TILE  = 16
NT    = D // TILE
ACC_W = 48
INV_D_Q32 = (1 << 32) // D
EPS_Q30   = 10737

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


def msb_seed_q12_12(v_msb):
    """Match RTL seed LUT (24-bit Q12.12)."""
    SEED = {31:2365, 30:3344, 29:4730, 28:6689, 27:9459, 26:13377,
            25:18919, 24:26755, 23:37837, 22:53510, 21:75674, 20:107020,
            19:151349, 18:214040, 17:302698, 16:428079, 15:605396,
            14:856159, 13:1210791, 12:1712317, 11:2421583, 10:3424635,
            9:4843165, 8:6849270, 7:9686330, 6:13698540}
    return SEED.get(v_msb, (1 << 24) - 1)


def golden_rmsnorm(x_m, x_e, g_m, g_e):
    """Match rmsnorm_bfp.sv pipeline lane-for-lane."""
    # Stage A: per-tile sum-of-squares
    tile_sums = np.zeros(NT, dtype=np.int64)
    tile_exps = np.zeros(NT, dtype=np.int64)
    for t in range(NT):
        tile_sums[t] = int(np.sum(x_m[t].astype(np.int64) ** 2))
        tile_exps[t] = 2 * int(x_e[t])

    # Cross-tile align + accumulate
    acc = 0
    acc_e = 0
    init = False
    for t in range(NT):
        if not init:
            acc = int(tile_sums[t]); acc_e = int(tile_exps[t]); init = True; continue
        d = int(tile_exps[t]) - acc_e
        if d > 0:
            if d > ACC_W - 1: acc = 0
            else:             acc = acc >> d
            acc_e = int(tile_exps[t])
        elif d < 0:
            nd = -d
            ts = int(tile_sums[t]) >> nd if nd <= ACC_W - 1 else 0
        else:
            ts = int(tile_sums[t])
        if d <= 0:
            acc = acc + (int(tile_sums[t]) if d == 0 else ts)
        else:
            acc = acc + int(tile_sums[t])

    # Stage B: mean_sq = acc * INV_D >> 32, normalize to Q2.30.
    # If (acc_e + lead) is odd, shift one less so the resulting exponent
    # expression (30 - acc_e - lead) is always even and inv_rms_real has
    # an integer power-of-two scale.  In that case v_q30 lives in [2^31, 2^32)
    # and NR sees a "Q3.29-like" value — the iteration converges fine since
    # the range is bounded.
    prod_invD = acc * INV_D_Q32
    mean_sq_int = prod_invD >> 32       # 48-bit unsigned
    if mean_sq_int == 0: lead = 0
    else: lead = mean_sq_int.bit_length() - 1
    # Adjust lead so (acc_e + lead) is even
    if (acc_e + lead) & 1: lead = lead - 1
    if lead >= 30:
        v_q30 = mean_sq_int >> (lead - 30)
    else:
        v_q30 = mean_sq_int << (30 - lead)
    v_q30 = v_q30 + EPS_Q30

    # Stage C: NR rsqrt
    if v_q30 == 0: v_msb = 0
    else: v_msb = v_q30.bit_length() - 1
    inv_rms = msb_seed_q12_12(v_msb)
    C = (3 * (1 << 30)) // 2
    INV_MAX = (1 << 24) - 1
    for _ in range(3):
        y_sq = inv_rms * inv_rms
        vy2 = v_q30 * y_sq
        vy2_q30 = vy2 >> 24
        if vy2_q30 >= (1 << 32): corr = 0
        else: corr = max(0, C - (vy2_q30 >> 1))
        inv_rms = min((inv_rms * corr + (1 << 29)) >> 30, INV_MAX)

    # e_inv_excess: how to translate the raw NR output (inv_rms_q12_int treated
    # as Q12.12) into a value with an explicit signed exponent.
    #   inv_rms_real = inv_rms_q12_int * 2^(e_inv_excess - 12)
    # Derivation: inv_rms_real = inv_rms_nr_real * 2^((30 - lead - acc_e)/2)
    # where inv_rms_nr_real = inv_rms_q12_int/2^12.  So
    #   e_inv_excess = (30 - lead - acc_e) / 2   (floor on odd)
    e_inv_rms = (30 - acc_e - lead) >> 1   # python >> floor-divides signed

    # Stage D: per-element y = x * g * inv_rms, re-tile-quantize
    out_m = np.zeros(D, dtype=np.int16)
    out_e = np.zeros(NT, dtype=np.int8)
    for t in range(NT):
        # Compute products for the tile
        prods = []
        for k in range(TILE):
            xg = int(x_m[t, k]) * int(g_m[t, k])    # 32-bit signed
            xgi = xg * inv_rms                        # 56-bit signed
            prods.append(xgi)
        # Find max leading bit
        max_lead = 0
        for p in prods:
            if p != 0:
                lp = abs(p).bit_length() - 1
                if lp > max_lead: max_lead = lp
        shift = max_lead - 14
        # Emit aligned mantissas
        for k in range(TILE):
            if shift >= 0:
                sh = prods[k] >> shift
            else:
                sh = prods[k] << (-shift)
            m = sh & 0xFFFF
            if m & 0x8000: m -= 0x10000
            out_m[t * TILE + k] = m
        out_e_wide = int(x_e[t]) + int(g_e[t]) + e_inv_rms + shift - 27
        out_e_wide = max(-128, min(127, out_e_wide))
        out_e[t] = out_e_wide
    return out_m, out_e


# Random x and gamma
x = np.random.randn(D) * 3.0
g = np.random.randn(D) * 0.8 + 1.0

x_m, x_e = tile_quantize_raw(x)
g_m, g_e = tile_quantize_raw(g)

y_m, y_e = golden_rmsnorm(x_m, x_e, g_m, g_e)

# FP reference for sanity
x_real = (x_m.astype(np.float64) / 32768.0) * np.power(2.0, x_e.astype(np.float64))[:, None]
x_real = x_real.flatten()
g_real = (g_m.astype(np.float64) / 32768.0) * np.power(2.0, g_e.astype(np.float64))[:, None]
g_real = g_real.flatten()
v_ref = np.mean(x_real * x_real) + 1e-5
inv_rms_ref = 1.0 / math.sqrt(v_ref)
y_ref = x_real * g_real * inv_rms_ref

# Reconstruct golden y_real from (out_m, out_e)
y_bfp = np.zeros(D)
for t in range(NT):
    for k in range(TILE):
        y_bfp[t*TILE + k] = int(y_m[t*TILE + k]) / 32768.0 * (2.0 ** int(y_e[t]))

print(f"D={D} NT={NT}  inv_rms_ref={inv_rms_ref:.6f}", file=sys.stderr)
print(f"  y_ref[0..3] = {y_ref[:4]}", file=sys.stderr)
print(f"  y_bfp[0..3] = {y_bfp[:4]}", file=sys.stderr)
rel = np.abs(y_ref - y_bfp) / (np.abs(y_ref) + 1e-9)
print(f"  rel err max/mean = {rel.max():.4e} / {rel.mean():.4e}", file=sys.stderr)


def write_hex(path, lines):
    with open(path, 'w') as f:
        for ln in lines: f.write(ln + "\n")

write_hex(os.path.join(OUT, "rmsnorm_bfp_xm.hex"),
          [f"{int(x_m[t, k]) & 0xFFFF:04x}" for t in range(NT) for k in range(TILE)])
write_hex(os.path.join(OUT, "rmsnorm_bfp_xe.hex"),
          [f"{int(x_e[t]) & 0xFF:02x}" for t in range(NT)])
write_hex(os.path.join(OUT, "rmsnorm_bfp_gm.hex"),
          [f"{int(g_m[t, k]) & 0xFFFF:04x}" for t in range(NT) for k in range(TILE)])
write_hex(os.path.join(OUT, "rmsnorm_bfp_ge.hex"),
          [f"{int(g_e[t]) & 0xFF:02x}" for t in range(NT)])
write_hex(os.path.join(OUT, "rmsnorm_bfp_ym.hex"),
          [f"{int(y_m[i]) & 0xFFFF:04x}" for i in range(D)])
write_hex(os.path.join(OUT, "rmsnorm_bfp_ye.hex"),
          [f"{int(y_e[t]) & 0xFF:02x}" for t in range(NT)])

with open(os.path.join(OUT, "rmsnorm_bfp_cfg.svh"), "w") as f:
    f.write(f"`define RMSBFP_D  {D}\n")
    f.write(f"`define RMSBFP_NT {NT}\n")

print("  wrote generated/rmsnorm_bfp_{xm,xe,gm,ge,ym,ye}.hex + cfg.svh", file=sys.stderr)
