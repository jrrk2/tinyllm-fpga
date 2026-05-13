#!/usr/bin/env python3
"""Block-FP softmax golden — matches softmax_bfp.sv exactly."""
import os, sys, math, numpy as np

N    = int(os.environ.get('N', '8'))
N_MAX = 64

np.random.seed(0)
OUT = "generated"
os.makedirs(OUT, exist_ok=True)

# Load exp LUT
exp_lut = []
with open(os.path.join(OUT, "exp_lut.hex")) as f:
    for line in f:
        s = line.split('//')[0].strip()
        if s: exp_lut.append(int(s, 16) & 0xFFFF)
exp_lut = np.array(exp_lut, dtype=np.uint16)
assert len(exp_lut) == 1024


def block_quantize_scalar_vec(v):
    """Quantize a vector to (m, e) with a SINGLE shared exponent."""
    v = np.asarray(v, dtype=np.float64)
    max_abs = max(np.abs(v).max(), 1e-300)
    e = int(np.floor(np.log2(max_abs)) + 1)
    e = max(-127, min(127, e))
    m = np.clip(np.round(v / (2**e) * 32768.0).astype(np.int64),
                -32768, 32767).astype(np.int16)
    return m, e


def golden_softmax(x_m, e_shared, n_elems):
    max_r = max(int(x_m[i]) for i in range(n_elems))
    e_lookup = []
    sum_e = 0
    for k in range(n_elems):
        diff = int(x_m[k]) - max_r
        # shamt = e_shared - 8
        shamt = e_shared - 8
        if shamt >= 0:
            ds = diff << shamt
        else:
            ds = diff >> (-shamt)
        if ds >= 0:        idx = 1023
        elif ds <= -1023:  idx = 0
        else:              idx = ds + 1023
        e_v = int(exp_lut[idx])
        e_lookup.append(e_v)
        sum_e += e_v
    # Newton-Raphson reciprocal: y_{n+1} = y_n * (2^33 - sum_e*y_n) >> 32
    if sum_e == 0: msb = 0
    else: msb = sum_e.bit_length() - 1
    y = 1 << (31 - msb)
    for _ in range(4):
        prod = sum_e * y
        diff64 = (1 << 33) - prod
        y = (y * diff64) >> 32
    inv_sum = y
    # norm_prods
    norm_prods = [e_lookup[k] * inv_sum for k in range(n_elems)]
    max_lead = 0
    for p in norm_prods:
        if p != 0:
            lp = abs(p).bit_length() - 1
            if lp > max_lead: max_lead = lp
    shift = max_lead - 14
    out_m = []
    for p in norm_prods:
        if shift >= 0: sh = p >> shift
        else:          sh = p << (-shift)
        m = sh & 0xFFFF
        if m & 0x8000: m -= 0x10000
        out_m.append(m)
    # out_e = shift - 17  (norm_prod represents prob*2^32; *2^-17 → Q1.15)
    out_e = shift - 17
    out_e = max(-128, min(127, out_e))
    return out_m, out_e


# Generate scores
x = np.random.randn(N) * 3.0
x_m, x_e = block_quantize_scalar_vec(x)

y_m, y_e = golden_softmax(x_m, x_e, N)

# FP reference
x_real = (x_m.astype(np.float64) / 32768.0) * (2.0 ** x_e)
y_ref = np.exp(x_real - x_real.max())
y_ref = y_ref / y_ref.sum()

y_bfp = np.zeros(N)
for k in range(N):
    y_bfp[k] = y_m[k] / 32768.0 * (2.0 ** y_e)

print(f"N={N}", file=sys.stderr)
print(f"  y_ref = {y_ref}", file=sys.stderr)
print(f"  y_bfp = {y_bfp}", file=sys.stderr)
print(f"  sum_y_bfp = {y_bfp.sum():.6f}  (should be ~1.0)", file=sys.stderr)


def write_hex(path, lines):
    with open(path, 'w') as f:
        for ln in lines: f.write(ln + "\n")

write_hex(os.path.join(OUT, "softmax_bfp_xm.hex"),
          [f"{int(x_m[i]) & 0xFFFF:04x}" for i in range(N)])
with open(os.path.join(OUT, "softmax_bfp_cfg.svh"), "w") as f:
    f.write(f"`define SMBFP_N    {N}\n")
    f.write(f"`define SMBFP_XE   {x_e & 0xff}\n")
write_hex(os.path.join(OUT, "softmax_bfp_ym.hex"),
          [f"{int(y_m[i]) & 0xFFFF:04x}" for i in range(N)])
with open(os.path.join(OUT, "softmax_bfp_ye.hex"), "w") as f:
    f.write(f"{y_e & 0xFF:02x}\n")

print("  wrote generated/softmax_bfp_*.hex + cfg.svh", file=sys.stderr)
