#!/usr/bin/env python3
"""Generate a block-FP matvec golden vector: random x and W tile-quantized,
then push through the same arithmetic that matvec_bfp_engine.sv will use.

Outputs:
  generated/matvec_bfp_x.hex   — D lines, each `MM_MM EE` (input mantissa hex
                                  followed by tile exponent hex, repeated for
                                  every element of that tile)
  generated/matvec_bfp_w.hex   — D × LANES weight values per tile, plus the
                                  tile's per-lane exponent
  generated/matvec_bfp_y.hex   — LANES lines, each `MM_MM EE` (golden output
                                  mantissa + exponent)
  generated/matvec_bfp_cfg.svh — `define MVBFP_D, MVBFP_LANES, MVBFP_NT
"""
import os, sys, numpy as np

D     = int(os.environ.get('D', '576'))    # input dim (multiple of TILE; env-override)
LANES = 16    # output dim
TILE  = 16
NT    = D // TILE

np.random.seed(0)
OUT = "generated"
os.makedirs(OUT, exist_ok=True)


def tile_quantize_raw(v):
    """Return (m_int int16, e_int8) tile-grouping along last axis."""
    v = np.asarray(v, dtype=np.float64)
    flat = v.reshape(-1, TILE)
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = (np.floor(np.log2(max_abs)).astype(np.int32) + 1).clip(-127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    m_int = np.clip(np.round(flat / m_scale * 32768.0).astype(np.int64),
                    -32768, 32767)
    return m_int.astype(np.int16), e.astype(np.int8)


# Random x ~ N(0, 1) * 5 to exercise dynamic range
x = np.random.randn(D) * 5.0
W = np.random.randn(LANES, D) * 0.3

x_m, x_e = tile_quantize_raw(x)         # shape (NT, TILE), (NT,)
# Weights tile-grouped per-row
W_m = np.zeros((LANES, NT, TILE), dtype=np.int16)
W_e = np.zeros((LANES, NT), dtype=np.int8)
for i in range(LANES):
    m_i, e_i = tile_quantize_raw(W[i])
    W_m[i] = m_i
    W_e[i] = e_i


def golden_matvec(x_m, x_e, W_m, W_e):
    """Bit-accurate reference matching matvec_bfp_engine.sv: tile MAC + 48b
    acc + barrel align + leading-bit normalize."""
    ACC_W = 48
    ACC_HI = (1 << (ACC_W - 1)) - 1
    ACC_LO = -(1 << (ACC_W - 1))

    out_m = np.zeros(LANES, dtype=np.int16)
    out_e = np.zeros(LANES, dtype=np.int8)
    for i in range(LANES):
        acc      = 0
        acc_e    = 0
        init     = False
        for t in range(NT):
            tile_sum = int(np.sum(x_m[t].astype(np.int64) * W_m[i, t].astype(np.int64)))
            tile_exp = int(x_e[t]) + int(W_e[i, t])
            if not init:
                acc = tile_sum
                acc_e = tile_exp
                init = True
                continue
            diff = tile_exp - acc_e
            if diff > 0:
                if diff > ACC_W - 1: acc = 0
                else:                acc = acc >> diff   # ASR for signed Python int
                acc_e = tile_exp
            elif diff < 0:
                nd = -diff
                if nd > ACC_W - 1: tile_sum = 0
                else:              tile_sum = tile_sum >> nd
            acc += tile_sum
            if acc > ACC_HI: acc = ACC_HI
            if acc < ACC_LO: acc = ACC_LO
        # Normalize: find leading bit of |acc|
        a = abs(acc) if acc != 0 else 0
        if a == 0:
            lead = 0
        else:
            lead = a.bit_length() - 1
        shift = lead - 14
        if shift >= 0:
            sh = acc >> shift
        else:
            sh = acc << (-shift)
        m_out = sh & 0xFFFF
        if m_out & 0x8000: m_out -= 0x10000
        # out_e = acc_e + shift - 15  (the -15 comes from the Q1.15 mantissa
        # scale combined with the 2^-30 in acc's representation).  Equivalent
        # derivation: real = acc * 2^(acc_e - 30) and we want real = m/32768 *
        # 2^e ⇒ e = acc_e - 30 + (lead+1) ⇒ e = acc_e - 30 + (shift + 15) =
        # acc_e + shift - 15.
        e_out = acc_e + shift - 15
        e_out = max(-128, min(127, e_out))
        out_m[i] = m_out
        out_e[i] = e_out
    return out_m, out_e


y_m, y_e = golden_matvec(x_m, x_e, W_m, W_e)
# Sanity: compute the "true" matvec from the FP-reconstructed x and W
x_real = (x_m.astype(np.float64) / 32768.0) * np.power(2.0, x_e.astype(np.float64))[:, None]
x_real = x_real.flatten()
W_real = np.zeros((LANES, D), dtype=np.float64)
for i in range(LANES):
    Wr = (W_m[i].astype(np.float64) / 32768.0) * np.power(2.0, W_e[i].astype(np.float64))[:, None]
    W_real[i] = Wr.flatten()
y_true_real = W_real @ x_real

y_from_bfp = y_m.astype(np.float64) / 32768.0 * np.power(2.0, y_e.astype(np.float64))

print(f"D={D} LANES={LANES} NT={NT}", file=sys.stderr)
print(f"  reference real y[0..3] = {y_true_real[:4]}", file=sys.stderr)
print(f"  golden BFP y[0..3]     = {y_from_bfp[:4]}", file=sys.stderr)
rel = np.abs(y_true_real - y_from_bfp) / (np.abs(y_true_real) + 1e-12)
print(f"  rel err max/mean       = {rel.max():.4e} / {rel.mean():.4e}", file=sys.stderr)


def write_hex(path, lines):
    with open(path, 'w') as f:
        for ln in lines:
            f.write(ln + "\n")


# Emit flat $readmemh-compatible hex files (one entry per line).
# x_m.hex: D lines, 4-hex int16 mantissa
write_hex(os.path.join(OUT, "matvec_bfp_xm.hex"),
          [f"{int(x_m[t, k]) & 0xFFFF:04x}" for t in range(NT) for k in range(TILE)])
# x_e.hex: NT lines, 2-hex int8 exponent (one per tile)
write_hex(os.path.join(OUT, "matvec_bfp_xe.hex"),
          [f"{int(x_e[t]) & 0xFF:02x}" for t in range(NT)])
# w_m.hex: D * LANES lines (row-major: element by element, then lanes)
write_hex(os.path.join(OUT, "matvec_bfp_wm.hex"),
          [f"{int(W_m[i, t, k]) & 0xFFFF:04x}"
           for t in range(NT) for k in range(TILE) for i in range(LANES)])
# w_e.hex: NT * LANES lines (per-tile per-lane exponent)
write_hex(os.path.join(OUT, "matvec_bfp_we.hex"),
          [f"{int(W_e[i, t]) & 0xFF:02x}"
           for t in range(NT) for i in range(LANES)])
# y_m.hex / y_e.hex: golden output
write_hex(os.path.join(OUT, "matvec_bfp_ym.hex"),
          [f"{int(y_m[i]) & 0xFFFF:04x}" for i in range(LANES)])
write_hex(os.path.join(OUT, "matvec_bfp_ye.hex"),
          [f"{int(y_e[i]) & 0xFF:02x}" for i in range(LANES)])

# cfg.svh
with open(os.path.join(OUT, "matvec_bfp_cfg.svh"), "w") as f:
    f.write(f"`define MVBFP_D     {D}\n")
    f.write(f"`define MVBFP_LANES {LANES}\n")
    f.write(f"`define MVBFP_NT    {NT}\n")

print("  wrote generated/matvec_bfp_{x,w,y}.hex + cfg.svh", file=sys.stderr)
