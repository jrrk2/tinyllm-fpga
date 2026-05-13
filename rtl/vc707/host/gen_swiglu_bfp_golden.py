#!/usr/bin/env python3
"""Generate block-FP swiglu golden vectors matching swiglu_bfp.sv exactly:
LUT-based silu with mantissa pre-shift to LUT scale, multiply by up,
per-tile re-quantize.
"""
import os, sys, math, numpy as np

D    = int(os.environ.get('D', '64'))
TILE = 16
NT   = D // TILE
LUT_SCALE_LOG2 = 5
SILU_LUT_SCALE = 1 << LUT_SCALE_LOG2

np.random.seed(0)
OUT = "generated"
os.makedirs(OUT, exist_ok=True)


# Reload silu LUT exactly as RTL sees it
silu_lut = []
with open(os.path.join(OUT, "silu_lut.hex")) as f:
    for line in f:
        s = line.split('//')[0].strip()
        if s:
            v = int(s, 16)
            if v >= 32768: v -= 65536
            silu_lut.append(v)
silu_lut = np.array(silu_lut, dtype=np.int16)
assert len(silu_lut) == 65536


def tile_quantize_raw(v):
    v = np.asarray(v, dtype=np.float64)
    flat = v.reshape(-1, TILE)
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = (np.floor(np.log2(max_abs)).astype(np.int32) + 1).clip(-127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    m_int = np.clip(np.round(flat / m_scale * 32768.0).astype(np.int64),
                    -32768, 32767)
    return m_int.astype(np.int16), e.astype(np.int8)


def golden_swiglu(gate_m, gate_e, up_m, up_e):
    out_m = np.zeros(D, dtype=np.int16)
    out_e = np.zeros(NT, dtype=np.int8)
    for t in range(NT):
        # Compute per-element products, track max leading bit
        prods = []
        max_lead = 0
        for k in range(TILE):
            # Pre-shift m_gate to Q1.15-at-LUT-scale: shift = e_gate - 5
            shamt = int(gate_e[t]) - LUT_SCALE_LOG2
            mg = int(gate_m[t, k])
            if shamt >= 0:
                idx = mg << shamt
            else:
                # ASR for signed
                idx = mg >> (-shamt)
            idx = max(-32768, min(32767, idx))
            silu_val = int(silu_lut[idx & 0xFFFF])
            # Product = silu_val * m_up
            prod = silu_val * int(up_m[t, k])
            prods.append(prod)
            if prod != 0:
                lp = abs(prod).bit_length() - 1
                if lp > max_lead: max_lead = lp
        # Tile-quantize: shift each prod by (max_lead - 14), output mant int16
        shift = max_lead - 14
        for k in range(TILE):
            if shift >= 0: sh = prods[k] >> shift
            else:          sh = prods[k] << (-shift)
            m = sh & 0xFFFF
            if m & 0x8000: m -= 0x10000
            out_m[t * TILE + k] = m
        # out_e = e_up + shift - 10
        oe = int(up_e[t]) + shift - 10
        out_e[t] = max(-128, min(127, oe))
    return out_m, out_e


gate = np.random.randn(D) * 4.0   # exercise wider range
up   = np.random.randn(D) * 1.0
gate_m, gate_e = tile_quantize_raw(gate)
up_m,   up_e   = tile_quantize_raw(up)

y_m, y_e = golden_swiglu(gate_m, gate_e, up_m, up_e)

# FP reference
gate_real = (gate_m.astype(np.float64) / 32768.0) * np.power(2.0, gate_e.astype(np.float64))[:, None]
gate_real = gate_real.flatten()
up_real   = (up_m  .astype(np.float64) / 32768.0) * np.power(2.0, up_e  .astype(np.float64))[:, None]
up_real   = up_real.flatten()
silu_ref  = gate_real / (1.0 + np.exp(-gate_real))
y_ref     = silu_ref * up_real

y_bfp = np.zeros(D)
for t in range(NT):
    for k in range(TILE):
        y_bfp[t*TILE + k] = int(y_m[t*TILE + k]) / 32768.0 * (2.0 ** int(y_e[t]))

print(f"D={D} NT={NT}", file=sys.stderr)
print(f"  y_ref[0..3] = {y_ref[:4]}", file=sys.stderr)
print(f"  y_bfp[0..3] = {y_bfp[:4]}", file=sys.stderr)
rel = np.abs(y_ref - y_bfp) / (np.abs(y_ref) + 1e-9)
print(f"  rel err max/mean = {rel.max():.4e} / {rel.mean():.4e}", file=sys.stderr)


def write_hex(path, lines):
    with open(path, 'w') as f:
        for ln in lines: f.write(ln + "\n")

write_hex(os.path.join(OUT, "swiglu_bfp_gatem.hex"),
          [f"{int(gate_m[t, k]) & 0xFFFF:04x}" for t in range(NT) for k in range(TILE)])
write_hex(os.path.join(OUT, "swiglu_bfp_gatee.hex"),
          [f"{int(gate_e[t]) & 0xFF:02x}" for t in range(NT)])
write_hex(os.path.join(OUT, "swiglu_bfp_upm.hex"),
          [f"{int(up_m[t, k]) & 0xFFFF:04x}" for t in range(NT) for k in range(TILE)])
write_hex(os.path.join(OUT, "swiglu_bfp_upe.hex"),
          [f"{int(up_e[t]) & 0xFF:02x}" for t in range(NT)])
write_hex(os.path.join(OUT, "swiglu_bfp_ym.hex"),
          [f"{int(y_m[i]) & 0xFFFF:04x}" for i in range(D)])
write_hex(os.path.join(OUT, "swiglu_bfp_ye.hex"),
          [f"{int(y_e[t]) & 0xFF:02x}" for t in range(NT)])

with open(os.path.join(OUT, "swiglu_bfp_cfg.svh"), "w") as f:
    f.write(f"`define SWBFP_D  {D}\n")
    f.write(f"`define SWBFP_NT {NT}\n")

print("  wrote generated/swiglu_bfp_*.hex + cfg.svh", file=sys.stderr)
