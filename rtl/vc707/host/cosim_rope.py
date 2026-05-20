#!/usr/bin/env python3
"""Python ↔ Verilator cosim for rope_bfp.

Stream a HD-element x vector through rope_bfp at position `pos`,
capture per-element rotated output, compare against numpy rope_fp.

rope_bfp's HD parameter defaults to 64.
"""
import ctypes, math, os, sys
import numpy as np

TILE    = 16
HD      = int(os.environ.get('HD',       64))
POS     = int(os.environ.get('POS',       3))
SEED    = int(os.environ.get('SEED',      0))
TOL_REL = float(os.environ.get('TOL_REL', 0.05))

assert HD % 2 == 0
assert HD % TILE == 0
H2 = HD // 2

SO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  "..", "sim", "cosim", "librp_cosim.so"))
if not os.path.exists(SO):
    sys.exit(f"missing {SO} — run 'make -C sim/cosim librp_cosim.so'")

lib = ctypes.CDLL(SO)
for f in ("rp_create", "rp_destroy", "rp_reset"):
    getattr(lib, f).restype = None
lib.rp_cycle.restype = ctypes.c_uint64
lib.rp_tick.restype  = None
lib.rp_tick.argtypes = [
    ctypes.c_uint8, ctypes.c_uint8,             # rst, start
    ctypes.c_int16, ctypes.c_int8,              # x  (m, e)
    ctypes.c_uint8, ctypes.c_uint16,            # in_valid, pos
    ctypes.POINTER(ctypes.c_int16),             # *out_y_mant
    ctypes.POINTER(ctypes.c_int8),              # *out_y_exp
    ctypes.POINTER(ctypes.c_uint8),             # *out_valid
    ctypes.POINTER(ctypes.c_uint8),             # *done
]


def tile_quantize(v):
    flat = np.asarray(v, dtype=np.float64).reshape(-1, TILE)
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = (np.floor(np.log2(max_abs)).astype(np.int32) + 1).clip(-127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    m = np.clip(np.round(flat / m_scale * 32768.0).astype(np.int64),
                -32768, 32767)
    return m.astype(np.int16), e.astype(np.int8)


def tile_decode(m, e):
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    return ((m.astype(np.float64) / 32768.0) * m_scale).flatten()


def fp_quantize_vec(v):
    m, e = tile_quantize(v)
    return tile_decode(m, e)


def rope_fp(x, pos, base=10000.0):
    """Mirrors gen_smollm_blockfp_bfp.py's rope_fp exactly."""
    out = np.array(x, dtype=np.float64)
    for j in range(H2):
        theta = 1.0 / (base ** (2*j/HD)) * pos
        c = math.cos(theta); s = math.sin(theta)
        a, b = x[j], x[j+H2]
        out[j]    = a*c - b*s
        out[j+H2] = b*c + a*s
    return fp_quantize_vec(out)


print(f"[cosim rp] HD={HD} pos={POS} seed={SEED}", file=sys.stderr)
np.random.seed(SEED)
x_real = np.random.uniform(-2.0, 2.0, HD)
x_m, x_e = tile_quantize(x_real)
x_dec = tile_decode(x_m, x_e)
ref_real = rope_fp(x_dec, POS)
ref_m, ref_e = tile_quantize(ref_real)
ref_values = tile_decode(ref_m, ref_e)

lib.rp_create()
lib.rp_reset()
out_m_p     = (ctypes.c_int16 * 1)(0)
out_e_p     = (ctypes.c_int8  * 1)(0)
out_valid_p = (ctypes.c_uint8 * 1)(0)
done_p      = (ctypes.c_uint8 * 1)(0)

# Pulse start with pos already set.
lib.rp_tick(0, 1, 0, 0, 0, POS, out_m_p, out_e_p, out_valid_p, done_p)

rtl_m = np.zeros(HD, dtype=np.int64)
rtl_e = np.zeros(HD, dtype=np.int64)
in_idx, out_idx = 0, 0
MAX_CYCLES = HD * 16 + 500
for c in range(MAX_CYCLES):
    if in_idx < HD:
        tile = in_idx // TILE
        idx  = in_idx %  TILE
        xm = int(x_m[tile, idx]); xe = int(x_e[tile])
        iv = 1
    else:
        xm = xe = 0; iv = 0
    lib.rp_tick(0, 0, xm, xe, iv, POS,
                out_m_p, out_e_p, out_valid_p, done_p)
    if iv: in_idx += 1
    if out_valid_p[0] and out_idx < HD:
        rtl_m[out_idx] = int(out_m_p[0])
        rtl_e[out_idx] = int(out_e_p[0])
        out_idx += 1
    if done_p[0] and out_idx >= HD:
        break
else:
    print(f"[cosim rp] WARN — hit MAX_CYCLES without done", file=sys.stderr)
lib.rp_destroy()
print(f"[cosim rp] in_idx={in_idx} out_idx={out_idx}", file=sys.stderr)

# rope_bfp emits per-element exps (NOT tile-shared) — compare element
# values directly.
mismatches = 0
first_bad = None
for i in range(HD):
    rv = int(rtl_m[i]) * (2.0 ** (int(rtl_e[i]) - 15))
    pv = ref_values[i]
    rel = abs(rv - pv) / max(abs(pv), 1e-6)
    lsb_e = max(int(rtl_e[i]), int(ref_e[i // TILE]))
    abs_floor = 8 * (2.0 ** (lsb_e - 15))
    if rel > TOL_REL and abs(rv - pv) > abs_floor:
        mismatches += 1
        if first_bad is None:
            first_bad = (i, int(rtl_m[i]), int(rtl_e[i]),
                         int(ref_m[i // TILE, i % TILE]), int(ref_e[i // TILE]),
                         rv, pv, rel)
if first_bad:
    i, rm, re_, pm, pe, rv, pv, rel = first_bad
    print(f"[cosim rp] FIRST MISMATCH at element {i}:")
    print(f"          x={x_dec[i]:+.4e}  ref={pv:+.6e} rtl={rv:+.6e}")
    print(f"          REF: m={pm:7d} e={pe:5d}   RTL: m={rm:7d} e={re_:5d}")
    print(f"          rel_err = {rel:.4e}")
    print(f"[cosim rp] {mismatches}/{HD} positions exceed tolerances")
    sys.exit(1)
else:
    print(f"[cosim rp] PASS — all {HD} positions within rel_tol={TOL_REL}")
