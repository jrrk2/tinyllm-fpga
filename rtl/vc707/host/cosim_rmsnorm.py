#!/usr/bin/env python3
"""Python ↔ Verilator cosim for rmsnorm_bfp.

Stream a tile-quantized vector x and its tile-quantized gammas through
rmsnorm_bfp, capture per-element (m, e), compare against numpy
rmsnorm_fp.  No in_ready handshake (DUT always accepts).
"""
import ctypes, os, sys
import numpy as np

TILE    = 16
N_ELEMS = int(os.environ.get('N_ELEMS', 576))
SEED    = int(os.environ.get('SEED',     0))
TOL_REL = float(os.environ.get('TOL_REL', 0.05))

assert N_ELEMS % TILE == 0

SO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  "..", "sim", "cosim", "librn_cosim.so"))
if not os.path.exists(SO):
    sys.exit(f"missing {SO} — run 'make -C sim/cosim librn_cosim.so'")

lib = ctypes.CDLL(SO)
for f in ("rn_create", "rn_destroy", "rn_reset"):
    getattr(lib, f).restype = None
lib.rn_cycle.restype = ctypes.c_uint64
lib.rn_tick.restype  = None
lib.rn_tick.argtypes = [
    ctypes.c_uint8,                   # rst
    ctypes.c_uint8,                   # start
    ctypes.c_int16, ctypes.c_int8,    # x  (m, e)
    ctypes.c_int16, ctypes.c_int8,    # g  (m, e)
    ctypes.c_uint8, ctypes.c_uint8,   # in_valid, last_elem
    ctypes.POINTER(ctypes.c_int16),   # *out_y_mant
    ctypes.POINTER(ctypes.c_int8),    # *out_y_exp
    ctypes.POINTER(ctypes.c_uint8),   # *out_valid
    ctypes.POINTER(ctypes.c_uint8),   # *done
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


def rmsnorm_fp(x, gamma, eps=1e-5):
    v = np.mean(x*x) + eps
    return fp_quantize_vec(x * gamma / np.sqrt(v))


# Stimulus
print(f"[cosim rn] n_elems={N_ELEMS} seed={SEED}", file=sys.stderr)
np.random.seed(SEED)
x_real = np.random.uniform(-2.0, 2.0, N_ELEMS)
g_real = np.random.uniform( 0.5, 1.5, N_ELEMS)
x_m, x_e = tile_quantize(x_real)
g_m, g_e = tile_quantize(g_real)
x_dec = tile_decode(x_m, x_e)
g_dec = tile_decode(g_m, g_e)
ref_real = rmsnorm_fp(x_dec, g_dec)
ref_m, ref_e = tile_quantize(ref_real)
ref_values = tile_decode(ref_m, ref_e)

# Drive RTL — no in_ready, just stream.
lib.rn_create()
lib.rn_reset()
out_m_p     = (ctypes.c_int16 * 1)(0)
out_e_p     = (ctypes.c_int8  * 1)(0)
out_valid_p = (ctypes.c_uint8 * 1)(0)
done_p      = (ctypes.c_uint8 * 1)(0)

# Pulse start.
lib.rn_tick(0, 1, 0, 0, 0, 0, 0, 0, out_m_p, out_e_p, out_valid_p, done_p)

rtl_m = np.zeros(N_ELEMS, dtype=np.int64)
rtl_e = np.zeros(N_ELEMS, dtype=np.int64)
in_idx = 0
out_idx = 0
MAX_CYCLES = N_ELEMS * 8 + 200
for c in range(MAX_CYCLES):
    if in_idx < N_ELEMS:
        tile = in_idx // TILE
        idx  = in_idx %  TILE
        xm = int(x_m[tile, idx]); xe = int(x_e[tile])
        gm = int(g_m[tile, idx]); ge = int(g_e[tile])
        iv = 1
        le = 1 if (in_idx == N_ELEMS - 1) else 0
    else:
        xm = xe = gm = ge = 0; iv = 0; le = 0
    lib.rn_tick(0, 0, xm, xe, gm, ge, iv, le,
                out_m_p, out_e_p, out_valid_p, done_p)
    if iv:
        in_idx += 1
    if out_valid_p[0] and out_idx < N_ELEMS:
        rtl_m[out_idx] = int(out_m_p[0])
        rtl_e[out_idx] = int(out_e_p[0])
        out_idx += 1
    if done_p[0] and out_idx >= N_ELEMS:
        break
else:
    print(f"[cosim rn] WARN — hit MAX_CYCLES without done", file=sys.stderr)
lib.rn_destroy()
print(f"[cosim rn] in_idx={in_idx} out_idx={out_idx}", file=sys.stderr)

# Compare
mismatches = 0
first_bad = None
for i in range(N_ELEMS):
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
    print(f"[cosim rn] FIRST MISMATCH at element {i} (tile {i//TILE}, idx {i%TILE}):")
    print(f"          x={x_dec[i]:+.4e} g={g_dec[i]:+.4e}  ref={pv:+.6e} rtl={rv:+.6e}")
    print(f"          REF: m={pm:7d} e={pe:5d}   RTL: m={rm:7d} e={re_:5d}")
    print(f"          rel_err = {rel:.4e}")
    print(f"[cosim rn] {mismatches}/{N_ELEMS} positions exceed tolerances")
    sys.exit(1)
else:
    print(f"[cosim rn] PASS — all {N_ELEMS} positions within rel_tol={TOL_REL}")
