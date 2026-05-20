#!/usr/bin/env python3
"""Python ↔ Verilator cosim for swiglu_bfp.

Stream random tile-quantized (gate, up) pairs through swiglu_bfp,
capture each output element's (m, e), compare against numpy's silu(g) * u.

Configurable: N_ELEMS (default 720, must be tile-aligned), SEED, TOL_REL.
"""
import ctypes, os, sys
import numpy as np

TILE   = 16
N_ELEMS = int(os.environ.get('N_ELEMS', 720))
SEED    = int(os.environ.get('SEED',     0))
TOL_REL = float(os.environ.get('TOL_REL', 0.05))   # silu LUT precision is coarser

assert N_ELEMS % TILE == 0

SO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  "..", "sim", "cosim", "libsw_cosim.so"))
if not os.path.exists(SO):
    sys.exit(f"missing {SO} — run 'make -C sim/cosim'")

lib = ctypes.CDLL(SO)
for f in ("sw_create", "sw_destroy", "sw_reset"):
    getattr(lib, f).restype = None
lib.sw_cycle.restype = ctypes.c_uint64
lib.sw_tick.restype  = None
lib.sw_tick.argtypes = [
    ctypes.c_uint8,                   # rst
    ctypes.c_uint8,                   # start
    ctypes.c_int16, ctypes.c_int8,    # gate (m, e)
    ctypes.c_int16, ctypes.c_int8,    # up   (m, e)
    ctypes.c_uint8, ctypes.c_uint8,   # in_valid, last_elem
    ctypes.POINTER(ctypes.c_uint8),   # *in_ready
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


def silu_fp(x):
    return fp_quantize_vec(x / (1.0 + np.exp(-x)))


# --------------------------------------------------------------------------
# Stimulus + reference
# --------------------------------------------------------------------------
print(f"[cosim sw] n_elems={N_ELEMS} seed={SEED}", file=sys.stderr)
np.random.seed(SEED)
g_real = np.random.uniform(-4.0, 4.0, N_ELEMS)
u_real = np.random.uniform(-2.0, 2.0, N_ELEMS)
g_m, g_e = tile_quantize(g_real)
u_m, u_e = tile_quantize(u_real)
# Decode back to float (the "actual" BFP value RTL will see).
g_dec = tile_decode(g_m, g_e)
u_dec = tile_decode(u_m, u_e)
# Python reference: mirrors gen_smollm_blockfp_bfp.py exactly.
silu_g = silu_fp(g_dec)
mlp_real = fp_quantize_vec(silu_g * u_dec)   # tile-quantized
ref_m, ref_e = tile_quantize(mlp_real)
ref_values = tile_decode(ref_m, ref_e)
print(f"[cosim sw] ref range: [{mlp_real.min():.4f}, {mlp_real.max():.4f}]",
      file=sys.stderr)


# --------------------------------------------------------------------------
# Drive RTL
# --------------------------------------------------------------------------
lib.sw_create()
lib.sw_reset()
in_ready  = (ctypes.c_uint8 * 1)(0)
out_m     = (ctypes.c_int16 * 1)(0)
out_e     = (ctypes.c_int8  * 1)(0)
out_valid = (ctypes.c_uint8 * 1)(0)
done      = (ctypes.c_uint8 * 1)(0)

# Pulse start.
lib.sw_tick(0, 1, 0, 0, 0, 0, 0, 0, in_ready, out_m, out_e, out_valid, done)

rtl_m = np.zeros(N_ELEMS, dtype=np.int64)
rtl_e = np.zeros(N_ELEMS, dtype=np.int64)
in_idx = 0
out_idx = 0
MAX_CYCLES = N_ELEMS * 8 + 100
for c in range(MAX_CYCLES):
    # Clean valid/ready: only drive in_valid when DUT is ready.
    # Otherwise leave inputs at zero so pipeline regs don't latch
    # element-N+1 garbage during tile-boundary emits.
    drive = (in_idx < N_ELEMS) and (in_ready[0] != 0)
    if drive:
        tile = in_idx // TILE
        idx  = in_idx %  TILE
        gm = int(g_m[tile, idx]); ge = int(g_e[tile])
        um = int(u_m[tile, idx]); ue = int(u_e[tile])
        iv = 1
        le = 1 if (in_idx == N_ELEMS - 1) else 0
    else:
        gm = ge = um = ue = 0
        iv = 0; le = 0
    lib.sw_tick(0, 0, gm, ge, um, ue, iv, le,
                in_ready, out_m, out_e, out_valid, done)
    if drive:
        in_idx += 1
    if out_valid[0] and out_idx < N_ELEMS:
        rtl_m[out_idx] = int(out_m[0])
        rtl_e[out_idx] = int(out_e[0])
        out_idx += 1
    if done[0] and out_idx >= N_ELEMS:
        break
else:
    print(f"[cosim sw] WARN — sim hit MAX_CYCLES={MAX_CYCLES} without done",
          file=sys.stderr)

lib.sw_destroy()
print(f"[cosim sw] in_idx={in_idx} out_idx={out_idx}", file=sys.stderr)


# --------------------------------------------------------------------------
# Compare per-element
# --------------------------------------------------------------------------
mismatches = 0
first_bad = None
# Absolute-error floor: 8 LSBs of THIS element's BFP scale (pure
# rounding noise on near-zero outputs).
for i in range(N_ELEMS):
    rv = int(rtl_m[i]) * (2.0 ** (int(rtl_e[i]) - 15))
    pv = ref_values[i]
    denom = max(abs(pv), 1e-6)
    rel = abs(rv - pv) / denom
    abs_err = abs(rv - pv)
    lsb_e = max(int(rtl_e[i]), int(ref_e[i // TILE]))
    abs_floor = 8 * (2.0 ** (lsb_e - 15))
    if rel > TOL_REL and abs_err > abs_floor:
        mismatches += 1
        if first_bad is None:
            first_bad = (i, int(rtl_m[i]), int(rtl_e[i]),
                         int(ref_m[i // TILE, i % TILE]), int(ref_e[i // TILE]),
                         rv, pv, rel)

if first_bad:
    i, rm, re_, pm, pe, rv, pv, rel = first_bad
    print(f"[cosim sw] FIRST MISMATCH at element {i} (tile {i//TILE}, idx {i%TILE}):")
    print(f"          g={g_dec[i]:+.4e}  u={u_dec[i]:+.4e}")
    print(f"          silu(g)*u (float)={silu_g[i]*u_dec[i]:+.6e}")
    print(f"          REF: m={pm:7d} e={pe:5d}  value={pv:+.6e}")
    print(f"          RTL: m={rm:7d} e={re_:5d}  value={rv:+.6e}")
    print(f"          rel_err = {rel:.4e}")
    print(f"[cosim sw] {mismatches}/{N_ELEMS} positions exceed rel_tol={TOL_REL}")
    sys.exit(1)
else:
    print(f"[cosim sw] PASS — all {N_ELEMS} positions within rel_tol={TOL_REL}")
