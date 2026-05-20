#!/usr/bin/env python3
"""Python ↔ Verilator cosim for matvec_bfp_engine.

Loads sim/cosim/libmv_cosim.so via ctypes, drives the RTL engine
cycle-by-cycle with stimulus generated in Python, and compares each
chunk's 16-lane output against numpy's matvec_hw_golden reference.

Prints the first chunk + per-lane (m, e) mismatch and exits non-zero
when divergence exceeds the tolerance.

Default stimulus: a single matvec with in_dim=720 (= NT=45, the
threshold where the old non-cosim selftest started reporting drift).
Override with env vars IN_DIM, SEED, N_CHUNKS, TOL_REL.

Usage:
    cd rtl/vc707/sim/cosim && make libmv_cosim.so
    cd rtl/vc707 && python3 host/cosim_matvec.py
"""
import ctypes, os, sys
import numpy as np

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TILE     = 16
LANES    = 16
IN_DIM   = int(os.environ.get('IN_DIM',   720))
N_CHUNKS = int(os.environ.get('N_CHUNKS',   1))
SEED     = int(os.environ.get('SEED',       0))
TOL_REL  = float(os.environ.get('TOL_REL', 0.01))

assert IN_DIM % TILE == 0, f"IN_DIM={IN_DIM} must be multiple of TILE={TILE}"
NT_IN = IN_DIM // TILE

# --------------------------------------------------------------------------
# Load the cosim shared library
# --------------------------------------------------------------------------
SO = os.path.join(os.path.dirname(__file__), "..", "sim", "cosim", "libmv_cosim.so")
SO = os.path.abspath(SO)
if not os.path.exists(SO):
    sys.exit(f"missing {SO} — run 'make -C sim/cosim'")

lib = ctypes.CDLL(SO)
lib.mv_create.restype  = None
lib.mv_destroy.restype = None
lib.mv_reset.restype   = None
lib.mv_cycle.restype   = ctypes.c_uint64
lib.mv_tick.restype    = None
lib.mv_tick.argtypes   = [
    ctypes.c_uint8,                                  # rst
    ctypes.c_uint8,                                  # start_matvec
    ctypes.c_int16,                                  # in_x_mant
    ctypes.c_int8,                                   # in_x_exp
    ctypes.c_uint8,                                  # in_valid
    ctypes.c_uint8,                                  # last_elem
    ctypes.POINTER(ctypes.c_uint32),                 # w_mant_words[8]
    ctypes.POINTER(ctypes.c_uint32),                 # w_exp_words[4]
    ctypes.POINTER(ctypes.c_uint8),                  # *out_valid
    ctypes.POINTER(ctypes.c_uint32),                 # out_mant_words[8]
    ctypes.POINTER(ctypes.c_uint32),                 # out_exp_words[4]
]

# --------------------------------------------------------------------------
# Stimulus generation — matches gen_smollm_blockfp_bfp.py's quantizer.
# --------------------------------------------------------------------------
def tile_quantize(v):
    flat = np.asarray(v, dtype=np.float64).reshape(-1, TILE)
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = (np.floor(np.log2(max_abs)).astype(np.int32) + 1).clip(-127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    m = np.clip(np.round(flat / m_scale * 32768.0).astype(np.int64),
                -32768, 32767)
    return m.astype(np.int16), e.astype(np.int8)


def matvec_hw_golden(x_m, x_e, W_m, W_e):
    """Mirrors gen_smollm_blockfp_bfp.py exactly.  Returns the per-output
    (m, e) tuple: m is 16-bit signed mantissa per output, e is shared
    per-lane (= per output) — NOT tile-shared.  RTL emits per-lane e."""
    D_out, nT = W_e.shape
    prods = x_m[None, :, :].astype(np.int64) * W_m[:].astype(np.int64)
    tile_sums = prods.sum(axis=2)
    tile_exps = x_e[None, :].astype(np.int64) + W_e.astype(np.int64)
    max_e = tile_exps.max(axis=1, keepdims=True)
    shift = max_e - tile_exps
    ALIGN_MAX = 47
    mask = shift <= ALIGN_MAX
    aligned = np.right_shift(tile_sums, np.clip(shift, 0, 63))
    aligned = np.where(mask, aligned, 0)
    acc = aligned.sum(axis=1)
    SAT_HI = (1 << 47) - 1
    SAT_LO = -(1 << 47)
    acc = np.clip(acc, SAT_LO, SAT_HI)
    # RTL-style per-lane normalize: lead_pos → shift to land mantissa
    # in [2^14, 2^15); e_out = max_e + (lead-14) - 15.
    m_out = np.zeros(D_out, dtype=np.int64)
    e_out = np.zeros(D_out, dtype=np.int64)
    for i in range(D_out):
        a = int(acc[i])
        if a == 0:
            m_out[i] = 0; e_out[i] = int(max_e[i, 0]) - 15
            continue
        absv = abs(a)
        lead = absv.bit_length() - 1  # position of leading 1
        sh = lead - 14
        if sh >= 0:
            m = a >> sh
        else:
            m = a << (-sh)
        # m is BFP_MANT_W=16-bit signed
        m &= 0xffff
        if m & 0x8000: m -= 0x10000
        m_out[i] = m
        e_out[i] = int(max_e[i, 0]) + sh - 15
    return m_out.astype(np.int64), e_out.astype(np.int64)


def pack_lane_mant(W_m_lanes):  # (LANES,) int16 → 256-bit as 8 uint32
    """Pack 16 lanes of 16-bit signed mantissas into 8 × 32-bit words
    matching Verilator's VlWide<8> layout for w_mant[255:0]."""
    bits = 0
    for L in range(LANES):
        v = int(W_m_lanes[L]) & 0xffff
        bits |= (v << (L * 16))
    words = (ctypes.c_uint32 * 8)()
    for i in range(8):
        words[i] = (bits >> (i * 32)) & 0xffffffff
    return words


def pack_lane_exp(W_e_lanes):  # (LANES,) int8 → 128-bit as 4 uint32
    bits = 0
    for L in range(LANES):
        v = int(W_e_lanes[L]) & 0xff
        bits |= (v << (L * 8))
    words = (ctypes.c_uint32 * 4)()
    for i in range(4):
        words[i] = (bits >> (i * 32)) & 0xffffffff
    return words


def unpack_out(words_m, words_e):
    """Verilator out_mant/out_exp wide words → per-lane (m, e) arrays."""
    bits_m = 0
    for i in range(8):
        bits_m |= (int(words_m[i]) & 0xffffffff) << (i * 32)
    bits_e = 0
    for i in range(4):
        bits_e |= (int(words_e[i]) & 0xffffffff) << (i * 32)
    m = np.zeros(LANES, dtype=np.int64)
    e = np.zeros(LANES, dtype=np.int64)
    for L in range(LANES):
        mv = (bits_m >> (L * 16)) & 0xffff
        if mv & 0x8000: mv -= 0x10000
        ev = (bits_e >> (L * 8 )) & 0xff
        if ev & 0x80: ev -= 0x100
        m[L] = mv; e[L] = ev
    return m, e


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
print(f"[cosim] in_dim={IN_DIM} NT={NT_IN} chunks={N_CHUNKS} seed={SEED}",
      file=sys.stderr)
np.random.seed(SEED)

# Generate stimulus: input vector x (random float in [-1, 1]) and one
# weight matrix W of shape (N_CHUNKS*LANES, IN_DIM).  All tile-quantized.
x_real = np.random.uniform(-1.0, 1.0, IN_DIM)
x_m, x_e = tile_quantize(x_real)  # x_m: (NT_IN, TILE), x_e: (NT_IN,)

D_out = N_CHUNKS * LANES
W_real = np.random.uniform(-1.0, 1.0, (D_out, IN_DIM))
W_m = np.zeros((D_out, NT_IN, TILE), dtype=np.int16)
W_e = np.zeros((D_out, NT_IN),       dtype=np.int8)
for i in range(D_out):
    m_i, e_i = tile_quantize(W_real[i])
    W_m[i] = m_i; e_i_int8 = e_i
    W_e[i] = e_i_int8

# Reference: numpy matvec.
ref_m, ref_e = matvec_hw_golden(x_m, x_e, W_m, W_e)

# Drive RTL.
lib.mv_create()
lib.mv_reset()
out_valid = (ctypes.c_uint8  * 1)(0)
out_mant  = (ctypes.c_uint32 * 8)()
out_exp   = (ctypes.c_uint32 * 4)()

rtl_m = np.zeros(D_out, dtype=np.int64)
rtl_e = np.zeros(D_out, dtype=np.int64)
chunks_done = 0

# Per chunk: pulse start_matvec, then stream IN_DIM elements with in_valid=1.
# last_elem fires on the very last input element.  Caller asserts in_valid=0
# on the cycle after to let the engine drain.
for chunk in range(N_CHUNKS):
    # Pulse start_matvec for one cycle BEFORE the first input element.
    lib.mv_tick(0, 1, 0, 0, 0, 0,
                pack_lane_mant(np.zeros(LANES, dtype=np.int16)),
                pack_lane_exp (np.zeros(LANES, dtype=np.int8 )),
                out_valid, out_mant, out_exp)
    # Stream IN_DIM elements.  w_mant/w_exp for this chunk = W[chunk*LANES :
    # (chunk+1)*LANES, col, lane_within_tile].
    rows = slice(chunk * LANES, (chunk + 1) * LANES)
    chunk_W_m = W_m[rows]    # (LANES, NT_IN, TILE)
    chunk_W_e = W_e[rows]    # (LANES, NT_IN)
    for col in range(IN_DIM):
        tile_idx = col // TILE
        idx_in_tile = col % TILE
        # 16 lanes' mantissas at (col): one per row.
        w_lanes_m = chunk_W_m[:, tile_idx, idx_in_tile]    # (LANES,) int16
        # Exp is per-tile shared — same value across the 16 cycles of one
        # tile.  Just pick from any lane (tile-shared).
        w_lanes_e = chunk_W_e[:, tile_idx]                  # (LANES,) int8
        last = (col == IN_DIM - 1)
        lib.mv_tick(0, 0,
                    int(x_m[tile_idx, idx_in_tile]), int(x_e[tile_idx]),
                    1, int(last),
                    pack_lane_mant(w_lanes_m),
                    pack_lane_exp (w_lanes_e),
                    out_valid, out_mant, out_exp)
        if out_valid[0]:
            m, e = unpack_out(out_mant, out_exp)
            rtl_m[rows] = m
            rtl_e[rows] = e
            chunks_done += 1
            break  # consume rest of pipeline below
    # Drain: a couple of in_valid=0 cycles so the pipeline can settle and
    # out_valid fires (matvec engine: out_valid pulses 1-2 cycles after
    # last_elem).
    for _ in range(8):
        if out_valid[0]: break
        lib.mv_tick(0, 0, 0, 0, 0, 0,
                    pack_lane_mant(np.zeros(LANES, dtype=np.int16)),
                    pack_lane_exp (np.zeros(LANES, dtype=np.int8 )),
                    out_valid, out_mant, out_exp)
        if out_valid[0]:
            m, e = unpack_out(out_mant, out_exp)
            rtl_m[rows] = m
            rtl_e[rows] = e
            chunks_done += 1
            break

lib.mv_destroy()

# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------
print(f"[cosim] chunks_done={chunks_done}/{N_CHUNKS}")
mismatches = 0
first_bad = None
for i in range(D_out):
    rv = int(rtl_m[i]) * (2.0 ** (int(rtl_e[i]) - 15))
    pv = int(ref_m[i]) * (2.0 ** (int(ref_e[i]) - 15))
    rel = abs(rv - pv) / max(abs(pv), 1e-12)
    if rel > TOL_REL:
        mismatches += 1
        if first_bad is None:
            first_bad = (i, int(rtl_m[i]), int(rtl_e[i]), int(ref_m[i]), int(ref_e[i]), rv, pv, rel)
if first_bad:
    i, rm, re_, pm, pe, rv, pv, rel = first_bad
    print(f"[cosim] FIRST MISMATCH at output position {i} (chunk {i//LANES}, lane {i%LANES}):")
    print(f"          RTL: m={rm:7d} e={re_:5d}  value={rv:+.6e}")
    print(f"          REF: m={pm:7d} e={pe:5d}  value={pv:+.6e}")
    print(f"          rel_err = {rel:.4e}")
    print(f"[cosim] {mismatches}/{D_out} positions exceed rel_tol={TOL_REL}")
    sys.exit(1)
else:
    print(f"[cosim] PASS — all {D_out} positions within rel_tol={TOL_REL}")
