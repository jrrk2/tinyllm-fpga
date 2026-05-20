#!/usr/bin/env python3
"""Block-FP weight baker for smollm_layer_bfp.sv.

Emits 7 weight matrices (Wq/Wk/Wv/Wo/Wg/Wu/Wd), 2 gammas (g1/g2), and KV
cache preload (K/V_INIT) as $readmemh hex files matching the RTL layout:

  Per matrix M of shape (D_out, D_in):
    rom_<M>_m hex (file <PREFIX>W<M>_m.hex):
      index = (chunk * D_in + col) * LANES + lane
        where chunk    = D_out / LANES           (0 .. CHUNKS_OUT-1)
              col      = 0 .. D_in-1
              lane     = 0 .. LANES-1            (= row within chunk)
      value = mantissa of W[chunk*LANES+lane, col]  (16-bit signed, masked)
    rom_<M>_e hex (file <PREFIX>W<M>_e.hex):
      index = (chunk * NT_in + tile) * LANES + lane
      value = shared exponent of tile (chunk*LANES+lane, tile)  (8-bit signed)

  Gammas: per-element mantissas (D entries) + per-tile exps (NT_D entries).

  KV cache: rope-rotated K + raw V at each preloaded timestep, sized
    MAX_CTX × H_KV × HD mantissas + MAX_CTX × NT_KV per-tile exps.

Outputs (default OUT=generated):  <OUT>/<PREFIX>{WQ,WK,WV,WO,WG,WU,WDN}_{m,e}.hex
                                  <OUT>/<PREFIX>{G1,G2}_{m,e}.hex
                                  <OUT>/<PREFIX>{K,V}_INIT_{m,e}.hex

Also emits <OUT>/<PREFIX>cfg.svh with `define LBFP_{D,HQ,HKV,HD,FFN,MAX_CTX}.
And a hidden-in test vector + golden hidden-out: <OUT>/<PREFIX>{HIN,HOUT}_{m,e}.hex
"""
import os, sys, math
import numpy as np

# --------------------------------------------------------------------------
# Config (defaults match smollm_layer_bfp.sv's small-dim parameters)
# --------------------------------------------------------------------------
D       = int(os.environ.get('D',       64))
H_Q     = int(os.environ.get('H_Q',     1))
H_KV    = int(os.environ.get('H_KV',    1))
HD      = int(os.environ.get('HD',      64))
FFN     = int(os.environ.get('FFN',     128))
MAX_CTX = int(os.environ.get('MAX_CTX', 4))
POS     = int(os.environ.get('POS',     3))
KV_POS  = int(os.environ.get('KV_POS',  3))
PREFIX  = os.environ.get('PREFIX', 'lbfp_')
OUT     = os.environ.get('OUT', 'generated')
TILE    = 16
LANES   = 16
SEED    = int(os.environ.get('SEED', 0))
np.random.seed(SEED)

NT_D    = (D + TILE - 1) // TILE
NT_FFN  = (FFN + TILE - 1) // TILE
NT_KV   = (H_KV * HD + TILE - 1) // TILE
CHUNKS_D    = D // LANES
CHUNKS_KV   = (H_KV * HD) // LANES
CHUNKS_FFN  = FFN // LANES

assert D % LANES == 0
assert (H_KV * HD) % LANES == 0
assert FFN % LANES == 0

os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
# Tile quantization (matches sim_rtl_fp_hw.py)
# --------------------------------------------------------------------------
def tile_quantize(v):
    """v: 1D vector (length divisible by TILE).  Returns (m_int, e_int)
       with m_int shape (D/TILE, TILE) int16, e_int shape (D/TILE,) int8."""
    flat = np.asarray(v, dtype=np.float64).reshape(-1, TILE)
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = (np.floor(np.log2(max_abs)).astype(np.int32) + 1).clip(-127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    m = np.clip(np.round(flat / m_scale * 32768.0).astype(np.int64),
                -32768, 32767)
    return m.astype(np.int16), e.astype(np.int8)


def tile_decode(m_int, e_int):
    """Inverse of tile_quantize: back to float64 1D vector."""
    m_scale = np.power(2.0, e_int.astype(np.float64))[:, None]
    return ((m_int.astype(np.float64) / 32768.0) * m_scale).flatten()


# --------------------------------------------------------------------------
# Quantize a weight matrix W (D_out, D_in) → m[(D_out, NT_in, TILE)],
# e[(D_out, NT_in)] (per-row tile-quantized along D_in).
# --------------------------------------------------------------------------
def quantize_W(W):
    D_out, D_in = W.shape
    assert D_in % TILE == 0
    NT_in = D_in // TILE
    m = np.zeros((D_out, NT_in, TILE), dtype=np.int16)
    e = np.zeros((D_out, NT_in),       dtype=np.int8)
    for i in range(D_out):
        mi, ei = tile_quantize(W[i])
        m[i] = mi; e[i] = ei
    return m, e


def write_hex(path, vals, width):
    """vals: iterable of ints.  width: hex digit count (4 for 16-bit, 2 for 8-bit)."""
    mask = (1 << (width * 4)) - 1
    with open(path, 'w') as f:
        for x in vals:
            f.write(f"{int(x) & mask:0{width}x}\n")


def emit_W(prefix_tag, W):
    """Emit rom_W<TAG>_m.hex + rom_W<TAG>_e.hex in the RTL's wide-packed
    layout — each line holds LANES mantissas (256 bits = 64 hex chars) for
    mantissa ROMs, LANES exponents (128 bits = 32 hex chars) for exponent
    ROMs.  This matches the int8 weight_streamer_brom.sv format that Vivado
    BRAM-infers as cascaded RAMB36 primitives.
       mantissa hex line N = concat(lane15 .. lane0) at entry N
       entry layout: row = chunk*LANES + lane, addr = chunk*D_in + col
    """
    m, e = quantize_W(W)
    D_out, D_in = W.shape
    NT_in = D_in // TILE
    CHUNKS_OUT = D_out // LANES
    # Mantissa wide entries: one 256-bit word per (chunk, col)
    mant_lines = []
    for chunk in range(CHUNKS_OUT):
        for col in range(D_in):
            word = 0
            for lane in range(LANES):
                row = chunk * LANES + lane
                tile = col // TILE
                idx_in_tile = col % TILE
                mv = int(m[row, tile, idx_in_tile]) & 0xFFFF
                word |= (mv << (lane * 16))
            mant_lines.append(f"{word:064x}")
    # Exponent wide entries: one 128-bit word per (chunk, tile)
    exp_lines = []
    for chunk in range(CHUNKS_OUT):
        for tile in range(NT_in):
            word = 0
            for lane in range(LANES):
                row = chunk * LANES + lane
                ev = int(e[row, tile]) & 0xFF
                word |= (ev << (lane * 8))
            exp_lines.append(f"{word:032x}")
    with open(os.path.join(OUT, f"{PREFIX}{prefix_tag}_m.hex"), 'w') as f:
        for ln in mant_lines: f.write(ln + "\n")
    with open(os.path.join(OUT, f"{PREFIX}{prefix_tag}_e.hex"), 'w') as f:
        for ln in exp_lines: f.write(ln + "\n")
    return m, e   # return for golden computation


def emit_gamma(prefix_tag, g):
    """Emit per-element mantissas (D entries) + per-tile exps (NT_D)."""
    m, e = tile_quantize(g)
    m_flat = m.flatten()
    write_hex(os.path.join(OUT, f"{PREFIX}{prefix_tag}_m.hex"), m_flat, 4)
    write_hex(os.path.join(OUT, f"{PREFIX}{prefix_tag}_e.hex"), e, 2)
    return m, e


def emit_kv_init(K, V):
    """K, V shape (MAX_CTX, H_KV*HD).  Emit per-element mantissas (one per
       cycle of cnt during KVWR_M) plus per-tile exponents (NT_KV per
       timestep)."""
    Kw = np.zeros(MAX_CTX * H_KV * HD, dtype=np.int16)
    Kw_e = np.zeros(MAX_CTX * NT_KV,    dtype=np.int8)
    Vw = np.zeros(MAX_CTX * H_KV * HD, dtype=np.int16)
    Vw_e = np.zeros(MAX_CTX * NT_KV,    dtype=np.int8)
    for t in range(MAX_CTX):
        km, ke = tile_quantize(K[t])
        Kw[t*H_KV*HD:(t+1)*H_KV*HD]    = km.flatten()
        Kw_e[t*NT_KV:(t+1)*NT_KV]      = ke
        vm, ve = tile_quantize(V[t])
        Vw[t*H_KV*HD:(t+1)*H_KV*HD]    = vm.flatten()
        Vw_e[t*NT_KV:(t+1)*NT_KV]      = ve
    write_hex(os.path.join(OUT, f"{PREFIX}K_INIT_m.hex"), Kw, 4)
    write_hex(os.path.join(OUT, f"{PREFIX}K_INIT_e.hex"), Kw_e, 2)
    write_hex(os.path.join(OUT, f"{PREFIX}V_INIT_m.hex"), Vw, 4)
    write_hex(os.path.join(OUT, f"{PREFIX}V_INIT_e.hex"), Vw_e, 2)


# --------------------------------------------------------------------------
# Generate small synthetic weights — magnitudes chosen so every tile has
# meaningful dynamic range (otherwise tiles full of zeros would have
# arbitrary exp and confuse the test).
# --------------------------------------------------------------------------
def randn(*shape, scale=0.5):
    return (np.random.randn(*shape) * scale).astype(np.float64)


Wq = randn(D,       D,   scale=0.3)
Wk = randn(H_KV*HD, D,   scale=0.3)
Wv = randn(H_KV*HD, D,   scale=0.3)
Wo = randn(D,       D,   scale=0.3)
Wg = randn(FFN,     D,   scale=0.3)
Wu = randn(FFN,     D,   scale=0.3)
Wd = randn(D,       FFN, scale=0.3)
g1 = np.ones(D, dtype=np.float64) + 0.05 * np.random.randn(D)
g2 = np.ones(D, dtype=np.float64) + 0.05 * np.random.randn(D)

# Quantize everything to BFP (the RTL only ever sees the quantized versions)
print(f"baking BFP weights D={D} HD={HD} FFN={FFN} MAX_CTX={MAX_CTX} PREFIX={PREFIX}",
      file=sys.stderr)

emit_W('WQ',  Wq);  emit_W('WK', Wk);  emit_W('WV', Wv);  emit_W('WO', Wo)
emit_W('WG',  Wg);  emit_W('WU', Wu);  emit_W('WDN', Wd)
emit_gamma('G1', g1); emit_gamma('G2', g2)

# --------------------------------------------------------------------------
# KV cache preload: timesteps 0..KV_POS-1 contain post-rope K and raw V.
# We synthesize them as random and let the test feed kv_pos=KV_POS so the
# RTL writes timestep KV_POS itself.  For golden-matching, the Python
# golden does the same thing.
# --------------------------------------------------------------------------
K_init = randn(MAX_CTX, H_KV*HD, scale=0.5)
V_init = randn(MAX_CTX, H_KV*HD, scale=0.5)
emit_kv_init(K_init, V_init)

# --------------------------------------------------------------------------
# Hidden-in test vector (D elements)
# --------------------------------------------------------------------------
h_in = randn(D, scale=1.0)
h_m, h_e = tile_quantize(h_in)
write_hex(os.path.join(OUT, f"{PREFIX}HIN_m.hex"), h_m.flatten(), 4)
write_hex(os.path.join(OUT, f"{PREFIX}HIN_e.hex"), h_e, 2)
h_in_q = tile_decode(h_m, h_e)

# Emit hidden-in as a case-statement LUT SVH for FPGA synth (Vivado mis-
# handles $readmemh into a comb-read array; case-stmt LUTs synthesize cleanly).
with open(os.path.join(OUT, f"{PREFIX}hidden_in_lut.svh"), 'w') as f:
    f.write(f"// AUTO-GENERATED by gen_smollm_blockfp_bfp.py — do not edit.\n")
    f.write(f"// BFP hidden_in: D={D} mantissas, NT={NT_D} per-tile exponents.\n")
    f.write(f"\nfunction automatic logic [15:0] {PREFIX}hin_m_lut(input int unsigned idx);\n")
    f.write(f"  case (idx)\n")
    flat_m = h_m.flatten()
    for i in range(D):
        f.write(f"    {i:5d}: {PREFIX}hin_m_lut = 16'h{int(flat_m[i]) & 0xFFFF:04x};\n")
    f.write(f"    default: {PREFIX}hin_m_lut = 16'h0000;\n")
    f.write(f"  endcase\nendfunction\n")
    f.write(f"\nfunction automatic logic [7:0] {PREFIX}hin_e_lut(input int unsigned idx);\n")
    f.write(f"  case (idx)\n")
    for t in range(NT_D):
        f.write(f"    {t:5d}: {PREFIX}hin_e_lut = 8'h{int(h_e[t]) & 0xFF:02x};\n")
    f.write(f"    default: {PREFIX}hin_e_lut = 8'h00;\n")
    f.write(f"  endcase\nendfunction\n")

# --------------------------------------------------------------------------
# Reference forward pass — mirrors sim_rtl_fp_hw.py's fwd_hw exactly, so
# the RTL output should match within block-FP quantization tolerance.
# --------------------------------------------------------------------------
ALIGN_MAX = 47


def fp_quantize_vec(v):
    m, e = tile_quantize(v)
    return tile_decode(m, e)


def matvec_hw_golden(x_m, x_e, W_m, W_e):
    """Hardware-faithful per-tile FP matvec — same as sim_rtl_fp_hw.py."""
    D_out, nT = W_e.shape
    prods = x_m[None, :, :].astype(np.int64) * W_m[:].astype(np.int64)
    tile_sums = prods.sum(axis=2)
    tile_exps = x_e[None, :].astype(np.int64) + W_e.astype(np.int64)
    max_e = tile_exps.max(axis=1, keepdims=True)
    shift = max_e - tile_exps
    mask = shift <= ALIGN_MAX
    shift_capped = np.clip(shift, 0, 63)
    aligned = np.right_shift(tile_sums, shift_capped)
    aligned = np.where(mask, aligned, 0)
    acc = aligned.sum(axis=1)
    SAT_HI = (1 << 47) - 1
    SAT_LO = -(1 << 47)
    acc = np.clip(acc, SAT_LO, SAT_HI)
    out_real = acc.astype(np.float64) * np.power(2.0, max_e[:, 0].astype(np.float64) - 30)
    return fp_quantize_vec(out_real)


def rmsnorm_fp(x, gamma, eps=1e-5):
    v = np.mean(x*x) + eps
    return fp_quantize_vec(x * gamma / np.sqrt(v))


def silu_fp(x):
    return fp_quantize_vec(x / (1.0 + np.exp(-x)))


def softmax_fp(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def rope_fp(x, pos, base=10000.0):
    HD2 = HD // 2
    out = np.array(x, dtype=np.float64)
    for j in range(HD2):
        theta = 1.0 / (base ** (2*j/HD)) * pos
        c = math.cos(theta); s = math.sin(theta)
        a, b = x[j], x[j+HD2]
        out[j]     = a*c - b*s
        out[j+HD2] = b*c + a*s
    return fp_quantize_vec(out)


# Quantize weights once for the golden compute
Wq_m, Wq_e = quantize_W(Wq)
Wk_m, Wk_e = quantize_W(Wk)
Wv_m, Wv_e = quantize_W(Wv)
Wo_m, Wo_e = quantize_W(Wo)
Wg_m, Wg_e = quantize_W(Wg)
Wu_m, Wu_e = quantize_W(Wu)
Wd_m, Wd_e = quantize_W(Wd)
g1_q = fp_quantize_vec(g1)
g2_q = fp_quantize_vec(g2)

# Decode preloaded KV from the hex (so we use exactly what the RTL sees)
kv_K = np.zeros((MAX_CTX, H_KV, HD))
kv_V = np.zeros((MAX_CTX, H_KV, HD))
for t in range(MAX_CTX):
    km, ke = tile_quantize(K_init[t])
    vm, ve = tile_quantize(V_init[t])
    flat_k = tile_decode(km, ke); flat_v = tile_decode(vm, ve)
    for h_i in range(H_KV):
        kv_K[t, h_i] = flat_k[h_i*HD:(h_i+1)*HD]
        kv_V[t, h_i] = flat_v[h_i*HD:(h_i+1)*HD]

grp = H_Q // H_KV
n1 = rmsnorm_fp(h_in_q, g1_q)
n1_m, n1_e = tile_quantize(n1)
q = matvec_hw_golden(n1_m, n1_e, Wq_m, Wq_e)
k = matvec_hw_golden(n1_m, n1_e, Wk_m, Wk_e)
v = matvec_hw_golden(n1_m, n1_e, Wv_m, Wv_e)
for h_i in range(H_Q):  q[h_i*HD:(h_i+1)*HD] = rope_fp(q[h_i*HD:(h_i+1)*HD], POS)
for h_i in range(H_KV): k[h_i*HD:(h_i+1)*HD] = rope_fp(k[h_i*HD:(h_i+1)*HD], POS)
# Write current step into KV cache slot KV_POS
for h_i in range(H_KV):
    kv_K[KV_POS, h_i] = k[h_i*HD:(h_i+1)*HD]
    kv_V[KV_POS, h_i] = v[h_i*HD:(h_i+1)*HD]
attn = np.zeros(D)
golden_scores = None
golden_probs  = None
for h_i in range(H_Q):
    kvh = h_i // grp
    scores = np.zeros(KV_POS + 1)
    for t in range(KV_POS + 1):
        scores[t] = float(np.dot(q[h_i*HD:(h_i+1)*HD], kv_K[t, kvh])) / math.sqrt(HD)
    sm = softmax_fp(scores)
    # Save the LAST head's scores/probs — the RTL layer's $writememh
    # of scores_m / probs_m fires once per head (per S_SM_WAIT → S_AV_PRIME
    # transition) and is overwritten each iteration, so the file holds
    # whichever head ran last.  Comparing against any other head's golden
    # would be apples-to-oranges.
    if h_i == H_Q - 1:
        golden_scores = scores.copy()
        golden_probs  = sm.copy()
    for t in range(KV_POS + 1):
        attn[h_i*HD:(h_i+1)*HD] += sm[t] * kv_V[t, kvh]
attn = fp_quantize_vec(attn)
# Save golden scores + probs for head 0 (single-head test config).
# Quantize scores to BFP at scores_shared_exp (max of per-timestep exponents);
# probs to BFP-Q1.15 (shared exp via softmax output).
# RTL dump format: 4 mantissas (MAX_CTX) each.
if golden_scores is not None:
    s_pad = np.zeros(MAX_CTX); s_pad[:KV_POS+1] = golden_scores
    p_pad = np.zeros(MAX_CTX); p_pad[:KV_POS+1] = golden_probs
    # scores: pad to next TILE multiple so tile_quantize can reshape;
    # take first MAX_CTX elements afterwards.  Originally assumed
    # MAX_CTX ≤ TILE — fails for smollm360's MAX_CTX=32.
    pad_to = ((MAX_CTX + TILE - 1) // TILE) * TILE
    s_padT = np.zeros(pad_to); s_padT[:MAX_CTX] = s_pad
    p_padT = np.zeros(pad_to); p_padT[:MAX_CTX] = p_pad
    sm_, se_ = tile_quantize(s_padT); pm_, pe_ = tile_quantize(p_padT)
    write_hex(os.path.join(OUT, f"{PREFIX}STAGE_SCORES_m.hex"), sm_.flatten()[:MAX_CTX], 4)
    write_hex(os.path.join(OUT, f"{PREFIX}STAGE_SCORES_e.hex"), [int(se_[0])]*MAX_CTX, 2)
    write_hex(os.path.join(OUT, f"{PREFIX}STAGE_PROBS_m.hex"),  pm_.flatten()[:MAX_CTX], 4)
    write_hex(os.path.join(OUT, f"{PREFIX}STAGE_PROBS_e.hex"),  [int(pe_[0])]*MAX_CTX, 2)
    sys.stderr.write(f"  golden scores: {golden_scores}\n")
    sys.stderr.write(f"  golden probs:  {golden_probs}\n")
a_m, a_e = tile_quantize(attn)
o = matvec_hw_golden(a_m, a_e, Wo_m, Wo_e)
h1 = fp_quantize_vec(h_in_q + o)
n2 = rmsnorm_fp(h1, g2_q)
n2_m, n2_e = tile_quantize(n2)
g_vec = matvec_hw_golden(n2_m, n2_e, Wg_m, Wg_e)
u_vec = matvec_hw_golden(n2_m, n2_e, Wu_m, Wu_e)
mlp = fp_quantize_vec(silu_fp(g_vec) * u_vec)
mlp_m, mlp_e = tile_quantize(mlp)
d_vec = matvec_hw_golden(mlp_m, mlp_e, Wd_m, Wd_e)
h_out = fp_quantize_vec(h1 + d_vec)

# Write golden hidden_out
h_out_m, h_out_e = tile_quantize(h_out)
write_hex(os.path.join(OUT, f"{PREFIX}HOUT_m.hex"), h_out_m.flatten(), 4)
write_hex(os.path.join(OUT, f"{PREFIX}HOUT_e.hex"), h_out_e, 2)

# Dump per-stage intermediates so RTL can diff against them.
def dump_stage(tag, arr):
    m, e = tile_quantize(arr)
    write_hex(os.path.join(OUT, f"{PREFIX}STAGE_{tag}_m.hex"), m.flatten(), 4)
    write_hex(os.path.join(OUT, f"{PREFIX}STAGE_{tag}_e.hex"), e, 2)

dump_stage('N1',  n1)
dump_stage('Q',   q)        # post-rope Q (since rope_fp overwrites in-place)
dump_stage('K',   k)        # post-rope K
dump_stage('V',   v)
dump_stage('ATTN', attn)
dump_stage('O',   o)
dump_stage('H1',  h1)
dump_stage('N2',  n2)
dump_stage('G',   g_vec)
dump_stage('U',   u_vec)
dump_stage('MLP', mlp)
dump_stage('D',   d_vec)
# Pre-rope Q/K dumps need a re-run before the in-place rope step.  Re-do.
np.random.seed(SEED)
# advance RNG to the same point as h_in
_ = (np.random.randn(D,D) * 0.3); _ = (np.random.randn(H_KV*HD,D) * 0.3)
_ = (np.random.randn(H_KV*HD,D) * 0.3); _ = (np.random.randn(D,D) * 0.3)
_ = (np.random.randn(FFN,D) * 0.3); _ = (np.random.randn(FFN,D) * 0.3)
_ = (np.random.randn(D,FFN) * 0.3)
_ = np.ones(D) + 0.05*np.random.randn(D); _ = np.ones(D) + 0.05*np.random.randn(D)
_ = (np.random.randn(MAX_CTX,H_KV*HD) * 0.5); _ = (np.random.randn(MAX_CTX,H_KV*HD) * 0.5)
_ = (np.random.randn(D) * 1.0)
# Recompute pre-rope:
n1_rerun = rmsnorm_fp(h_in_q, g1_q)
n1m_r, n1e_r = tile_quantize(n1_rerun)
q_pre = matvec_hw_golden(n1m_r, n1e_r, Wq_m, Wq_e)
k_pre = matvec_hw_golden(n1m_r, n1e_r, Wk_m, Wk_e)
dump_stage('QPRE', q_pre)
dump_stage('KPRE', k_pre)

# Emit cfg.svh
with open(os.path.join(OUT, f"{PREFIX}cfg.svh"), 'w') as f:
    f.write(f"`define LBFP_D       {D}\n")
    f.write(f"`define LBFP_HQ      {H_Q}\n")
    f.write(f"`define LBFP_HKV     {H_KV}\n")
    f.write(f"`define LBFP_HD      {HD}\n")
    f.write(f"`define LBFP_FFN     {FFN}\n")
    f.write(f"`define LBFP_MAX_CTX {MAX_CTX}\n")
    f.write(f"`define LBFP_POS     {POS}\n")
    f.write(f"`define LBFP_KV_POS  {KV_POS}\n")

print(f"wrote {OUT}/{PREFIX}*.hex + {PREFIX}cfg.svh", file=sys.stderr)
print(f"hidden_out (golden):  min={h_out.min():.3f}  max={h_out.max():.3f}",
      file=sys.stderr)
