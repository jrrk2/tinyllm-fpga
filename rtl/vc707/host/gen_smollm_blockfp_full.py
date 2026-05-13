#!/usr/bin/env python3
"""Full-scale block-FP weight baker for SmolLM2-135M.

Extracts all 30 transformer layers from HuggingFace SmolLM2-135M, tile-
quantizes to BFP (TILE=16 mantissas / 8-bit shared exp), packs LANES=16
mantissas (resp. exponents) per wide BRAM entry, and emits one .hex file
per matrix kind with all NL layers concatenated.  Companion to the small-
dim selftest baker (host/gen_smollm_blockfp_bfp.py) — this one is for the
full Verilator multilayer harness + the future FPGA build.

Output files (default OUT=generated, PREFIX=lbfp_full_):

  Weight matrices (wide entries, 256-bit mantissa / 128-bit exp):
    <PREFIX>WQ_m.hex   shape (NL * CHUNKS_D   * D, 256)
    <PREFIX>WQ_e.hex   shape (NL * CHUNKS_D   * NT_D, 128)
    <PREFIX>WK_m.hex   shape (NL * CHUNKS_KV  * D, 256)
    <PREFIX>WK_e.hex   shape (NL * CHUNKS_KV  * NT_D, 128)
    <PREFIX>WV_m.hex   ...   <PREFIX>WV_e.hex
    <PREFIX>WO_m.hex   ...   <PREFIX>WO_e.hex
    <PREFIX>WG_m.hex   shape (NL * CHUNKS_FFN * D, 256)
    <PREFIX>WG_e.hex   shape (NL * CHUNKS_FFN * NT_D, 128)
    <PREFIX>WU_m.hex   ...   <PREFIX>WU_e.hex
    <PREFIX>WDN_m.hex  shape (NL * CHUNKS_D   * FFN, 256)
    <PREFIX>WDN_e.hex  shape (NL * CHUNKS_D   * NT_FFN, 128)

  Per-layer norm gammas (narrow, 16-bit mant + 8-bit exp per element):
    <PREFIX>G1_m.hex   NL * D entries
    <PREFIX>G1_e.hex   NL * NT_D entries
    <PREFIX>G2_m.hex   NL * D entries
    <PREFIX>G2_e.hex   NL * NT_D entries

  Decoder side:
    <PREFIX>EMBED_m.hex    VOCAB_SIZE × D wide (256-bit) entries
    <PREFIX>EMBED_e.hex    VOCAB_SIZE × NT_D wide (128-bit) entries
    <PREFIX>NORM_W_m.hex   D entries (final pre-lm_head norm weight)
    <PREFIX>NORM_W_e.hex   NT_D entries

  Golden reference:
    <PREFIX>GOLDEN_TOKENS.txt   one token-id per line; produced by the
                                 same BFP forward used by sim_rtl_fp_hw.py
                                 so RTL can autoregress and compare.

Also emits <PREFIX>cfg.svh with `define LBFP_FULL_{D,HQ,HKV,HD,FFN,NL,MAX_CTX,VOCAB}.
"""
import os, sys, math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL   = os.environ.get('MODEL',   'HuggingFaceTB/SmolLM2-135M')
D       = int(os.environ.get('D',       576))
H_Q     = int(os.environ.get('H_Q',     9))
H_KV    = int(os.environ.get('H_KV',    3))
HD      = int(os.environ.get('HD',      64))
FFN     = int(os.environ.get('FFN',     1536))
NL      = int(os.environ.get('NL',      30))
MAX_CTX = int(os.environ.get('MAX_CTX', 64))
PROMPT  = os.environ.get('PROMPT',  'Once upon a time')
N_GEN   = int(os.environ.get('N_GEN',   15))
PREFIX  = os.environ.get('PREFIX',  'lbfp_full_')
OUT     = os.environ.get('OUT',     'generated')
TILE    = 16
LANES   = 16
BFP_MANT_W = 16
BFP_EXP_W  = 8

NT_D    = D // TILE
NT_FFN  = FFN // TILE
NT_KV   = (H_KV * HD) // TILE
CHUNKS_D    = D // LANES
CHUNKS_KV   = (H_KV * HD) // LANES
CHUNKS_FFN  = FFN // LANES
H2          = HD // 2

assert D % LANES == 0
assert (H_KV * HD) % LANES == 0
assert FFN % LANES == 0
assert D % TILE == 0
assert FFN % TILE == 0
assert (H_KV * HD) % TILE == 0

os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
# Tile quantization — bit-for-bit identical to sim_rtl_fp_hw.py
# --------------------------------------------------------------------------
def tile_quantize(v):
    flat = np.asarray(v, dtype=np.float64).reshape(-1, TILE)
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = (np.floor(np.log2(max_abs)).astype(np.int32) + 1).clip(-127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    m = np.clip(np.round(flat / m_scale * 32768.0).astype(np.int64),
                -32768, 32767)
    return m.astype(np.int16), e.astype(np.int8)


def tile_decode(m_int, e_int):
    m_scale = np.power(2.0, e_int.astype(np.float64))[:, None]
    return ((m_int.astype(np.float64) / 32768.0) * m_scale).flatten()


def fp_quantize_vec(v):
    m, e = tile_quantize(v)
    return tile_decode(m, e)


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


# --------------------------------------------------------------------------
# Hex emitters — wide-packed, NL layers concatenated.
# --------------------------------------------------------------------------
def write_lines(path, lines):
    with open(path, 'w') as f:
        for ln in lines:
            f.write(ln + "\n")


def emit_W_concat(prefix_tag, m_per_layer, e_per_layer, D_in):
    """m_per_layer: list of (D_out, NT_in, TILE) int16 arrays, one per layer.
       e_per_layer: list of (D_out, NT_in)       int8  arrays, one per layer.
       Emits two .hex files (mant + exp) with NL layers concatenated."""
    D_out, NT_in, _ = m_per_layer[0].shape
    CHUNKS_OUT = D_out // LANES
    mant_lines, exp_lines = [], []
    for li in range(len(m_per_layer)):
        m = m_per_layer[li]; e = e_per_layer[li]
        # Mantissa wide entries: one 256-bit word per (chunk, col)
        for chunk in range(CHUNKS_OUT):
            for col in range(D_in):
                word = 0
                for lane in range(LANES):
                    row  = chunk * LANES + lane
                    tile = col // TILE
                    idx  = col % TILE
                    mv = int(m[row, tile, idx]) & 0xFFFF
                    word |= (mv << (lane * BFP_MANT_W))
                mant_lines.append(f"{word:064x}")
        # Exponent wide entries: one 128-bit word per (chunk, tile)
        for chunk in range(CHUNKS_OUT):
            for tile in range(NT_in):
                word = 0
                for lane in range(LANES):
                    row = chunk * LANES + lane
                    ev = int(e[row, tile]) & 0xFF
                    word |= (ev << (lane * BFP_EXP_W))
                exp_lines.append(f"{word:032x}")
    write_lines(os.path.join(OUT, f"{PREFIX}{prefix_tag}_m.hex"), mant_lines)
    write_lines(os.path.join(OUT, f"{PREFIX}{prefix_tag}_e.hex"), exp_lines)


def emit_gamma_concat(prefix_tag, g_per_layer):
    """g_per_layer: list of (D,) float vectors, one per layer.
       Emits per-element mantissa hex + per-tile exponent hex (narrow)."""
    mant_lines, exp_lines = [], []
    for g in g_per_layer:
        m, e = tile_quantize(g)
        for v in m.flatten():
            mant_lines.append(f"{int(v) & 0xFFFF:04x}")
        for v in e.flatten():
            exp_lines.append(f"{int(v) & 0xFF:02x}")
    write_lines(os.path.join(OUT, f"{PREFIX}{prefix_tag}_m.hex"), mant_lines)
    write_lines(os.path.join(OUT, f"{PREFIX}{prefix_tag}_e.hex"), exp_lines)


def emit_table_wide(prefix_tag, T):
    """T: (rows, D) float matrix.  Emit wide-packed (LANES per entry) hex
       just like emit_W_concat but for a single-layer table (embedding /
       lm_head)."""
    rows, D_cols = T.shape
    m, e = quantize_W(T)  # (rows, NT_D, TILE) and (rows, NT_D)
    CHUNKS_OUT = rows // LANES
    assert rows % LANES == 0, f"{prefix_tag}: rows {rows} not LANES-aligned"
    NT_in = D_cols // TILE
    mant_lines, exp_lines = [], []
    for chunk in range(CHUNKS_OUT):
        for col in range(D_cols):
            word = 0
            for lane in range(LANES):
                row  = chunk * LANES + lane
                tile = col // TILE
                idx  = col % TILE
                mv = int(m[row, tile, idx]) & 0xFFFF
                word |= (mv << (lane * BFP_MANT_W))
            mant_lines.append(f"{word:064x}")
    for chunk in range(CHUNKS_OUT):
        for tile in range(NT_in):
            word = 0
            for lane in range(LANES):
                row = chunk * LANES + lane
                ev = int(e[row, tile]) & 0xFF
                word |= (ev << (lane * BFP_EXP_W))
            exp_lines.append(f"{word:032x}")
    write_lines(os.path.join(OUT, f"{PREFIX}{prefix_tag}_m.hex"), mant_lines)
    write_lines(os.path.join(OUT, f"{PREFIX}{prefix_tag}_e.hex"), exp_lines)


# --------------------------------------------------------------------------
# Hardware-faithful BFP forward (matches sim_rtl_fp_hw.py exactly so the
# golden tokens produced here are what the RTL must reproduce).
# --------------------------------------------------------------------------
ALIGN_MAX = 47


def matvec_hw_golden(x_m, x_e, W_m, W_e):
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
    out = np.array(x, dtype=np.float64)
    for j in range(H2):
        theta = 1.0 / (base ** (2*j/HD)) * pos
        c = math.cos(theta); s = math.sin(theta)
        a, b = x[j], x[j+H2]
        out[j]    = a*c - b*s
        out[j+H2] = b*c + a*s
    return fp_quantize_vec(out)


def fwd_layer(h, lw, pos, kv, kv_pos):
    grp = H_Q // H_KV
    n1 = rmsnorm_fp(h, lw['g1'])
    n1_m, n1_e = tile_quantize(n1)
    q = matvec_hw_golden(n1_m, n1_e, *lw['Wq'])
    k = matvec_hw_golden(n1_m, n1_e, *lw['Wk'])
    v = matvec_hw_golden(n1_m, n1_e, *lw['Wv'])
    for h_i in range(H_Q):  q[h_i*HD:(h_i+1)*HD] = rope_fp(q[h_i*HD:(h_i+1)*HD], pos)
    for h_i in range(H_KV): k[h_i*HD:(h_i+1)*HD] = rope_fp(k[h_i*HD:(h_i+1)*HD], pos)
    for h_i in range(H_KV):
        kv['k'][kv_pos, h_i] = k[h_i*HD:(h_i+1)*HD]
        kv['v'][kv_pos, h_i] = v[h_i*HD:(h_i+1)*HD]
    attn = np.zeros(D)
    for h_i in range(H_Q):
        kvh = h_i // grp
        scores = np.zeros(kv_pos+1)
        for t in range(kv_pos+1):
            scores[t] = float(np.dot(q[h_i*HD:(h_i+1)*HD], kv['k'][t, kvh])) / math.sqrt(HD)
        sm = softmax_fp(scores)
        for t in range(kv_pos+1):
            attn[h_i*HD:(h_i+1)*HD] += sm[t] * kv['v'][t, kvh]
    attn = fp_quantize_vec(attn)
    a_m, a_e = tile_quantize(attn)
    o = matvec_hw_golden(a_m, a_e, *lw['Wo'])
    h1 = fp_quantize_vec(h + o)
    n2 = rmsnorm_fp(h1, lw['g2'])
    n2_m, n2_e = tile_quantize(n2)
    g_vec = matvec_hw_golden(n2_m, n2_e, *lw['Wg'])
    u_vec = matvec_hw_golden(n2_m, n2_e, *lw['Wu'])
    mlp = fp_quantize_vec(silu_fp(g_vec) * u_vec)
    mlp_m, mlp_e = tile_quantize(mlp)
    d_vec = matvec_hw_golden(mlp_m, mlp_e, *lw['Wd'])
    return fp_quantize_vec(h1 + d_vec)


# --------------------------------------------------------------------------
# Main: extract → quantize → emit hex → run BFP forward → save golden
# --------------------------------------------------------------------------
def main():
    print(f"[bake] loading {MODEL} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

    print(f"[bake] extracting weights NL={NL} D={D} FFN={FFN} ...", file=sys.stderr)
    layers_quant = []   # for hex emit (per-layer m,e arrays)
    layers_fwd   = []   # for golden forward (m,e tuples by matrix name)
    g1_list, g2_list = [], []
    with torch.no_grad():
        for li in range(NL):
            L = model.model.layers[li]
            row = {}
            for nm, sub in [('Wq', L.self_attn.q_proj),
                            ('Wk', L.self_attn.k_proj),
                            ('Wv', L.self_attn.v_proj),
                            ('Wo', L.self_attn.o_proj),
                            ('Wg', L.mlp.gate_proj),
                            ('Wu', L.mlp.up_proj),
                            ('Wd', L.mlp.down_proj)]:
                W = sub.weight.detach().cpu().numpy().astype(np.float64)
                m, e = quantize_W(W)
                row[nm] = (m, e)
            row['g1'] = fp_quantize_vec(L.input_layernorm.weight.detach().cpu().numpy().astype(np.float64))
            row['g2'] = fp_quantize_vec(L.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float64))
            layers_quant.append(row)
            g1_list.append(row['g1']); g2_list.append(row['g2'])
            print(f"  L{li:02d}", end='\r', file=sys.stderr)
        embed = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float64)
        norm_w = model.model.norm.weight.detach().cpu().numpy().astype(np.float64)
    VOCAB = embed.shape[0]
    print(f"\n[bake] VOCAB={VOCAB}", file=sys.stderr)

    # Emit per-matrix hex (NL layers concatenated)
    for tag, D_in, D_out in [('WQ',  D,   D),
                              ('WK',  D,   H_KV*HD),
                              ('WV',  D,   H_KV*HD),
                              ('WO',  D,   D),
                              ('WG',  D,   FFN),
                              ('WU',  D,   FFN),
                              ('WDN', FFN, D)]:
        nm_key = {'WQ':'Wq','WK':'Wk','WV':'Wv','WO':'Wo',
                  'WG':'Wg','WU':'Wu','WDN':'Wd'}[tag]
        m_pl = [layers_quant[li][nm_key][0] for li in range(NL)]
        e_pl = [layers_quant[li][nm_key][1] for li in range(NL)]
        emit_W_concat(tag, m_pl, e_pl, D_in)
        print(f"  emit {tag}: {m_pl[0].shape[0]} rows × {D_in} cols × {NL} layers",
              file=sys.stderr)
    emit_gamma_concat('G1', g1_list)
    emit_gamma_concat('G2', g2_list)
    print(f"  emit G1/G2: {NL} layers × {D} elements", file=sys.stderr)

    # Empty KV-init files (NL*MAX_CTX*H_KV*HD zero mantissas + NL*MAX_CTX*NT_KV
    # zero exponents).  KV starts empty in autoregress; the BFP forward fills
    # slot kv_pos at each step.  Verilator's --x-initial 0 already
    # zero-initializes the arrays — these files just suppress $readmem
    # not-found warnings.
    kv_m_count = NL * MAX_CTX * H_KV * HD
    kv_e_count = NL * MAX_CTX * NT_KV
    write_lines(os.path.join(OUT, f"{PREFIX}K_INIT_m.hex"), ['0000'] * kv_m_count)
    write_lines(os.path.join(OUT, f"{PREFIX}V_INIT_m.hex"), ['0000'] * kv_m_count)
    write_lines(os.path.join(OUT, f"{PREFIX}K_INIT_e.hex"), ['00']   * kv_e_count)
    write_lines(os.path.join(OUT, f"{PREFIX}V_INIT_e.hex"), ['00']   * kv_e_count)
    print(f"  emit KV-init zeros: {kv_m_count} mant + {kv_e_count} exp", file=sys.stderr)

    # Embed table (VOCAB × D) — pad VOCAB to LANES multiple
    pad = (LANES - VOCAB % LANES) % LANES
    if pad:
        embed_padded = np.vstack([embed, np.zeros((pad, D))])
    else:
        embed_padded = embed
    emit_table_wide('EMBED', embed_padded)
    print(f"  emit EMBED: {embed_padded.shape[0]} rows × {D} cols (padded from {VOCAB})", file=sys.stderr)

    # Narrow per-element embed lookup for the RTL embed_lookup_bfp module.
    # Layout: rom indexed by token_id*D + col (mantissas) or token_id*NT_D + tile
    # (exponents).  $readmemh reads one value per hex line.
    emb_m_lines = []
    emb_e_lines = []
    for r in range(embed_padded.shape[0]):
        m, e = tile_quantize(embed_padded[r])
        for v in m.flatten():
            emb_m_lines.append(f"{int(v) & 0xFFFF:04x}")
        for v in e.flatten():
            emb_e_lines.append(f"{int(v) & 0xFF:02x}")
    write_lines(os.path.join(OUT, f"{PREFIX}EMBED_LOOKUP_m.hex"), emb_m_lines)
    write_lines(os.path.join(OUT, f"{PREFIX}EMBED_LOOKUP_e.hex"), emb_e_lines)
    print(f"  emit EMBED_LOOKUP_{{m,e}}.hex (narrow per-element, for RTL)", file=sys.stderr)

    # Prompt token ids — one per line — used by tb_full_bfp.cpp.
    with open(os.path.join(OUT, f"{PREFIX}PROMPT_TOKENS.txt"), 'w') as f:
        for tid in tok(PROMPT, return_tensors='pt').input_ids[0].tolist():
            f.write(f"{tid}\n")

    # Final norm_w (D-element gamma)
    nw_m, nw_e = tile_quantize(norm_w)
    write_lines(os.path.join(OUT, f"{PREFIX}NORM_W_m.hex"),
                [f"{int(v)&0xFFFF:04x}" for v in nw_m.flatten()])
    write_lines(os.path.join(OUT, f"{PREFIX}NORM_W_e.hex"),
                [f"{int(v)&0xFF:02x}" for v in nw_e.flatten()])
    print(f"  emit NORM_W: {D} elements", file=sys.stderr)

    # --------------------------------------------------------------------
    # Run the BFP forward pass and save the golden token sequence.
    # Mirrors sim_rtl_fp_hw.py's main loop bit-for-bit.
    # --------------------------------------------------------------------
    print(f"[bake] running BFP forward pass to save golden tokens ...", file=sys.stderr)
    embed_q = np.array([fp_quantize_vec(embed[i].astype(np.float64))
                        for i in range(VOCAB)])
    norm_w_q = fp_quantize_vec(norm_w)

    # Repack layers in the form fwd_layer expects: (m, e) tuples.
    layers_fwd = []
    for li in range(NL):
        lq = layers_quant[li]
        layers_fwd.append({
            'Wq': lq['Wq'], 'Wk': lq['Wk'], 'Wv': lq['Wv'], 'Wo': lq['Wo'],
            'Wg': lq['Wg'], 'Wu': lq['Wu'], 'Wd': lq['Wd'],
            'g1': lq['g1'], 'g2': lq['g2'],
        })

    ids = tok(PROMPT, return_tensors='pt').input_ids[0].tolist()
    kv = [{'k': np.zeros((MAX_CTX, H_KV, HD), dtype=np.float64),
           'v': np.zeros((MAX_CTX, H_KV, HD), dtype=np.float64)}
          for _ in range(NL)]
    generated = list(ids)
    print(f"\nprompt = {PROMPT!r}", file=sys.stderr)
    print(f"{PROMPT}", end='', flush=True, file=sys.stderr)
    for step in range(len(ids) + N_GEN):
        tid = ids[step] if step < len(ids) else generated[-1]
        h = embed_q[tid]
        for li in range(NL):
            h = fwd_layer(h, layers_fwd[li], pos=step, kv=kv[li], kv_pos=step)
        if step >= len(ids) - 1:
            h_normed = (h / np.sqrt(np.mean(h*h) + 1e-5)) * norm_w_q
            logits = h_normed @ embed.T
            nid = int(np.argmax(logits))
            if step >= len(ids):
                print(tok.decode([nid]), end='', flush=True, file=sys.stderr)
                generated.append(nid)
    print(file=sys.stderr)

    # Save golden tokens (one per line) for RTL to compare against.
    with open(os.path.join(OUT, f"{PREFIX}GOLDEN_TOKENS.txt"), 'w') as f:
        for tid in generated:
            f.write(f"{tid}\n")
    print(f"[bake] wrote {len(generated)} golden tokens to {PREFIX}GOLDEN_TOKENS.txt",
          file=sys.stderr)

    # cfg.svh
    with open(os.path.join(OUT, f"{PREFIX}cfg.svh"), 'w') as f:
        f.write(f"`define LBFP_FULL_D       {D}\n")
        f.write(f"`define LBFP_FULL_HQ      {H_Q}\n")
        f.write(f"`define LBFP_FULL_HKV     {H_KV}\n")
        f.write(f"`define LBFP_FULL_HD      {HD}\n")
        f.write(f"`define LBFP_FULL_FFN     {FFN}\n")
        f.write(f"`define LBFP_FULL_NL      {NL}\n")
        f.write(f"`define LBFP_FULL_MAX_CTX {MAX_CTX}\n")
        f.write(f"`define LBFP_FULL_VOCAB   {VOCAB + pad}\n")
        f.write(f"`define LBFP_FULL_NPROMPT {len(ids)}\n")
        f.write(f"`define LBFP_FULL_NGEN    {N_GEN}\n")

    print(f"[bake] done. files in {OUT}/{PREFIX}*", file=sys.stderr)


if __name__ == '__main__':
    main()
