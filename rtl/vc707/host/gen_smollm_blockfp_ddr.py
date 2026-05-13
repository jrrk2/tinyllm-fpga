#!/usr/bin/env python3
"""Bake the full SmolLM2-135M block-FP weight set into a single DDR3
image (.bin) that mirrors lbfp_ddr3.svh's byte-offset map.  The image
is uploaded to DDR3 by the host before pulsing the autoregress start
bit; weight_streamer_bfp_mt.sv then fetches per-matvec chunks via AXI.

Output: generated/lbfp_full_DDR3.bin  (~286 MB)

Each weight matrix region is wide-packed in BFP:
  mantissas: LANES (=16) × 16 b = 256 b per (chunk, col) entry,
             NL × CHUNKS_OUT × D_in entries per matrix.
  exponents: LANES (=16) × 8 b = 128 b per (chunk, tile) entry,
             NL × CHUNKS_OUT × NT_in entries per matrix.

Per-matrix region is 4 KB-aligned (matches LBFP_DDR3_ALIGN).
"""
import os, sys, math, struct
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------
# Config — must match host/gen_smollm_blockfp_full.py.
# --------------------------------------------------------------------------
MODEL  = os.environ.get('MODEL',  'HuggingFaceTB/SmolLM2-135M')
D      = int(os.environ.get('D',      576))
H_Q    = int(os.environ.get('H_Q',    9))
H_KV   = int(os.environ.get('H_KV',   3))
HD     = int(os.environ.get('HD',     64))
FFN    = int(os.environ.get('FFN',    1536))
NL     = int(os.environ.get('NL',     30))
OUT    = os.environ.get('OUT',    'generated')
TILE   = 16
LANES  = 16
ALIGN  = 4096
BYTES_M_PER_COL  = 32   # 16 mantissas × 2 B
BYTES_E_PER_TILE = 16   # 16 exps × 1 B

NT_D    = D // TILE
NT_FFN  = FFN // TILE
NT_KV   = (H_KV * HD) // TILE
CHUNKS_D    = D // LANES
CHUNKS_KV   = (H_KV * HD) // LANES
CHUNKS_FFN  = FFN // LANES

os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------
# Tile quantization — identical to gen_smollm_blockfp_full.py.
# --------------------------------------------------------------------------
def tile_quantize(v):
    flat = np.asarray(v, dtype=np.float64).reshape(-1, TILE)
    max_abs = np.maximum(np.abs(flat).max(axis=1), 1e-300)
    e = (np.floor(np.log2(max_abs)).astype(np.int32) + 1).clip(-127, 127)
    m_scale = np.power(2.0, e.astype(np.float64))[:, None]
    m = np.clip(np.round(flat / m_scale * 32768.0).astype(np.int64),
                -32768, 32767)
    return m.astype(np.int16), e.astype(np.int8)


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
# Pack a per-layer matrix (m, e) into the wide-packed DDR3 byte sequence
# expected by weight_streamer_bfp_mt.sv.  Mantissa block first (32 B per
# col entry), then exponent block (16 B per tile entry).  Streamer
# fetches contiguous bytes via AXI bursts.
#
# Mantissa entry @ (chunk, col):
#   bytes 0..31 = LE pack of 16 mantissas (lane 0 → bytes 0..1, lane 1 → 2..3, ...)
# Exponent entry @ (chunk, tile):
#   bytes 0..15 = LE pack of 16 exps (lane 0 → byte 0, lane 1 → byte 1, ...)
# --------------------------------------------------------------------------
def pack_layer_mantissas(m, e):
    """m: (D_out, NT_in, TILE) int16.  Return bytes for the LAYER mantissa
    region (CHUNKS_OUT × D_in × 32 B)."""
    D_out, NT_in, _ = m.shape
    CHUNKS_OUT = D_out // LANES
    D_in = NT_in * TILE
    buf = bytearray(CHUNKS_OUT * D_in * BYTES_M_PER_COL)
    for chunk in range(CHUNKS_OUT):
        for col in range(D_in):
            tile = col // TILE
            idx  = col % TILE
            base = (chunk * D_in + col) * BYTES_M_PER_COL
            for lane in range(LANES):
                row = chunk * LANES + lane
                mv = int(m[row, tile, idx]) & 0xFFFF
                struct.pack_into('<H', buf, base + lane * 2, mv)
    return bytes(buf)


def pack_layer_exponents(e):
    """e: (D_out, NT_in) int8.  Return bytes for the LAYER exponent region
    (CHUNKS_OUT × NT_in × 16 B)."""
    D_out, NT_in = e.shape
    CHUNKS_OUT = D_out // LANES
    buf = bytearray(CHUNKS_OUT * NT_in * BYTES_E_PER_TILE)
    for chunk in range(CHUNKS_OUT):
        for tile in range(NT_in):
            base = (chunk * NT_in + tile) * BYTES_E_PER_TILE
            for lane in range(LANES):
                row = chunk * LANES + lane
                buf[base + lane] = int(e[row, tile]) & 0xFF
    return bytes(buf)


def align4k(n):
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def main():
    print(f"[ddr] loading {MODEL} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

    print(f"[ddr] extracting + quantizing weights ...", file=sys.stderr)
    layers_q = []
    g1_list, g2_list = [], []
    with torch.no_grad():
        for li in range(NL):
            L = model.model.layers[li]
            row = {}
            for nm, sub in [('Wq', L.self_attn.q_proj), ('Wk', L.self_attn.k_proj),
                            ('Wv', L.self_attn.v_proj), ('Wo', L.self_attn.o_proj),
                            ('Wg', L.mlp.gate_proj),    ('Wu', L.mlp.up_proj),
                            ('Wd', L.mlp.down_proj)]:
                W = sub.weight.detach().cpu().numpy().astype(np.float64)
                row[nm] = quantize_W(W)
            row['g1'] = tile_quantize(L.input_layernorm.weight.detach().cpu().numpy().astype(np.float64))
            row['g2'] = tile_quantize(L.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float64))
            layers_q.append(row)
            g1_list.append(row['g1']); g2_list.append(row['g2'])
            print(f"  L{li:02d}", end='\r', file=sys.stderr)
        embed = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float64)
        norm_w = model.model.norm.weight.detach().cpu().numpy().astype(np.float64)
    VOCAB = embed.shape[0]
    print(f"\n[ddr] VOCAB={VOCAB}", file=sys.stderr)

    # Build per-matrix mantissa + exp regions (NL layers concatenated).
    regions = []
    def add_region(name, data):
        size_aligned = align4k(len(data))
        pad = size_aligned - len(data)
        regions.append((name, data + b'\x00' * pad))
        print(f"  {name:14s}  {len(data):>11,} B  → {size_aligned:>11,} B (aligned)", file=sys.stderr)

    # Per-matrix concat across layers (NL × per-layer-bytes).
    for tag, mat_key, D_out in [('WQ',  'Wq', D),
                                ('WK',  'Wk', H_KV*HD),
                                ('WV',  'Wv', H_KV*HD),
                                ('WO',  'Wo', D),
                                ('WG',  'Wg', FFN),
                                ('WU',  'Wu', FFN),
                                ('WDN', 'Wd', D)]:
        m_buf = b''
        e_buf = b''
        for li in range(NL):
            m_buf += pack_layer_mantissas(layers_q[li][mat_key][0],
                                           layers_q[li][mat_key][1])
            e_buf += pack_layer_exponents(layers_q[li][mat_key][1])
        add_region(f'{tag}_m', m_buf)
        add_region(f'{tag}_e', e_buf)

    # Gammas: narrow per-element mantissas, narrow per-tile exps.
    g1_m_buf = b''; g1_e_buf = b''
    g2_m_buf = b''; g2_e_buf = b''
    for li in range(NL):
        m, e = g1_list[li]
        g1_m_buf += m.tobytes()    # 16-bit signed
        g1_e_buf += e.tobytes()    # 8-bit signed
        m, e = g2_list[li]
        g2_m_buf += m.tobytes()
        g2_e_buf += e.tobytes()
    add_region('G1_m', g1_m_buf)
    add_region('G1_e', g1_e_buf)
    add_region('G2_m', g2_m_buf)
    add_region('G2_e', g2_e_buf)

    # EMBED wide-packed (for lm_head matvec).
    pad = (LANES - VOCAB % LANES) % LANES
    embed_padded = np.vstack([embed, np.zeros((pad, D))]) if pad else embed
    emb_m_data, emb_e_data = quantize_W(embed_padded)
    emb_m_buf = pack_layer_mantissas(emb_m_data, emb_e_data)
    emb_e_buf = pack_layer_exponents(emb_e_data)
    add_region('EMBED_m', emb_m_buf)
    add_region('EMBED_e', emb_e_buf)

    # EMBED row-wise narrow (for embed_lookup_bfp's per-row read).
    lu_m_buf = b''; lu_e_buf = b''
    for r in range(embed_padded.shape[0]):
        m, e = tile_quantize(embed_padded[r])
        lu_m_buf += m.tobytes()
        lu_e_buf += e.tobytes()
    add_region('EMBED_LU_m', lu_m_buf)
    add_region('EMBED_LU_e', lu_e_buf)

    # Final norm gamma.
    nw_m, nw_e = tile_quantize(norm_w)
    add_region('NORM_W_m', nw_m.tobytes())
    add_region('NORM_W_e', nw_e.tobytes())

    # ----------------------------------------------------------------
    # Assemble the image — each region appended in lbfp_ddr3.svh order,
    # 4 KB-aligned.  Emit base-offset map alongside.
    # ----------------------------------------------------------------
    img = bytearray()
    offsets = {}
    for name, data in regions:
        offsets[name] = len(img)
        img.extend(data)
    out_path = os.path.join(OUT, 'lbfp_full_DDR3.bin')
    with open(out_path, 'wb') as f:
        f.write(img)
    print(f"\n[ddr] wrote {out_path}  ({len(img):,} B = {len(img)/1024/1024:.1f} MB)",
          file=sys.stderr)
    with open(os.path.join(OUT, 'lbfp_full_DDR3.offsets.txt'), 'w') as f:
        f.write(f"# Region offsets in {out_path}\n")
        for name, off in offsets.items():
            f.write(f"  {name:14s}  0x{off:08X}  ({off:,} B)\n")
    print(f"[ddr] offsets in lbfp_full_DDR3.offsets.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
