// lbfp_ddr3.svh — DDR3 byte-offset map for the full SmolLM2-135M
// block-FP weight set, consumed by weight_streamer_bfp_mt.sv.
//
//   The host (over Ethernet) uploads a single .bin file produced by
//   host/gen_smollm_blockfp_ddr.py to DDR3 starting at offset 0.  Within
//   that image each weight matrix occupies a fixed byte range; this
//   header is `included by autoregress_bfp_top.sv to wire the right
//   matrix_base into the streamer per matvec call.
//
//   Layout (all offsets in bytes from DDR3 base, NL=30 layers
//   concatenated per matrix, wide-packed BFP — 32 B / mantissa entry,
//   16 B / exponent entry):
//
//     0x00_0000  WQ_m     NL * CHUNKS_D   * D   * 32 B   = 30 * 36 * 576 * 32 =  19,906,560 B
//     0x12_FE00  WQ_e     NL * CHUNKS_D   * NT_D * 16 B  = 30 * 36 * 36  * 16 =     622,080 B
//     0x13_9300  WK_m     NL * CHUNKS_KV  * D   * 32 B   = 30 * 12 * 576 * 32 =   6,635,520 B
//     ...
//
//   For simplicity we 4 KB-page-align each matrix.  Total weight image
//   ≈ 286 MB (well within VC707's 1 GB DDR3).

`ifndef LBFP_DDR3_SVH
`define LBFP_DDR3_SVH

// 32 B = 256 b mantissa entry; 16 B = 128 b exp entry; round up to 4 KB.
// 16 mantissas × 16 b = 256 b
`define LBFP_BYTES_M_PER_COL       32
// 16 exps × 8 b = 128 b
`define LBFP_BYTES_E_PER_TILE      16
`define LBFP_DDR3_ALIGN            4096

// Per-matrix region sizes (NL=30 layers concatenated, computed below).
// in_dim = number of input columns / tiles for the matvec.
// out_dim = D_out (used for CHUNKS_OUT = D_out / LANES).
`define LBFP_REGION_SIZE_M(D_out, D_in)  \
    (((`LBFP_FULL_NL * ((D_out) / 16) * (D_in) * `LBFP_BYTES_M_PER_COL) + `LBFP_DDR3_ALIGN - 1) \
     & ~(`LBFP_DDR3_ALIGN - 1))
`define LBFP_REGION_SIZE_E(D_out, D_in)  \
    (((`LBFP_FULL_NL * ((D_out) / 16) * ((D_in) / 16) * `LBFP_BYTES_E_PER_TILE) + `LBFP_DDR3_ALIGN - 1) \
     & ~(`LBFP_DDR3_ALIGN - 1))

// Base offsets — laid out sequentially.  Each matrix's region is
// {mantissa block, exp block}, both 4 KB-aligned.

`define LBFP_BASE_WQ_M        30'h00_0000
`define LBFP_BASE_WQ_E        (`LBFP_BASE_WQ_M + 30'(`LBFP_REGION_SIZE_M(`LBFP_FULL_D, `LBFP_FULL_D)))

`define LBFP_BASE_WK_M        (`LBFP_BASE_WQ_E + 30'(`LBFP_REGION_SIZE_E(`LBFP_FULL_D, `LBFP_FULL_D)))
`define LBFP_BASE_WK_E        (`LBFP_BASE_WK_M + 30'(`LBFP_REGION_SIZE_M(`LBFP_FULL_HKV * `LBFP_FULL_HD, `LBFP_FULL_D)))

`define LBFP_BASE_WV_M        (`LBFP_BASE_WK_E + 30'(`LBFP_REGION_SIZE_E(`LBFP_FULL_HKV * `LBFP_FULL_HD, `LBFP_FULL_D)))
`define LBFP_BASE_WV_E        (`LBFP_BASE_WV_M + 30'(`LBFP_REGION_SIZE_M(`LBFP_FULL_HKV * `LBFP_FULL_HD, `LBFP_FULL_D)))

`define LBFP_BASE_WO_M        (`LBFP_BASE_WV_E + 30'(`LBFP_REGION_SIZE_E(`LBFP_FULL_HKV * `LBFP_FULL_HD, `LBFP_FULL_D)))
`define LBFP_BASE_WO_E        (`LBFP_BASE_WO_M + 30'(`LBFP_REGION_SIZE_M(`LBFP_FULL_D, `LBFP_FULL_D)))

`define LBFP_BASE_WG_M        (`LBFP_BASE_WO_E + 30'(`LBFP_REGION_SIZE_E(`LBFP_FULL_D, `LBFP_FULL_D)))
`define LBFP_BASE_WG_E        (`LBFP_BASE_WG_M + 30'(`LBFP_REGION_SIZE_M(`LBFP_FULL_FFN, `LBFP_FULL_D)))

`define LBFP_BASE_WU_M        (`LBFP_BASE_WG_E + 30'(`LBFP_REGION_SIZE_E(`LBFP_FULL_FFN, `LBFP_FULL_D)))
`define LBFP_BASE_WU_E        (`LBFP_BASE_WU_M + 30'(`LBFP_REGION_SIZE_M(`LBFP_FULL_FFN, `LBFP_FULL_D)))

`define LBFP_BASE_WDN_M       (`LBFP_BASE_WU_E + 30'(`LBFP_REGION_SIZE_E(`LBFP_FULL_FFN, `LBFP_FULL_D)))
`define LBFP_BASE_WDN_E       (`LBFP_BASE_WDN_M + 30'(`LBFP_REGION_SIZE_M(`LBFP_FULL_D, `LBFP_FULL_FFN)))

// Gamma matrices: NL × D × 16 b mantissas + NL × NT_D × 8 b exps (narrow).
// Padding matches host/gen_smollm_blockfp_ddr.py uniform 4 KB alignment.
`define LBFP_BASE_G1_M        (`LBFP_BASE_WDN_E + 30'(`LBFP_REGION_SIZE_E(`LBFP_FULL_D, `LBFP_FULL_FFN)))
// NL × D × 2 B = 34,560 → 36,864 (4 KB-aligned)
`define LBFP_BASE_G1_E        (`LBFP_BASE_G1_M + 30'h0_9000)
// NL × NT_D × 1 B = 1,080 → 4,096
`define LBFP_BASE_G2_M        (`LBFP_BASE_G1_E + 30'h0_1000)
`define LBFP_BASE_G2_E        (`LBFP_BASE_G2_M + 30'h0_9000)

// EMBED table (49152 × 576): wide-packed for lm_head AND narrow row-wise
// for embed_lookup.  Both stored separately in DDR3.
`define LBFP_BASE_EMBED_M     (`LBFP_BASE_G2_E + 30'h0_1000)
`define LBFP_BASE_EMBED_E     (`LBFP_BASE_EMBED_M + 30'(`LBFP_FULL_VOCAB / 16 * `LBFP_FULL_D * `LBFP_BYTES_M_PER_COL))
`define LBFP_BASE_EMBED_LU_M  (`LBFP_BASE_EMBED_E + 30'(`LBFP_FULL_VOCAB / 16 * (`LBFP_FULL_D / 16) * `LBFP_BYTES_E_PER_TILE))
`define LBFP_BASE_EMBED_LU_E  (`LBFP_BASE_EMBED_LU_M + 30'(`LBFP_FULL_VOCAB * `LBFP_FULL_D * 2))

// norm_w (final RMS gamma), 576 × 2 B mantissas + 36 × 1 B exps.
// EMBED_LU_e padded to 64 B per row by the baker for AXI alignment, so
// total = VOCAB × 64 B (not VOCAB × NT_D).
`define LBFP_BASE_NORM_W_M    (`LBFP_BASE_EMBED_LU_E + 30'(`LBFP_FULL_VOCAB * 64))
`define LBFP_BASE_NORM_W_E    (`LBFP_BASE_NORM_W_M + 30'h1_0000)

// Total DDR3 image size in 128-bit entries (one entry per mock_axi_slave
// mem[] slot).  Conservatively rounded up — extra entries init to 0.
// 64 Mi × 16 B = 1024 MB.  Bumped from 24 Mi because smollm360 image is
// ~860 MB (53 Mi entries) — overshot the previous budget designed for
// smollm135's ~320 MB image.
`define LBFP_DDR3_ENTRIES   (64 * 1024 * 1024)

// matvec_id encoding consumed by weight_streamer_bfp_mt.sv:
//   0=Q, 1=K, 2=V, 3=O, 4=G, 5=U, 6=DN
`define LBFP_MV_ID_Q    3'd0
`define LBFP_MV_ID_K    3'd1
`define LBFP_MV_ID_V    3'd2
`define LBFP_MV_ID_O    3'd3
`define LBFP_MV_ID_G    3'd4
`define LBFP_MV_ID_U    3'd5
`define LBFP_MV_ID_DN   3'd6

`endif  // LBFP_DDR3_SVH
