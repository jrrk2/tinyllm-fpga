// smollm_layer_bfp.sv — block-FP version of a single SmolLM2 transformer
// layer.  Wires the 6 verified block-FP leaf ops (matvec_bfp_engine,
// rmsnorm_bfp, rope_bfp, softmax_bfp, swiglu_bfp, residual_bfp) into a
// complete pre-norm transformer block.  All hidden state, weights, and
// activations are in the block-FP format defined in bfp_format.svh:
//   value = mantissa / 2^15 * 2^exp     (TILE=16 mantissas per shared exp).
//
// FSM:
//   S_RMSNORM1 → S_QMV → S_KMV → S_VMV → S_ROPE_Q → S_ROPE_K → S_KV_WRITE
//             → for head: S_QK → S_SM → S_AV
//             → S_OMV → S_RES1
//             → S_RMSNORM2 → S_GMV → S_UMV → S_SWG → S_DMV → S_RES2
//             → S_DONE
//
// One shared matvec engine (LANES=16) handles all 7 weight matrix
// multiplies plus the per-head QK^T scores and AV; chunked by output
// dimension so the 16 output lanes fill (out_dim / 16) times per matvec.
// Weights and KV cache init from $readmemh — DDR3 streamer can replace.
//
// All inter-engine wiring is registered (single non-blocking driver per
// signal) to avoid multi-driver synthesis errors and to keep timing
// predictable for VC707 (50 MHz core clock).

`include "bfp_format.svh"

`default_nettype none

module smollm_layer_bfp #(
  parameter int    D       = 64,
  parameter int    H_Q     = 1,
  parameter int    H_KV    = 1,
  parameter int    HD      = 64,
  parameter int    FFN     = 128,
  parameter int    MAX_CTX = 4,
  // Number of layers stored in the weight / KV-cache ROMs (time-mux).
  // NL=1 → single-layer selftest (weights addressed without layer offset);
  // NL>1 → multilayer mode where layer_idx selects the active layer's
  // ROM bank.  KV cache likewise scales by NL.
  parameter int    NL      = 1,
  parameter        PREFIX  = "lbfp_",
  // When STREAM_WEIGHTS=1, the 7 weight matrices are served by an
  // internal weight_streamer_bfp_mt instance that fetches chunks from
  // DDR3 via the AXI master ports below.  $readmemh ROMs are dropped.
  // Gammas + KV cache remain on-chip (small).  Default 0 = selftest.
  parameter bit    STREAM_WEIGHTS = 1'b0,
  parameter int    AXI_ADDR_WIDTH = 30,
  parameter int    AXI_ID_WIDTH   = 5
)(
  input  wire                                clk,
  input  wire                                rst,
  input  wire                                start,
  input  wire [10:0]                         pos,
  input  wire [6:0]                          kv_pos,       // up to MAX_CTX-1
  input  wire [4:0]                          layer_idx,    // 0..NL-1
  input  wire signed [D*BFP_MANT_W-1:0]      hidden_in_m,
  input  wire signed [(D/BFP_TILE)*BFP_EXP_W-1:0] hidden_in_e,
  output logic signed [D*BFP_MANT_W-1:0]     hidden_out_m,
  output logic signed [(D/BFP_TILE)*BFP_EXP_W-1:0] hidden_out_e,
  output logic                               done,
  // -------------------------------------------------------------------
  // DDR3 streamer interface (used iff STREAM_WEIGHTS=1).
  // Caller (autoregress_bfp_top / multilayer_tm) computes the layer-
  // qualified byte base for each of the 7 matrices and presents them
  // here; the layer applies chunk_idx * in_dim_bytes internally via the
  // streamer.  AXI master ports route to MIG ui_clk domain.
  // -------------------------------------------------------------------
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WQ_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WQ_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WK_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WK_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WV_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WV_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WO_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WO_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WG_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WG_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WU_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WU_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WDN_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WDN_e,
  input  wire                                clk_axi,
  input  wire                                rst_axi,
  output wire                                m_axi_arvalid,
  input  wire                                m_axi_arready,
  output wire [AXI_ID_WIDTH-1:0]             m_axi_arid,
  output wire [AXI_ADDR_WIDTH-1:0]           m_axi_araddr,
  output wire [7:0]                          m_axi_arlen,
  output wire [2:0]                          m_axi_arsize,
  output wire [1:0]                          m_axi_arburst,
  output wire                                m_axi_arlock,
  output wire [3:0]                          m_axi_arcache,
  output wire [2:0]                          m_axi_arprot,
  output wire [3:0]                          m_axi_arqos,
  input  wire                                m_axi_rvalid,
  output wire                                m_axi_rready,
  input  wire [AXI_ID_WIDTH-1:0]             m_axi_rid,
  input  wire [511:0]                        m_axi_rdata,
  input  wire [1:0]                          m_axi_rresp,
  input  wire                                m_axi_rlast,
  // Host write port for the weight / gamma / KV-init BRAMs.  Lets the host
  // patch BFP weights post-bitstream without re-synth (matches the existing
  // scale-brom flow for the int8 path).
  //   wr_kind: 0=WQ_m 1=WQ_e 2=WK_m 3=WK_e 4=WV_m 5=WV_e 6=WO_m 7=WO_e
  //            8=WG_m 9=WG_e 10=WU_m 11=WU_e 12=WDN_m 13=WDN_e
  //            14=G1_m 15=G1_e 16=G2_m 17=G2_e
  //            18=K_INIT_m 19=K_INIT_e 20=V_INIT_m 21=V_INIT_e
  //   wr_addr: per-rom entry index
  //   wr_data: 16-bit mantissa (lower 16) or 8-bit exponent (lower 8)
  //   wr_en : 1-cycle pulse
  input  wire [4:0]                          wr_kind,
  input  wire [17:0]                         wr_addr,
  input  wire [15:0]                         wr_data,
  input  wire                                wr_en,
  // Debug taps (synthesis-friendly — only used by sim testbench)
  output logic [6:0]                         dbg_state,
  output logic [11:0]                        dbg_cnt,
  output logic [6:0]                         dbg_chunk
);

  // ---------------------------------------------------------------------------
  // Derived sizes
  // ---------------------------------------------------------------------------
  localparam int NT_D    = (D + BFP_TILE - 1) / BFP_TILE;
  localparam int NT_KV   = (H_KV*HD + BFP_TILE - 1) / BFP_TILE;
  localparam int NT_FFN  = (FFN + BFP_TILE - 1) / BFP_TILE;
  localparam int LANES   = 16;
  localparam int CHUNKS_D   = D    / LANES;
  localparam int CHUNKS_KV  = (H_KV*HD) / LANES;
  localparam int CHUNKS_FFN = FFN  / LANES;
  localparam int CHUNKS_HD  = HD   / LANES;

  // Counter widths (max of D, FFN, H_KV*HD)
  localparam int CW_D    = $clog2(D    + 1);
  localparam int CW_FFN  = $clog2(FFN  + 1);
  localparam int CW_KV   = $clog2(H_KV*HD + 1);
  localparam int CW_HD   = $clog2(HD   + 1);
  localparam int CW_CTX  = $clog2(MAX_CTX + 1);

  // Wide-packed ROM widths: each entry holds LANES mantissas (or exponents)
  // concatenated, so one BRAM read per cycle returns all 16 lanes worth
  // of data — single-port access matches what Vivado can infer / cascade
  // RAMB18/36 primitives for.  The previous per-lane unpacked array was
  // a 16-port read (one address per lane), which Vivado refuses to BRAM-
  // infer (max 2 read ports) and is too big for distributed RAM (>128 Kb).
  localparam int LANE_M_W = LANES * BFP_MANT_W;     // 256
  localparam int LANE_E_W = LANES * BFP_EXP_W;      // 128

  // Number of wide entries per layer (one entry per col / per tile, not
  // per lane).  Total ROM size = NL * per-layer.
  localparam int WQ_M_PL   = CHUNKS_D   * D;
  localparam int WK_M_PL   = CHUNKS_KV  * D;
  localparam int WV_M_PL   = CHUNKS_KV  * D;
  localparam int WO_M_PL   = CHUNKS_D   * D;
  localparam int WG_M_PL   = CHUNKS_FFN * D;
  localparam int WU_M_PL   = CHUNKS_FFN * D;
  localparam int WDN_M_PL  = CHUNKS_D   * FFN;
  localparam int WQ_E_PL   = CHUNKS_D   * NT_D;
  localparam int WK_E_PL   = CHUNKS_KV  * NT_D;
  localparam int WV_E_PL   = CHUNKS_KV  * NT_D;
  localparam int WO_E_PL   = CHUNKS_D   * NT_D;
  localparam int WG_E_PL   = CHUNKS_FFN * NT_D;
  localparam int WU_E_PL   = CHUNKS_FFN * NT_D;
  localparam int WDN_E_PL  = CHUNKS_D   * NT_FFN;

  localparam int WQ_M_ENT  = NL * WQ_M_PL;
  localparam int WK_M_ENT  = NL * WK_M_PL;
  localparam int WV_M_ENT  = NL * WV_M_PL;
  localparam int WO_M_ENT  = NL * WO_M_PL;
  localparam int WG_M_ENT  = NL * WG_M_PL;
  localparam int WU_M_ENT  = NL * WU_M_PL;
  localparam int WDN_M_ENT = NL * WDN_M_PL;
  localparam int WQ_E_ENT  = NL * WQ_E_PL;
  localparam int WK_E_ENT  = NL * WK_E_PL;
  localparam int WV_E_ENT  = NL * WV_E_PL;
  localparam int WO_E_ENT  = NL * WO_E_PL;
  localparam int WG_E_ENT  = NL * WG_E_PL;
  localparam int WU_E_ENT  = NL * WU_E_PL;
  localparam int WDN_E_ENT = NL * WDN_E_PL;

  // ---------------------------------------------------------------------------
  // Weight ROMs.  `ram_style = "block"` forces Vivado to use RAMB18/RAMB36
  // primitives (cascaded for the 256-bit-wide mantissa output / 128-bit
  // exponent output) rather than dissolving to FFs.
  // ---------------------------------------------------------------------------
  (* ram_style = "block" *) logic [LANE_M_W-1:0] rom_WQ_m  [0:WQ_M_ENT-1];
  (* ram_style = "block" *) logic [LANE_M_W-1:0] rom_WK_m  [0:WK_M_ENT-1];
  (* ram_style = "block" *) logic [LANE_M_W-1:0] rom_WV_m  [0:WV_M_ENT-1];
  (* ram_style = "block" *) logic [LANE_M_W-1:0] rom_WO_m  [0:WO_M_ENT-1];
  (* ram_style = "block" *) logic [LANE_M_W-1:0] rom_WG_m  [0:WG_M_ENT-1];
  (* ram_style = "block" *) logic [LANE_M_W-1:0] rom_WU_m  [0:WU_M_ENT-1];
  (* ram_style = "block" *) logic [LANE_M_W-1:0] rom_WDN_m [0:WDN_M_ENT-1];

  (* ram_style = "block" *) logic [LANE_E_W-1:0] rom_WQ_e  [0:WQ_E_ENT-1];
  (* ram_style = "block" *) logic [LANE_E_W-1:0] rom_WK_e  [0:WK_E_ENT-1];
  (* ram_style = "block" *) logic [LANE_E_W-1:0] rom_WV_e  [0:WV_E_ENT-1];
  (* ram_style = "block" *) logic [LANE_E_W-1:0] rom_WO_e  [0:WO_E_ENT-1];
  (* ram_style = "block" *) logic [LANE_E_W-1:0] rom_WG_e  [0:WG_E_ENT-1];
  (* ram_style = "block" *) logic [LANE_E_W-1:0] rom_WU_e  [0:WU_E_ENT-1];
  (* ram_style = "block" *) logic [LANE_E_W-1:0] rom_WDN_e [0:WDN_E_ENT-1];

  // Gamma / KV cache: scaled by NL for multilayer time-mux.  Indexed
  // by {layer_idx, position, lane}.
  (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] rom_G1_m [0:NL*D-1];
  (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] rom_G2_m [0:NL*D-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] rom_G1_e [0:NL*NT_D-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] rom_G2_e [0:NL*NT_D-1];

  (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] kv_k_m [0:NL*MAX_CTX*H_KV*HD-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] kv_k_e [0:NL*MAX_CTX*NT_KV-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] kv_v_e [0:NL*MAX_CTX*NT_KV-1];

  // kv_v_m is wide-packed (LANES mantissas per entry) so AV_DRIVE's
  // 16-lane parallel read becomes a single BRAM read.  Original flat
  // layout had 16 simultaneous reads → too many ports for BRAM inference,
  // and at 5.9 Mbit total Vivado refused to dissolve to FFs (one Synth
  // 8-3391 error per `make`).  Wide-packing keeps total bits the same
  // but presents a 1-write + 1-read access pattern.
  localparam int KVV_CHUNKS_PER_KVPOS = (H_KV * HD) / LANES;          // 12 for full SmolLM2
  (* ram_style = "block" *) logic [LANES*BFP_MANT_W-1:0] kv_v_m_chk [0:NL*MAX_CTX*KVV_CHUNKS_PER_KVPOS-1];

  // When STREAM_WEIGHTS=1, the 14 weight $readmemh's are dropped (weights
  // come from DDR3 via the streamer below); gammas + KV cache continue
  // to load from hex.  The W?_? rom_* arrays are still declared above
  // for elaboration symmetry but go unused (synthesis prunes them).
`ifdef MICROGPT_WEIGHT_DIR
  initial begin
    if (!STREAM_WEIGHTS) begin
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WQ_m.hex"},  rom_WQ_m);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WK_m.hex"},  rom_WK_m);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WV_m.hex"},  rom_WV_m);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WO_m.hex"},  rom_WO_m);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WG_m.hex"},  rom_WG_m);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WU_m.hex"},  rom_WU_m);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WDN_m.hex"}, rom_WDN_m);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WQ_e.hex"},  rom_WQ_e);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WK_e.hex"},  rom_WK_e);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WV_e.hex"},  rom_WV_e);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WO_e.hex"},  rom_WO_e);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WG_e.hex"},  rom_WG_e);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WU_e.hex"},  rom_WU_e);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "WDN_e.hex"}, rom_WDN_e);
    end
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "G1_m.hex"},  rom_G1_m);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "G2_m.hex"},  rom_G2_m);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "G1_e.hex"},  rom_G1_e);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "G2_e.hex"},  rom_G2_e);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "K_INIT_m.hex"}, kv_k_m);
    // V_INIT_m no longer loaded — kv_v_m_chk is wide-packed; baker
    // value is all-zero anyway and Vivado zero-inits BRAM on bitstream.
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "K_INIT_e.hex"}, kv_k_e);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "V_INIT_e.hex"}, kv_v_e);
  end
`else
  initial begin
    if (!STREAM_WEIGHTS) begin
      $readmemh({PREFIX, "WQ_m.hex"},  rom_WQ_m);
      $readmemh({PREFIX, "WK_m.hex"},  rom_WK_m);
      $readmemh({PREFIX, "WV_m.hex"},  rom_WV_m);
      $readmemh({PREFIX, "WO_m.hex"},  rom_WO_m);
      $readmemh({PREFIX, "WG_m.hex"},  rom_WG_m);
      $readmemh({PREFIX, "WU_m.hex"},  rom_WU_m);
      $readmemh({PREFIX, "WDN_m.hex"}, rom_WDN_m);
      $readmemh({PREFIX, "WQ_e.hex"},  rom_WQ_e);
      $readmemh({PREFIX, "WK_e.hex"},  rom_WK_e);
      $readmemh({PREFIX, "WV_e.hex"},  rom_WV_e);
      $readmemh({PREFIX, "WO_e.hex"},  rom_WO_e);
      $readmemh({PREFIX, "WG_e.hex"},  rom_WG_e);
      $readmemh({PREFIX, "WU_e.hex"},  rom_WU_e);
      $readmemh({PREFIX, "WDN_e.hex"}, rom_WDN_e);
    end
    $readmemh({PREFIX, "G1_m.hex"},  rom_G1_m);
    $readmemh({PREFIX, "G2_m.hex"},  rom_G2_m);
    $readmemh({PREFIX, "G1_e.hex"},  rom_G1_e);
    $readmemh({PREFIX, "G2_e.hex"},  rom_G2_e);
    $readmemh({PREFIX, "K_INIT_m.hex"}, kv_k_m);
    // V_INIT_m no longer loaded (see header note on kv_v_m_chk).
    $readmemh({PREFIX, "K_INIT_e.hex"}, kv_k_e);
    $readmemh({PREFIX, "V_INIT_e.hex"}, kv_v_e);
  end
`endif

  // ---------------------------------------------------------------------------
  // Host write port (wr_kind / wr_addr / wr_data / wr_en): currently NOT
  // wired internally.  The wide-packed ROM layout that Vivado needs for
  // BRAM inference (256-bit-wide mantissa entries cascaded across multiple
  // RAMB36 primitives) is incompatible with a single-mantissa byte-write
  // port.  To restore writability the host would have to use a 2-cycle
  // read-modify-write helper FSM, or the build flow would emit per-lane
  // BRAM wrappers (cf. gen_brom_sv.py for the int8 path).
  //
  // For now hex files are the source of truth: regenerate via
  //   make USE_BFP=1 bfp-sim     # rebake lbfp_*.hex
  // and resynthesize.  The wr_* wrapper port is retained for ABI compat
  // with the int8 scale-write regmap path, but writes are dropped here.
  // ---------------------------------------------------------------------------
  /* verilator lint_off UNUSEDSIGNAL */
  wire _unused_wr = &{1'b0, wr_kind, wr_addr, wr_data, wr_en, 1'b0};
  /* verilator lint_on UNUSEDSIGNAL */

  // ---------------------------------------------------------------------------
  // Working buffers
  // ---------------------------------------------------------------------------
  logic signed [BFP_MANT_W-1:0] hin_m   [0:D-1];
  logic signed [BFP_EXP_W -1:0] hin_e   [0:NT_D-1];

  logic signed [BFP_MANT_W-1:0] n1_m    [0:D-1];
  logic signed [BFP_EXP_W -1:0] n1_e    [0:NT_D-1];

  logic signed [BFP_MANT_W-1:0] q_m     [0:D-1];
  logic signed [BFP_EXP_W -1:0] q_e     [0:NT_D-1];

  logic signed [BFP_MANT_W-1:0] k_m     [0:H_KV*HD-1];
  logic signed [BFP_EXP_W -1:0] k_e     [0:NT_KV-1];
  logic signed [BFP_MANT_W-1:0] v_m     [0:H_KV*HD-1];
  logic signed [BFP_EXP_W -1:0] v_e     [0:NT_KV-1];

  logic signed [BFP_MANT_W-1:0] q_rot_m [0:D-1];
  logic signed [BFP_EXP_W -1:0] q_rot_e [0:D-1];
  logic signed [BFP_MANT_W-1:0] k_rot_m [0:H_KV*HD-1];
  logic signed [BFP_EXP_W -1:0] k_rot_e [0:H_KV*HD-1];

  logic signed [BFP_MANT_W-1:0] scores_m  [0:MAX_CTX-1];
  logic signed [BFP_EXP_W -1:0] qk_score_e[0:MAX_CTX-1];
  logic signed [BFP_EXP_W -1:0] scores_e_shared;
  logic signed [BFP_MANT_W-1:0] probs_m   [0:MAX_CTX-1];
  logic signed [BFP_EXP_W -1:0] probs_e_shared;

  logic signed [BFP_MANT_W-1:0] attn_m  [0:D-1];
  logic signed [BFP_EXP_W -1:0] attn_e  [0:NT_D-1];

  logic signed [BFP_MANT_W-1:0] o_m     [0:D-1];
  logic signed [BFP_EXP_W -1:0] o_e     [0:NT_D-1];

  logic signed [BFP_MANT_W-1:0] h1_m    [0:D-1];
  logic signed [BFP_EXP_W -1:0] h1_e    [0:NT_D-1];

  logic signed [BFP_MANT_W-1:0] n2_m    [0:D-1];
  logic signed [BFP_EXP_W -1:0] n2_e    [0:NT_D-1];

  logic signed [BFP_MANT_W-1:0] g_m     [0:FFN-1];
  logic signed [BFP_EXP_W -1:0] g_e     [0:NT_FFN-1];
  logic signed [BFP_MANT_W-1:0] u_m     [0:FFN-1];
  logic signed [BFP_EXP_W -1:0] u_e     [0:NT_FFN-1];
  logic signed [BFP_MANT_W-1:0] mlp_m   [0:FFN-1];
  logic signed [BFP_EXP_W -1:0] mlp_e   [0:NT_FFN-1];

  logic signed [BFP_MANT_W-1:0] d_m     [0:D-1];
  logic signed [BFP_EXP_W -1:0] d_e     [0:NT_D-1];

  logic signed [BFP_MANT_W-1:0] hout_m  [0:D-1];
  logic signed [BFP_EXP_W -1:0] hout_e  [0:NT_D-1];

  // ---------------------------------------------------------------------------
  // FSM
  // ---------------------------------------------------------------------------
  typedef enum logic [6:0] {
    S_IDLE,
    S_LATCH_IN,
    S_NORM1, S_NORM1_WAIT,
    S_QMV_PRIME, S_QMV_DRIVE, S_QMV_DRAIN, S_QMV_REQ, S_QMV_NEXT,
    S_KMV_PRIME, S_KMV_DRIVE, S_KMV_DRAIN, S_KMV_REQ, S_KMV_NEXT,
    S_VMV_PRIME, S_VMV_DRIVE, S_VMV_DRAIN, S_VMV_REQ, S_VMV_NEXT,
    S_ROPEQ, S_ROPEQ_WAIT, S_ROPEQ_RQ_A, S_ROPEQ_RQ_B,
    S_ROPEK, S_ROPEK_WAIT, S_ROPEK_RQ_A, S_ROPEK_RQ_B,
    S_KVWR_M, S_KVWR_E,
    S_QK_PRIME, S_QK_DRIVE, S_QK_DRAIN,
    S_SM_DRIVE, S_SM_WAIT,
    S_AV_PRIME, S_AV_EMAX_SCAN, S_AV_DRIVE, S_AV_DRAIN, S_AV_REQ, S_AV_NEXT,
    S_OMV_PRIME, S_OMV_DRIVE, S_OMV_DRAIN, S_OMV_REQ, S_OMV_NEXT,
    S_RES1, S_RES1_WAIT,
    S_NORM2, S_NORM2_WAIT,
    S_GMV_PRIME, S_GMV_DRIVE, S_GMV_DRAIN, S_GMV_REQ, S_GMV_NEXT,
    S_UMV_PRIME, S_UMV_DRIVE, S_UMV_DRAIN, S_UMV_REQ, S_UMV_NEXT,
    S_SWG, S_SWG_WAIT,
    S_DMV_PRIME, S_DMV_DRIVE, S_DMV_DRAIN, S_DMV_REQ, S_DMV_NEXT,
    S_RES2, S_RES2_WAIT,
    S_DONE
  } state_t;
  state_t state;

  // Generic counters
  logic [11:0]      cnt;
  logic [6:0]       chunk;
  logic [4:0]       head_idx;
  logic [4:0]       kv_t;
  logic signed [BFP_EXP_W-1:0] score_emax;

  // ---------------------------------------------------------------------------
  // Per-matvec requant pipeline (2-stage).  S_*_DRAIN now latches the engine
  // outputs and computes the max of the two 8-lane halves of mv_out_e; the
  // next-cycle S_*_REQ takes that pair, finalises emax, applies
  // requant_mant to all 16 lanes, and writes the chunk's mantissa+exp.
  // Splits the original 58-LUT-level chain (15-deep emax cascade + barrel
  // shift) into two ~7-LUT-level stages so each clock fits in the 20 ns
  // budget at 50 MHz core_clk.
  // ---------------------------------------------------------------------------
  logic signed [LANES*BFP_MANT_W-1:0] mv_out_m_drain_r;
  logic signed [LANES*BFP_EXP_W -1:0] mv_out_e_drain_r;
  logic signed [BFP_EXP_W-1:0]        emax_h0_r, emax_h1_r;

  // ---------------------------------------------------------------------------
  // Structural pipeline barrier — Vivado's retimer was pulling the
  // FSM-gated emax pipeline backward into the matvec engine's MAC,
  // producing a 27 ns cnt → q_m chain.  This module-scope always_ff
  // latches the engine output ONCE more (1 extra cycle) with no
  // FSM-state condition, so the synthesiser cannot collapse it as
  // "redundant".  The S_*_DRAIN stages then consume the latched copies
  // — chain becomes cnt → engine → mv_out_{m,e}_lat FF → emax FFs →
  // requant FFs → q_m, with hard register barriers at each step.
  // ---------------------------------------------------------------------------
  logic                                       mv_out_valid_lat;
  logic signed [LANES*BFP_MANT_W-1:0]         mv_out_m_lat;
  logic signed [LANES*BFP_EXP_W -1:0]         mv_out_e_lat;
  always_ff @(posedge clk) begin
    if (rst) begin
      mv_out_valid_lat <= 1'b0;
      mv_out_m_lat     <= '0;
      mv_out_e_lat     <= '0;
    end else begin
      mv_out_valid_lat <= mv_out_valid;
      if (mv_out_valid) begin
        mv_out_m_lat <= mv_out_m;
        mv_out_e_lat <= mv_out_e;
      end
    end
  end
  // Comb-derived signed view of the matvec engine's lane-0 exp output.
  // The part-select mv_out_e[0+:8] is unsigned by SV rules, so explicit
  // re-typing is needed for signed comparisons against score_emax.
  wire signed [BFP_EXP_W-1:0] mv_lane0_e = $signed(mv_out_e[0 +: BFP_EXP_W]);
  wire signed [BFP_MANT_W-1:0] mv_lane0_m = $signed(mv_out_m[0 +: BFP_MANT_W]);

  // Sub-engine reset (one-shot pulses between matvecs / etc.)
  logic eng_rst;

  // ---------------------------------------------------------------------------
  // Engine instances (all driven via registered signals below)
  // ---------------------------------------------------------------------------
  logic                                      mv_start;
  logic signed [BFP_MANT_W-1:0]              mv_x_m;
  logic signed [BFP_EXP_W -1:0]              mv_x_e;
  logic                                      mv_valid, mv_last;
  logic signed [LANES*BFP_MANT_W-1:0]        mv_w_m;
  logic signed [LANES*BFP_EXP_W -1:0]        mv_w_e;
  wire  signed [LANES*BFP_MANT_W-1:0]        mv_out_m;
  wire  signed [LANES*BFP_EXP_W -1:0]        mv_out_e;
  wire                                       mv_out_valid;

  // mv_w_m_eff / mv_w_e_eff: muxed weight bus presented to the matvec
  // engine.  In selftest mode (STREAM_WEIGHTS=0) these forward the
  // registered mv_w_m / mv_w_e values read from on-chip ROMs.  In
  // streaming mode the streamer's bank_m / bank_e outputs replace
  // them — but ONLY for the 7 W?_* matvecs (Q/K/V/O/G/U/DN).  QK and
  // AV matvecs read kv_k_m / kv_v_m / k_m / v_m (BRAM-backed) and
  // must keep mv_w_m as their weight source.  `is_stream_matvec` is
  // set in each W?_PRIME and cleared in QK_PRIME / AV_PRIME, holding
  // through DRIVE / DRAIN / REQ.
  logic is_stream_matvec;
  wire signed [LANES*BFP_MANT_W-1:0]         mv_w_m_eff;
  wire signed [LANES*BFP_EXP_W -1:0]         mv_w_e_eff;
  wire        [255:0]                        ws_weight_m_out;
  wire        [127:0]                        ws_weight_e_out;
  assign mv_w_m_eff = (STREAM_WEIGHTS && is_stream_matvec) ? $signed(ws_weight_m_out) : mv_w_m;
  assign mv_w_e_eff = (STREAM_WEIGHTS && is_stream_matvec) ? $signed(ws_weight_e_out) : mv_w_e;

  // is_stream_matvec: set when entering a W?_* matvec via its PRIME,
  // cleared when entering QK_PRIME or AV_PRIME.  Held through DRIVE /
  // DRAIN / REQ / NEXT so the engine consumes streamer data only when
  // appropriate.  Always_ff is intentionally separate from the main
  // FSM so the flag isn't tied to the existing case-table indentation.

  always_ff @(posedge clk) begin
    if (rst) is_stream_matvec <= 1'b0;
    else begin
      case (state)
        S_QMV_PRIME, S_KMV_PRIME, S_VMV_PRIME, S_OMV_PRIME,
        S_GMV_PRIME, S_UMV_PRIME, S_DMV_PRIME: is_stream_matvec <= 1'b1;
        S_QK_PRIME, S_AV_PRIME:                is_stream_matvec <= 1'b0;
        default: ;  // hold through DRIVE / DRAIN / REQ / NEXT
      endcase
    end
  end

  matvec_bfp_engine #(.LANES(LANES)) i_mv (
    .clk(clk), .rst(rst | eng_rst),
    .start_matvec(mv_start),
    .in_x_mant(mv_x_m), .in_x_exp(mv_x_e),
    .in_valid(mv_valid), .last_elem(mv_last),
    .w_mant(mv_w_m_eff), .w_exp(mv_w_e_eff),
    .out_mant(mv_out_m), .out_exp(mv_out_e), .out_valid(mv_out_valid)
  );

  logic                                rn_start, rn_valid, rn_last;
  logic signed [BFP_MANT_W-1:0]        rn_x_m, rn_g_m;
  logic signed [BFP_EXP_W -1:0]        rn_x_e, rn_g_e;
  wire  signed [BFP_MANT_W-1:0]        rn_y_m;
  wire  signed [BFP_EXP_W -1:0]        rn_y_e;
  wire                                 rn_y_valid, rn_done;

  rmsnorm_bfp #(.D(D)) i_rn (
    .clk(clk), .rst(rst),
    .start(rn_start),
    .in_x_mant(rn_x_m), .in_x_exp(rn_x_e),
    .in_g_mant(rn_g_m), .in_g_exp(rn_g_e),
    .in_valid(rn_valid), .last_elem(rn_last),
    .out_y_mant(rn_y_m), .out_y_exp(rn_y_e),
    .out_valid(rn_y_valid), .done(rn_done)
  );

  logic                                rp_start, rp_valid;
  logic signed [BFP_MANT_W-1:0]        rp_x_m;
  logic signed [BFP_EXP_W -1:0]        rp_x_e;
  wire  signed [BFP_MANT_W-1:0]        rp_y_m;
  wire  signed [BFP_EXP_W -1:0]        rp_y_e;
  wire                                 rp_y_valid, rp_done;

  rope_bfp #(.HD(HD)) i_rp (
    .clk(clk), .rst(rst),
    .start(rp_start),
    .in_x_mant(rp_x_m), .in_x_exp(rp_x_e), .in_valid(rp_valid),
    .pos(pos),
    .out_y_mant(rp_y_m), .out_y_exp(rp_y_e),
    .out_valid(rp_y_valid), .done(rp_done)
  );

  // SwiGLU engine — same comb-drive pattern as residual (load+emit
  // alternation with in_ready back-pressure at tile boundaries).
  logic                                sg_start;
  logic                                sg_drive;
  logic signed [BFP_MANT_W-1:0]        sg_g_m_c, sg_u_m_c;
  logic signed [BFP_EXP_W -1:0]        sg_g_e_c, sg_u_e_c;
  logic                                sg_valid_c, sg_last_c;
  wire                                 sg_in_ready;
  wire  signed [BFP_MANT_W-1:0]        sg_y_m;
  wire  signed [BFP_EXP_W -1:0]        sg_y_e;
  wire                                 sg_y_valid, sg_done;

  always_comb sg_drive = (state == S_SWG);
  always_comb begin
    sg_g_m_c   = '0; sg_g_e_c = '0;
    sg_u_m_c   = '0; sg_u_e_c = '0;
    sg_valid_c = 1'b0;
    sg_last_c  = 1'b0;
    if (sg_drive) begin
      sg_g_m_c   = g_m[cnt[CW_FFN-1:0]];
      sg_g_e_c   = g_e[cnt[CW_FFN-1:0] / BFP_TILE];
      sg_u_m_c   = u_m[cnt[CW_FFN-1:0]];
      sg_u_e_c   = u_e[cnt[CW_FFN-1:0] / BFP_TILE];
      // NB: gate in_valid by in_ready (mirrors the standalone testbench).
      // This makes the 1-cycle valid_r pipeline inside swiglu_bfp line up
      // with the comb up_mant_r / silu_at_lut registration — otherwise the
      // first prod_buf write after S_EMIT→S_LOAD would store the stale
      // (last-S_EMIT-cycle) input.
      sg_valid_c = sg_in_ready && (cnt < FFN);
      sg_last_c  = sg_in_ready && (cnt == FFN-1);
    end
  end

  swiglu_bfp #(.D(FFN)) i_sg (
    .clk(clk), .rst(rst),
    .start(sg_start),
    .in_gate_mant(sg_g_m_c), .in_gate_exp(sg_g_e_c),
    .in_up_mant(sg_u_m_c),   .in_up_exp(sg_u_e_c),
    .in_valid(sg_valid_c), .last_elem(sg_last_c),
    .in_ready(sg_in_ready),
    .out_y_mant(sg_y_m), .out_y_exp(sg_y_e),
    .out_valid(sg_y_valid), .done(sg_done)
  );

  logic                                sm_start, sm_valid;
  logic [CW_CTX-1:0]                   sm_n;
  logic signed [BFP_MANT_W-1:0]        sm_x_m;
  logic signed [BFP_EXP_W -1:0]        sm_x_e;
  wire  signed [BFP_MANT_W-1:0]        sm_y_m;
  wire  signed [BFP_EXP_W -1:0]        sm_y_e;
  wire                                 sm_y_valid, sm_done;

  softmax_bfp #(.N_MAX(MAX_CTX)) i_sm (
    .clk(clk), .rst(rst),
    .start(sm_start),
    .n_elems(sm_n),
    .in_x_mant(sm_x_m), .in_x_exp(sm_x_e), .in_valid(sm_valid),
    .out_y_mant(sm_y_m), .out_y_exp(sm_y_e),
    .out_valid(sm_y_valid), .done(sm_done)
  );

  // Residual engine — combinational input drive so the engine's S_LOAD→
  // S_EMIT back-pressure (in_ready dropping at tile boundary) directly stops
  // cnt advancement WITHOUT losing the next-to-be-consumed sample.  See the
  // commented timing trace in S_RES1 / S_RES2 for why a registered drive
  // off-by-ones at tile boundaries.
  logic                                rs_start;
  logic                                rs_drive_res1;     // 1 in S_RES1
  logic                                rs_drive_res2;     // 1 in S_RES2
  logic signed [BFP_MANT_W-1:0]        rs_a_m_c, rs_b_m_c;
  logic signed [BFP_EXP_W -1:0]        rs_a_e_c, rs_b_e_c;
  logic                                rs_valid_c, rs_last_c;
  wire                                 rs_in_ready;
  wire  signed [BFP_MANT_W-1:0]        rs_y_m;
  wire  signed [BFP_EXP_W -1:0]        rs_y_e;
  wire                                 rs_y_valid, rs_done;

  always_comb rs_drive_res1 = (state == S_RES1);
  always_comb rs_drive_res2 = (state == S_RES2);
  always_comb begin
    rs_a_m_c   = '0; rs_a_e_c = '0;
    rs_b_m_c   = '0; rs_b_e_c = '0;
    rs_valid_c = 1'b0;
    rs_last_c  = 1'b0;
    if (rs_drive_res1) begin
      rs_a_m_c   = hin_m[cnt[CW_D-1:0]];
      rs_a_e_c   = hin_e[cnt[CW_D-1:0] / BFP_TILE];
      rs_b_m_c   = o_m  [cnt[CW_D-1:0]];
      rs_b_e_c   = o_e  [cnt[CW_D-1:0] / BFP_TILE];
      rs_valid_c = rs_in_ready && (cnt < D);
      rs_last_c  = rs_in_ready && (cnt == D-1);
    end else if (rs_drive_res2) begin
      rs_a_m_c   = h1_m[cnt[CW_D-1:0]];
      rs_a_e_c   = h1_e[cnt[CW_D-1:0] / BFP_TILE];
      rs_b_m_c   = d_m [cnt[CW_D-1:0]];
      rs_b_e_c   = d_e [cnt[CW_D-1:0] / BFP_TILE];
      rs_valid_c = rs_in_ready && (cnt < D);
      rs_last_c  = rs_in_ready && (cnt == D-1);
    end
  end

  residual_bfp #(.D(D)) i_rs (
    .clk(clk), .rst(rst),
    .start(rs_start),
    .in_a_mant(rs_a_m_c), .in_a_exp(rs_a_e_c),
    .in_b_mant(rs_b_m_c), .in_b_exp(rs_b_e_c),
    .in_valid(rs_valid_c), .last_elem(rs_last_c), .in_ready(rs_in_ready),
    .out_y_mant(rs_y_m), .out_y_exp(rs_y_e),
    .out_valid(rs_y_valid), .done(rs_done)
  );

  // ---------------------------------------------------------------------------
  // Output bus packing
  // ---------------------------------------------------------------------------
  always_comb begin
    for (int i = 0; i < D; i++)
      hidden_out_m[i*BFP_MANT_W +: BFP_MANT_W] = hout_m[i];
    for (int t = 0; t < NT_D; t++)
      hidden_out_e[t*BFP_EXP_W  +: BFP_EXP_W ] = hout_e[t];
  end

  // ---------------------------------------------------------------------------
  // K-group mapping helper (for GQA)
  // ---------------------------------------------------------------------------
  function automatic int kv_grp_of;
    input int h;
    kv_grp_of = (H_Q == H_KV) ? h : (h * H_KV) / H_Q;
  endfunction

  // ---------------------------------------------------------------------------
  // matvec chunk re-tile-quantize: matvec_bfp_engine emits a per-lane
  // exponent in mv_out_e but BFP requires one shared exponent per TILE.  When
  // LANES == BFP_TILE (=16) each chunk's LANES outputs form exactly one tile.
  // This helper aligns each lane's mantissa to max(per-lane exp) and returns
  // the shared tile mantissas + tile exp.
  // ---------------------------------------------------------------------------
  // mv_out_m/mv_out_e are the engine's per-lane outputs.  After computing
  // emax = max(mv_out_e[i]), we shift each mantissa right by (emax - mv_out_e[i]).
  function automatic logic signed [BFP_MANT_W-1:0] requant_mant;
    input logic signed [BFP_MANT_W-1:0]  m;
    input logic signed [BFP_EXP_W -1:0]  e_lane;
    input logic signed [BFP_EXP_W -1:0]  e_max;
    logic signed [7:0] sh;
    sh = e_max - e_lane;
    if (sh >= 16)      requant_mant = 16'sd0;
    else if (sh >= 0)  requant_mant = m >>> sh[3:0];
    else               requant_mant = m;
  endfunction

  // ---------------------------------------------------------------------------
  // DDR3 weight streamer (used iff STREAM_WEIGHTS=1).
  //
  //   Active matvec is identified by ws_matvec_id:
  //     0=Q  1=K  2=V  3=O  4=G  5=U  6=DN
  //
  //   Per chunk, the layer FSM pulses ws_load_req with ws_chunk_idx=chunk
  //   and waits for ws_ready before transitioning into the matvec DRIVE
  //   phase.  Then ws_rd_col / ws_rd_tile (driven from `cnt`) deliver
  //   weight_m_out / weight_e_out one cycle later — same pipeline as the
  //   on-chip ROM read it replaces.
  // ---------------------------------------------------------------------------
  logic [2:0]                       ws_matvec_id;
  logic                             ws_load_req;
  wire                              ws_ready, ws_busy_unused;
  logic [AXI_ADDR_WIDTH-1:0]        ws_base_m_mux, ws_base_e_mux;
  logic [11:0]                      ws_in_dim_mux, ws_in_dim_tiles_mux;
  logic [11:0]                      ws_rd_col, ws_rd_tile;

  // 4-phase per-chunk load handshake.  Mirrors smollm_layer.sv's MV_LOAD_
  // {REQ,HOLD,WAIT} pattern: pulse load_req, wait for ws_ready to fall
  // (HOLD — avoids reading stale ready=1 from a previous chunk), then
  // wait for it to rise (WAIT — fresh tile is in bank).  Stays in WSP_READY
  // once data is in bank, until *_NEXT resets it for the next chunk.
  typedef enum logic [1:0] {
    WSP_KICK, WSP_HOLD, WSP_WAIT, WSP_READY
  } ws_phase_t;
  ws_phase_t ws_phase;

  // rd_col / rd_tile track cnt during DRIVE: the streamer has a 1-cycle
  // BRAM read latency, matching the existing `mv_w_m <= rom[cnt]` pipeline
  // alignment exactly.  Outside DRIVE the values don't matter.
  always_comb ws_rd_col  = cnt;
  always_comb ws_rd_tile = cnt >> 4;

  always_comb begin
    // Default — Q.
    ws_base_m_mux        = ws_base_WQ_m;
    ws_base_e_mux        = ws_base_WQ_e;
    ws_in_dim_mux        = 12'(D);
    ws_in_dim_tiles_mux  = 12'(NT_D);
    unique case (ws_matvec_id)
      3'd0: begin ws_base_m_mux = ws_base_WQ_m;  ws_base_e_mux = ws_base_WQ_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd1: begin ws_base_m_mux = ws_base_WK_m;  ws_base_e_mux = ws_base_WK_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd2: begin ws_base_m_mux = ws_base_WV_m;  ws_base_e_mux = ws_base_WV_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd3: begin ws_base_m_mux = ws_base_WO_m;  ws_base_e_mux = ws_base_WO_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd4: begin ws_base_m_mux = ws_base_WG_m;  ws_base_e_mux = ws_base_WG_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd5: begin ws_base_m_mux = ws_base_WU_m;  ws_base_e_mux = ws_base_WU_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd6: begin ws_base_m_mux = ws_base_WDN_m; ws_base_e_mux = ws_base_WDN_e;
                  ws_in_dim_mux = 12'(FFN); ws_in_dim_tiles_mux = 12'(NT_FFN); end
      default:;
    endcase
  end

  generate
    if (STREAM_WEIGHTS) begin : g_stream
      weight_streamer_bfp_mt #(
        .AXI_DATA_WIDTH (512),
        .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH),
        .AXI_ID_WIDTH   (AXI_ID_WIDTH),
        .IN_DIM_MAX     (FFN > D ? FFN : D),
        .IN_DIM_BITS    (12),
        .CHUNK_BITS     (7)
      ) i_ws (
        .clk_core      (clk),
        .rst_core      (rst),
        .matrix_base_m (ws_base_m_mux),
        .matrix_base_e (ws_base_e_mux),
        .chunk_idx     (chunk),                // declared below; alias
        .in_dim        (ws_in_dim_mux),
        .in_dim_tiles  (ws_in_dim_tiles_mux),
        .load_req      (ws_load_req),
        .ready         (ws_ready),
        .busy          (ws_busy_unused),
        .rd_col        (ws_rd_col),
        .weight_m_out  (ws_weight_m_out),
        .rd_tile       (ws_rd_tile),
        .weight_e_out  (ws_weight_e_out),
        .clk_axi       (clk_axi),
        .rst_axi       (rst_axi),
        .m_axi_arvalid (m_axi_arvalid),
        .m_axi_arready (m_axi_arready),
        .m_axi_arid    (m_axi_arid),
        .m_axi_araddr  (m_axi_araddr),
        .m_axi_arlen   (m_axi_arlen),
        .m_axi_arsize  (m_axi_arsize),
        .m_axi_arburst (m_axi_arburst),
        .m_axi_arlock  (m_axi_arlock),
        .m_axi_arcache (m_axi_arcache),
        .m_axi_arprot  (m_axi_arprot),
        .m_axi_arqos   (m_axi_arqos),
        .m_axi_rvalid  (m_axi_rvalid),
        .m_axi_rready  (m_axi_rready),
        .m_axi_rid     (m_axi_rid),
        .m_axi_rdata   (m_axi_rdata),
        .m_axi_rresp   (m_axi_rresp),
        .m_axi_rlast   (m_axi_rlast)
      );
    end else begin : g_no_stream
      // Tie off AXI master ports — no streamer, no traffic.
      assign m_axi_arvalid = 1'b0;
      assign m_axi_arid    = '0;
      assign m_axi_araddr  = '0;
      assign m_axi_arlen   = '0;
      assign m_axi_arsize  = 3'd6;
      assign m_axi_arburst = 2'b01;
      assign m_axi_arlock  = 1'b0;
      assign m_axi_arcache = 4'b0011;
      assign m_axi_arprot  = 3'b000;
      assign m_axi_arqos   = 4'b0000;
      assign m_axi_rready  = 1'b1;
      assign ws_weight_m_out = '0;
      assign ws_weight_e_out = '0;
      assign ws_ready        = 1'b1;     // never gates the FSM
      assign ws_busy_unused  = 1'b0;
      /* verilator lint_off UNUSEDSIGNAL */
      wire _unused_axi = &{1'b0, m_axi_arready, m_axi_rvalid, m_axi_rid,
                           m_axi_rdata, m_axi_rresp, m_axi_rlast, clk_axi,
                           rst_axi, 1'b0};
      /* verilator lint_on UNUSEDSIGNAL */
    end
  endgenerate

  // ---------------------------------------------------------------------------
  // Main FSM — all writes use non-blocking assignment; engine inputs are
  // driven from registers so they pipeline-align with mv_valid one cycle
  // later (matvec engine samples on the clock edge after we register).
  // ---------------------------------------------------------------------------
  integer ii;       // for-loop scratch in always_ff (synth-safe with int loop body)
  logic [4:0] head_grp;
  logic [11:0] av_row_base;      // current AV row offset = head*HD + chunk*LANES
  // Common w_e for AV — max of kv_v_e[t][chunk] over t in [0..kv_pos].
  // Computed per-chunk in S_AV_PRIME; used by S_AV_DRIVE to align each
  // timestep's V mantissa before feeding the matvec engine.
  logic signed [BFP_EXP_W-1:0] av_emax;
  // Sequential scan counter for AV_EMAX_SCAN — walks t=0..kv_pos-1
  // (one cycle each) to compute av_emax from kv_v_e without a 64-port
  // comb fan-in (which Vivado can't infer as BRAM and won't dissolve
  // at full SmolLM2 dims).
  logic [CW_CTX-1:0] av_scan_cnt;
  // Residual / SwiGLU output write-index — independent of cnt (which tracks
  // input drive) because the engines emit interleaved with loading.
  logic [11:0] out_cnt;

  always_ff @(posedge clk) begin
    if (rst) begin
      state           <= S_IDLE;
      done            <= 1'b0;
      eng_rst         <= 1'b1;
      cnt             <= '0;
      chunk           <= '0;
      head_idx        <= '0;
      kv_t            <= '0;
      score_emax      <= -8'sd128;
      mv_out_m_drain_r<= '0;
      mv_out_e_drain_r<= '0;
      emax_h0_r       <= -8'sd128;
      emax_h1_r       <= -8'sd128;
      mv_start        <= 1'b0;
      mv_valid        <= 1'b0;
      mv_last         <= 1'b0;
      mv_x_m          <= '0;
      mv_x_e          <= '0;
      mv_w_m          <= '0;
      mv_w_e          <= '0;
      rn_start        <= 1'b0;
      rn_valid        <= 1'b0;
      rn_last         <= 1'b0;
      rn_x_m          <= '0;
      rn_x_e          <= '0;
      rn_g_m          <= '0;
      rn_g_e          <= '0;
      rp_start        <= 1'b0;
      rp_valid        <= 1'b0;
      rp_x_m          <= '0;
      rp_x_e          <= '0;
      sg_start        <= 1'b0;
      sm_start        <= 1'b0;
      sm_valid        <= 1'b0;
      sm_n            <= '0;
      sm_x_m          <= '0;
      sm_x_e          <= '0;
      rs_start        <= 1'b0;
      scores_e_shared <= '0;
      probs_e_shared  <= '0;
      head_grp        <= '0;
      av_row_base     <= '0;
      av_scan_cnt     <= '0;
      out_cnt         <= '0;
      ws_matvec_id    <= 3'd0;
      ws_load_req     <= 1'b0;
      ws_phase        <= WSP_KICK;
    end else begin
      // Default pulses → 0
      mv_start <= 1'b0;
      mv_valid <= 1'b0;
      mv_last  <= 1'b0;
      rn_start <= 1'b0;
      rn_valid <= 1'b0;
      rn_last  <= 1'b0;
      rp_start <= 1'b0;
      rp_valid <= 1'b0;
      sg_start <= 1'b0;
      sm_start <= 1'b0;
      sm_valid <= 1'b0;
      rs_start <= 1'b0;
      eng_rst  <= 1'b0;
      done     <= 1'b0;
      ws_load_req <= 1'b0;

      case (state)
        // -------------------------------------------------------------------
        S_IDLE: if (start) begin
          state   <= S_LATCH_IN;
          cnt     <= '0;
          eng_rst <= 1'b1;
          if (STREAM_WEIGHTS) ws_phase <= WSP_KICK;
        end

        // Copy hidden_in bus into BRAM-style arrays one element/cycle.
        S_LATCH_IN: begin
          hin_m[cnt[CW_D-1:0]] <= hidden_in_m[cnt[CW_D-1:0]*BFP_MANT_W +: BFP_MANT_W];
          if (cnt < NT_D)
            hin_e[cnt[CW_D-1:0]] <= hidden_in_e[cnt[CW_D-1:0]*BFP_EXP_W +: BFP_EXP_W];
          if (cnt == D-1) begin
            state    <= S_NORM1;
            cnt      <= '0;
            rn_start <= 1'b1;
          end else cnt <= cnt + 1'b1;
        end

        // -------------------------------------------------------------------
        // RMSNorm1 — feed hin + G1, capture norm1 (n1).
        // -------------------------------------------------------------------
        S_NORM1: begin
          rn_valid <= 1'b1;
          rn_x_m   <= hin_m[cnt[CW_D-1:0]];
          rn_x_e   <= hin_e[cnt[CW_D-1:0] / BFP_TILE];
          rn_g_m   <= rom_G1_m[layer_idx * D    + cnt[CW_D-1:0]];
          rn_g_e   <= rom_G1_e[layer_idx * NT_D + cnt[CW_D-1:0] / BFP_TILE];
          rn_last  <= (cnt == D-1);
          if (cnt == D-1) begin
            state <= S_NORM1_WAIT;
            cnt   <= '0;
          end else cnt <= cnt + 1'b1;
        end

        S_NORM1_WAIT: begin
          if (rn_y_valid) begin
            n1_m[cnt[CW_D-1:0]] <= rn_y_m;
            if (cnt[3:0] == 4'd0)
              n1_e[cnt[CW_D-1:0] / BFP_TILE] <= rn_y_e;
            if (cnt == D-1) begin
              cnt      <= '0;
              chunk    <= '0;
              eng_rst  <= 1'b1;
              state    <= S_QMV_PRIME;
            end else cnt <= cnt + 1'b1;
          end
        end

        // ===================================================================
        // Q matvec — 7 phases (PRIME/DRIVE/DRAIN/NEXT) — input dim = D, out_dim = D
        // ===================================================================
        S_QMV_PRIME: begin
          if (STREAM_WEIGHTS && ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd0;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1;
            cnt      <= '0;
            state    <= S_QMV_DRIVE;
          end
        end
        S_QMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= n1_m[cnt[CW_D-1:0]];
          mv_x_e   <= n1_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_w_m   <= rom_WQ_m[layer_idx * WQ_M_PL + chunk * D    + cnt[CW_D-1:0]];
          mv_w_e   <= rom_WQ_e[layer_idx * WQ_E_PL + chunk * NT_D + cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_QMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_QMV_DRAIN: if (mv_out_valid_lat) begin : qmv_pipeA
          // Stage 1: latch + compute max of two 8-lane halves.
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_QMV_REQ;
        end
        S_QMV_REQ: begin : qmv_pipeB
          // Stage 2: final max + 16-lane parallel requant + write.
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          for (ii = 0; ii < LANES; ii++)
            q_m[chunk * LANES + ii] <= requant_mant(
              mv_out_m_drain_r[ii*BFP_MANT_W +: BFP_MANT_W],
              mv_out_e_drain_r[ii*BFP_EXP_W  +: BFP_EXP_W ],
              emax_f);
          q_e[chunk] <= emax_f;
          state <= S_QMV_NEXT;
        end
        S_QMV_NEXT: begin
          if (STREAM_WEIGHTS) ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_D - 1) begin
            chunk   <= '0;
            eng_rst <= 1'b1;
            state   <= S_KMV_PRIME;
          end else begin
            chunk   <= chunk + 1'b1;
            eng_rst <= 1'b1;
            state   <= S_QMV_PRIME;
          end
        end

        // ===================================================================
        // K matvec
        // ===================================================================
        S_KMV_PRIME: begin
          if (STREAM_WEIGHTS && ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd1;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_KMV_DRIVE;
          end
        end
        S_KMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= n1_m[cnt[CW_D-1:0]];
          mv_x_e   <= n1_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_w_m   <= rom_WK_m[layer_idx * WK_M_PL + chunk * D    + cnt[CW_D-1:0]];
          mv_w_e   <= rom_WK_e[layer_idx * WK_E_PL + chunk * NT_D + cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_KMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_KMV_DRAIN: if (mv_out_valid_lat) begin : kmv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_KMV_REQ;
        end
        S_KMV_REQ: begin : kmv_pipeB
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          for (ii = 0; ii < LANES; ii++)
            k_m[chunk * LANES + ii] <= requant_mant(
              mv_out_m_drain_r[ii*BFP_MANT_W +: BFP_MANT_W],
              mv_out_e_drain_r[ii*BFP_EXP_W  +: BFP_EXP_W ],
              emax_f);
          k_e[chunk] <= emax_f;
          state <= S_KMV_NEXT;
        end
        S_KMV_NEXT: begin
          if (STREAM_WEIGHTS) ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_KV - 1) begin
            chunk <= '0; eng_rst <= 1'b1; state <= S_VMV_PRIME;
          end else begin
            chunk <= chunk + 1'b1; eng_rst <= 1'b1; state <= S_KMV_PRIME;
          end
        end

        // ===================================================================
        // V matvec
        // ===================================================================
        S_VMV_PRIME: begin
          if (STREAM_WEIGHTS && ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd2;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_VMV_DRIVE;
          end
        end
        S_VMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= n1_m[cnt[CW_D-1:0]];
          mv_x_e   <= n1_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_w_m   <= rom_WV_m[layer_idx * WV_M_PL + chunk * D    + cnt[CW_D-1:0]];
          mv_w_e   <= rom_WV_e[layer_idx * WV_E_PL + chunk * NT_D + cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_VMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_VMV_DRAIN: if (mv_out_valid_lat) begin : vmv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_VMV_REQ;
        end
        S_VMV_REQ: begin : vmv_pipeB
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          for (ii = 0; ii < LANES; ii++)
            v_m[chunk * LANES + ii] <= requant_mant(
              mv_out_m_drain_r[ii*BFP_MANT_W +: BFP_MANT_W],
              mv_out_e_drain_r[ii*BFP_EXP_W  +: BFP_EXP_W ],
              emax_f);
          v_e[chunk] <= emax_f;
          state <= S_VMV_NEXT;
        end
        S_VMV_NEXT: begin
          if (STREAM_WEIGHTS) ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_KV - 1) begin
            chunk <= '0; head_idx <= '0; cnt <= '0;
            eng_rst <= 1'b1; rp_start <= 1'b1; state <= S_ROPEQ;
          end else begin
            chunk <= chunk + 1'b1; eng_rst <= 1'b1; state <= S_VMV_PRIME;
          end
        end

        // ===================================================================
        // RoPE Q — H_Q heads
        // ===================================================================
        S_ROPEQ: begin
          rp_valid <= 1'b1;
          rp_x_m   <= q_m[head_idx * HD + cnt[CW_HD-1:0]];
          rp_x_e   <= q_e[(head_idx * HD + cnt[CW_HD-1:0]) / BFP_TILE];
          if (cnt == HD-1) begin
            state <= S_ROPEQ_WAIT; cnt <= '0;
          end else cnt <= cnt + 1'b1;
        end
        S_ROPEQ_WAIT: begin
          if (rp_y_valid) begin
            q_rot_m[head_idx * HD + cnt[CW_HD-1:0]] <= rp_y_m;
            q_rot_e[head_idx * HD + cnt[CW_HD-1:0]] <= rp_y_e;
            cnt <= cnt + 1'b1;
          end
          if (rp_done) begin
            if (head_idx == H_Q - 1) begin
              cnt   <= '0;
              state <= S_ROPEQ_RQ_A;
            end else begin
              head_idx <= head_idx + 1'b1; cnt <= '0;
              rp_start <= 1'b1; state <= S_ROPEQ;
            end
          end
        end

        // Re-tile-quantize q_rot — 2-stage pipeline.
        // Stage A: scan the tile's q_rot_e[base..base+15], compute the
        //          max of each 8-element half; latch into emax_h{0,1}_r.
        // Stage B: final emax = max(halves); apply requant_mant to all
        //          16 elements; write q_m + q_e; advance cnt.
        S_ROPEQ_RQ_A: begin : rq_q_a
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          automatic int base;
          base = cnt[$clog2(NT_D+1)-1:0] * BFP_TILE;
          e_lo = -8'sd128;
          for (ii = 0; ii < 8; ii++)
            if (base + ii < D)
              if (q_rot_e[base + ii] > e_lo) e_lo = q_rot_e[base + ii];
          e_hi = -8'sd128;
          for (ii = 8; ii < BFP_TILE; ii++)
            if (base + ii < D)
              if (q_rot_e[base + ii] > e_hi) e_hi = q_rot_e[base + ii];
          emax_h0_r <= e_lo;
          emax_h1_r <= e_hi;
          state <= S_ROPEQ_RQ_B;
        end
        S_ROPEQ_RQ_B: begin : rq_q_b
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          automatic int base;
          base = cnt[$clog2(NT_D+1)-1:0] * BFP_TILE;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          for (ii = 0; ii < BFP_TILE; ii++) begin
            if (base + ii < D)
              q_m[base + ii] <= requant_mant(q_rot_m[base + ii],
                                             q_rot_e[base + ii], emax_f);
          end
          q_e[cnt[$clog2(NT_D+1)-1:0]] <= emax_f;
          if (cnt == NT_D - 1) begin
            head_idx <= '0; cnt <= '0;
            rp_start <= 1'b1; state <= S_ROPEK;
          end else begin
            cnt   <= cnt + 1'b1;
            state <= S_ROPEQ_RQ_A;
          end
        end

        // RoPE K — H_KV heads
        S_ROPEK: begin
          rp_valid <= 1'b1;
          rp_x_m   <= k_m[head_idx * HD + cnt[CW_HD-1:0]];
          rp_x_e   <= k_e[(head_idx * HD + cnt[CW_HD-1:0]) / BFP_TILE];
          if (cnt == HD-1) begin
            state <= S_ROPEK_WAIT; cnt <= '0;
          end else cnt <= cnt + 1'b1;
        end
        S_ROPEK_WAIT: begin
          if (rp_y_valid) begin
            k_rot_m[head_idx * HD + cnt[CW_HD-1:0]] <= rp_y_m;
            k_rot_e[head_idx * HD + cnt[CW_HD-1:0]] <= rp_y_e;
            cnt <= cnt + 1'b1;
          end
          if (rp_done) begin
            if (head_idx == H_KV - 1) begin
              cnt   <= '0;
              state <= S_ROPEK_RQ_A;
            end else begin
              head_idx <= head_idx + 1'b1; cnt <= '0;
              rp_start <= 1'b1; state <= S_ROPEK;
            end
          end
        end

        // Re-tile-quantize k_rot — 2-stage pipeline (mirror of S_ROPEQ_RQ).
        S_ROPEK_RQ_A: begin : rq_k_a
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          automatic int base;
          base = cnt[$clog2(NT_KV+1)-1:0] * BFP_TILE;
          e_lo = -8'sd128;
          for (ii = 0; ii < 8; ii++)
            if (base + ii < H_KV*HD)
              if (k_rot_e[base + ii] > e_lo) e_lo = k_rot_e[base + ii];
          e_hi = -8'sd128;
          for (ii = 8; ii < BFP_TILE; ii++)
            if (base + ii < H_KV*HD)
              if (k_rot_e[base + ii] > e_hi) e_hi = k_rot_e[base + ii];
          emax_h0_r <= e_lo;
          emax_h1_r <= e_hi;
          state <= S_ROPEK_RQ_B;
        end
        S_ROPEK_RQ_B: begin : rq_k_b
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          automatic int base;
          base = cnt[$clog2(NT_KV+1)-1:0] * BFP_TILE;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          for (ii = 0; ii < BFP_TILE; ii++) begin
            if (base + ii < H_KV*HD)
              k_m[base + ii] <= requant_mant(k_rot_m[base + ii],
                                              k_rot_e[base + ii], emax_f);
          end
          k_e[cnt[$clog2(NT_KV+1)-1:0]] <= emax_f;
          if (cnt == NT_KV - 1) begin
            cnt <= '0;
            state <= S_KVWR_M;
          end else begin
            cnt   <= cnt + 1'b1;
            state <= S_ROPEK_RQ_A;
          end
        end

        // ===================================================================
        // KV-cache write: store k_rot and v (no rope for V) at slot kv_pos.
        // Phase M: stream HD*H_KV mantissas.  Phase E: collapse rope
        // per-element exps into one shared tile-exp via max.
        // ===================================================================
        // After re-tile-quant, k_m/k_e is the post-rope K in per-tile BFP.
        // Just copy mantissas + per-tile exponents into the KV cache slot.
        S_KVWR_M: begin : kvwr_m_blk
          kv_k_m[layer_idx * MAX_CTX * H_KV * HD + kv_pos * H_KV * HD + cnt[CW_KV-1:0]] <= k_m[cnt[CW_KV-1:0]];
          // Write the wide-packed kv_v_m_chk entry every 16 cycles
          // (once per chunk of 16 lanes).  The packed entry assembles
          // v_m[chunk_base..chunk_base+15] combinationally from v_m
          // (which is small FFs — 16-port read is cheap).
          if (cnt[3:0] == 4'd15) begin
            automatic logic [LANES*BFP_MANT_W-1:0] packed_v;
            for (int gl = 0; gl < LANES; gl++)
              packed_v[gl*BFP_MANT_W +: BFP_MANT_W] = v_m[cnt[CW_KV-1:4] * LANES + gl];
            kv_v_m_chk[layer_idx * MAX_CTX * KVV_CHUNKS_PER_KVPOS
                       + kv_pos * KVV_CHUNKS_PER_KVPOS + cnt[CW_KV-1:4]] <= packed_v;
          end
          if (cnt == H_KV*HD - 1) begin
            cnt   <= '0; state <= S_KVWR_E;
          end else cnt <= cnt + 1'b1;
        end
        S_KVWR_E: begin
          kv_k_e[layer_idx * MAX_CTX * NT_KV + kv_pos * NT_KV + cnt[CW_KV-1:0]] <= k_e[cnt[CW_KV-1:0]];
          kv_v_e[layer_idx * MAX_CTX * NT_KV + kv_pos * NT_KV + cnt[CW_KV-1:0]] <= v_e[cnt[CW_KV-1:0]];
          if (cnt == NT_KV - 1) begin
            cnt <= '0; head_idx <= '0; kv_t <= '0;
            score_emax <= -8'sd128;
            eng_rst <= 1'b1;
            state <= S_QK_PRIME;
          end else cnt <= cnt + 1'b1;
        end

        // ===================================================================
        // Attention QK^T — one timestep at a time.  Engine reused: stream
        // Q[head] mantissas, with K[grp,t] mantissas on lane 0 (other lanes 0).
        // ===================================================================
        S_QK_PRIME: begin
          head_grp <= 5'(kv_grp_of(head_idx));
          mv_start <= 1'b1; cnt <= '0; state <= S_QK_DRIVE;
        end
        S_QK_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= q_m[head_idx * HD + cnt[CW_HD-1:0]];
          mv_x_e   <= q_e[(head_idx * HD + cnt[CW_HD-1:0]) / BFP_TILE];
          mv_last  <= (cnt == HD-1);
          for (ii = 0; ii < LANES; ii++) begin
            if (ii == 0) begin
              if (kv_t == kv_pos) begin
                mv_w_m[0 +: BFP_MANT_W] <= k_m[head_grp * HD + cnt[CW_HD-1:0]];
                mv_w_e[0 +: BFP_EXP_W ] <= k_e[(head_grp * HD + cnt[CW_HD-1:0]) / BFP_TILE];
              end else begin
                mv_w_m[0 +: BFP_MANT_W] <= kv_k_m[layer_idx * MAX_CTX * H_KV*HD + kv_t * H_KV*HD + head_grp * HD + cnt[CW_HD-1:0]];
                mv_w_e[0 +: BFP_EXP_W ] <= kv_k_e[layer_idx * MAX_CTX * NT_KV  + kv_t * NT_KV + (head_grp * HD + cnt[CW_HD-1:0]) / BFP_TILE];
              end
            end else begin
              mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= '0;
              mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <= '0;
            end
          end
          if (cnt == HD-1) state <= S_QK_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_QK_DRAIN: if (mv_out_valid) begin
`ifdef LBFP_STAGE_DUMP
          $display("[QK_DRAIN] kv_t=%0d mv_lane0_m=%0d mv_lane0_e=%0d score_emax=%0d this_>max=%0d",
                   kv_t, mv_lane0_m, mv_lane0_e, score_emax,
                   (mv_lane0_e > score_emax));
`endif
          scores_m  [kv_t] <= mv_lane0_m;
          qk_score_e[kv_t] <= mv_lane0_e;
          if (mv_lane0_e > score_emax) score_emax <= mv_lane0_e;
          if (kv_t == kv_pos) begin
            scores_e_shared <= (mv_lane0_e > score_emax) ? mv_lane0_e : score_emax;
            cnt      <= '0;
            eng_rst  <= 1'b1;
            sm_start <= 1'b1;          // fire start one cycle before first valid
            state    <= S_SM_DRIVE;
          end else begin
            kv_t    <= kv_t + 1'b1;
            eng_rst <= 1'b1;
            state   <= S_QK_PRIME;
          end
        end

        // -------------------------------------------------------------------
        // Softmax: pre-align each score mantissa to scores_e_shared, apply
        // 1/sqrt(HD) = 2^-3 (HD=64) by tweaking the shared exponent.  Stream
        // (kv_pos+1) values into softmax_bfp.
        // -------------------------------------------------------------------
        S_SM_DRIVE: begin
          sm_valid <= 1'b1;
          sm_n     <= CW_CTX'(kv_pos + 1);
          // Pre-shift mantissa
          begin : sm_drv
            automatic logic signed [BFP_EXP_W:0] sh;
            sh = scores_e_shared - qk_score_e[cnt[CW_CTX-1:0]];
            if (sh >= 16) sm_x_m <= '0;
            else          sm_x_m <= $signed(scores_m[cnt[CW_CTX-1:0]]) >>> sh[3:0];
          end
          // 1/sqrt(HD=64) = 2^-3
          sm_x_e <= scores_e_shared - 8'sd3;
          if (cnt == kv_pos) begin
            state <= S_SM_WAIT; cnt <= '0;
          end else cnt <= cnt + 1'b1;
        end
        S_SM_WAIT: begin
          if (sm_y_valid) begin
            probs_m[cnt[CW_CTX-1:0]] <= sm_y_m;
            probs_e_shared           <= sm_y_e;
            cnt <= cnt + 1'b1;
          end
          if (sm_done) begin
            // Begin AV: HD outputs in CHUNKS_HD chunks of LANES
            chunk        <= '0;
            kv_t         <= '0;
            av_row_base  <= head_idx * HD;
            eng_rst      <= 1'b1;
            state        <= S_AV_PRIME;
          end
        end

        // ===================================================================
        // AV — for each chunk of HD outputs, matvec input = probs (length
        // kv_pos+1), weight row = V[grp, t, j_base..j_base+LANES-1].
        // ===================================================================
        // S_AV_PRIME: init head_grp + seed av_emax with the current-step
        // V tile exponent.  Then S_AV_EMAX_SCAN walks past timesteps
        // 0..kv_pos-1 one per cycle to fold each kv_v_e[t] into the max.
        // Replaces the original MAX_CTX-wide combinational fan-in,
        // which made kv_v_e_reg unsynthesizable at NL=30 / MAX_CTX=64.
        S_AV_PRIME: begin : av_prime_blk
          automatic int tile_idx;
          tile_idx = (kv_grp_of(head_idx) * HD + chunk * LANES) / BFP_TILE;
          head_grp    <= 5'(kv_grp_of(head_idx));
          av_emax     <= $signed(v_e[tile_idx]);   // seed with current step
          av_scan_cnt <= '0;
          state       <= S_AV_EMAX_SCAN;
        end
        S_AV_EMAX_SCAN: begin : av_emax_scan
          automatic logic signed [BFP_EXP_W-1:0] e_t;
          automatic int tile_idx;
          tile_idx = (head_grp * HD + chunk * LANES) / BFP_TILE;
          if (av_scan_cnt < kv_pos) begin
            e_t = $signed(kv_v_e[layer_idx * MAX_CTX * NT_KV
                                  + av_scan_cnt[CW_CTX-1:0] * NT_KV + tile_idx]);
            if (e_t > av_emax) av_emax <= e_t;
            av_scan_cnt <= av_scan_cnt + 1'b1;
          end else begin
            // Scan complete; kick the engine and enter DRIVE next cycle.
            mv_start <= 1'b1;
            cnt      <= '0;
            state    <= S_AV_DRIVE;
          end
        end
        // matvec engine requires input dim to be a multiple of TILE so that
        // tile_done can fire AND all 16 elements of a tile share one w_e.
        // For AV the per-timestep V values have different per-tile exps, so
        // we pre-shift each timestep's V mantissa to av_emax and feed av_emax
        // as the shared w_e for every cycle (including the post-kv_pos
        // padding zeros).
        S_AV_DRIVE: begin : av_drive_blk
          automatic int tile_idx;
          automatic logic signed [BFP_EXP_W-1:0] v_e_this;
          automatic logic signed [BFP_EXP_W-1:0] shamt;
          // Read the full 16-lane wide-packed V chunk for this (t, head_grp, chunk).
          // Slice lanes via combinational mux of the registered 256-bit value.
          automatic logic [LANES*BFP_MANT_W-1:0] v_chk;
          tile_idx = (head_grp * HD + chunk * LANES) / BFP_TILE;
          v_chk = kv_v_m_chk[layer_idx * MAX_CTX * KVV_CHUNKS_PER_KVPOS
                             + cnt[CW_CTX-1:0] * KVV_CHUNKS_PER_KVPOS
                             + head_grp * (HD/LANES) + chunk];
          mv_valid <= 1'b1;
          mv_x_e   <= probs_e_shared;
          if (cnt <= kv_pos) begin
            mv_x_m <= probs_m[cnt[CW_CTX-1:0]];
            if (cnt[CW_CTX-1:0] == kv_pos) v_e_this = $signed(v_e[tile_idx]);
            else v_e_this = $signed(kv_v_e[layer_idx * MAX_CTX * NT_KV + cnt[CW_CTX-1:0] * NT_KV + tile_idx]);
            shamt = av_emax - v_e_this;
            for (ii = 0; ii < LANES; ii++) begin
              automatic logic signed [BFP_MANT_W-1:0] m_raw;
              if (cnt[CW_CTX-1:0] == kv_pos)
                m_raw = $signed(v_m[head_grp * HD + chunk * LANES + ii]);
              else
                m_raw = $signed(v_chk[ii*BFP_MANT_W +: BFP_MANT_W]);
              if (shamt >= 16)     mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= '0;
              else if (shamt >= 0) mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= m_raw >>> shamt[3:0];
              else                 mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= m_raw;
              mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <= av_emax;
            end
          end else begin
            // Padding cycles past kv_pos.  Mantissas zero; w_e == av_emax so
            // the matvec engine's tile_exp ends up at probs_e + av_emax (the
            // intended scale).
            mv_x_m <= '0;
            for (ii = 0; ii < LANES; ii++) begin
              mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= '0;
              mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <= av_emax;
            end
          end
          mv_last <= (cnt == BFP_TILE - 1);
          if (cnt == BFP_TILE - 1) state <= S_AV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_AV_DRAIN: if (mv_out_valid_lat) begin : av_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_AV_REQ;
        end
        S_AV_REQ: begin : av_pipeB
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          for (ii = 0; ii < LANES; ii++)
            attn_m[av_row_base + chunk * LANES + ii] <= requant_mant(
              mv_out_m_drain_r[ii*BFP_MANT_W +: BFP_MANT_W],
              mv_out_e_drain_r[ii*BFP_EXP_W  +: BFP_EXP_W ],
              emax_f);
          attn_e[(av_row_base + chunk * LANES) / BFP_TILE] <= emax_f;
          state <= S_AV_NEXT;
        end
        S_AV_NEXT: begin
          if (chunk == CHUNKS_HD - 1) begin
            // head done — next head, or move to O matvec
            if (head_idx == H_Q - 1) begin
              chunk    <= '0; cnt <= '0; eng_rst <= 1'b1; state <= S_OMV_PRIME;
            end else begin
              head_idx <= head_idx + 1'b1;
              chunk    <= '0; cnt <= '0; kv_t <= '0;
              score_emax <= -8'sd128;
              eng_rst <= 1'b1;
              state <= S_QK_PRIME;
            end
          end else begin
            chunk   <= chunk + 1'b1;
            eng_rst <= 1'b1;
            state   <= S_AV_PRIME;
          end
        end

        // ===================================================================
        // O matvec (D inputs, D outputs)
        // ===================================================================
        S_OMV_PRIME: begin
          if (STREAM_WEIGHTS && ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd3;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_OMV_DRIVE;
          end
        end
        S_OMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= attn_m[cnt[CW_D-1:0]];
          mv_x_e   <= attn_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_w_m   <= rom_WO_m[layer_idx * WO_M_PL + chunk * D    + cnt[CW_D-1:0]];
          mv_w_e   <= rom_WO_e[layer_idx * WO_E_PL + chunk * NT_D + cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_OMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_OMV_DRAIN: if (mv_out_valid_lat) begin : omv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_OMV_REQ;
        end
        S_OMV_REQ: begin : omv_pipeB
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          for (ii = 0; ii < LANES; ii++)
            o_m[chunk * LANES + ii] <= requant_mant(
              mv_out_m_drain_r[ii*BFP_MANT_W +: BFP_MANT_W],
              mv_out_e_drain_r[ii*BFP_EXP_W  +: BFP_EXP_W ],
              emax_f);
          o_e[chunk] <= emax_f;
          state <= S_OMV_NEXT;
        end
        S_OMV_NEXT: begin
          if (STREAM_WEIGHTS) ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_D - 1) begin
            chunk   <= '0; cnt <= '0; out_cnt <= '0;
            eng_rst <= 1'b1; rs_start <= 1'b1; state <= S_RES1;
          end else begin
            chunk   <= chunk + 1'b1;
            eng_rst <= 1'b1; state <= S_OMV_PRIME;
          end
        end

        // ===================================================================
        // Residual 1: h1 = hin + o.
        //
        // rs_a_*/rs_b_*/rs_valid/rs_last are driven COMBINATIONALLY from cnt
        // (see always_comb above and the rs_drive_res1 flag).  The engine's
        // in_ready gate naturally pauses cnt during S_EMIT phases.  When
        // in_valid && in_ready both high THIS cycle, the engine WILL consume
        // the comb data into its registers this same cycle — so cnt can
        // advance without losing data at tile boundaries.
        // ===================================================================
        S_RES1: begin
          // (comb drive is via rs_drive_res1 / cnt above)
          if (rs_in_ready && rs_valid_c) begin
            if (cnt == D-1) state <= S_RES1_WAIT;
            else            cnt <= cnt + 1'b1;
          end
          // Capture outputs concurrently — engine emits during S_LOAD gaps.
          if (rs_y_valid) begin
            h1_m[out_cnt[CW_D-1:0]] <= rs_y_m;
            if (out_cnt[3:0] == 4'd0)
              h1_e[out_cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            out_cnt <= out_cnt + 1'b1;
          end
        end
        S_RES1_WAIT: begin
          if (rs_y_valid) begin
            h1_m[out_cnt[CW_D-1:0]] <= rs_y_m;
            if (out_cnt[3:0] == 4'd0)
              h1_e[out_cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            out_cnt <= out_cnt + 1'b1;
          end
          if (rs_done) begin
            state    <= S_NORM2;
            eng_rst  <= 1'b1;
            cnt      <= '0;
            out_cnt  <= '0;
            rn_start <= 1'b1;
          end
        end

        // ===================================================================
        // RMSNorm2
        // ===================================================================
        S_NORM2: begin
          rn_valid <= 1'b1;
          rn_x_m   <= h1_m[cnt[CW_D-1:0]];
          rn_x_e   <= h1_e[cnt[CW_D-1:0] / BFP_TILE];
          rn_g_m   <= rom_G2_m[layer_idx * D    + cnt[CW_D-1:0]];
          rn_g_e   <= rom_G2_e[layer_idx * NT_D + cnt[CW_D-1:0] / BFP_TILE];
          rn_last  <= (cnt == D-1);
          if (cnt == D-1) begin
            state <= S_NORM2_WAIT; cnt <= '0;
          end else cnt <= cnt + 1'b1;
        end
        S_NORM2_WAIT: begin
          if (rn_y_valid) begin
            n2_m[cnt[CW_D-1:0]] <= rn_y_m;
            if (cnt[3:0] == 4'd0)
              n2_e[cnt[CW_D-1:0] / BFP_TILE] <= rn_y_e;
            if (cnt == D-1) begin
              cnt   <= '0; chunk <= '0; eng_rst <= 1'b1; state <= S_GMV_PRIME;
            end else cnt <= cnt + 1'b1;
          end
        end

        // ===================================================================
        // Gate matvec (D in, FFN out)
        // ===================================================================
        S_GMV_PRIME: begin
          if (STREAM_WEIGHTS && ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd4;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_GMV_DRIVE;
          end
        end
        S_GMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= n2_m[cnt[CW_D-1:0]];
          mv_x_e   <= n2_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_w_m   <= rom_WG_m[layer_idx * WG_M_PL + chunk * D    + cnt[CW_D-1:0]];
          mv_w_e   <= rom_WG_e[layer_idx * WG_E_PL + chunk * NT_D + cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_GMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_GMV_DRAIN: if (mv_out_valid_lat) begin : gmv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_GMV_REQ;
        end
        S_GMV_REQ: begin : gmv_pipeB
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          for (ii = 0; ii < LANES; ii++)
            g_m[chunk * LANES + ii] <= requant_mant(
              mv_out_m_drain_r[ii*BFP_MANT_W +: BFP_MANT_W],
              mv_out_e_drain_r[ii*BFP_EXP_W  +: BFP_EXP_W ],
              emax_f);
          g_e[chunk] <= emax_f;
          state <= S_GMV_NEXT;
        end
        S_GMV_NEXT: begin
          if (STREAM_WEIGHTS) ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_FFN - 1) begin
            chunk <= '0; eng_rst <= 1'b1; state <= S_UMV_PRIME;
          end else begin
            chunk <= chunk + 1'b1; eng_rst <= 1'b1; state <= S_GMV_PRIME;
          end
        end

        // ===================================================================
        // Up matvec (D in, FFN out)
        // ===================================================================
        S_UMV_PRIME: begin
          if (STREAM_WEIGHTS && ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd5;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_UMV_DRIVE;
          end
        end
        S_UMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= n2_m[cnt[CW_D-1:0]];
          mv_x_e   <= n2_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_w_m   <= rom_WU_m[layer_idx * WU_M_PL + chunk * D    + cnt[CW_D-1:0]];
          mv_w_e   <= rom_WU_e[layer_idx * WU_E_PL + chunk * NT_D + cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_UMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_UMV_DRAIN: if (mv_out_valid_lat) begin : umv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_UMV_REQ;
        end
        S_UMV_REQ: begin : umv_pipeB
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          for (ii = 0; ii < LANES; ii++)
            u_m[chunk * LANES + ii] <= requant_mant(
              mv_out_m_drain_r[ii*BFP_MANT_W +: BFP_MANT_W],
              mv_out_e_drain_r[ii*BFP_EXP_W  +: BFP_EXP_W ],
              emax_f);
          u_e[chunk] <= emax_f;
          state <= S_UMV_NEXT;
        end
        S_UMV_NEXT: begin
          if (STREAM_WEIGHTS) ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_FFN - 1) begin
            chunk <= '0; cnt <= '0; out_cnt <= '0;
            eng_rst <= 1'b1; sg_start <= 1'b1; state <= S_SWG;
          end else begin
            chunk <= chunk + 1'b1; eng_rst <= 1'b1; state <= S_UMV_PRIME;
          end
        end

        // ===================================================================
        // SwiGLU (FFN -> FFN).  Comb drive via sg_drive flag.
        // ===================================================================
        S_SWG: begin
          if (sg_in_ready && sg_valid_c) begin
            if (cnt == FFN-1) state <= S_SWG_WAIT;
            else              cnt <= cnt + 1'b1;
          end
          if (sg_y_valid) begin
            mlp_m[out_cnt[CW_FFN-1:0]] <= sg_y_m;
            if (out_cnt[3:0] == 4'd0)
              mlp_e[out_cnt[CW_FFN-1:0] / BFP_TILE] <= sg_y_e;
            out_cnt <= out_cnt + 1'b1;
          end
        end
        S_SWG_WAIT: begin
          if (sg_y_valid) begin
            mlp_m[out_cnt[CW_FFN-1:0]] <= sg_y_m;
            if (out_cnt[3:0] == 4'd0)
              mlp_e[out_cnt[CW_FFN-1:0] / BFP_TILE] <= sg_y_e;
            out_cnt <= out_cnt + 1'b1;
          end
          if (sg_done) begin
            chunk <= '0; eng_rst <= 1'b1; state <= S_DMV_PRIME;
          end
        end

        // ===================================================================
        // Down matvec (FFN in, D out)
        // ===================================================================
        S_DMV_PRIME: begin
          if (STREAM_WEIGHTS && ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd6;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_DMV_DRIVE;
          end
        end
        S_DMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= mlp_m[cnt[CW_FFN-1:0]];
          mv_x_e   <= mlp_e[cnt[CW_FFN-1:0] / BFP_TILE];
          mv_w_m   <= rom_WDN_m[layer_idx * WDN_M_PL + chunk * FFN    + cnt[CW_FFN-1:0]];
          mv_w_e   <= rom_WDN_e[layer_idx * WDN_E_PL + chunk * NT_FFN + cnt[CW_FFN-1:0] / BFP_TILE];
          mv_last  <= (cnt == FFN-1);
          if (cnt == FFN-1) state <= S_DMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_DMV_DRAIN: if (mv_out_valid_lat) begin : dmv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_DMV_REQ;
        end
        S_DMV_REQ: begin : dmv_pipeB
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          for (ii = 0; ii < LANES; ii++)
            d_m[chunk * LANES + ii] <= requant_mant(
              mv_out_m_drain_r[ii*BFP_MANT_W +: BFP_MANT_W],
              mv_out_e_drain_r[ii*BFP_EXP_W  +: BFP_EXP_W ],
              emax_f);
          d_e[chunk] <= emax_f;
          state <= S_DMV_NEXT;
        end
        S_DMV_NEXT: begin
          if (STREAM_WEIGHTS) ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_D - 1) begin
            chunk    <= '0; cnt <= '0; out_cnt <= '0;
            eng_rst  <= 1'b1; rs_start <= 1'b1; state <= S_RES2;
          end else begin
            chunk    <= chunk + 1'b1;
            eng_rst  <= 1'b1; state <= S_DMV_PRIME;
          end
        end

        // ===================================================================
        // Residual 2: hout = h1 + d.  Comb drive via rs_drive_res2.
        // ===================================================================
        S_RES2: begin
          if (rs_in_ready && rs_valid_c) begin
            if (cnt == D-1) state <= S_RES2_WAIT;
            else            cnt <= cnt + 1'b1;
          end
          if (rs_y_valid) begin
            hout_m[out_cnt[CW_D-1:0]] <= rs_y_m;
            if (out_cnt[3:0] == 4'd0)
              hout_e[out_cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            out_cnt <= out_cnt + 1'b1;
          end
        end
        S_RES2_WAIT: begin
          if (rs_y_valid) begin
            hout_m[out_cnt[CW_D-1:0]] <= rs_y_m;
            if (out_cnt[3:0] == 4'd0)
              hout_e[out_cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            out_cnt <= out_cnt + 1'b1;
          end
          if (rs_done) state <= S_DONE;
        end

        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  always_comb dbg_state = state;
  always_comb dbg_cnt   = cnt;
  always_comb dbg_chunk = chunk;

`ifdef LBFP_STAGE_DUMP
  // Sim-only intermediate dumps.  $writememh executes on the cycle the
  // FSM transitions OUT of each post-stage state.  Vivado will warn-skip
  // these; Verilator writes the listed array to a hex file.
  logic [5:0] prev_state;
  always_ff @(posedge clk) begin
    if (rst) prev_state <= 6'h3f;
    else begin
      prev_state <= state;
      if (prev_state == 6'd3 && state == 6'd4) begin
        $writememh("rtl_n1_m.hex", n1_m);
        $writememh("rtl_n1_e.hex", n1_e);
      end
      if (prev_state == 6'd7 && state == 6'd8) begin
        $writememh("rtl_qpre_m.hex", q_m);
        $writememh("rtl_qpre_e.hex", q_e);
      end
      if (prev_state == 6'd11 && state == 6'd12) begin
        $writememh("rtl_kpre_m.hex", k_m);
        $writememh("rtl_kpre_e.hex", k_e);
      end
      if (prev_state == 6'd15 && state == 6'd16) begin
        $writememh("rtl_v_m.hex", v_m);
        $writememh("rtl_v_e.hex", v_e);
      end
      // State numbers shifted by +2 vs original because of new S_ROPEQ_RQ + S_ROPEK_RQ states.
      // S_KVWR_M = 22 (was 20) — state right after RQ/RoPE is complete.
      if (state == 6'd22 && prev_state == 6'd21) begin
        // Post-rope-requant Q + K
        $writememh("rtl_q_m.hex", q_m);
        $writememh("rtl_q_e.hex", q_e);
        $writememh("rtl_k_m.hex", k_m);
        $writememh("rtl_k_e.hex", k_e);
      end
      // After attention (transition to S_OMV_PRIME)
      if (prev_state == 6'd32 && state == 6'd33) begin
        $writememh("rtl_attn_m.hex", attn_m);
        $writememh("rtl_attn_e.hex", attn_e);
      end
      // After softmax wait (S_SM_WAIT=28 → S_AV_PRIME=29): scores + probs ready
      if (prev_state == 6'd28 && state == 6'd29) begin
        $writememh("rtl_scores_m.hex", scores_m);
        $writememh("rtl_qkscore_e.hex", qk_score_e);
        $writememh("rtl_probs_m.hex", probs_m);
        $display("scores_e_shared=%0d probs_e_shared=%0d",
                 $signed(scores_e_shared), $signed(probs_e_shared));
      end
      // After O matvec (transition to S_RES1)
      if (prev_state == 6'd36 && state == 6'd37) begin
        $writememh("rtl_o_m.hex", o_m);
        $writememh("rtl_o_e.hex", o_e);
      end
      // After RES1 (transition to S_NORM2)
      if (prev_state == 6'd38 && state == 6'd39) begin
        $writememh("rtl_h1_m.hex", h1_m);
        $writememh("rtl_h1_e.hex", h1_e);
      end
      // After NORM2 (transition to S_GMV)
      if (prev_state == 6'd40 && state == 6'd41) begin
        $writememh("rtl_n2_m.hex", n2_m);
        $writememh("rtl_n2_e.hex", n2_e);
      end
      // After GMV
      if (prev_state == 6'd44 && state == 6'd45) begin
        $writememh("rtl_g_m.hex", g_m);
        $writememh("rtl_g_e.hex", g_e);
      end
      // After UMV
      if (prev_state == 6'd48 && state == 6'd49) begin
        $writememh("rtl_u_m.hex", u_m);
        $writememh("rtl_u_e.hex", u_e);
      end
      // After SWG
      if (prev_state == 6'd50 && state == 6'd51) begin
        $writememh("rtl_mlp_m.hex", mlp_m);
        $writememh("rtl_mlp_e.hex", mlp_e);
      end
      // After DMV
      if (prev_state == 6'd54 && state == 6'd55) begin
        $writememh("rtl_d_m.hex", d_m);
        $writememh("rtl_d_e.hex", d_e);
      end
    end
  end
`endif

endmodule

`default_nettype wire
