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
  parameter        PREFIX  = "lbfp_"
)(
  input  wire                                clk,
  input  wire                                rst,
  input  wire                                start,
  input  wire [10:0]                         pos,
  input  wire [4:0]                          kv_pos,
  input  wire signed [D*BFP_MANT_W-1:0]      hidden_in_m,
  input  wire signed [(D/BFP_TILE)*BFP_EXP_W-1:0] hidden_in_e,
  output logic signed [D*BFP_MANT_W-1:0]     hidden_out_m,
  output logic signed [(D/BFP_TILE)*BFP_EXP_W-1:0] hidden_out_e,
  output logic                               done,
  // Debug taps (synthesis-friendly — only used by sim testbench)
  output logic [5:0]                         dbg_state,
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

  // Weight ROM sizes (per chunk × per col × per lane)
  localparam int WQ_M_SIZE  = CHUNKS_D   * D   * LANES;
  localparam int WK_M_SIZE  = CHUNKS_KV  * D   * LANES;
  localparam int WV_M_SIZE  = CHUNKS_KV  * D   * LANES;
  localparam int WO_M_SIZE  = CHUNKS_D   * D   * LANES;
  localparam int WG_M_SIZE  = CHUNKS_FFN * D   * LANES;
  localparam int WU_M_SIZE  = CHUNKS_FFN * D   * LANES;
  localparam int WDN_M_SIZE = CHUNKS_D   * FFN * LANES;

  localparam int WQ_E_SIZE  = CHUNKS_D   * NT_D   * LANES;
  localparam int WK_E_SIZE  = CHUNKS_KV  * NT_D   * LANES;
  localparam int WV_E_SIZE  = CHUNKS_KV  * NT_D   * LANES;
  localparam int WO_E_SIZE  = CHUNKS_D   * NT_D   * LANES;
  localparam int WG_E_SIZE  = CHUNKS_FFN * NT_D   * LANES;
  localparam int WU_E_SIZE  = CHUNKS_FFN * NT_D   * LANES;
  localparam int WDN_E_SIZE = CHUNKS_D   * NT_FFN * LANES;

  // ---------------------------------------------------------------------------
  // Weight ROMs
  // ---------------------------------------------------------------------------
  logic signed [BFP_MANT_W-1:0] rom_WQ_m  [0:WQ_M_SIZE-1];
  logic signed [BFP_MANT_W-1:0] rom_WK_m  [0:WK_M_SIZE-1];
  logic signed [BFP_MANT_W-1:0] rom_WV_m  [0:WV_M_SIZE-1];
  logic signed [BFP_MANT_W-1:0] rom_WO_m  [0:WO_M_SIZE-1];
  logic signed [BFP_MANT_W-1:0] rom_WG_m  [0:WG_M_SIZE-1];
  logic signed [BFP_MANT_W-1:0] rom_WU_m  [0:WU_M_SIZE-1];
  logic signed [BFP_MANT_W-1:0] rom_WDN_m [0:WDN_M_SIZE-1];

  logic signed [BFP_EXP_W -1:0] rom_WQ_e  [0:WQ_E_SIZE-1];
  logic signed [BFP_EXP_W -1:0] rom_WK_e  [0:WK_E_SIZE-1];
  logic signed [BFP_EXP_W -1:0] rom_WV_e  [0:WV_E_SIZE-1];
  logic signed [BFP_EXP_W -1:0] rom_WO_e  [0:WO_E_SIZE-1];
  logic signed [BFP_EXP_W -1:0] rom_WG_e  [0:WG_E_SIZE-1];
  logic signed [BFP_EXP_W -1:0] rom_WU_e  [0:WU_E_SIZE-1];
  logic signed [BFP_EXP_W -1:0] rom_WDN_e [0:WDN_E_SIZE-1];

  logic signed [BFP_MANT_W-1:0] rom_G1_m [0:D-1];
  logic signed [BFP_MANT_W-1:0] rom_G2_m [0:D-1];
  logic signed [BFP_EXP_W -1:0] rom_G1_e [0:NT_D-1];
  logic signed [BFP_EXP_W -1:0] rom_G2_e [0:NT_D-1];

  logic signed [BFP_MANT_W-1:0] kv_k_m [0:MAX_CTX*H_KV*HD-1];
  logic signed [BFP_MANT_W-1:0] kv_v_m [0:MAX_CTX*H_KV*HD-1];
  logic signed [BFP_EXP_W -1:0] kv_k_e [0:MAX_CTX*NT_KV-1];
  logic signed [BFP_EXP_W -1:0] kv_v_e [0:MAX_CTX*NT_KV-1];

`ifdef MICROGPT_WEIGHT_DIR
  initial begin
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
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "G1_m.hex"},  rom_G1_m);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "G2_m.hex"},  rom_G2_m);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "G1_e.hex"},  rom_G1_e);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "G2_e.hex"},  rom_G2_e);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "K_INIT_m.hex"}, kv_k_m);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "V_INIT_m.hex"}, kv_v_m);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "K_INIT_e.hex"}, kv_k_e);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "V_INIT_e.hex"}, kv_v_e);
  end
`else
  initial begin
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
    $readmemh({PREFIX, "G1_m.hex"},  rom_G1_m);
    $readmemh({PREFIX, "G2_m.hex"},  rom_G2_m);
    $readmemh({PREFIX, "G1_e.hex"},  rom_G1_e);
    $readmemh({PREFIX, "G2_e.hex"},  rom_G2_e);
    $readmemh({PREFIX, "K_INIT_m.hex"}, kv_k_m);
    $readmemh({PREFIX, "V_INIT_m.hex"}, kv_v_m);
    $readmemh({PREFIX, "K_INIT_e.hex"}, kv_k_e);
    $readmemh({PREFIX, "V_INIT_e.hex"}, kv_v_e);
  end
`endif

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
  typedef enum logic [5:0] {
    S_IDLE,
    S_LATCH_IN,
    S_NORM1, S_NORM1_WAIT,
    S_QMV_PRIME, S_QMV_DRIVE, S_QMV_DRAIN, S_QMV_NEXT,
    S_KMV_PRIME, S_KMV_DRIVE, S_KMV_DRAIN, S_KMV_NEXT,
    S_VMV_PRIME, S_VMV_DRIVE, S_VMV_DRAIN, S_VMV_NEXT,
    S_ROPEQ, S_ROPEQ_WAIT, S_ROPEQ_RQ,
    S_ROPEK, S_ROPEK_WAIT, S_ROPEK_RQ,
    S_KVWR_M, S_KVWR_E,
    S_QK_PRIME, S_QK_DRIVE, S_QK_DRAIN,
    S_SM_DRIVE, S_SM_WAIT,
    S_AV_PRIME, S_AV_DRIVE, S_AV_DRAIN, S_AV_NEXT,
    S_OMV_PRIME, S_OMV_DRIVE, S_OMV_DRAIN, S_OMV_NEXT,
    S_RES1, S_RES1_WAIT,
    S_NORM2, S_NORM2_WAIT,
    S_GMV_PRIME, S_GMV_DRIVE, S_GMV_DRAIN, S_GMV_NEXT,
    S_UMV_PRIME, S_UMV_DRIVE, S_UMV_DRAIN, S_UMV_NEXT,
    S_SWG, S_SWG_WAIT,
    S_DMV_PRIME, S_DMV_DRIVE, S_DMV_DRAIN, S_DMV_NEXT,
    S_RES2, S_RES2_WAIT,
    S_DONE
  } state_t;
  state_t state;

  // Generic counters
  logic [11:0]      cnt;
  logic [6:0]       chunk;
  logic [4:0]       head_idx;
  logic [4:0]       kv_t;
  logic [BFP_EXP_W-1:0] score_emax;

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

  matvec_bfp_engine #(.LANES(LANES)) i_mv (
    .clk(clk), .rst(rst | eng_rst),
    .start_matvec(mv_start),
    .in_x_mant(mv_x_m), .in_x_exp(mv_x_e),
    .in_valid(mv_valid), .last_elem(mv_last),
    .w_mant(mv_w_m), .w_exp(mv_w_e),
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

  logic                                sg_start, sg_valid, sg_last;
  logic signed [BFP_MANT_W-1:0]        sg_g_m, sg_u_m;
  logic signed [BFP_EXP_W -1:0]        sg_g_e, sg_u_e;
  wire                                 sg_in_ready;
  wire  signed [BFP_MANT_W-1:0]        sg_y_m;
  wire  signed [BFP_EXP_W -1:0]        sg_y_e;
  wire                                 sg_y_valid, sg_done;

  swiglu_bfp #(.D(FFN)) i_sg (
    .clk(clk), .rst(rst),
    .start(sg_start),
    .in_gate_mant(sg_g_m), .in_gate_exp(sg_g_e),
    .in_up_mant(sg_u_m),   .in_up_exp(sg_u_e),
    .in_valid(sg_valid), .last_elem(sg_last),
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

  logic                                rs_start, rs_valid, rs_last;
  logic signed [BFP_MANT_W-1:0]        rs_a_m, rs_b_m;
  logic signed [BFP_EXP_W -1:0]        rs_a_e, rs_b_e;
  wire                                 rs_in_ready;
  wire  signed [BFP_MANT_W-1:0]        rs_y_m;
  wire  signed [BFP_EXP_W -1:0]        rs_y_e;
  wire                                 rs_y_valid, rs_done;

  residual_bfp #(.D(D)) i_rs (
    .clk(clk), .rst(rst),
    .start(rs_start),
    .in_a_mant(rs_a_m), .in_a_exp(rs_a_e),
    .in_b_mant(rs_b_m), .in_b_exp(rs_b_e),
    .in_valid(rs_valid), .last_elem(rs_last), .in_ready(rs_in_ready),
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
  // Main FSM — all writes use non-blocking assignment; engine inputs are
  // driven from registers so they pipeline-align with mv_valid one cycle
  // later (matvec engine samples on the clock edge after we register).
  // ---------------------------------------------------------------------------
  integer ii;       // for-loop scratch in always_ff (synth-safe with int loop body)
  logic [4:0] head_grp;
  logic [11:0] av_row_base;      // current AV row offset = head*HD + chunk*LANES

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
      sg_valid        <= 1'b0;
      sg_last         <= 1'b0;
      sg_g_m          <= '0;
      sg_g_e          <= '0;
      sg_u_m          <= '0;
      sg_u_e          <= '0;
      sm_start        <= 1'b0;
      sm_valid        <= 1'b0;
      sm_n            <= '0;
      sm_x_m          <= '0;
      sm_x_e          <= '0;
      rs_start        <= 1'b0;
      rs_valid        <= 1'b0;
      rs_last         <= 1'b0;
      rs_a_m          <= '0;
      rs_a_e          <= '0;
      rs_b_m          <= '0;
      rs_b_e          <= '0;
      scores_e_shared <= '0;
      probs_e_shared  <= '0;
      head_grp        <= '0;
      av_row_base     <= '0;
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
      sg_valid <= 1'b0;
      sg_last  <= 1'b0;
      sm_start <= 1'b0;
      sm_valid <= 1'b0;
      rs_start <= 1'b0;
      rs_valid <= 1'b0;
      rs_last  <= 1'b0;
      eng_rst  <= 1'b0;
      done     <= 1'b0;

      case (state)
        // -------------------------------------------------------------------
        S_IDLE: if (start) begin
          state   <= S_LATCH_IN;
          cnt     <= '0;
          eng_rst <= 1'b1;
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
          rn_g_m   <= rom_G1_m[cnt[CW_D-1:0]];
          rn_g_e   <= rom_G1_e[cnt[CW_D-1:0] / BFP_TILE];
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
          mv_start <= 1'b1;
          cnt      <= '0;
          state    <= S_QMV_DRIVE;
        end
        S_QMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= n1_m[cnt[CW_D-1:0]];
          mv_x_e   <= n1_e[cnt[CW_D-1:0] / BFP_TILE];
          for (ii = 0; ii < LANES; ii++) begin
            mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <=
              rom_WQ_m[((chunk * D + cnt[CW_D-1:0]) * LANES) + ii];
            mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <=
              rom_WQ_e[((chunk * NT_D + cnt[CW_D-1:0]/BFP_TILE) * LANES) + ii];
          end
          mv_last <= (cnt == D-1);
          if (cnt == D-1) state <= S_QMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_QMV_DRAIN: if (mv_out_valid) begin
          begin : qmv_requant
            automatic logic signed [BFP_EXP_W-1:0] emax;
            emax = $signed(mv_out_e[0 +: BFP_EXP_W]);
            for (ii = 1; ii < LANES; ii++)
              if ($signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]) > emax)
                emax = $signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]);
            for (ii = 0; ii < LANES; ii++)
              q_m[chunk * LANES + ii] <= requant_mant(
                mv_out_m[ii*BFP_MANT_W +: BFP_MANT_W],
                mv_out_e[ii*BFP_EXP_W  +: BFP_EXP_W ],
                emax);
            q_e[chunk] <= emax;
          end
          state <= S_QMV_NEXT;
        end
        S_QMV_NEXT: begin
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
          mv_start <= 1'b1; cnt <= '0; state <= S_KMV_DRIVE;
        end
        S_KMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= n1_m[cnt[CW_D-1:0]];
          mv_x_e   <= n1_e[cnt[CW_D-1:0] / BFP_TILE];
          for (ii = 0; ii < LANES; ii++) begin
            mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <=
              rom_WK_m[((chunk * D + cnt[CW_D-1:0]) * LANES) + ii];
            mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <=
              rom_WK_e[((chunk * NT_D + cnt[CW_D-1:0]/BFP_TILE) * LANES) + ii];
          end
          mv_last <= (cnt == D-1);
          if (cnt == D-1) state <= S_KMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_KMV_DRAIN: if (mv_out_valid) begin
          begin : kmv_requant
            automatic logic signed [BFP_EXP_W-1:0] emax;
            emax = $signed(mv_out_e[0 +: BFP_EXP_W]);
            for (ii = 1; ii < LANES; ii++)
              if ($signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]) > emax)
                emax = $signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]);
            for (ii = 0; ii < LANES; ii++)
              k_m[chunk * LANES + ii] <= requant_mant(
                mv_out_m[ii*BFP_MANT_W +: BFP_MANT_W],
                mv_out_e[ii*BFP_EXP_W  +: BFP_EXP_W ],
                emax);
            k_e[chunk] <= emax;
          end
          state <= S_KMV_NEXT;
        end
        S_KMV_NEXT: begin
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
          mv_start <= 1'b1; cnt <= '0; state <= S_VMV_DRIVE;
        end
        S_VMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= n1_m[cnt[CW_D-1:0]];
          mv_x_e   <= n1_e[cnt[CW_D-1:0] / BFP_TILE];
          for (ii = 0; ii < LANES; ii++) begin
            mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <=
              rom_WV_m[((chunk * D + cnt[CW_D-1:0]) * LANES) + ii];
            mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <=
              rom_WV_e[((chunk * NT_D + cnt[CW_D-1:0]/BFP_TILE) * LANES) + ii];
          end
          mv_last <= (cnt == D-1);
          if (cnt == D-1) state <= S_VMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_VMV_DRAIN: if (mv_out_valid) begin
          begin : vmv_requant
            automatic logic signed [BFP_EXP_W-1:0] emax;
            emax = $signed(mv_out_e[0 +: BFP_EXP_W]);
            for (ii = 1; ii < LANES; ii++)
              if ($signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]) > emax)
                emax = $signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]);
            for (ii = 0; ii < LANES; ii++)
              v_m[chunk * LANES + ii] <= requant_mant(
                mv_out_m[ii*BFP_MANT_W +: BFP_MANT_W],
                mv_out_e[ii*BFP_EXP_W  +: BFP_EXP_W ],
                emax);
            v_e[chunk] <= emax;
          end
          state <= S_VMV_NEXT;
        end
        S_VMV_NEXT: begin
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
              state <= S_ROPEQ_RQ;
            end else begin
              head_idx <= head_idx + 1'b1; cnt <= '0;
              rp_start <= 1'b1; state <= S_ROPEQ;
            end
          end
        end

        // Re-tile-quantize q_rot (per-element exp) into per-tile q_m / q_e.
        // For each tile: emax = max(q_rot_e in tile); for each elem in tile,
        // q_m = q_rot_m >>> (emax - q_rot_e_i); q_e = emax.  cnt steps tile.
        S_ROPEQ_RQ: begin
          begin : rq_q
            automatic logic signed [BFP_EXP_W-1:0] emax;
            automatic int base;
            base = cnt[$clog2(NT_D+1)-1:0] * BFP_TILE;
            emax = -8'sd128;
            for (ii = 0; ii < BFP_TILE; ii++)
              if (base + ii < D)
                if (q_rot_e[base + ii] > emax) emax = q_rot_e[base + ii];
            for (ii = 0; ii < BFP_TILE; ii++) begin
              if (base + ii < D)
                q_m[base + ii] <= requant_mant(q_rot_m[base + ii],
                                               q_rot_e[base + ii], emax);
            end
            q_e[cnt[$clog2(NT_D+1)-1:0]] <= emax;
          end
          if (cnt == NT_D - 1) begin
            head_idx <= '0; cnt <= '0;
            rp_start <= 1'b1; state <= S_ROPEK;
          end else cnt <= cnt + 1'b1;
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
              state <= S_ROPEK_RQ;
            end else begin
              head_idx <= head_idx + 1'b1; cnt <= '0;
              rp_start <= 1'b1; state <= S_ROPEK;
            end
          end
        end

        // Re-tile-quantize k_rot.  Writes back per-tile (m, e) into k_m / k_e
        // — overwriting the pre-rope K vector since QK uses post-rope K only.
        S_ROPEK_RQ: begin
          begin : rq_k
            automatic logic signed [BFP_EXP_W-1:0] emax;
            automatic int base;
            base = cnt[$clog2(NT_KV+1)-1:0] * BFP_TILE;
            emax = -8'sd128;
            for (ii = 0; ii < BFP_TILE; ii++)
              if (base + ii < H_KV*HD)
                if (k_rot_e[base + ii] > emax) emax = k_rot_e[base + ii];
            for (ii = 0; ii < BFP_TILE; ii++) begin
              if (base + ii < H_KV*HD)
                k_m[base + ii] <= requant_mant(k_rot_m[base + ii],
                                                k_rot_e[base + ii], emax);
            end
            k_e[cnt[$clog2(NT_KV+1)-1:0]] <= emax;
          end
          if (cnt == NT_KV - 1) begin
            cnt <= '0;
            state <= S_KVWR_M;
          end else cnt <= cnt + 1'b1;
        end

        // ===================================================================
        // KV-cache write: store k_rot and v (no rope for V) at slot kv_pos.
        // Phase M: stream HD*H_KV mantissas.  Phase E: collapse rope
        // per-element exps into one shared tile-exp via max.
        // ===================================================================
        // After re-tile-quant, k_m/k_e is the post-rope K in per-tile BFP.
        // Just copy mantissas + per-tile exponents into the KV cache slot.
        S_KVWR_M: begin
          kv_k_m[kv_pos * H_KV * HD + cnt[CW_KV-1:0]] <= k_m[cnt[CW_KV-1:0]];
          kv_v_m[kv_pos * H_KV * HD + cnt[CW_KV-1:0]] <= v_m[cnt[CW_KV-1:0]];
          if (cnt == H_KV*HD - 1) begin
            cnt   <= '0; state <= S_KVWR_E;
          end else cnt <= cnt + 1'b1;
        end
        S_KVWR_E: begin
          kv_k_e[kv_pos * NT_KV + cnt[CW_KV-1:0]] <= k_e[cnt[CW_KV-1:0]];
          kv_v_e[kv_pos * NT_KV + cnt[CW_KV-1:0]] <= v_e[cnt[CW_KV-1:0]];
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
                mv_w_m[0 +: BFP_MANT_W] <= kv_k_m[kv_t * H_KV*HD + head_grp * HD + cnt[CW_HD-1:0]];
                mv_w_e[0 +: BFP_EXP_W ] <= kv_k_e[kv_t * NT_KV + (head_grp * HD + cnt[CW_HD-1:0]) / BFP_TILE];
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
          scores_m  [kv_t] <= mv_out_m[0 +: BFP_MANT_W];
          qk_score_e[kv_t] <= mv_out_e[0 +: BFP_EXP_W ];
          if (mv_out_e[0 +: BFP_EXP_W] > score_emax)
            score_emax <= mv_out_e[0 +: BFP_EXP_W];
          if (kv_t == kv_pos) begin
            scores_e_shared <= (mv_out_e[0 +: BFP_EXP_W] > score_emax)
                              ? mv_out_e[0 +: BFP_EXP_W] : score_emax;
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
        S_AV_PRIME: begin
          mv_start <= 1'b1; cnt <= '0; state <= S_AV_DRIVE;
          head_grp <= 5'(kv_grp_of(head_idx));
        end
        // matvec engine requires input dim to be a multiple of TILE so that
        // tile_done can fire.  Pad with zeros past kv_pos to the next TILE
        // boundary; mv_last fires on the final padded cycle.
        S_AV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_e   <= probs_e_shared;
          if (cnt <= kv_pos) begin
            mv_x_m <= probs_m[cnt[CW_CTX-1:0]];
            for (ii = 0; ii < LANES; ii++) begin
              if (cnt[CW_CTX-1:0] == kv_pos) begin
                mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <=
                  v_m[head_grp * HD + chunk * LANES + ii];
                mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <=
                  v_e[(head_grp * HD + chunk * LANES + ii) / BFP_TILE];
              end else begin
                mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <=
                  kv_v_m[cnt[CW_CTX-1:0] * H_KV*HD + head_grp * HD + chunk * LANES + ii];
                mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <=
                  kv_v_e[cnt[CW_CTX-1:0] * NT_KV
                         + (head_grp * HD + chunk * LANES + ii) / BFP_TILE];
              end
            end
          end else begin
            // Padding: drive zero mantissas (any exp works; weight zero
            // contributes nothing to the MAC)
            mv_x_m <= '0;
            for (ii = 0; ii < LANES; ii++) begin
              mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= '0;
              mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <= '0;
            end
          end
          mv_last <= (cnt == BFP_TILE - 1);
          if (cnt == BFP_TILE - 1) state <= S_AV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_AV_DRAIN: if (mv_out_valid) begin
          begin : av_requant
            automatic logic signed [BFP_EXP_W-1:0] emax;
            emax = $signed(mv_out_e[0 +: BFP_EXP_W]);
            for (ii = 1; ii < LANES; ii++)
              if ($signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]) > emax)
                emax = $signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]);
            for (ii = 0; ii < LANES; ii++)
              attn_m[av_row_base + chunk * LANES + ii] <= requant_mant(
                mv_out_m[ii*BFP_MANT_W +: BFP_MANT_W],
                mv_out_e[ii*BFP_EXP_W  +: BFP_EXP_W ],
                emax);
            attn_e[(av_row_base + chunk * LANES) / BFP_TILE] <= emax;
          end
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
          mv_start <= 1'b1; cnt <= '0; state <= S_OMV_DRIVE;
        end
        S_OMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= attn_m[cnt[CW_D-1:0]];
          mv_x_e   <= attn_e[cnt[CW_D-1:0] / BFP_TILE];
          for (ii = 0; ii < LANES; ii++) begin
            mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <=
              rom_WO_m[((chunk * D + cnt[CW_D-1:0]) * LANES) + ii];
            mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <=
              rom_WO_e[((chunk * NT_D + cnt[CW_D-1:0]/BFP_TILE) * LANES) + ii];
          end
          mv_last <= (cnt == D-1);
          if (cnt == D-1) state <= S_OMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_OMV_DRAIN: if (mv_out_valid) begin
          begin : omv_requant
            automatic logic signed [BFP_EXP_W-1:0] emax;
            emax = $signed(mv_out_e[0 +: BFP_EXP_W]);
            for (ii = 1; ii < LANES; ii++)
              if ($signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]) > emax)
                emax = $signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]);
            for (ii = 0; ii < LANES; ii++)
              o_m[chunk * LANES + ii] <= requant_mant(
                mv_out_m[ii*BFP_MANT_W +: BFP_MANT_W],
                mv_out_e[ii*BFP_EXP_W  +: BFP_EXP_W ],
                emax);
            o_e[chunk] <= emax;
          end
          state <= S_OMV_NEXT;
        end
        S_OMV_NEXT: begin
          if (chunk == CHUNKS_D - 1) begin
            chunk   <= '0; cnt <= '0;
            eng_rst <= 1'b1; rs_start <= 1'b1; state <= S_RES1;
          end else begin
            chunk   <= chunk + 1'b1;
            eng_rst <= 1'b1; state <= S_OMV_PRIME;
          end
        end

        // ===================================================================
        // Residual 1: h1 = hin + o
        // ===================================================================
        S_RES1: begin
          rs_valid <= rs_in_ready;
          rs_a_m   <= hin_m[cnt[CW_D-1:0]];
          rs_a_e   <= hin_e[cnt[CW_D-1:0] / BFP_TILE];
          rs_b_m   <= o_m  [cnt[CW_D-1:0]];
          rs_b_e   <= o_e  [cnt[CW_D-1:0] / BFP_TILE];
          rs_last  <= rs_in_ready && (cnt == D-1);
          if (rs_in_ready) begin
            if (cnt == D-1) begin
              state <= S_RES1_WAIT; cnt <= '0;
            end else cnt <= cnt + 1'b1;
          end
        end
        S_RES1_WAIT: begin
          if (rs_y_valid) begin
            h1_m[cnt[CW_D-1:0]] <= rs_y_m;
            if (cnt[3:0] == 4'd0)
              h1_e[cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            cnt <= cnt + 1'b1;
          end
          if (rs_done) begin
            state    <= S_NORM2;
            eng_rst  <= 1'b1;
            cnt      <= '0;
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
          rn_g_m   <= rom_G2_m[cnt[CW_D-1:0]];
          rn_g_e   <= rom_G2_e[cnt[CW_D-1:0] / BFP_TILE];
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
          mv_start <= 1'b1; cnt <= '0; state <= S_GMV_DRIVE;
        end
        S_GMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= n2_m[cnt[CW_D-1:0]];
          mv_x_e   <= n2_e[cnt[CW_D-1:0] / BFP_TILE];
          for (ii = 0; ii < LANES; ii++) begin
            mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <=
              rom_WG_m[((chunk * D + cnt[CW_D-1:0]) * LANES) + ii];
            mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <=
              rom_WG_e[((chunk * NT_D + cnt[CW_D-1:0]/BFP_TILE) * LANES) + ii];
          end
          mv_last <= (cnt == D-1);
          if (cnt == D-1) state <= S_GMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_GMV_DRAIN: if (mv_out_valid) begin
          begin : gmv_requant
            automatic logic signed [BFP_EXP_W-1:0] emax;
            emax = $signed(mv_out_e[0 +: BFP_EXP_W]);
            for (ii = 1; ii < LANES; ii++)
              if ($signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]) > emax)
                emax = $signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]);
            for (ii = 0; ii < LANES; ii++)
              g_m[chunk * LANES + ii] <= requant_mant(
                mv_out_m[ii*BFP_MANT_W +: BFP_MANT_W],
                mv_out_e[ii*BFP_EXP_W  +: BFP_EXP_W ],
                emax);
            g_e[chunk] <= emax;
          end
          state <= S_GMV_NEXT;
        end
        S_GMV_NEXT: begin
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
          mv_start <= 1'b1; cnt <= '0; state <= S_UMV_DRIVE;
        end
        S_UMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= n2_m[cnt[CW_D-1:0]];
          mv_x_e   <= n2_e[cnt[CW_D-1:0] / BFP_TILE];
          for (ii = 0; ii < LANES; ii++) begin
            mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <=
              rom_WU_m[((chunk * D + cnt[CW_D-1:0]) * LANES) + ii];
            mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <=
              rom_WU_e[((chunk * NT_D + cnt[CW_D-1:0]/BFP_TILE) * LANES) + ii];
          end
          mv_last <= (cnt == D-1);
          if (cnt == D-1) state <= S_UMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_UMV_DRAIN: if (mv_out_valid) begin
          begin : umv_requant
            automatic logic signed [BFP_EXP_W-1:0] emax;
            emax = $signed(mv_out_e[0 +: BFP_EXP_W]);
            for (ii = 1; ii < LANES; ii++)
              if ($signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]) > emax)
                emax = $signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]);
            for (ii = 0; ii < LANES; ii++)
              u_m[chunk * LANES + ii] <= requant_mant(
                mv_out_m[ii*BFP_MANT_W +: BFP_MANT_W],
                mv_out_e[ii*BFP_EXP_W  +: BFP_EXP_W ],
                emax);
            u_e[chunk] <= emax;
          end
          state <= S_UMV_NEXT;
        end
        S_UMV_NEXT: begin
          if (chunk == CHUNKS_FFN - 1) begin
            chunk <= '0; cnt <= '0; eng_rst <= 1'b1; sg_start <= 1'b1; state <= S_SWG;
          end else begin
            chunk <= chunk + 1'b1; eng_rst <= 1'b1; state <= S_UMV_PRIME;
          end
        end

        // ===================================================================
        // SwiGLU (FFN -> FFN)
        // ===================================================================
        S_SWG: begin
          sg_valid <= sg_in_ready;
          sg_g_m   <= g_m[cnt[CW_FFN-1:0]];
          sg_g_e   <= g_e[cnt[CW_FFN-1:0] / BFP_TILE];
          sg_u_m   <= u_m[cnt[CW_FFN-1:0]];
          sg_u_e   <= u_e[cnt[CW_FFN-1:0] / BFP_TILE];
          sg_last  <= sg_in_ready && (cnt == FFN-1);
          if (sg_in_ready) begin
            if (cnt == FFN-1) begin
              state <= S_SWG_WAIT; cnt <= '0;
            end else cnt <= cnt + 1'b1;
          end
        end
        S_SWG_WAIT: begin
          if (sg_y_valid) begin
            mlp_m[cnt[CW_FFN-1:0]] <= sg_y_m;
            if (cnt[3:0] == 4'd0)
              mlp_e[cnt[CW_FFN-1:0] / BFP_TILE] <= sg_y_e;
            cnt <= cnt + 1'b1;
          end
          if (sg_done) begin
            chunk <= '0; eng_rst <= 1'b1; state <= S_DMV_PRIME;
          end
        end

        // ===================================================================
        // Down matvec (FFN in, D out)
        // ===================================================================
        S_DMV_PRIME: begin
          mv_start <= 1'b1; cnt <= '0; state <= S_DMV_DRIVE;
        end
        S_DMV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= mlp_m[cnt[CW_FFN-1:0]];
          mv_x_e   <= mlp_e[cnt[CW_FFN-1:0] / BFP_TILE];
          for (ii = 0; ii < LANES; ii++) begin
            mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <=
              rom_WDN_m[((chunk * FFN + cnt[CW_FFN-1:0]) * LANES) + ii];
            mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <=
              rom_WDN_e[((chunk * NT_FFN + cnt[CW_FFN-1:0]/BFP_TILE) * LANES) + ii];
          end
          mv_last <= (cnt == FFN-1);
          if (cnt == FFN-1) state <= S_DMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_DMV_DRAIN: if (mv_out_valid) begin
          begin : dmv_requant
            automatic logic signed [BFP_EXP_W-1:0] emax;
            emax = $signed(mv_out_e[0 +: BFP_EXP_W]);
            for (ii = 1; ii < LANES; ii++)
              if ($signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]) > emax)
                emax = $signed(mv_out_e[ii*BFP_EXP_W +: BFP_EXP_W]);
            for (ii = 0; ii < LANES; ii++)
              d_m[chunk * LANES + ii] <= requant_mant(
                mv_out_m[ii*BFP_MANT_W +: BFP_MANT_W],
                mv_out_e[ii*BFP_EXP_W  +: BFP_EXP_W ],
                emax);
            d_e[chunk] <= emax;
          end
          state <= S_DMV_NEXT;
        end
        S_DMV_NEXT: begin
          if (chunk == CHUNKS_D - 1) begin
            chunk    <= '0; cnt <= '0;
            eng_rst  <= 1'b1; rs_start <= 1'b1; state <= S_RES2;
          end else begin
            chunk    <= chunk + 1'b1;
            eng_rst  <= 1'b1; state <= S_DMV_PRIME;
          end
        end

        // ===================================================================
        // Residual 2: hout = h1 + d
        // ===================================================================
        S_RES2: begin
          rs_valid <= rs_in_ready;
          rs_a_m   <= h1_m[cnt[CW_D-1:0]];
          rs_a_e   <= h1_e[cnt[CW_D-1:0] / BFP_TILE];
          rs_b_m   <= d_m [cnt[CW_D-1:0]];
          rs_b_e   <= d_e [cnt[CW_D-1:0] / BFP_TILE];
          rs_last  <= rs_in_ready && (cnt == D-1);
          if (rs_in_ready) begin
            if (cnt == D-1) begin
              state <= S_RES2_WAIT; cnt <= '0;
            end else cnt <= cnt + 1'b1;
          end
        end
        S_RES2_WAIT: begin
          if (rs_y_valid) begin
            hout_m[cnt[CW_D-1:0]] <= rs_y_m;
            if (cnt[3:0] == 4'd0)
              hout_e[cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            cnt <= cnt + 1'b1;
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

endmodule

`default_nettype wire
