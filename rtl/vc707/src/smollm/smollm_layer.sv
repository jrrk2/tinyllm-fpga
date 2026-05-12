// smollm_layer.sv — orchestrates one SmolLM2-style transformer layer.
//
// Wires the 5 validated leaf ops (matvec_int8_engine, rmsnorm, rope,
// swiglu, softmax_q15) into a complete pre-norm transformer block:
//
//   pre-attn-norm → Q/K/V proj → RoPE Q,K → KV cache write
//                 → Q·Kᵀ/√HD scores → softmax → ·V → O proj → +residual
//   post-attn-norm → gate, up proj → SiLU(gate)*up → down proj → +residual
//
// Single 16-lane matvec engine reused 7 times; out_dim>16 paths run
// multiple chunks.  Test config: D=HD=64, FFN=128, MAX_CTX=4, H_Q=H_KV=1.
// Test inputs (`hidden_in` not used; instead packed-localparam test data
// via `\`include "layer_test_data.svh"`) — for FPGA bring-up we'd take
// data via port and stream weights from BRAM/DDR3.

`default_nettype none

module smollm_layer #(
  parameter int    D       = 576,
  parameter int    H_Q     = 9,
  parameter int    H_KV    = 3,
  parameter int    HD      = 64,
  parameter int    FFN     = 1536,
  parameter int    MAX_CTX = 4,
  parameter int    NL      = 1,           // # layers sharing this instance (time-mux)
  parameter        PREFIX  = "layer_",
  // DDR3 byte offsets of each weight matrix's chunk-0 WITHIN ONE LAYER.
  // The streamer's matrix_base = layer_base_addr + BASE_<matrix>, so the
  // wrapper picks layer_base_addr = layer_idx * LAYER_BYTES at runtime.
  parameter [29:0] BASE_Q    = 30'h00_0000,
  parameter [29:0] BASE_K    = 30'h08_0000,
  parameter [29:0] BASE_V    = 30'h10_0000,
  parameter [29:0] BASE_O    = 30'h18_0000,
  parameter [29:0] BASE_GATE = 30'h20_0000,
  parameter [29:0] BASE_UP   = 30'h28_0000,
  parameter [29:0] BASE_DOWN = 30'h30_0000
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 start,
  input  wire [10:0]          pos,
  input  wire [4:0]           kv_pos,
  // Time-multiplex selectors (default 0 = single-layer behaviour).
  input  wire [4:0]           layer_idx,    // 0..NL-1
  input  wire [29:0]          layer_base_addr,
  // Hidden state: 16-bit Q1.15 with PER-LAYER block-FP scale = 2^p_hid.
  input  wire signed [D*16-1:0]  hidden_in,
  output logic signed [D*16-1:0] hidden_out,
  output logic                done,
  input  wire [23:0]          resid1_factor,
  input  wire [23:0]          resid2_factor,
  input  wire signed [3:0]    sh_h_in_to_h1,
  input  wire signed [3:0]    sh_h1_to_h_out,
  // Scale-aware SwiGLU/Attn-AV factors (per-layer, from tm_layer_swiglu_attn.svh).
  // SiLU LUT is at SILU_LUT_SCALE=32; AV result rescales from lsc[v]→lsc[attn].
  input  wire [15:0]          sg_gate_in_factor,   // lsc[gate]/SILU_LUT_SCALE  Q1.15
  input  wire [15:0]          sg_up_in_factor,     // lsc[up]/SILU_LUT_SCALE    Q1.15
  input  wire [23:0]          sg_mlp_out_factor,   // SILU_LUT_SCALE^2/lsc[mlp] Q16.8
  input  wire [23:0]          attn_factor,         // lsc[v]/lsc[attn]          Q16.8
  // Host-write port for per-row scale brom overrides.  Defaults still
  // load from .INIT_xx params at config; runtime writes overlay specific
  // (matrix, entry) slots.  kind: 0=Q 1=K 2=V 3=O 4=GATE 5=UP 6=DOWN.
  input  wire [3:0]           scale_wr_kind,
  input  wire [15:0]          scale_wr_addr,
  input  wire [15:0]          scale_wr_data,
  input  wire                 scale_wr_en
`ifdef MICROGPT_DDR3_WEIGHTS
  ,
  // ILA probe taps — named exports, wired to ila instances in the top.
  output wire  [2:0]    ila_ws_state_axi,        // streamer FSM (clk_axi)
  output wire           ila_start_load_axi,
  output wire  [1:0]    ila_tile_idx,
  output wire  [6:0]    ila_beat_idx,
  output logic [4:0]    ila_state,
  output logic [2:0]    ila_mv_phase,
  output logic [10:0]   ila_cnt,
  output logic [6:0]    ila_chunk,
  output logic [2:0]    ila_ws_matvec_id,
  output logic          ila_ws_load_req,
  output logic          ila_ws_ready,
  output logic [10:0]   ila_ws_rd_addr,
  output logic [127:0]  ila_ws_weight_data,
  output logic [127:0]  ila_eng_w,
  output logic [15:0]   ila_eng_in_value,
  output logic          ila_eng_in_valid,
  output logic          ila_eng_in_last,
  output logic          ila_eng_acc_clear,
  output logic          ila_eng_out_valid,
  // Diagnostics:
  //   dbg_first_araddr      : first AXI read address the streamer issues
  //   dbg_first_rdata       : first AXI beat returned (= layer_DDR3 entries 0..3)
  //   dbg_first_ar/r_seen   : sticky flags
  //   dbg_eng_w_packed[i]   : eng_w at S_M_Q chunk 0 MV_DRIVE cnt=i+1 (= W[i])
  //   dbg_wd_packed[i]      : ws_weight_data at same instant
  //   dbg_in_value_packed[i]: eng_in_value at same instant
  //   dbg_snap_done_o       : 1 when 4 captures have been taken
  output logic [29:0]                dbg_first_araddr,
  output logic [511:0]               dbg_first_rdata,
  output logic                       dbg_first_ar_seen,
  output logic                       dbg_first_r_seen,
  output logic [511:0]               dbg_eng_w_packed,
  output logic [511:0]               dbg_wd_packed,
  output logic [63:0]                dbg_in_value_packed,
  output logic                       dbg_snap_done_o,
  // AXI clock + reset — drives weight_streamer_mt's loader at MIG's ui_clk
  // rate.  In Verilator sim we tie clk_axi=clk and rst_axi=rst.
  input  wire                        clk_axi,
  input  wire                        rst_axi,
  // AXI4 read master to MIG.
  output logic                       m_axi_arvalid,
  input  wire                        m_axi_arready,
  output logic [4:0]                 m_axi_arid,
  output logic [29:0]                m_axi_araddr,
  output logic [7:0]                 m_axi_arlen,
  output logic [2:0]                 m_axi_arsize,
  output logic [1:0]                 m_axi_arburst,
  output logic                       m_axi_arlock,
  output logic [3:0]                 m_axi_arcache,
  output logic [2:0]                 m_axi_arprot,
  output logic [3:0]                 m_axi_arqos,
  input  wire                        m_axi_rvalid,
  output wire                        m_axi_rready,
  input  wire  [4:0]                 m_axi_rid,
  input  wire  [511:0]               m_axi_rdata,
  input  wire  [1:0]                 m_axi_rresp,
  input  wire                        m_axi_rlast
`endif
`ifdef DEBUG
  ,
  // Trace ports for the Verilator integration test.  Synthesised away
  // for FPGA builds (DEBUG is a Verilator-only define).
  output logic signed [D*16-1:0]      trace_norm1,
  output logic signed [D*16-1:0]      trace_q,
  output logic signed [H_KV*HD*16-1:0] trace_k,
  output logic signed [H_KV*HD*16-1:0] trace_v,
  output logic signed [D*16-1:0]      trace_q_rot,
  output logic signed [H_KV*HD*16-1:0] trace_k_rot,
  output logic signed [D*16-1:0]      trace_attn,
  output logic signed [D*16-1:0]      trace_hidden1,
  output logic signed [D*16-1:0]      trace_norm2,
  output logic signed [FFN*16-1:0]    trace_mlp_inter
`endif
);

  // Test data — only LAYER_POS is in the SVH; everything else loads from
  // hex files via $readmemh into unpacked arrays.  Packed-localparam selects
  // crash Vivado at full FFN width.
`include "layer_test_data.svh"

  // Hex-loaded ROMs.  Phase D-real: weights now live in the
  // weight_streamer_brom instance (Verilator) / weight_streamer.sv (FPGA).
  // Per-layer ROMs sized NL × per_layer_count for time-mux.
  //
  // Scales (rom_S_*): explicit RAMB36E1 primitives via host-generated
  //   brom_S_*.sv wrappers — Vivado was flattening the case-statement
  //   mux trees into the matvec engine cone (~100K LUTs in i_eng).
  // Gammas, KV-cache init: $readmemh-into-array; Vivado infers BRAM
  //   (single-port reads, no multi-lane access).
  logic signed [15:0]  rom_GAMMA1  [0:NL*D-1];
  logic signed [15:0]  rom_GAMMA2  [0:NL*D-1];
  logic signed [15:0]  rom_K_INIT  [0:NL*MAX_CTX*H_KV*HD-1];
  logic signed [15:0]  rom_V_INIT  [0:NL*MAX_CTX*H_KV*HD-1];

`ifdef MICROGPT_WEIGHT_DIR
  initial begin
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "GAMMA1.hex"},       rom_GAMMA1);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "GAMMA2.hex"},       rom_GAMMA2);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "K_CACHE_INIT.hex"}, rom_K_INIT);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "V_CACHE_INIT.hex"}, rom_V_INIT);
  end
`else
  initial begin
    $readmemh({PREFIX, "GAMMA1.hex"},       rom_GAMMA1);
    $readmemh({PREFIX, "GAMMA2.hex"},       rom_GAMMA2);
    $readmemh({PREFIX, "K_CACHE_INIT.hex"}, rom_K_INIT);
    $readmemh({PREFIX, "V_CACHE_INIT.hex"}, rom_V_INIT);
  end
`endif

  // ----------------------------------------------------------------------
  // Scale ROM brom instances (one per matrix).  Read latency: 1 cycle.
  // The MV_SCALE FSM presents a comb addr on cycle T; brom captures it
  // at the edge T→T+1; data appears on cycle T+1.  An MV_SCALE_PRIME
  // bubble at the start of each chunk's scale phase aligns the lane
  // counter with the data arrival.
  // ----------------------------------------------------------------------
  localparam int SQ_AW   = $clog2(NL*D);
  localparam int SKV_AW  = $clog2(NL*H_KV*HD);
  localparam int SFFN_AW = $clog2(NL*FFN);

  logic [SQ_AW-1:0]   addr_s_q,    addr_s_o,    addr_s_down;
  logic [SKV_AW-1:0]  addr_s_k,    addr_s_v;
  logic [SFFN_AW-1:0] addr_s_gate, addr_s_up;
  wire  [15:0]        data_s_q, data_s_k, data_s_v, data_s_o;
  wire  [15:0]        data_s_gate, data_s_up, data_s_down;

  // Scale-BRAM writes from host: scale_wr_kind selects matrix
  // (0=Q 1=K 2=V 3=O 4=GATE 5=UP 6=DOWN), addr is the per-matrix entry,
  // data is the 16-bit replacement scale.  Defaults still come from
  // .INIT_xx params; runtime writes overlay specific entries.
  wire wr_en_q    = scale_wr_en && (scale_wr_kind == 4'd0);
  wire wr_en_k    = scale_wr_en && (scale_wr_kind == 4'd1);
  wire wr_en_v    = scale_wr_en && (scale_wr_kind == 4'd2);
  wire wr_en_o    = scale_wr_en && (scale_wr_kind == 4'd3);
  wire wr_en_gate = scale_wr_en && (scale_wr_kind == 4'd4);
  wire wr_en_up   = scale_wr_en && (scale_wr_kind == 4'd5);
  wire wr_en_down = scale_wr_en && (scale_wr_kind == 4'd6);

  brom_SCALE_Q    i_brom_S_Q    (.clk(clk), .addr(addr_s_q),    .data(data_s_q),
    .wr_addr(scale_wr_addr[SQ_AW-1:0]),   .wr_data(scale_wr_data), .wr_en(wr_en_q));
  brom_SCALE_K    i_brom_S_K    (.clk(clk), .addr(addr_s_k),    .data(data_s_k),
    .wr_addr(scale_wr_addr[SKV_AW-1:0]),  .wr_data(scale_wr_data), .wr_en(wr_en_k));
  brom_SCALE_V    i_brom_S_V    (.clk(clk), .addr(addr_s_v),    .data(data_s_v),
    .wr_addr(scale_wr_addr[SKV_AW-1:0]),  .wr_data(scale_wr_data), .wr_en(wr_en_v));
  brom_SCALE_O    i_brom_S_O    (.clk(clk), .addr(addr_s_o),    .data(data_s_o),
    .wr_addr(scale_wr_addr[SQ_AW-1:0]),   .wr_data(scale_wr_data), .wr_en(wr_en_o));
  brom_SCALE_GATE i_brom_S_GATE (.clk(clk), .addr(addr_s_gate), .data(data_s_gate),
    .wr_addr(scale_wr_addr[SFFN_AW-1:0]), .wr_data(scale_wr_data), .wr_en(wr_en_gate));
  brom_SCALE_UP   i_brom_S_UP   (.clk(clk), .addr(addr_s_up),   .data(data_s_up),
    .wr_addr(scale_wr_addr[SFFN_AW-1:0]), .wr_data(scale_wr_data), .wr_en(wr_en_up));
  brom_SCALE_DOWN i_brom_S_DOWN (.clk(clk), .addr(addr_s_down), .data(data_s_down),
    .wr_addr(scale_wr_addr[SQ_AW-1:0]),   .wr_data(scale_wr_data), .wr_en(wr_en_down));

  // ------------------------------------------------------------------------
  //  Weight streamer.
  //    matvec_id: 0=Q 1=K 2=V 3=O 4=GATE 5=UP 6=DOWN
  //  Sim build (default): weight_streamer_brom — $readmemh BRAM, no AXI.
  //  FPGA build (`MICROGPT_DDR3_WEIGHTS): weight_streamer_mt — N-bank
  //    AXI loader fed by MIG.  Per-matrix matrix_base + in_dim are
  //    derived from matvec_id.
  // ------------------------------------------------------------------------
  logic [2:0]   ws_matvec_id;
  logic [6:0]   ws_chunk_idx;
  logic         ws_load_req;
  logic         ws_ready;
  logic [10:0]  ws_rd_addr;
  // Forward-declared so the always_comb / always_ff blocks below
  // (which xvlog/xsim require to see the type before use) can use these.
  typedef enum logic [3:0] {
    MV_LOAD_REQ, MV_LOAD_HOLD, MV_LOAD_WAIT, MV_CLEAR, MV_DRIVE,
    MV_DRAIN, MV_SCALE_PRIME, MV_SCALE, MV_WAIT, MV_CAPTURE
  } mv_phase_t;
  mv_phase_t mv_phase;
  logic [10:0] cnt;
  logic [6:0]  chunk;
  typedef enum logic [4:0] {
    S_PRE_INIT,
    S_IDLE,
    S_NORM1, S_M_Q, S_M_K, S_M_V,
    S_ROPE_Q, S_ROPE_K, S_KV_WRITE,
    S_ATTN_QK, S_ATTN_SOFTMAX, S_ATTN_AV,
    S_M_O, S_RESID1,
    S_NORM2,
    S_M_GATE, S_M_UP, S_SWIGLU, S_M_DOWN,
    S_RESID2, S_DONE
  } state_t;
  state_t state;
  logic [127:0] ws_weight_data;

`ifdef MICROGPT_DDR3_WEIGHTS
  // Per-matvec matrix_base + in_dim, selected by ws_matvec_id.
  logic [29:0] ws_matrix_base;
  logic [10:0] ws_in_dim;
  always_comb begin
    case (ws_matvec_id)
      3'd0: begin ws_matrix_base = layer_base_addr + BASE_Q;    ws_in_dim = 11'(D);   end
      3'd1: begin ws_matrix_base = layer_base_addr + BASE_K;    ws_in_dim = 11'(D);   end
      3'd2: begin ws_matrix_base = layer_base_addr + BASE_V;    ws_in_dim = 11'(D);   end
      3'd3: begin ws_matrix_base = layer_base_addr + BASE_O;    ws_in_dim = 11'(D);   end
      3'd4: begin ws_matrix_base = layer_base_addr + BASE_GATE; ws_in_dim = 11'(D);   end
      3'd5: begin ws_matrix_base = layer_base_addr + BASE_UP;   ws_in_dim = 11'(D);   end
      3'd6: begin ws_matrix_base = layer_base_addr + BASE_DOWN; ws_in_dim = 11'(FFN); end
      default: begin ws_matrix_base = '0; ws_in_dim = '0; end
    endcase
  end

  weight_streamer_mt #(
    .AXI_DATA_WIDTH(512), .AXI_ADDR_WIDTH(30), .AXI_ID_WIDTH(5),
    .IN_DIM_BITS(11), .CHUNK_BITS(7),
    .MAX_TILES(3), .TILE_ENTRIES(512), .BURST_LEN_LOG2(7)
  ) i_ws (
    .clk_core(clk),  .rst_core(rst),
    .clk_axi (clk_axi), .rst_axi(rst_axi),
    .matrix_base (ws_matrix_base),
    .chunk_idx   (ws_chunk_idx),
    .in_dim      (ws_in_dim),
    .load_req    (ws_load_req),
    .ready       (ws_ready),
    .busy        (),
    .rd_addr     (ws_rd_addr),
    .weight_data (ws_weight_data),
    .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
    .m_axi_arid(m_axi_arid),       .m_axi_araddr(m_axi_araddr),
    .m_axi_arlen(m_axi_arlen),     .m_axi_arsize(m_axi_arsize),
    .m_axi_arburst(m_axi_arburst), .m_axi_arlock(m_axi_arlock),
    .m_axi_arcache(m_axi_arcache), .m_axi_arprot(m_axi_arprot),
    .m_axi_arqos(m_axi_arqos),
    .m_axi_rvalid(m_axi_rvalid),   .m_axi_rready(m_axi_rready),
    .m_axi_rid(m_axi_rid),         .m_axi_rdata(m_axi_rdata),
    .m_axi_rresp(m_axi_rresp),     .m_axi_rlast(m_axi_rlast),
    .dbg_first_araddr (dbg_first_araddr),
    .dbg_first_rdata  (dbg_first_rdata),
    .dbg_first_ar_seen(dbg_first_ar_seen),
    .dbg_first_r_seen (dbg_first_r_seen),
    .ila_ws_state      (ila_ws_state_axi),
    .ila_start_load_axi(ila_start_load_axi),
    .ila_tile_idx      (ila_tile_idx),
    .ila_beat_idx      (ila_beat_idx)
  );
`else
  weight_streamer_brom #(
    .D(D), .H_KV(H_KV), .HD(HD), .FFN(FFN), .PREFIX(PREFIX)
  ) i_ws (
    .clk(clk), .rst(rst),
    .matvec_id   (ws_matvec_id),
    .chunk_idx   (ws_chunk_idx),
    .load_req    (ws_load_req),
    .ready       (ws_ready),
    .rd_addr     (ws_rd_addr),
    .weight_data (ws_weight_data)
  );
`endif

  localparam int CHUNK_Q  = D   / 16;     // 4
  localparam int CHUNK_KV = (H_KV*HD) / 16;  // 4
  localparam int CHUNK_FFN = FFN / 16;    // 8
  localparam int HD_SHIFT = $clog2(HD) / 2;   // 1/sqrt(HD): >>3 for HD=64

  // Matvec drive index → streamer read address.
  //   sim path (brom):    weight_data is combinational on rd_addr, so
  //                       `eng_w <= ws_weight_data` with rd_addr=cnt aligns
  //                       weight K with src K at cycle K+1.
  //   FPGA path (mt+AXI): weight_data is registered (1-cycle latency from
  //                       rd_addr), so we lead rd_addr by 1 during MV_DRIVE.
  //                       In MV_CLEAR (cnt=0), rd_addr=0 fetches W[0]; in
  //                       MV_DRIVE iteration K (cnt=K-1 entering), rd_addr=K
  //                       fetches W[K] for the next iteration while
  //                       weight_data carries W[K-1] for this iteration.
  // Sim path (brom): weight_data is combinational on rd_addr — `rd_addr = cnt`
  //   so eng_w <= weight_data captures W[cnt] each MV_DRIVE cycle.
  // FPGA path (mt+AXI): weight_data is REGISTERED for BRAM inference, so we
  //   lead rd_addr by 1 during MV_DRIVE and force rd_addr=0 during MV_CLEAR
  //   so the registered weight_data lands W[cnt] in time for cycle cnt's MAC.
`ifdef MICROGPT_DDR3_WEIGHTS
  always_comb begin
    case (mv_phase)
      MV_CLEAR: ws_rd_addr = 11'd0;
      MV_DRIVE: ws_rd_addr = cnt + 11'd1;
      default:  ws_rd_addr = cnt;
    endcase
  end
`else
  assign ws_rd_addr = cnt;
`endif

  // hidden_in is currently unused (test data comes from the SVH's
  // LAYER_HIDDEN_IN_PACKED localparam).  Reserved for FPGA-time wiring.

  // ------------------------------------------------------------------------
  //  Shared matvec engine
  // ------------------------------------------------------------------------
  logic signed [15:0]       eng_in_value;
  logic                     eng_in_valid;
  logic                     eng_in_last;
  logic [16*8-1:0]          eng_w;

  // ------------------------------------------------------------------------
  //  Diagnostic: snapshot the FIRST few eng_w values seen during the run,
  //  plus the first ws_weight_data slice.  Read out via host regmap to
  //  verify the streamer→engine path delivers correct weights.
  //  Captured on the FIRST 4 MV_DRIVE iterations (state==S_M_Q chunk 0).
  // ------------------------------------------------------------------------
`ifdef MICROGPT_DDR3_WEIGHTS
  logic [127:0]       dbg_eng_w_snap [0:3];
  logic [127:0]       dbg_wd_snap    [0:3];
  logic signed [15:0] dbg_in_snap    [0:3];
  logic               dbg_snap_done;
  always_ff @(posedge clk) begin
    if (rst) begin
      dbg_snap_done <= 1'b0;
      for (int i=0; i<4; i++) begin
        dbg_eng_w_snap[i] <= '0;
        dbg_wd_snap[i]    <= '0;
        dbg_in_snap[i]    <= '0;
      end
    end else if (!dbg_snap_done && state == S_M_Q && mv_phase == MV_DRIVE
                                 && chunk == 7'd0 && cnt >= 11'd1 && cnt <= 11'd4) begin
      dbg_eng_w_snap[cnt - 1] <= eng_w;
      dbg_wd_snap   [cnt - 1] <= ws_weight_data;
      dbg_in_snap   [cnt - 1] <= eng_in_value;
      if (cnt == 11'd4) dbg_snap_done <= 1'b1;
    end
  end
  assign dbg_eng_w_packed   = {dbg_eng_w_snap[3], dbg_eng_w_snap[2],
                               dbg_eng_w_snap[1], dbg_eng_w_snap[0]};
  assign dbg_wd_packed      = {dbg_wd_snap[3],    dbg_wd_snap[2],
                               dbg_wd_snap[1],    dbg_wd_snap[0]};
  assign dbg_in_value_packed = {dbg_in_snap[3], dbg_in_snap[2],
                                dbg_in_snap[1], dbg_in_snap[0]};
  assign dbg_snap_done_o    = dbg_snap_done;
`endif
  logic [16*16-1:0]         eng_scale;
  logic                     eng_scale_valid;
  logic                     eng_acc_clear;
  logic [16*16-1:0]         eng_out;
  logic                     eng_out_valid;

`ifdef VERILATOR_DEBUG_SCALE
  // Dump first few gamma reads + rms inputs for layer 0 to check
  // brom-comb-wire alignment.  rms1_in_x and rms1_in_gamma at the SAME
  // cycle should reflect the SAME index (cnt-1, since both are 1-cycle
  // delayed from the comb assigns the prior cycle).
  always_ff @(posedge clk) begin
    if (state == S_NORM1 && nm_phase == NM_LOAD && layer_idx == 0
                          && cnt < 11'd6) begin
      $display("G1-FEED cnt=%0d addr_g1=0x%04h data_g1=0x%04h rms1_x=0x%04h v=%0d", cnt, addr_gamma1, data_gamma1, rms1_in_x, rms1_in_valid);
    end
    if (state == S_M_Q && mv_phase == MV_DRIVE && layer_idx == 0
                       && chunk == 0 && cnt < 11'd6) begin
      $display("MQ-DRIVE cnt=%0d norm1_buf=0x%04h eng_in=0x%04h eng_w15=0x%04h", cnt, norm1_buf[cnt], eng_in_value, eng_w[15:0]);
    end
  end
`endif

  matvec_int8_engine #(.LANES(16), .ACC_W(40)) i_eng (
    .clk(clk), .rst(rst),
    .in_value   (eng_in_value),  .in_valid   (eng_in_valid),
    .in_last    (eng_in_last),   .w_int8     (eng_w),
    .scale_q15  (eng_scale),     .scale_valid(eng_scale_valid),
    .out_value  (eng_out),       .out_valid  (eng_out_valid),
    .acc_clear  (eng_acc_clear)
  );

  // ------------------------------------------------------------------------
  //  Two RMSNorm instances — one per call site.
  //
  //  in_gamma is a WIRE driven directly from the gamma brom's RAMB36E1
  //  output (no intermediate FF) so the BRAM's 1-cycle internal latency
  //  is the only cycle of read latency.  The FSM (NM_LOAD) presents the
  //  brom addr combinationally and drives in_x/in_valid one cycle ahead
  //  of the matching gamma data.
  // ------------------------------------------------------------------------
  localparam int GAMMA_AW = $clog2(NL*D);
  logic [GAMMA_AW-1:0] addr_gamma1, addr_gamma2;
  wire signed [15:0]   data_gamma1, data_gamma2;
  brom_GAMMA1 i_brom_GAMMA1 (.clk(clk), .addr(addr_gamma1), .data(data_gamma1));
  brom_GAMMA2 i_brom_GAMMA2 (.clk(clk), .addr(addr_gamma2), .data(data_gamma2));

  // BRAM addresses driven combinationally from cnt; the brom's internal
  // RAMB36E1 addr register provides the 1-cycle latency that the
  // original $readmemh + sync-read pattern relied on.  No extra FSM
  // bubble is needed because rms*_in_gamma is a wire (not a register),
  // so the BRAM's internal reg is the only stage between cnt and
  // rmsnorm's input.
  always_comb begin
    addr_gamma1 = GAMMA_AW'(layer_idx*D + cnt);
    addr_gamma2 = GAMMA_AW'(layer_idx*D + cnt);
  end

  logic                rms1_start, rms1_in_valid, rms1_out_valid, rms1_done;
  logic signed [15:0]  rms1_in_x, rms1_out_y;
  wire  signed [15:0]  rms1_in_gamma = data_gamma1;
  rmsnorm #(.D(D)) i_rms1 (
    .clk, .rst, .start(rms1_start),
    .in_x(rms1_in_x), .in_gamma(rms1_in_gamma), .in_valid(rms1_in_valid),
    .out_y(rms1_out_y), .out_valid(rms1_out_valid), .done(rms1_done)
  );

  logic                rms2_start, rms2_in_valid, rms2_out_valid, rms2_done;
  logic signed [15:0]  rms2_in_x, rms2_out_y;
  wire  signed [15:0]  rms2_in_gamma = data_gamma2;
  rmsnorm #(.D(D)) i_rms2 (
    .clk, .rst, .start(rms2_start),
    .in_x(rms2_in_x), .in_gamma(rms2_in_gamma), .in_valid(rms2_in_valid),
    .out_y(rms2_out_y), .out_valid(rms2_out_valid), .done(rms2_done)
  );

  // ------------------------------------------------------------------------
  //  Single rope (reused for Q and K)
  // ------------------------------------------------------------------------
  logic                rope_start, rope_in_valid, rope_out_valid, rope_done;
  logic signed [15:0]  rope_in_x, rope_out_y;
  rope #(.HEAD_DIM(HD), .MAX_CTX(MAX_CTX==4 ? 2048 : MAX_CTX)) i_rope (
    // MAX_CTX needs to match the freq-table compile (32 entries → HEAD_DIM=64).
    // We set MAX_CTX=2048 (default) since `pos` only uses log2(2048)=11 bits
    // and our test pos≤3 fits easily.  The freq table ROM is HEAD_DIM-only.
    .clk, .rst, .start(rope_start),
    .pos(pos),
    .in_x(rope_in_x), .in_valid(rope_in_valid),
    .out_y(rope_out_y), .out_valid(rope_out_valid), .done(rope_done)
  );

  // ------------------------------------------------------------------------
  //  Single softmax (N = MAX_CTX = 4 for this test)
  // ------------------------------------------------------------------------
  logic                sm_start, sm_in_valid, sm_out_valid, sm_done;
  logic signed [15:0]  sm_in_x, sm_out_y;
  softmax_q15 #(.N(MAX_CTX)) i_sm (
    .clk, .rst, .start(sm_start),
    .in_x(sm_in_x), .in_valid(sm_in_valid),
    .out_y(sm_out_y), .out_valid(sm_out_valid), .done(sm_done)
  );

  // ------------------------------------------------------------------------
  //  Swiglu — purely streaming, 1-cycle latency
  // ------------------------------------------------------------------------
  logic                sg_in_valid, sg_out_valid;
  logic signed [15:0]  sg_in_gate, sg_in_up, sg_out_y;
  swiglu i_sg (
    .clk, .rst,
    .in_gate(sg_in_gate), .in_up(sg_in_up), .in_valid(sg_in_valid),
    .gate_in_factor (sg_gate_in_factor),
    .up_in_factor   (sg_up_in_factor),
    .mlp_out_factor (sg_mlp_out_factor),
    .out_y(sg_out_y), .out_valid(sg_out_valid)
  );

  // ------------------------------------------------------------------------
  //  KV cache (per-position × per-head × HD entries, Q1.15)
  //  Initialised after reset by an FSM state (S_PRE_INIT) walking through
  //  the packed-localparam pre-state.  No `initial` blocks — Vivado's
  //  support for them on writable arrays is unreliable.
  // ------------------------------------------------------------------------
  // Per-layer KV caches sized NL × MAX_CTX*H_KV*HD.  Each layer keeps
  // its own state across token generations.
  logic signed [15:0] k_cache [0:NL*MAX_CTX*H_KV*HD-1];
  logic signed [15:0] v_cache [0:NL*MAX_CTX*H_KV*HD-1];

  // ------------------------------------------------------------------------
  //  Result buffers
  // ------------------------------------------------------------------------
  logic signed [15:0] norm1_buf  [0:D-1];
  logic signed [15:0] q_buf      [0:D-1];
  logic signed [15:0] k_buf      [0:H_KV*HD-1];
  logic signed [15:0] v_buf      [0:H_KV*HD-1];
  logic signed [15:0] q_rot_buf  [0:D-1];
  logic signed [15:0] k_rot_buf  [0:H_KV*HD-1];
  logic signed [15:0] scores_buf [0:MAX_CTX-1];
  logic signed [15:0] sm_buf     [0:MAX_CTX-1];
  logic signed [15:0] attn_buf   [0:D-1];
  logic signed [15:0] o_buf      [0:D-1];
  // hidden1 is 16-bit Q1.15 at scale 2^h1_p2 (block-FP).
  logic signed [15:0] hidden1_buf[0:D-1];
  logic signed [15:0] norm2_buf  [0:D-1];
  logic signed [15:0] gate_buf   [0:FFN-1];
  logic signed [15:0] up_buf     [0:FFN-1];
  logic signed [15:0] mlp_buf    [0:FFN-1];
  logic signed [15:0] down_buf   [0:D-1];

  // Pack hidden_out from buffers (always present — production output).
  //   hidden_out_24 = hidden1_buf_24 + (down_buf_int16 * resid2_factor)
  //                                            ↑ folds bus scale to Q15.9
  // Saturation to 24-bit signed.  Trace outputs only emitted when DEBUG.
  genvar gi;
  generate
    for (gi = 0; gi < D; gi++) begin : g_hidden_out
      // hidden_out = sat16( (hidden1 << sh_h1_to_h_out_signed) +
      //                     ((down_int16 * resid2_factor_q24) >> 8 then
      //                      shifted to h_out's scale via factor encoding) )
      // resid2_factor encodes (s_down / s_h_out * 2^8), so r2_delta is
      // already at h_out's scale.  Only hidden1 needs cascading rescale.
      logic signed [40:0] r2_prod;
      logic signed [32:0] r2_delta;
      logic signed [31:0] h1_aligned;
      logic signed [33:0] r2_sum;
      assign r2_prod   = $signed(down_buf[gi]) * $signed({1'b0, resid2_factor});
      assign r2_delta  = $signed(r2_prod >>> 8);
      // Rescale hidden1: positive sh = right shift; negative = left shift.
      assign h1_aligned = (sh_h1_to_h_out >= 0) ?
              ($signed({{16{hidden1_buf[gi][15]}}, hidden1_buf[gi]}) >>> sh_h1_to_h_out) :
              ($signed({{16{hidden1_buf[gi][15]}}, hidden1_buf[gi]}) <<< (-sh_h1_to_h_out));
      assign r2_sum = $signed({{2{h1_aligned[31]}}, h1_aligned})
                    + $signed({{1{r2_delta[32]}}, r2_delta});
      assign hidden_out[gi*16 +: 16] = (state != S_DONE) ? '0 :
            (r2_sum >  34'sd32767)  ?  16'sh7FFF :
            (r2_sum < -34'sd32768)  ?  16'sh8000 :
            r2_sum[15:0];
    end
`ifdef DEBUG
    for (gi = 0; gi < D; gi++) begin : g_trace_d
      assign trace_norm1  [gi*16 +: 16] = norm1_buf  [gi];
      assign trace_q      [gi*16 +: 16] = q_buf      [gi];
      assign trace_q_rot  [gi*16 +: 16] = q_rot_buf  [gi];
      assign trace_attn   [gi*16 +: 16] = attn_buf   [gi];
      assign trace_hidden1[gi*16 +: 16] = hidden1_buf[gi];
      assign trace_norm2  [gi*16 +: 16] = norm2_buf  [gi];
    end
    for (gi = 0; gi < H_KV*HD; gi++) begin : g_trace_kv
      assign trace_k    [gi*16 +: 16] = k_buf    [gi];
      assign trace_v    [gi*16 +: 16] = v_buf    [gi];
      assign trace_k_rot[gi*16 +: 16] = k_rot_buf[gi];
    end
    for (gi = 0; gi < FFN; gi++) begin : g_trace_ffn
      assign trace_mlp_inter[gi*16 +: 16] = mlp_buf[gi];
    end
`endif
  endgenerate

  // ------------------------------------------------------------------------
  //  FSM
  // ------------------------------------------------------------------------
  // (state_t / state forward-declared earlier with mv_phase)
  // walks 0..NL*MAX_CTX*H_KV*HD-1 during S_PRE_INIT (post-reset KV load)
  localparam int KV_INIT_WORDS = NL * MAX_CTX * H_KV * HD;
  logic [$clog2(KV_INIT_WORDS):0] init_cnt;

  // Sub-phase trackers (per major state)
  typedef enum logic [2:0] {
    NM_IDLE, NM_LOAD, NM_OUTPUT
  } nm_phase_t;
  nm_phase_t nm_phase;

  // MV_LOAD_HOLD waits for ws_ready to go LOW after we pulse load_req
  // (the streamer takes 1 cycle to react).  Without this, on chunk 1+
  // the streamer was still showing ready=1 from the previous chunk, the
  // layer skipped MV_LOAD_WAIT, and MV_DRIVE consumed stale bank data
  // before the new chunk's tiles had been loaded.
  //
  // MV_CAPTURE writes the matvec result one lane per cycle into DST_BUF
  // (single-port BRAM-friendly).  The previous "NBA all 16 lanes in one
  // cycle" pattern forced every *_buf array to dissolve into registers,
  // which blew the LUT budget at SmolLM real dims.
  // (mv_phase_t / mv_phase / cnt / chunk forward-declared earlier)
  logic [255:0] eng_out_r;       // captured at MV_WAIT, drained one lane/cycle
  logic [3:0]   cap_lane;        // 0..15 walking write index in MV_CAPTURE
  logic [3:0]   scale_lane;      // 0..15 walking read index in MV_SCALE

  typedef enum logic [2:0] {
    RP_IDLE, RP_LOAD, RP_OUTPUT
  } rp_phase_t;
  rp_phase_t rp_phase;

  typedef enum logic [2:0] {
    SM_IDLE, SM_LOAD, SM_OUTPUT
  } sm_phase_t;
  sm_phase_t sm_phase;

  typedef enum logic [2:0] {
    AT_INIT, AT_DOT_PRIME, AT_DOT_DRIVE, AT_DOT_FINISH,
    AT_AV_INIT, AT_AV_PRIME, AT_AV_ACC, AT_AV_FINISH
  } at_phase_t;
  at_phase_t at_phase;

  // BRAM read pipeline for k_cache / v_cache.  A separate always_ff
  // outside the main FSM issues a synchronous read every cycle from
  // {k,v}_rd_addr; data is consumed one cycle later via {k,v}_rd_data.
  // Vivado infers RAMB36 in single-port mode (write in S_K/V_WRITE,
  // read in AT_DOT/AV_*).
  localparam int KV_AW = $clog2(NL*MAX_CTX*H_KV*HD);
  logic [KV_AW-1:0]   k_rd_addr, v_rd_addr;
  logic signed [15:0] k_rd_data, v_rd_data;
  always_ff @(posedge clk) k_rd_data <= k_cache[k_rd_addr];
  always_ff @(posedge clk) v_rd_data <= v_cache[v_rd_addr];

  typedef enum logic [1:0] {
    SG_RUN, SG_DRAIN
  } sg_phase_t;
  sg_phase_t sg_phase;

  // (cnt / chunk forward-declared earlier with mv_phase)
  // state_t / state forward-declared at top for the same reason.

  // Attention scratch
  logic [3:0]  head_idx;   // up to H_Q=9
  logic [10:0] t_idx;      // up to MAX_CTX-1 (could grow with longer context)
  logic [10:0] dot_cnt;    // up to HD-1
  logic signed [39:0] dot_acc;
  logic [10:0] av_cnt;     // up to HD-1
  logic signed [39:0] av_acc;

  // ------------------------------------------------------------------------
  //  Helper: pack 16 INT8 weights for one matvec drive cycle.
  //  Selects bytes from the relevant W_*_PACKED localparam given (chunk, k).
  //  WPACKED layout: lane-major, each lane has in_dim INT8s.
  //  Byte index for (lane, k) within W_*_PACKED = (lane * in_dim + k) * 8.
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  //  Always_ff
  // ------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      state    <= S_PRE_INIT;
      init_cnt <= '0;
      done     <= 1'b0;
      cnt      <= '0;
      chunk    <= '0;
      head_idx <= '0;
      t_idx    <= '0;
      dot_cnt  <= '0;
      dot_acc  <= '0;
      av_cnt   <= '0;
      av_acc   <= '0;
      nm_phase <= NM_IDLE;
      mv_phase <= MV_LOAD_REQ;
      rp_phase <= RP_IDLE;
      ws_load_req  <= 1'b0;
      ws_matvec_id <= '0;
      ws_chunk_idx <= '0;
      sm_phase <= SM_IDLE;
      at_phase <= AT_INIT;
      sg_phase <= SG_RUN;
      rms1_start    <= 1'b0;
      rms1_in_valid <= 1'b0;
      rms2_start    <= 1'b0;
      rms2_in_valid <= 1'b0;
      rope_start    <= 1'b0;
      rope_in_valid <= 1'b0;
      sm_start      <= 1'b0;
      sm_in_valid   <= 1'b0;
      sg_in_valid   <= 1'b0;
      eng_in_valid    <= 1'b0;
      eng_in_last     <= 1'b0;
      eng_scale_valid <= 1'b0;
      eng_acc_clear   <= 1'b0;
    end else begin
      // Default-low pulses
      rms1_start    <= 1'b0;
      rms2_start    <= 1'b0;
      rope_start    <= 1'b0;
      sm_start      <= 1'b0;
      rms1_in_valid <= 1'b0;
      rms2_in_valid <= 1'b0;
      rope_in_valid <= 1'b0;
      sm_in_valid   <= 1'b0;
      sg_in_valid   <= 1'b0;
      eng_in_valid    <= 1'b0;
      eng_in_last     <= 1'b0;
      eng_scale_valid <= 1'b0;
      eng_acc_clear   <= 1'b0;
      ws_load_req     <= 1'b0;     // default-low pulse

      case (state)
        // ----------------------------------------------------------------
        //  Pre-init: load KV cache from packed-localparam pre-state.
        //  Runs once after reset, MAX_CTX*H_KV*HD cycles total.
        // ----------------------------------------------------------------
        S_PRE_INIT: begin
          // Walk all NL × MAX_CTX*H_KV*HD entries so every layer's KV
          // cache is initialised post-reset.
          k_cache[init_cnt] <= rom_K_INIT[init_cnt];
          v_cache[init_cnt] <= rom_V_INIT[init_cnt];
          if (init_cnt == KV_INIT_WORDS - 1) state <= S_IDLE;
          init_cnt <= init_cnt + 1'b1;
        end

        // ----------------------------------------------------------------
        S_IDLE: begin
          done <= 1'b0;
          if (start) begin
            state    <= S_NORM1;
            nm_phase <= NM_IDLE;
            cnt      <= '0;
          end
        end

        // ----------------------------------------------------------------
        //  RMSNorm 1: hidden_in (from packed localparam) × γ₁  → norm1_buf
        // ----------------------------------------------------------------
        S_NORM1: begin
          case (nm_phase)
            NM_IDLE: begin
              rms1_start <= 1'b1;
              cnt        <= '0;
              nm_phase   <= NM_LOAD;
            end
            NM_LOAD: begin
              // hidden_in is already 16-bit Q1.15 — RMSNorm is scale-invariant,
              // so we feed it directly (gamma is calibrated for h_in's scale).
              rms1_in_x <= $signed(hidden_in[cnt*16 +: 16]);
              // rms1_in_gamma is a wire from brom_GAMMA1's BRAM output
              // (driven by addr_gamma1 = layer_idx*D + cnt, comb).
              rms1_in_valid <= 1'b1;
              cnt           <= cnt + 1'b1;
              if (cnt == D-1) begin
                nm_phase <= NM_OUTPUT;
                cnt      <= '0;
              end
            end
            NM_OUTPUT: begin
              if (rms1_out_valid) begin
                norm1_buf[cnt] <= rms1_out_y;
                cnt            <= cnt + 1'b1;
                if (cnt == D-1) begin
                  state    <= S_M_Q;
                  cnt      <= '0;
                  chunk    <= '0;
                  mv_phase <= MV_LOAD_REQ;
                end
              end
            end
            default: nm_phase <= NM_IDLE;
          endcase
        end

        // ----------------------------------------------------------------
        //  Generic matvec runner template.  Implementation per-state below
        //  (S_M_Q, S_M_K, S_M_V, S_M_O, S_M_GATE, S_M_UP, S_M_DOWN).
        //  Pattern:
        //    MV_CLEAR : pulse acc_clear;       cnt<-0
        //    MV_DRIVE : in_value=src[cnt], eng_w=W slice (chunk,cnt), in_valid=1; cnt+=1
        //    MV_DRAIN : 1 cycle wait
        //    MV_SCALE : eng_scale=scale slice (chunk), scale_valid=1
        //    MV_WAIT  : on out_valid → capture 16 lanes into dest_buf[chunk*16+:16];
        //               if last chunk: → next state; else chunk+=1 → MV_CLEAR
        // ----------------------------------------------------------------

`define MATVEC_STATE(STATE_NAME, NEXT_STATE, IN_DIM, OUT_DIM, SCALE_STRIDE, SRC_BUF, MATRIX_ID, ADDR_SIG, DATA_SIG, DST_BUF) \
        STATE_NAME: begin                                                                         \
          case (mv_phase)                                                                         \
            MV_LOAD_REQ: begin                                                                    \
              ws_matvec_id <= MATRIX_ID;                                                          \
              ws_chunk_idx <= chunk;                                                              \
              ws_load_req  <= 1'b1;                                                               \
              mv_phase     <= MV_LOAD_HOLD;                                                       \
            end                                                                                   \
            MV_LOAD_HOLD: begin                                                                   \
              /* Wait for streamer to drop ready (1 cycle after our pulse). */                    \
              if (!ws_ready) mv_phase <= MV_LOAD_WAIT;                                            \
            end                                                                                   \
            MV_LOAD_WAIT: begin                                                                   \
              if (ws_ready) mv_phase <= MV_CLEAR;                                                 \
            end                                                                                   \
            MV_CLEAR: begin                                                                       \
              eng_acc_clear <= 1'b1;                                                              \
              cnt           <= '0;                                                                \
              mv_phase      <= MV_DRIVE;                                                          \
            end                                                                                   \
            MV_DRIVE: begin                                                                       \
              eng_in_value <= SRC_BUF[cnt];                                                       \
              /* Phase D-real: streamer combinationally returns 16 lanes for ws_rd_addr=cnt */    \
              eng_w        <= ws_weight_data;                                                     \
              eng_in_valid <= 1'b1;                                                               \
              eng_in_last  <= (cnt == IN_DIM-1);                                                  \
              cnt          <= cnt + 1'b1;                                                         \
              if (cnt == IN_DIM-1) mv_phase <= MV_DRAIN;                                          \
            end                                                                                   \
            MV_DRAIN: begin                                                                   \
              scale_lane <= 4'd0;                                                                 \
              /* Pre-issue brom addr for lane 0; data appears 1 cycle  */                         \
              /* later when MV_SCALE consumes lane 0.                  */                         \
              ADDR_SIG   <= layer_idx*SCALE_STRIDE + chunk*16 + 4'd0;                             \
              mv_phase   <= MV_SCALE_PRIME;                                                       \
            end                                                                                   \
            MV_SCALE_PRIME: begin                                                                 \
              /* Pre-issue brom addr for lane 1.  Data for lane 0 will  */                        \
              /* land on the brom output at the next edge.              */                        \
              ADDR_SIG   <= layer_idx*SCALE_STRIDE + chunk*16 + 4'd1;                             \
              scale_lane <= 4'd0;                                                                 \
              mv_phase   <= MV_SCALE;                                                             \
            end                                                                                   \
            MV_SCALE: begin                                                                       \
              /* DATA_SIG this cycle == scale[chunk*16 + scale_lane],   */                        \
              /* registered through the brom's RAMB36E1 (1-cycle read). */                        \
              eng_scale[scale_lane*16 +: 16] <= DATA_SIG;                                         \
              ADDR_SIG <= layer_idx*SCALE_STRIDE + chunk*16 + scale_lane + 4'd2;                  \
              scale_lane <= scale_lane + 4'd1;                                                    \
              if (scale_lane == 4'd15) begin                                                      \
                eng_scale_valid <= 1'b1;                                                          \
                mv_phase        <= MV_WAIT;                                                       \
              end                                                                                 \
            end                                                                                   \
            MV_WAIT: begin                                                                        \
              if (eng_out_valid) begin                                                            \
                eng_out_r <= eng_out;                                                             \
                cap_lane  <= 4'd0;                                                                \
                mv_phase  <= MV_CAPTURE;                                                          \
              end                                                                                 \
            end                                                                                   \
            MV_CAPTURE: begin                                                                     \
              /* One lane/cycle write so Vivado infers single-port BRAM. */                       \
              DST_BUF[chunk*16 + cap_lane] <= eng_out_r[cap_lane*16 +: 16];                       \
              cap_lane <= cap_lane + 4'd1;                                                        \
              if (cap_lane == 4'd15) begin                                                        \
                if (chunk == (OUT_DIM/16) - 1) begin                                              \
                  chunk    <= '0;                                                                 \
                  mv_phase <= MV_LOAD_REQ;                                                        \
                  state    <= NEXT_STATE;                                                         \
                  cnt      <= '0;                                                                 \
                  av_cnt   <= '0;                                                                 \
                end else begin                                                                    \
                  chunk    <= chunk + 1'b1;                                                       \
                  mv_phase <= MV_LOAD_REQ;                                                        \
                end                                                                               \
              end                                                                                 \
            end                                                                                   \
            default: mv_phase <= MV_LOAD_REQ;                                                     \
          endcase                                                                                 \
        end

        `MATVEC_STATE(S_M_Q,    S_M_K,      D,   D,        D,       norm1_buf, 3'd0, addr_s_q,    data_s_q,    q_buf)
        `MATVEC_STATE(S_M_K,    S_M_V,      D,   H_KV*HD,  H_KV*HD, norm1_buf, 3'd1, addr_s_k,    data_s_k,    k_buf)
        `MATVEC_STATE(S_M_V,    S_ROPE_Q,   D,   H_KV*HD,  H_KV*HD, norm1_buf, 3'd2, addr_s_v,    data_s_v,    v_buf)
        `MATVEC_STATE(S_M_O,    S_RESID1,   D,   D,        D,       attn_buf,  3'd3, addr_s_o,    data_s_o,    o_buf)
        `MATVEC_STATE(S_M_GATE, S_M_UP,     D,   FFN,      FFN,     norm2_buf, 3'd4, addr_s_gate, data_s_gate, gate_buf)
        `MATVEC_STATE(S_M_UP,   S_SWIGLU,   D,   FFN,      FFN,     norm2_buf, 3'd5, addr_s_up,   data_s_up,   up_buf)
        `MATVEC_STATE(S_M_DOWN, S_RESID2,   FFN, D,        D,       mlp_buf,   3'd6, addr_s_down, data_s_down, down_buf)

`undef MATVEC_STATE

        // ----------------------------------------------------------------
        //  RoPE Q — loops H_Q heads.  Each head: HD elements of
        //  q_buf[h*HD..] → q_rot_buf[h*HD..], using rope at the same `pos`.
        // ----------------------------------------------------------------
        S_ROPE_Q: begin
          case (rp_phase)
            RP_IDLE: begin
              rope_start <= 1'b1;
              cnt        <= '0;
              rp_phase   <= RP_LOAD;
            end
            RP_LOAD: begin
              rope_in_x     <= q_buf[head_idx*HD + cnt];
              rope_in_valid <= 1'b1;
              cnt           <= cnt + 1'b1;
              if (cnt == HD-1) begin
                rp_phase <= RP_OUTPUT;
                cnt      <= '0;
              end
            end
            RP_OUTPUT: begin
              if (rope_out_valid) begin
                q_rot_buf[head_idx*HD + cnt] <= rope_out_y;
                cnt                          <= cnt + 1'b1;
                if (cnt == HD-1) begin
                  if (head_idx == H_Q-1) begin
                    head_idx <= '0;
                    state    <= S_ROPE_K;
                  end else begin
                    head_idx <= head_idx + 1'b1;
                  end
                  rp_phase <= RP_IDLE;
                  cnt      <= '0;
                end
              end
            end
            default: rp_phase <= RP_IDLE;
          endcase
        end

        // ----------------------------------------------------------------
        //  RoPE K — loops H_KV heads (typically fewer than H_Q under GQA).
        // ----------------------------------------------------------------
        S_ROPE_K: begin
          case (rp_phase)
            RP_IDLE: begin
              rope_start <= 1'b1;
              cnt        <= '0;
              rp_phase   <= RP_LOAD;
            end
            RP_LOAD: begin
              rope_in_x     <= k_buf[head_idx*HD + cnt];
              rope_in_valid <= 1'b1;
              cnt           <= cnt + 1'b1;
              if (cnt == HD-1) begin
                rp_phase <= RP_OUTPUT;
                cnt      <= '0;
              end
            end
            RP_OUTPUT: begin
              if (rope_out_valid) begin
                k_rot_buf[head_idx*HD + cnt] <= rope_out_y;
                cnt                          <= cnt + 1'b1;
                if (cnt == HD-1) begin
                  if (head_idx == H_KV-1) begin
                    head_idx <= '0;
                    state    <= S_KV_WRITE;
                  end else begin
                    head_idx <= head_idx + 1'b1;
                  end
                  rp_phase <= RP_IDLE;
                  cnt      <= '0;
                end
              end
            end
            default: rp_phase <= RP_IDLE;
          endcase
        end

        // ----------------------------------------------------------------
        //  KV cache write: H_KV*HD cycles, one element per cycle
        //  (Verilator strict-mode rejects NBA-to-indexed-array inside
        //  procedural for-loops, so we sequence with `cnt`).
        // ----------------------------------------------------------------
        S_KV_WRITE: begin
          k_cache[layer_idx*(MAX_CTX*H_KV*HD) + kv_pos * H_KV * HD + cnt] <= k_rot_buf[cnt];
          v_cache[layer_idx*(MAX_CTX*H_KV*HD) + kv_pos * H_KV * HD + cnt] <= v_buf[cnt];
          cnt <= cnt + 1'b1;
          if (cnt == H_KV*HD - 1) begin
            state    <= S_ATTN_QK;
            head_idx <= '0;
            t_idx    <= '0;
            at_phase <= AT_INIT;
            dot_cnt  <= '0;
            dot_acc  <= '0;
            cnt      <= '0;
          end
        end

        // ----------------------------------------------------------------
        //  Attention scoring: scores[t] = (Q_h scaled · K_cache[t,kv_h]) >> 15
        //  For test config H_Q=H_KV=1 → kv_h=0, head_idx loops just once.
        // ----------------------------------------------------------------
        S_ATTN_QK: begin
          case (at_phase)
            AT_INIT: begin
              dot_cnt  <= '0;
              dot_acc  <= '0;
              // Issue the FIRST read address for this (head, t_idx).
              // k_rd_data will hold k[base+0] two cycles later in the
              // first AT_DOT_DRIVE iteration.
              begin
                logic [3:0] kv_h;
                kv_h      = head_idx / (H_Q / H_KV);
                k_rd_addr <= KV_AW'(layer_idx*(MAX_CTX*H_KV*HD) + t_idx*H_KV*HD + kv_h*HD);
              end
              at_phase <= AT_DOT_PRIME;
            end
            AT_DOT_PRIME: begin
              // Bubble: pre-issue the next address (base+1).
              k_rd_addr <= k_rd_addr + 1'b1;
              at_phase  <= AT_DOT_DRIVE;
            end
            AT_DOT_DRIVE: begin
              // k_rd_data == k[base + dot_cnt] (registered BRAM read).
              // q_rot_buf is small enough to remain a comb-readable array.
              begin
                logic signed [15:0] q_e;
                logic signed [31:0] q_e_s32;
                logic signed [23:0] q_scaled;     // 24-bit, range OK
                logic signed [47:0] prod;
                q_e      = q_rot_buf[head_idx*HD + dot_cnt];
                q_e_s32  = $signed(q_e);
                q_scaled = q_e_s32 >>> HD_SHIFT;
                prod     = $signed(q_scaled) * $signed(k_rd_data);
                dot_acc  <= dot_acc + prod[39:0];
              end
              k_rd_addr <= k_rd_addr + 1'b1;        // preroll one ahead
              dot_cnt   <= dot_cnt + 1'b1;
              if (dot_cnt == HD-1) at_phase <= AT_DOT_FINISH;
            end
            AT_DOT_FINISH: begin
              // shifted = dot_acc >>> 15, saturate to Q1.15
              begin
                logic signed [39:0] shifted;
                shifted = dot_acc >>> 15;
                if      (shifted >  40'sd32767)  scores_buf[t_idx] <=  16'sd32767;
                else if (shifted < -40'sd32768)  scores_buf[t_idx] <= -16'sd32768;
                else                             scores_buf[t_idx] <= shifted[15:0];
              end
              if (t_idx == kv_pos) begin
                // All scores done — proceed to softmax
                state    <= S_ATTN_SOFTMAX;
                sm_phase <= SM_IDLE;
                cnt      <= '0;
              end else begin
                t_idx    <= t_idx + 1'b1;
                at_phase <= AT_INIT;
              end
            end
            default: at_phase <= AT_INIT;
          endcase
        end

        // ----------------------------------------------------------------
        //  Softmax over scores_buf[0..MAX_CTX-1] (kv_pos+1 real, rest = 0
        //  which represents 0 in Q1.15; only valid when pos==MAX_CTX-1).
        // ----------------------------------------------------------------
        S_ATTN_SOFTMAX: begin
          case (sm_phase)
            SM_IDLE: begin
              sm_start <= 1'b1;
              cnt      <= '0;
              sm_phase <= SM_LOAD;
            end
            SM_LOAD: begin
              sm_in_x     <= scores_buf[cnt];
              sm_in_valid <= 1'b1;
              cnt         <= cnt + 1'b1;
              if (cnt == MAX_CTX-1) begin
                sm_phase <= SM_OUTPUT;
                cnt      <= '0;
              end
            end
            SM_OUTPUT: begin
              if (sm_out_valid) begin
                sm_buf[cnt] <= sm_out_y;
                cnt         <= cnt + 1'b1;
                if (cnt == MAX_CTX-1) begin
                  // Move to AV for the CURRENT head (preserve head_idx —
                  // this is per-head pipeline; head_idx advances after
                  // AV finishes for this head).
                  state    <= S_ATTN_AV;
                  at_phase <= AT_AV_INIT;
                  av_cnt   <= '0;
                  av_acc   <= '0;
                end
              end
            end
            default: sm_phase <= SM_IDLE;
          endcase
        end

        // ----------------------------------------------------------------
        //  Weighted V: attn[h*HD+e] = sat( Σ_t sm[t] * v_cache[t,kv_h,e] >> 15 )
        // ----------------------------------------------------------------
        S_ATTN_AV: begin
          case (at_phase)
            AT_AV_INIT: begin
              t_idx    <= '0;
              av_acc   <= '0;
              // Issue the FIRST read address (t=0); v_rd_data appears 2
              // cycles later as the AT_AV_ACC loop begins.
              begin
                logic [3:0] kv_h;
                kv_h      = head_idx / (H_Q / H_KV);
                v_rd_addr <= KV_AW'(layer_idx*(MAX_CTX*H_KV*HD) + kv_h*HD + av_cnt);
              end
              at_phase <= AT_AV_PRIME;
            end
            AT_AV_PRIME: begin
              // Bubble: pre-issue read for t=1.  Stride between successive
              // timesteps is H_KV*HD (since v_cache is t-major).
              v_rd_addr <= v_rd_addr + (H_KV*HD);
              at_phase  <= AT_AV_ACC;
            end
            AT_AV_ACC: begin
              // v_rd_data == v[t_idx, kv_h, av_cnt]
              begin
                logic signed [15:0] sm_t;
                logic signed [31:0] prod;
                sm_t   = sm_buf[t_idx];
                prod   = $signed(sm_t) * $signed(v_rd_data);
                av_acc <= av_acc + {{8{prod[31]}}, prod};
              end
              v_rd_addr <= v_rd_addr + (H_KV*HD);  // preroll one timestep ahead
              if (t_idx == kv_pos) begin
                at_phase <= AT_AV_FINISH;
              end else begin
                t_idx    <= t_idx + 1'b1;
              end
            end
            AT_AV_FINISH: begin
              // Capture this element of the head, advance e or move to next head
              begin
                // av_acc carries V at lsc[v] scale.  Apply attn_factor
                // (Q16.8 = lsc[v]/lsc[attn]*256) so attn_buf lands at
                // lsc[attn] scale, matching the matvec_O input.  Without
                // this rescale the MLP path (and ultimately hidden_out)
                // ends up at the wrong scale.
                logic signed [39:0] shifted;
                logic signed [63:0] rescaled;
                shifted  = av_acc >>> 15;
                rescaled = $signed(shifted) * $signed({40'b0, attn_factor});
                rescaled = rescaled >>> 8;
                if      (rescaled >  64'sd32767)  attn_buf[head_idx*HD + av_cnt] <=  16'sd32767;
                else if (rescaled < -64'sd32768)  attn_buf[head_idx*HD + av_cnt] <= -16'sd32768;
                else                              attn_buf[head_idx*HD + av_cnt] <=  rescaled[15:0];
              end
              if (av_cnt == HD-1) begin
                // Done with all elements of this head.
                if (head_idx == H_Q-1) begin
                  // All H_Q heads done — proceed to O-projection.
                  state    <= S_M_O;
                  chunk    <= '0;
                  mv_phase <= MV_LOAD_REQ;
                  head_idx <= '0;
                end else begin
                  // Next head: rerun scoring → softmax → AV for h+1.
                  head_idx <= head_idx + 1'b1;
                  state    <= S_ATTN_QK;
                  at_phase <= AT_INIT;
                  t_idx    <= '0;
                  dot_cnt  <= '0;
                  dot_acc  <= '0;
                end
              end else begin
                av_cnt   <= av_cnt + 1'b1;
                at_phase <= AT_AV_INIT;
              end
            end
            default: at_phase <= AT_AV_INIT;
          endcase
        end

        // ----------------------------------------------------------------
        //  Residual 1: hidden1_buf = hidden_in + o (D cycles, 1 elem each)
        // ----------------------------------------------------------------
        S_RESID1: begin
          begin
            // hidden1 = sat16( (hidden_in << sh_h_in_to_h1_signed) +
            //                  ((o_int16 * resid1_factor) >> 8 at h1's scale) )
            logic signed [40:0] r1_prod;
            logic signed [32:0] r1_delta;
            logic signed [15:0] hi;
            logic signed [31:0] hi_aligned;
            logic signed [33:0] r1_sum;
            hi       = $signed(hidden_in[cnt*16 +: 16]);
            r1_prod  = $signed(o_buf[cnt]) * $signed({1'b0, resid1_factor});
            r1_delta = $signed(r1_prod >>> 8);
            // Block-FP rescale of hidden_in to h1's scale.  Positive sh =
            // right shift (h1 has bigger scale, hidden gets smaller int).
            hi_aligned = (sh_h_in_to_h1 >= 0) ?
                ($signed({{16{hi[15]}}, hi}) >>> sh_h_in_to_h1) :
                ($signed({{16{hi[15]}}, hi}) <<< (-sh_h_in_to_h1));
            r1_sum = $signed({{2{hi_aligned[31]}}, hi_aligned})
                   + $signed({{1{r1_delta[32]}}, r1_delta});
            if      (r1_sum >  34'sd32767)  hidden1_buf[cnt] <=  16'sh7FFF;
            else if (r1_sum < -34'sd32768)  hidden1_buf[cnt] <=  16'sh8000;
            else                            hidden1_buf[cnt] <=  r1_sum[15:0];
          end
          cnt <= cnt + 1'b1;
          if (cnt == D-1) begin
            state    <= S_NORM2;
            nm_phase <= NM_IDLE;
            cnt      <= '0;
          end
        end

        // ----------------------------------------------------------------
        //  RMSNorm 2: hidden1 × γ₂ → norm2_buf
        // ----------------------------------------------------------------
        S_NORM2: begin
          case (nm_phase)
            NM_IDLE: begin
              rms2_start <= 1'b1;
              cnt        <= '0;
              nm_phase   <= NM_LOAD;
            end
            NM_LOAD: begin
              // hidden1_buf is 16-bit (block-FP) — feed directly.
              rms2_in_x <= hidden1_buf[cnt];
              // rms2_in_gamma is a wire from brom_GAMMA2's BRAM output.
              rms2_in_valid <= 1'b1;
              cnt           <= cnt + 1'b1;
              if (cnt == D-1) begin
                nm_phase <= NM_OUTPUT;
                cnt      <= '0;
              end
            end
            NM_OUTPUT: begin
              if (rms2_out_valid) begin
                norm2_buf[cnt] <= rms2_out_y;
                cnt            <= cnt + 1'b1;
                if (cnt == D-1) begin
                  state    <= S_M_GATE;
                  cnt      <= '0;
                  chunk    <= '0;
                  mv_phase <= MV_LOAD_REQ;
                end
              end
            end
            default: nm_phase <= NM_IDLE;
          endcase
        end

        // ----------------------------------------------------------------
        //  SwiGLU: streaming, 1-cycle latency.  Drive (gate, up, valid)
        //  for FFN cycles → capture FFN out_valid pulses.
        // ----------------------------------------------------------------
        S_SWIGLU: begin
          case (sg_phase)
            SG_RUN: begin
              if (cnt < FFN) begin
                sg_in_gate  <= gate_buf[cnt];
                sg_in_up    <= up_buf  [cnt];
                sg_in_valid <= 1'b1;
                cnt         <= cnt + 1'b1;
              end
              // Capture out_valid pulses (offset by 1 from drives)
              if (sg_out_valid) begin
                mlp_buf[av_cnt] <= sg_out_y;
                av_cnt          <= av_cnt + 1'b1;
                if (av_cnt == FFN-1) begin
                  state    <= S_M_DOWN;
                  cnt      <= '0;
                  av_cnt   <= '0;
                  chunk    <= '0;
                  mv_phase <= MV_LOAD_REQ;
                  sg_phase <= SG_RUN;
                end
              end
            end
            default: sg_phase <= SG_RUN;
          endcase
        end

        // ----------------------------------------------------------------
        //  Residual 2: hidden_out = hidden1 + down  (combinational via
        //  the generate block on hidden_out, gated by state==S_DONE)
        // ----------------------------------------------------------------
        S_RESID2: begin
          // No buffered residual2; hidden_out is computed combinationally
          // from hidden1_buf + down_buf in the genvar block above.
          state <= S_DONE;
        end

        // ----------------------------------------------------------------
        S_DONE: begin
          done <= 1'b1;
          // Time-multiplex handshake: when the wrapper drops `start` after
          // capturing hidden_out, return to S_IDLE so the next layer's
          // call can begin (KV cache persists — no PRE_INIT this time).
          if (!start) state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

`ifdef MICROGPT_DDR3_WEIGHTS
  // ------------------------------------------------------------------------
  //  Named ILA taps — driven from the FSM/datapath signals that already exist
  //  in this module.  Routed to the top-level ila_core instance.
  // ------------------------------------------------------------------------
  assign ila_state          = state;
  assign ila_mv_phase       = mv_phase;
  assign ila_cnt            = cnt;
  assign ila_chunk          = chunk;
  assign ila_ws_matvec_id   = ws_matvec_id;
  assign ila_ws_load_req    = ws_load_req;
  assign ila_ws_ready       = ws_ready;
  assign ila_ws_rd_addr     = ws_rd_addr;
  assign ila_ws_weight_data = ws_weight_data;
  assign ila_eng_w          = eng_w;
  assign ila_eng_in_value   = eng_in_value;
  assign ila_eng_in_valid   = eng_in_valid;
  assign ila_eng_in_last    = eng_in_last;
  assign ila_eng_acc_clear  = eng_acc_clear;
  assign ila_eng_out_valid  = eng_out_valid;
`endif

endmodule

`default_nettype wire
