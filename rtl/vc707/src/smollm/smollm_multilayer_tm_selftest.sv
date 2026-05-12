// smollm_multilayer_tm_selftest.sv — FPGA wrapper around smollm_multilayer_tm.
//
// Drives one NL-layer time-multiplexed forward pass at boot using the
// hidden_in case-statement ROM, latches RESULT_LANES of hidden_out for
// the host regmap, exposes a `restart` input.
//
// Mirrors smollm_layer_selftest.sv but instantiates the multilayer-tm
// wrapper instead of a bare smollm_layer.  AXI master ports + clk_axi
// forwarded out to the top.

`ifdef MICROGPT_DDR3_WEIGHTS
  `include "tm_layer_ddr3_bases.svh"
`endif

`default_nettype none

module smollm_multilayer_tm_selftest #(
  // SmolLM2-135M real dims by default; the time-mux refactor + buffer
  // BRAM inference gets one layer's compute to ~30 K LUTs, so NL=30
  // (sharing one instance) fits VC707 with plenty of headroom.
  parameter int D       = 576,
  parameter int H_Q     = 9,
  parameter int H_KV    = 3,
  parameter int HD      = 64,
  parameter int FFN     = 1536,
  parameter int MAX_CTX = 4,
  parameter int NL      = 30,
  // Expose ALL D lanes so the host can decode the full hidden_out
  // through SmolLM2's lm_head.  Wired into a wider regmap window in
  // vc707_microgpt_eth (288 32-bit words = 576 16-bit lanes).
  parameter int RESULT_LANES = 576
)(
  input  wire                          clk,
  input  wire                          rst,
  input  wire                          restart,
  output logic [RESULT_LANES*16-1:0]   result,
  output logic                         done,
  // Host-driven layer snapshot select (0..NL-1, or NL for live final).
  input  wire [4:0]                    snapshot_layer_sel,
  // Runtime factor-override write port (CDC'd from eth_clk regmap).
  input  wire [4:0]                    factor_wr_layer,
  input  wire [31:0]                   factor_wr_data,
  input  wire                          factor_wr_en_swiglu_lo,
  input  wire                          factor_wr_en_swiglu_mlp,
  input  wire                          factor_wr_en_attn,
  input  wire [6:0]                    factor_rd_sel,
  output wire [31:0]                   factor_rd_data

`ifdef MICROGPT_DDR3_WEIGHTS
  ,
  input  wire                          clk_axi,
  input  wire                          rst_axi,
  output wire                          m_axi_arvalid,
  input  wire                          m_axi_arready,
  output wire [4:0]                    m_axi_arid,
  output wire [29:0]                   m_axi_araddr,
  output wire [7:0]                    m_axi_arlen,
  output wire [2:0]                    m_axi_arsize,
  output wire [1:0]                    m_axi_arburst,
  output wire                          m_axi_arlock,
  output wire [3:0]                    m_axi_arcache,
  output wire [2:0]                    m_axi_arprot,
  output wire [3:0]                    m_axi_arqos,
  input  wire                          m_axi_rvalid,
  output wire                          m_axi_rready,
  input  wire  [4:0]                   m_axi_rid,
  input  wire  [511:0]                 m_axi_rdata,
  input  wire  [1:0]                   m_axi_rresp,
  input  wire                          m_axi_rlast,

  // Debug taps forwarded from inner smollm_multilayer_tm.  Always
  // present under DDR3 build; top-level uses MICROGPT_LAYER_DEBUG to
  // decide whether to wire them into the regmap.
  output wire  [29:0]                  dbg_first_araddr,
  output wire  [511:0]                 dbg_first_rdata,
  output wire                          dbg_first_ar_seen,
  output wire                          dbg_first_r_seen,
  output wire  [511:0]                 dbg_eng_w_packed,
  output wire  [511:0]                 dbg_wd_packed,
  output wire  [63:0]                  dbg_in_value_packed,
  output wire                          dbg_snap_done_o,
  output wire  [4:0]                   ila_state,
  output wire  [2:0]                   ila_mv_phase,
  output wire  [10:0]                  ila_cnt,
  output wire  [6:0]                   ila_chunk,
  output wire  [2:0]                   ila_ws_matvec_id,
  output wire                          ila_ws_load_req,
  output wire                          ila_ws_ready,
  output wire  [10:0]                  ila_ws_rd_addr,
  output wire  [127:0]                 ila_ws_weight_data,
  output wire  [127:0]                 ila_eng_w,
  output wire  [15:0]                  ila_eng_in_value,
  output wire                          ila_eng_in_valid,
  output wire                          ila_eng_in_last,
  output wire                          ila_eng_acc_clear,
  output wire                          ila_eng_out_valid,
  output wire  [2:0]                   ila_ws_state_axi,
  output wire                          ila_start_load_axi,
  output wire  [1:0]                   ila_tile_idx,
  output wire  [6:0]                   ila_beat_idx,
  output wire  [2:0]                   ila_ml_state,
  output wire  [4:0]                   ila_ml_layer_idx
`endif
);

  logic                ml_start;
  logic [10:0]         ml_pos;
  logic [4:0]          ml_kv_pos;
  wire  [D*16-1:0]     ml_hidden_in;
  wire  [D*16-1:0]     ml_hidden_out;
  logic                ml_done;

  // Hidden_in baked in as a case-statement ROM (16-bit Q1.15 at layer 0's
  // h_in_p2 scale — emitted by gen_smollm_calib.py).
`include "layer_hidden_in_packed.svh"
  genvar gh;
  generate
    for (gh = 0; gh < D; gh++)
      assign ml_hidden_in[gh*16 +: 16] = layer_hidden_in_lut(gh);
  endgenerate

  smollm_multilayer_tm #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX),
    .NL(NL),
    .PREFIX("tm_layer_")
`ifdef MICROGPT_DDR3_WEIGHTS
    ,
    .LAYER_BYTES(`MICROGPT_TM_LAYER_BYTES),
    .BASE_Q   (`MICROGPT_TM_BASE_Q),
    .BASE_K   (`MICROGPT_TM_BASE_K),
    .BASE_V   (`MICROGPT_TM_BASE_V),
    .BASE_O   (`MICROGPT_TM_BASE_O),
    .BASE_GATE(`MICROGPT_TM_BASE_GATE),
    .BASE_UP  (`MICROGPT_TM_BASE_UP),
    .BASE_DOWN(`MICROGPT_TM_BASE_DOWN)
`endif
  ) i_ml (
    .clk(clk), .rst(rst | restart),
    .start(ml_start),
    .pos(ml_pos), .kv_pos(ml_kv_pos),
    .hidden_in(ml_hidden_in),
    .hidden_out(ml_hidden_out),
    .done(ml_done),
    .snapshot_layer_sel(snapshot_layer_sel),
    .factor_wr_layer        (factor_wr_layer),
    .factor_wr_data         (factor_wr_data),
    .factor_wr_en_swiglu_lo (factor_wr_en_swiglu_lo),
    .factor_wr_en_swiglu_mlp(factor_wr_en_swiglu_mlp),
    .factor_wr_en_attn      (factor_wr_en_attn),
    .factor_rd_sel          (factor_rd_sel),
    .factor_rd_data         (factor_rd_data),
    // factor_ram only resets on system rst, NOT on user restart — so the
    // runtime tweaks survive REG_RESTART triggering a new inference run.
    .factor_ram_por_init    (rst)
`ifdef MICROGPT_DDR3_WEIGHTS
    ,
    .clk_axi(clk_axi), .rst_axi(rst_axi),
    .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
    .m_axi_arid(m_axi_arid),       .m_axi_araddr(m_axi_araddr),
    .m_axi_arlen(m_axi_arlen),     .m_axi_arsize(m_axi_arsize),
    .m_axi_arburst(m_axi_arburst), .m_axi_arlock(m_axi_arlock),
    .m_axi_arcache(m_axi_arcache), .m_axi_arprot(m_axi_arprot),
    .m_axi_arqos(m_axi_arqos),
    .m_axi_rvalid(m_axi_rvalid),   .m_axi_rready(m_axi_rready),
    .m_axi_rid(m_axi_rid),         .m_axi_rdata(m_axi_rdata),
    .m_axi_rresp(m_axi_rresp),     .m_axi_rlast(m_axi_rlast),
    // Debug forwards
    .dbg_first_araddr   (dbg_first_araddr   ),
    .dbg_first_rdata    (dbg_first_rdata    ),
    .dbg_first_ar_seen  (dbg_first_ar_seen  ),
    .dbg_first_r_seen   (dbg_first_r_seen   ),
    .dbg_eng_w_packed   (dbg_eng_w_packed   ),
    .dbg_wd_packed      (dbg_wd_packed      ),
    .dbg_in_value_packed(dbg_in_value_packed),
    .dbg_snap_done_o    (dbg_snap_done_o    ),
    .ila_state          (ila_state          ),
    .ila_mv_phase       (ila_mv_phase       ),
    .ila_cnt            (ila_cnt            ),
    .ila_chunk          (ila_chunk          ),
    .ila_ws_matvec_id   (ila_ws_matvec_id   ),
    .ila_ws_load_req    (ila_ws_load_req    ),
    .ila_ws_ready       (ila_ws_ready       ),
    .ila_ws_rd_addr     (ila_ws_rd_addr     ),
    .ila_ws_weight_data (ila_ws_weight_data ),
    .ila_eng_w          (ila_eng_w          ),
    .ila_eng_in_value   (ila_eng_in_value   ),
    .ila_eng_in_valid   (ila_eng_in_valid   ),
    .ila_eng_in_last    (ila_eng_in_last    ),
    .ila_eng_acc_clear  (ila_eng_acc_clear  ),
    .ila_eng_out_valid  (ila_eng_out_valid  ),
    .ila_ws_state_axi   (ila_ws_state_axi   ),
    .ila_start_load_axi (ila_start_load_axi ),
    .ila_tile_idx       (ila_tile_idx       ),
    .ila_beat_idx       (ila_beat_idx       ),
    .ila_ml_state       (ila_ml_state       ),
    .ila_ml_layer_idx   (ila_ml_layer_idx   )
`endif
  );

  always_ff @(posedge clk) begin
    if (rst | restart) begin
      ml_start    <= 1'b0;
      ml_pos      <= 11'd3;
      ml_kv_pos   <= 5'd3;
      result      <= '0;
      done        <= 1'b0;
    end else begin
      if (!done) ml_start <= 1'b1;
      if (ml_done && !done) begin
        done     <= 1'b1;
        ml_start <= 1'b0;
      end
      // Continuously copy ml_hidden_out → result once inference is done.
      // The multilayer wrapper drives ml_hidden_out from the per-layer
      // snapshot RAM (selected by snapshot_layer_sel) — without this
      // continuous copy the result was frozen at the original done edge
      // and snap_sel changes had no effect.
      if (done || (ml_done && !done))
        result <= ml_hidden_out[RESULT_LANES*16-1:0];
    end
  end

endmodule

`default_nettype wire
