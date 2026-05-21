// tb_freeze_bfp.sv — Verilator harness to DEMONSTRATE the logic-analyser
// freeze in smollm_multilayer_tm_bfp without the full embed/decode pipeline.
//
// The C++ driver (tb_freeze_bfp.cpp) feeds:
//   - hidden_in_m/e : the Python BFP golden's layer-0 INPUT for a chosen
//     token-step (i.e. the embedding of that token), so the frozen layer's
//     hidden_out is directly comparable to the golden's per-layer dump.
//   - pos / kv_pos  : the token position for that step.
//   - freeze controls: either a counter match (freeze_en + snap_layer_sel +
//     snap_step_sel) or an absolute-cycle trigger (trig_cyc_en + trig_cyc).
//
// It then pulses `start` and waits for `done`.  When the engine freezes, the
// multilayer drives hidden_out <= h_state (the frozen layer's output), and
// asserts dbg_frozen; dbg_cur_layer / dbg_cyc report where it stopped.  The
// driver checks (a) it froze at the requested layer, (b) hidden_out matches
// the golden's layer-<snap_layer_sel> hidden for that step.
//
// Weights load from the sim_shadow $readmemh BRAMs (../generated/lbfp_*.hex,
// LBFP_STREAMER_SELFTEST) since the AXI streamer handshake is tied off.

`include "lbfp_full_cfg.svh"
`include "bfp_format.svh"
`include "lbfp_ddr3.svh"   // LBFP_BASE_* region offsets + LBFP_DDR3_ENTRIES

`default_nettype none

module tb_freeze_bfp (
  input  wire                                clk,
  input  wire                                rst,
  input  wire                                start,
  input  wire [10:0]                         pos,
  input  wire [6:0]                          kv_pos,
  input  wire signed [`LBFP_FULL_D*BFP_MANT_W-1:0]            hidden_in_m,
  input  wire signed [(`LBFP_FULL_D/BFP_TILE)*BFP_EXP_W-1:0]  hidden_in_e,
  // Freeze controls (mirror the regmap):
  input  wire                                freeze_en,        // 0x065
  input  wire [4:0]                          snap_layer_sel,   // 0x00A
  input  wire [10:0]                         snap_step_sel,    // 0x064
  input  wire                                trig_cyc_en,      // 0x067
  input  wire [31:0]                         trig_cyc,         // 0x066
  // Status + frozen hidden state for the driver to check:
  output wire                                done,
  output wire                                dbg_frozen,       // 0x072[5]
  output wire [31:0]                         dbg_cyc,          // 0x071
  output wire [4:0]                          dbg_cur_layer,    // 0x072[4:0]
  output wire signed [`LBFP_FULL_D*BFP_MANT_W-1:0]            hidden_out_m,
  output wire signed [(`LBFP_FULL_D/BFP_TILE)*BFP_EXP_W-1:0]  hidden_out_e
);

  localparam int D       = `LBFP_FULL_D;
  localparam int H_Q     = `LBFP_FULL_HQ;
  localparam int H_KV    = `LBFP_FULL_HKV;
  localparam int HD      = `LBFP_FULL_HD;
  localparam int FFN     = `LBFP_FULL_FFN;
  localparam int NL      = `LBFP_FULL_NL;
  localparam int MAX_CTX = `LBFP_FULL_MAX_CTX;

  // Minimal start/done shim around the multilayer (no embed/decode).
  logic lay_start;
  logic done_r;
  typedef enum logic [1:0] { S_IDLE, S_LAY, S_LAY_WAIT, S_DONE } st_t;
  st_t state;
  always_ff @(posedge clk) begin
    if (rst) begin
      state <= S_IDLE; lay_start <= 1'b0; done_r <= 1'b0;
    end else begin
      lay_start <= 1'b0; done_r <= 1'b0;
      case (state)
        S_IDLE:     if (start) begin lay_start <= 1'b1; state <= S_LAY; end
        S_LAY:      state <= S_LAY_WAIT;
        S_LAY_WAIT: if (lay_done) state <= S_DONE;
        S_DONE:     begin done_r <= 1'b1; state <= S_IDLE; end
        default:    state <= S_IDLE;
      endcase
    end
  end
  assign done = done_r;

  wire lay_done;

  // AXI fabric between the multilayer's weight streamer and the mock DDR3.
  wire        arvalid, arready, arlock;
  wire [4:0]  arid, rid;
  wire [29:0] araddr;
  wire [7:0]  arlen;
  wire [2:0]  arsize, arprot;
  wire [1:0]  arburst, rresp;
  wire [3:0]  arcache, arqos;
  wire        rvalid, rready, rlast;
  wire [511:0] rdata;

  smollm_multilayer_tm_bfp #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD),
    .FFN(FFN), .MAX_CTX(MAX_CTX), .NL(NL), .PREFIX("../generated/lbfp_full_")
  ) i_lay (
    .clk(clk), .rst(rst), .start(lay_start),
    .pos(pos), .kv_pos(kv_pos),
    .hidden_in_m(hidden_in_m), .hidden_in_e(hidden_in_e),
    .hidden_out_m(hidden_out_m), .hidden_out_e(hidden_out_e),
    .done(lay_done),
    .weight_hash(/*unused*/),
    .wr_kind(5'd0), .wr_addr(18'd0), .wr_data(16'd0), .wr_en(1'b0),
    .clk_wr(clk), .wr_rdata(/*unused*/),
    // Freeze / logic-analyser controls under test:
    .snap_layer_sel(snap_layer_sel), .snap_step_sel(snap_step_sel),
    .freeze_en(freeze_en),
    .trig_cyc_en(trig_cyc_en), .trig_cyc(trig_cyc),
    .dbg_cyc(dbg_cyc), .dbg_cur_layer(dbg_cur_layer), .dbg_frozen(dbg_frozen),
    // Real per-layer weights streamed from the mock DDR3 (all NL layers run).
    .ws_base_WQ_m(`LBFP_BASE_WQ_M),   .ws_base_WQ_e(`LBFP_BASE_WQ_E),
    .ws_base_WK_m(`LBFP_BASE_WK_M),   .ws_base_WK_e(`LBFP_BASE_WK_E),
    .ws_base_WV_m(`LBFP_BASE_WV_M),   .ws_base_WV_e(`LBFP_BASE_WV_E),
    .ws_base_WO_m(`LBFP_BASE_WO_M),   .ws_base_WO_e(`LBFP_BASE_WO_E),
    .ws_base_WG_m(`LBFP_BASE_WG_M),   .ws_base_WG_e(`LBFP_BASE_WG_E),
    .ws_base_WU_m(`LBFP_BASE_WU_M),   .ws_base_WU_e(`LBFP_BASE_WU_E),
    .ws_base_WDN_m(`LBFP_BASE_WDN_M), .ws_base_WDN_e(`LBFP_BASE_WDN_E),
    .clk_axi(clk), .rst_axi(rst),
    .m_axi_arvalid(arvalid), .m_axi_arready(arready),
    .m_axi_arid(arid),       .m_axi_araddr(araddr),
    .m_axi_arlen(arlen),     .m_axi_arsize(arsize),
    .m_axi_arburst(arburst), .m_axi_arlock(arlock),
    .m_axi_arcache(arcache), .m_axi_arprot(arprot),
    .m_axi_arqos(arqos),
    .m_axi_rvalid(rvalid),   .m_axi_rready(rready),
    .m_axi_rid(rid),         .m_axi_rdata(rdata),
    .m_axi_rresp(rresp),     .m_axi_rlast(rlast)
  );

  mock_axi_slave #(
    .AXI_DATA_WIDTH(512), .AXI_ADDR_WIDTH(30), .AXI_ID_WIDTH(5),
    .MEM_ENTRIES   (`LBFP_DDR3_ENTRIES),
    .INIT_HEX_FILE ("../generated/lbfp_full_DDR3.hex")
  ) i_mem (
    .clk(clk), .rst(rst),
    .m_axi_arvalid(arvalid), .m_axi_arready(arready),
    .m_axi_arid(arid),       .m_axi_araddr(araddr),
    .m_axi_arlen(arlen),     .m_axi_arsize(arsize),
    .m_axi_arburst(arburst), .m_axi_arlock(arlock),
    .m_axi_arcache(arcache), .m_axi_arprot(arprot),
    .m_axi_arqos(arqos),
    .m_axi_rvalid(rvalid),   .m_axi_rready(rready),
    .m_axi_rid(rid),         .m_axi_rdata(rdata),
    .m_axi_rresp(rresp),     .m_axi_rlast(rlast)
  );

endmodule

`default_nettype wire
