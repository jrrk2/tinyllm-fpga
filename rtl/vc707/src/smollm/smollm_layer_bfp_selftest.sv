// smollm_layer_bfp_selftest.sv — FPGA wrapper around smollm_layer_bfp.
//
// Bake-then-boot pattern (same shape as smollm_layer_selftest.sv):
//   - hidden_in baked into a case-statement lookup function (BFP mantissas
//     and per-tile exponents), then driven into the layer.
//   - weights/gammas/KV cache come from $readmemh hex files baked by
//     host/gen_smollm_blockfp_bfp.py (BRAM-loaded).
//   - One-shot start at boot; result snapshot of the first RESULT_LANES
//     hidden_out mantissas is latched and exposed for the host regmap.
//
// Small-dim config (D=64, H_Q=H_KV=1, HD=64, FFN=128, MAX_CTX=4) fits the
// VC707's BRAM budget after $readmemh weight ROMs are inferred.

`include "bfp_format.svh"

`default_nettype none

module smollm_layer_bfp_selftest #(
  parameter int D           = 64,
  parameter int H_Q         = 1,
  parameter int H_KV        = 1,
  parameter int HD          = 64,
  parameter int FFN         = 128,
  parameter int MAX_CTX     = 4,
  parameter int RESULT_LANES = 16    // first 16 hidden_out mantissas
)(
  input  wire                                clk,
  input  wire                                rst,
  input  wire                                restart,
  // Host weight-write port (forwarded to the inner layer's BRAM write port)
  input  wire [4:0]                          wr_kind,
  input  wire [17:0]                         wr_addr,
  input  wire [15:0]                         wr_data,
  input  wire                                wr_en,
  output logic [RESULT_LANES*BFP_MANT_W-1:0] result_m,
  output logic [(D/BFP_TILE)*BFP_EXP_W-1:0]  result_e,
  output logic                               done
);

  localparam int NT_D = D / BFP_TILE;

  logic                                  lay_start;
  logic [10:0]                           lay_pos;
  logic [6:0]                            lay_kv_pos;
  wire  signed [D*BFP_MANT_W-1:0]        lay_hidden_in_m;
  wire  signed [NT_D*BFP_EXP_W-1:0]      lay_hidden_in_e;
  wire  signed [D*BFP_MANT_W-1:0]        lay_hidden_out_m;
  wire  signed [NT_D*BFP_EXP_W-1:0]      lay_hidden_out_e;
  logic                                  lay_done;
  wire  [6:0]                            ignore_state;
  wire  [11:0]                           ignore_cnt;
  wire  [6:0]                            ignore_chunk;

  // Hidden_in / hidden_in_exp baked as case-statement lookup functions.
  // Same pattern as the existing layer_hidden_in_packed.svh — Vivado will
  // synthesize these as combinational LUT trees, avoiding the $readmemh-
  // into-comb-read-array hazard.
`include "lbfp_hidden_in_lut.svh"

  genvar gh;
  generate
    for (gh = 0; gh < D; gh++)
      assign lay_hidden_in_m[gh*BFP_MANT_W +: BFP_MANT_W] = lbfp_hin_m_lut(gh);
    for (gh = 0; gh < NT_D; gh++)
      assign lay_hidden_in_e[gh*BFP_EXP_W  +: BFP_EXP_W ] = lbfp_hin_e_lut(gh);
  endgenerate

  smollm_layer_bfp #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD),
    .FFN(FFN), .MAX_CTX(MAX_CTX), .PREFIX("lbfp_"),
    .STREAM_WEIGHTS(1'b0)
  ) i_layer (
    .clk(clk), .rst(rst | restart),
    .start(lay_start),
    .pos(lay_pos), .kv_pos(lay_kv_pos),
    .layer_idx(5'd0),         // single-layer selftest: always layer 0
    .hidden_in_m(lay_hidden_in_m),
    .hidden_in_e(lay_hidden_in_e),
    .hidden_out_m(lay_hidden_out_m),
    .hidden_out_e(lay_hidden_out_e),
    .done(lay_done),
    .wr_kind(wr_kind), .wr_addr(wr_addr),
    .wr_data(wr_data), .wr_en(wr_en),
    // DDR3 streamer ports tied off (selftest)
    .ws_base_WQ_m('0), .ws_base_WQ_e('0),
    .ws_base_WK_m('0), .ws_base_WK_e('0),
    .ws_base_WV_m('0), .ws_base_WV_e('0),
    .ws_base_WO_m('0), .ws_base_WO_e('0),
    .ws_base_WG_m('0), .ws_base_WG_e('0),
    .ws_base_WU_m('0), .ws_base_WU_e('0),
    .ws_base_WDN_m('0), .ws_base_WDN_e('0),
    .clk_axi(clk), .rst_axi(rst),
    .m_axi_arvalid(), .m_axi_arready(1'b0), .m_axi_arid(), .m_axi_araddr(),
    .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(), .m_axi_arlock(),
    .m_axi_arcache(), .m_axi_arprot(), .m_axi_arqos(),
    .m_axi_rvalid(1'b0), .m_axi_rready(),
    .m_axi_rid('0), .m_axi_rdata('0), .m_axi_rresp('0), .m_axi_rlast(1'b0),
    .dbg_state(ignore_state),
    .dbg_cnt(ignore_cnt),
    .dbg_chunk(ignore_chunk)
  );

  always_ff @(posedge clk) begin
    if (rst | restart) begin
      lay_start  <= 1'b0;
      lay_pos    <= 11'd3;
      lay_kv_pos <= 7'd3;
      result_m   <= '0;
      result_e   <= '0;
      done       <= 1'b0;
    end else begin
      if (!done) lay_start <= 1'b1;
      if (lay_done && !done) begin
        result_m  <= lay_hidden_out_m[RESULT_LANES*BFP_MANT_W-1:0];
        result_e  <= lay_hidden_out_e;
        done      <= 1'b1;
        lay_start <= 1'b0;
      end
    end
  end

endmodule

`default_nettype wire
