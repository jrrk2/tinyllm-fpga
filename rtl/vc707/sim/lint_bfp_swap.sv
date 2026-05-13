// Stub module that exercises the same instantiation pattern as the
// MICROGPT_USE_BFP branch of vc707_microgpt_eth.sv after the swap to
// autoregress_bfp_top.  Lets Verilator lint-check the port connections
// without pulling in Xilinx primitives.
`include "../generated/lbfp_full_cfg.svh"
`include "bfp_format.svh"
module lint_bfp_swap (
  input  wire        clk,
  input  wire        rst,
  input  wire        restart,
  output wire [9215:0] lay_result,
  output wire          lay_done_core,
  output wire [31:0]   factor_rd_data_core
);
  localparam int LBFP_NSTEPS = `LBFP_FULL_NPROMPT + `LBFP_FULL_NGEN;
  wire [LBFP_NSTEPS*16-1:0] bfp_result_tokens;

  reg bfp_start_r = 1'b0;
  always_ff @(posedge clk) begin
    if (rst | restart)       bfp_start_r <= 1'b0;
    else if (!lay_done_core) bfp_start_r <= 1'b1;
    else                     bfp_start_r <= 1'b0;
  end

  autoregress_bfp_top #(
    .D       (`LBFP_FULL_D),
    .H_Q     (`LBFP_FULL_HQ),
    .H_KV    (`LBFP_FULL_HKV),
    .HD      (`LBFP_FULL_HD),
    .FFN     (`LBFP_FULL_FFN),
    .NL      (`LBFP_FULL_NL),
    .MAX_CTX (`LBFP_FULL_MAX_CTX),
    .VOCAB   (`LBFP_FULL_VOCAB),
    .N_PROMPT(`LBFP_FULL_NPROMPT),
    .N_GEN   (`LBFP_FULL_NGEN),
    .PREFIX  ("../generated/lbfp_full_"),
    .STREAM_WEIGHTS(1'b0), .STREAM_LOOKUP(1'b0)
  ) i_lay_st (
    .clk(clk), .rst(rst | restart), .start(bfp_start_r),
    .done(lay_done_core), .result_tokens(bfp_result_tokens),
    .ws_base_WQ_m('0), .ws_base_WQ_e('0),
    .ws_base_WK_m('0), .ws_base_WK_e('0),
    .ws_base_WV_m('0), .ws_base_WV_e('0),
    .ws_base_WO_m('0), .ws_base_WO_e('0),
    .ws_base_WG_m('0), .ws_base_WG_e('0),
    .ws_base_WU_m('0), .ws_base_WU_e('0),
    .ws_base_WDN_m('0), .ws_base_WDN_e('0),
    .ws_base_EMBED_m('0), .ws_base_EMBED_e('0),
    .ws_base_EMBED_LU_m('0), .ws_base_EMBED_LU_e('0),
    .clk_axi(clk), .rst_axi(rst | restart),
    .m_axi_arvalid(), .m_axi_arready(1'b0), .m_axi_arid(), .m_axi_araddr(),
    .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(), .m_axi_arlock(),
    .m_axi_arcache(), .m_axi_arprot(), .m_axi_arqos(),
    .m_axi_rvalid(1'b0), .m_axi_rready(),
    .m_axi_rid('0), .m_axi_rdata('0), .m_axi_rresp('0), .m_axi_rlast(1'b0)
  );
  assign lay_result = {{(9216 - LBFP_NSTEPS*16){1'b0}}, bfp_result_tokens};
  assign factor_rd_data_core = '0;
endmodule
