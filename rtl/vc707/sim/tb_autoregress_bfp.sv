// tb_autoregress_bfp.sv — Verilator wrapper exercising the on-chip
// autoregressive BFP token generator (BRAM mode).  Trivially small:
// a single `go` pulse kicks off the loop, then we wait for `done` and
// snapshot the 19-token result bus.  The C++ harness reads result_tokens
// and decodes.

`include "../generated/lbfp_full_cfg.svh"
`include "bfp_format.svh"

`default_nettype none

module tb_autoregress_bfp (
  input  wire                                                 clk,
  input  wire                                                 rst,
  input  wire                                                 go,
  output wire                                                 done,
  output wire [(`LBFP_FULL_NPROMPT + `LBFP_FULL_NGEN)*16-1:0] result_tokens
);

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
  ) dut (
    .clk(clk), .rst(rst), .start(go),
    .done(done), .result_tokens(result_tokens),
    .ws_base_WQ_m('0), .ws_base_WQ_e('0),
    .ws_base_WK_m('0), .ws_base_WK_e('0),
    .ws_base_WV_m('0), .ws_base_WV_e('0),
    .ws_base_WO_m('0), .ws_base_WO_e('0),
    .ws_base_WG_m('0), .ws_base_WG_e('0),
    .ws_base_WU_m('0), .ws_base_WU_e('0),
    .ws_base_WDN_m('0), .ws_base_WDN_e('0),
    .ws_base_EMBED_m('0), .ws_base_EMBED_e('0),
    .ws_base_EMBED_LU_m('0), .ws_base_EMBED_LU_e('0),
    .clk_axi(clk), .rst_axi(rst),
    .m_axi_arvalid(), .m_axi_arready(1'b0), .m_axi_arid(), .m_axi_araddr(),
    .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(), .m_axi_arlock(),
    .m_axi_arcache(), .m_axi_arprot(), .m_axi_arqos(),
    .m_axi_rvalid(1'b0), .m_axi_rready(),
    .m_axi_rid('0), .m_axi_rdata('0), .m_axi_rresp('0), .m_axi_rlast(1'b0)
  );

endmodule

`default_nettype wire
