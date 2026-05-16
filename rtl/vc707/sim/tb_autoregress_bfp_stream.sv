// tb_autoregress_bfp_stream.sv — Verilator wrapper for the streaming
// (STREAM_WEIGHTS=1, STREAM_LOOKUP=1) BFP autoregress stack.  AXI master
// is routed to mock_axi_slave backed by lbfp_full_DDR3.hex.  C++ harness
// (tb_autoregress_bfp_stream.cpp) drives clock / reset / go and reads
// result_tokens at done.

`include "../generated/lbfp_full_cfg.svh"
`include "bfp_format.svh"
`include "lbfp_ddr3.svh"

`default_nettype none

module tb_autoregress_bfp_stream (
  input  wire                                                 clk,
  input  wire                                                 rst,
  input  wire                                                 go,
  output wire                                                 done,
  output wire [(`LBFP_FULL_NPROMPT + `LBFP_FULL_NGEN)*16-1:0] result_tokens
);

  // AXI fabric between DUT master and mock slave.
  wire        arvalid, arready, arlock;
  wire [4:0]  arid, rid;
  wire [29:0] araddr;
  wire [7:0]  arlen;
  wire [2:0]  arsize, arprot;
  wire [1:0]  arburst, rresp;
  wire [3:0]  arcache, arqos;
  wire        rvalid, rready, rlast;
  wire [511:0] rdata;

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
    .STREAM_LOOKUP(1'b1),
    .AXI_ADDR_WIDTH(30),   .AXI_ID_WIDTH(5)
  ) dut (
    .clk(clk), .rst(rst), .start(go),
    .done(done), .result_tokens(result_tokens),
    .ws_base_WQ_m       (`LBFP_BASE_WQ_M),
    .ws_base_WQ_e       (`LBFP_BASE_WQ_E),
    .ws_base_WK_m       (`LBFP_BASE_WK_M),
    .ws_base_WK_e       (`LBFP_BASE_WK_E),
    .ws_base_WV_m       (`LBFP_BASE_WV_M),
    .ws_base_WV_e       (`LBFP_BASE_WV_E),
    .ws_base_WO_m       (`LBFP_BASE_WO_M),
    .ws_base_WO_e       (`LBFP_BASE_WO_E),
    .ws_base_WG_m       (`LBFP_BASE_WG_M),
    .ws_base_WG_e       (`LBFP_BASE_WG_E),
    .ws_base_WU_m       (`LBFP_BASE_WU_M),
    .ws_base_WU_e       (`LBFP_BASE_WU_E),
    .ws_base_WDN_m      (`LBFP_BASE_WDN_M),
    .ws_base_WDN_e      (`LBFP_BASE_WDN_E),
    .ws_base_EMBED_m    (`LBFP_BASE_EMBED_M),
    .ws_base_EMBED_e    (`LBFP_BASE_EMBED_E),
    .ws_base_EMBED_LU_m (`LBFP_BASE_EMBED_LU_M),
    .ws_base_EMBED_LU_e (`LBFP_BASE_EMBED_LU_E),
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
    .AXI_DATA_WIDTH(512),
    .AXI_ADDR_WIDTH(30),
    .AXI_ID_WIDTH  (5),
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
