// tb_smollm_layer_ddr3_dut.sv — Verilator wrapper that exercises
// smollm_layer in `MICROGPT_DDR3_WEIGHTS mode against a hex-loaded
// mock_axi_slave.  Validates that the AXI streamer path produces the
// same trace_* outputs as the brom path (tested separately by
// tb_smollm_layer).

`include "layer_ddr3_bases.svh"

`default_nettype none

module tb_smollm_layer_ddr3_dut #(
  parameter int    D       = 576,
  parameter int    H_Q     = 9,
  parameter int    H_KV    = 3,
  parameter int    HD      = 64,
  parameter int    FFN     = 1536,
  parameter int    MAX_CTX = 4
)(
  input  wire                          clk,
  input  wire                          rst,
  input  wire                          start,
  input  wire [10:0]                   pos,
  input  wire [4:0]                    kv_pos,
  input  wire signed [D*16-1:0]        hidden_in,
  output wire signed [D*16-1:0]        hidden_out,
  output wire                          done

`ifdef DEBUG
  ,
  output wire signed [D*16-1:0]        trace_norm1,
  output wire signed [D*16-1:0]        trace_q,
  output wire signed [H_KV*HD*16-1:0]  trace_k,
  output wire signed [H_KV*HD*16-1:0]  trace_v,
  output wire signed [D*16-1:0]        trace_q_rot,
  output wire signed [H_KV*HD*16-1:0]  trace_k_rot,
  output wire signed [D*16-1:0]        trace_attn,
  output wire signed [D*16-1:0]        trace_hidden1,
  output wire signed [D*16-1:0]        trace_norm2,
  output wire signed [FFN*16-1:0]      trace_mlp_inter
`endif
);

  // AXI between layer and mock memory
  wire        arvalid, arready, arlock, rvalid, rready, rlast;
  wire [4:0]  arid, rid;
  wire [29:0] araddr;
  wire [7:0]  arlen;
  wire [2:0]  arsize, arprot;
  wire [1:0]  arburst, rresp;
  wire [3:0]  arcache, arqos;
  wire [511:0] rdata;

  smollm_layer #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX),
    .PREFIX("layer_"),
    .BASE_Q   (`MICROGPT_LAYER_BASE_Q),
    .BASE_K   (`MICROGPT_LAYER_BASE_K),
    .BASE_V   (`MICROGPT_LAYER_BASE_V),
    .BASE_O   (`MICROGPT_LAYER_BASE_O),
    .BASE_GATE(`MICROGPT_LAYER_BASE_GATE),
    .BASE_UP  (`MICROGPT_LAYER_BASE_UP),
    .BASE_DOWN(`MICROGPT_LAYER_BASE_DOWN)
  ) i_layer (
    .clk(clk), .rst(rst),
    .start(start), .pos(pos), .kv_pos(kv_pos),
    .hidden_in(hidden_in), .hidden_out(hidden_out), .done(done),

    // Single-clock test: tie clk_axi=clk, rst_axi=rst.
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

`ifdef DEBUG
    ,
    .trace_norm1(trace_norm1), .trace_q(trace_q),
    .trace_k(trace_k),         .trace_v(trace_v),
    .trace_q_rot(trace_q_rot), .trace_k_rot(trace_k_rot),
    .trace_attn(trace_attn),   .trace_hidden1(trace_hidden1),
    .trace_norm2(trace_norm2), .trace_mlp_inter(trace_mlp_inter)
`endif
  );

  mock_axi_slave #(
    .AXI_DATA_WIDTH(512),
    .AXI_ADDR_WIDTH(30),
    .AXI_ID_WIDTH(5),
    .MEM_ENTRIES(`MICROGPT_LAYER_DDR3_ENTRIES),
`ifdef MICROGPT_WEIGHT_DIR
    .INIT_HEX_FILE({`MICROGPT_WEIGHT_DIR, "/layer_DDR3.hex"})
`else
    .INIT_HEX_FILE("layer_DDR3.hex")
`endif
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
