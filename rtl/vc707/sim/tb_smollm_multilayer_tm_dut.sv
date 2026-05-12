// tb_smollm_multilayer_tm_dut.sv — Verilator wrapper exercising the
// time-multiplexed multilayer (smollm_multilayer_tm) against a hex-loaded
// mock_axi_slave.  Validates the full TM data path: layer_idx-driven
// scale/gamma/KV indexing, layer_base_addr-offset DDR3 fetches, hidden
// state cascading across NL iterations.

`include "tm_layer_ddr3_bases.svh"

`default_nettype none

module tb_smollm_multilayer_tm_dut #(
  parameter int    D       = 128,
  parameter int    H_Q     = 2,
  parameter int    H_KV    = 1,
  parameter int    HD      = 64,
  parameter int    FFN     = 128,
  parameter int    MAX_CTX = 4,
  parameter int    NL      = `MICROGPT_TM_NL
)(
  input  wire                          clk,
  input  wire                          rst,
  input  wire                          start,
  input  wire [10:0]                   pos,
  input  wire [4:0]                    kv_pos,
  input  wire signed [D*16-1:0]        hidden_in,
  output wire signed [D*16-1:0]        hidden_out,
  output wire                          done
);

  // AXI between layer's streamer and mock memory
  wire        arvalid, arready, arlock, rvalid, rready, rlast;
  wire [4:0]  arid, rid;
  wire [29:0] araddr;
  wire [7:0]  arlen;
  wire [2:0]  arsize, arprot;
  wire [1:0]  arburst, rresp;
  wire [3:0]  arcache, arqos;
  wire [511:0] rdata;

  smollm_multilayer_tm #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX),
    .NL(NL),
    .PREFIX("tm_layer_"),
    .LAYER_BYTES(`MICROGPT_TM_LAYER_BYTES),
    .BASE_Q   (`MICROGPT_TM_BASE_Q),
    .BASE_K   (`MICROGPT_TM_BASE_K),
    .BASE_V   (`MICROGPT_TM_BASE_V),
    .BASE_O   (`MICROGPT_TM_BASE_O),
    .BASE_GATE(`MICROGPT_TM_BASE_GATE),
    .BASE_UP  (`MICROGPT_TM_BASE_UP),
    .BASE_DOWN(`MICROGPT_TM_BASE_DOWN)
  ) i_ml (
    .clk(clk), .rst(rst),
    .start(start), .pos(pos), .kv_pos(kv_pos),
    .hidden_in(hidden_in),
    .hidden_out(hidden_out),
    .done(done),
    // Verilator test: NL → live final hidden_state (legacy behaviour).
    .snapshot_layer_sel(5'd31),
    .factor_wr_layer         (5'd0),
    .factor_wr_data          (32'd0),
    .factor_wr_en_swiglu_lo  (1'b0),
    .factor_wr_en_swiglu_mlp (1'b0),
    .factor_wr_en_attn       (1'b0),
    .factor_rd_sel           (7'b0),
    .factor_rd_data          (/* unused */),
    .factor_ram_por_init     (rst),
    .scale_wr_kind           (4'b0),
    .scale_wr_addr           (16'b0),
    .scale_wr_data           (16'b0),
    .scale_wr_en             (1'b0),
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
  );

  mock_axi_slave #(
    .AXI_DATA_WIDTH(512),
    .AXI_ADDR_WIDTH(30),
    .AXI_ID_WIDTH(5),
    .MEM_ENTRIES(`MICROGPT_TM_DDR3_ENTRIES),
`ifdef MICROGPT_WEIGHT_DIR
    .INIT_HEX_FILE({`MICROGPT_WEIGHT_DIR, "/tm_layer_DDR3.hex"})
`else
    .INIT_HEX_FILE("tm_layer_DDR3.hex")
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
