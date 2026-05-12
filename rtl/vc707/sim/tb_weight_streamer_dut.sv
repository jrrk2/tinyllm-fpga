// tb_weight_streamer_dut.sv — Verilator wrapper that ties weight_streamer
// to mock_axi_slave so the FPGA-side streamer can be unit-tested without
// MIG/DDR3 hardware.

`default_nettype none

module tb_weight_streamer_dut (
  input  wire                  clk,
  input  wire                  rst,
  input  wire [29:0]           matrix_base,
  input  wire [6:0]            chunk_idx,
  input  wire [10:0]           in_dim,
  input  wire                  load_req,
  output wire                  ready,
  output wire                  busy,
  input  wire [10:0]           rd_addr,
  output wire [127:0]          weight_data
);

  wire        arvalid, arready, arlock, rvalid, rready, rlast;
  wire [4:0]  arid, rid;
  wire [29:0] araddr;
  wire [7:0]  arlen;
  wire [2:0]  arsize, arprot;
  wire [1:0]  arburst, rresp;
  wire [3:0]  arcache, arqos;
  wire [511:0] rdata;

  weight_streamer_mt #(
    .AXI_DATA_WIDTH(512),
    .AXI_ADDR_WIDTH(30),
    .AXI_ID_WIDTH(5),
    .IN_DIM_BITS(11),
    .CHUNK_BITS(7),
    .MAX_TILES(3),
    .TILE_ENTRIES(512),
    .BURST_LEN_LOG2(7)
  ) i_ws (
    .clk(clk), .rst(rst),
    .matrix_base(matrix_base),
    .chunk_idx(chunk_idx),
    .in_dim(in_dim),
    .load_req(load_req),
    .ready(ready),
    .busy(busy),
    .rd_addr(rd_addr),
    .weight_data(weight_data),

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
    .MEM_ENTRIES(8192)         // 128 KiB of 128-bit entries
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
