// tb_idle_scan_crc_dut.sv — ties idle_scan_crc to the read-only mock_axi_slave
// (mem[i] = i) so the scan master's AXI read FSM + rolling hash can be unit-
// tested against a computable reference, without the full top / DDR3.
`default_nettype none
module tb_idle_scan_crc_dut (
  input  wire        clk,
  input  wire        rst,
  input  wire [29:0] base,
  input  wire [31:0] len,
  input  wire        trig,
  output wire        busy,
  output wire        done,
  output wire [31:0] crc
);
  wire        arvalid, arready, arlock, rvalid, rready, rlast;
  wire [4:0]  arid, rid;
  wire [29:0] araddr;
  wire [7:0]  arlen;
  wire [2:0]  arsize, arprot;
  wire [1:0]  arburst, rresp;
  wire [3:0]  arcache, arqos;
  wire [511:0]rdata;

  idle_scan_crc #(.AXI_ADDR_WIDTH(30), .AXI_DATA_WIDTH(512), .AXI_ID_WIDTH(5)) dut (
    .clk(clk), .rst(rst),
    .base(base), .len(len), .trig(trig),
    .busy(busy), .done(done), .crc(crc),
    .m_axi_arvalid(arvalid), .m_axi_arready(arready), .m_axi_arid(arid),
    .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize),
    .m_axi_arburst(arburst), .m_axi_arlock(arlock), .m_axi_arcache(arcache),
    .m_axi_arprot(arprot), .m_axi_arqos(arqos),
    .m_axi_rvalid(rvalid), .m_axi_rready(rready), .m_axi_rid(rid),
    .m_axi_rdata(rdata), .m_axi_rresp(rresp), .m_axi_rlast(rlast)
  );

  mock_axi_slave #(.AXI_DATA_WIDTH(512), .AXI_ADDR_WIDTH(30), .AXI_ID_WIDTH(5),
                   .MEM_ENTRIES(8192)) slave (
    .clk(clk), .rst(rst),
    .m_axi_arvalid(arvalid), .m_axi_arready(arready), .m_axi_arid(arid),
    .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize),
    .m_axi_arburst(arburst), .m_axi_arlock(arlock), .m_axi_arcache(arcache),
    .m_axi_arprot(arprot), .m_axi_arqos(arqos),
    .m_axi_rvalid(rvalid), .m_axi_rready(rready), .m_axi_rid(rid),
    .m_axi_rdata(rdata), .m_axi_rresp(rresp), .m_axi_rlast(rlast)
  );
endmodule
`default_nettype wire
