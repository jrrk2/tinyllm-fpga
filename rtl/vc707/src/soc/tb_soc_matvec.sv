// tb_soc_matvec.sv — PicoSoC weight-feed DEBUG-MODE proof at the matvec level.
//
// The PicoSoC firmware (fw_matvec.c) feeds a real matvec_bfp_engine beat-by-beat
// over iomem: an IDENTITY weight matrix (diagonal = 1.0) and a known x vector.
// With identity weights the matvec output must equal x (within BFP rounding),
// so no requant golden is needed.  This proves the SoC can deliver weights one
// beat at a time and the engine computes correctly off that (slow, gapped,
// MIG-free) feed — the mechanism that will isolate the streamer<->MIG race.

`include "bfp_format.svh"
`default_nettype none

module tb_soc_matvec (
  input  wire        clk,
  input  wire        resetn,
  output wire        out_captured,
  output wire signed [16*BFP_MANT_W-1:0] out_mant,
  output wire signed [16*BFP_EXP_W -1:0] out_exp
);

  wire        iomem_valid, iomem_ready;
  wire [3:0]  iomem_wstrb;
  wire [31:0] iomem_addr, iomem_wdata, iomem_rdata;
  wire        ser_tx;

  picosoc_noflash soc (
    .clk(clk), .resetn(resetn),
    .iomem_valid(iomem_valid), .iomem_ready(iomem_ready),
    .iomem_wstrb(iomem_wstrb), .iomem_addr(iomem_addr),
    .iomem_wdata(iomem_wdata), .iomem_rdata(iomem_rdata),
    .irq_5(1'b0), .irq_6(1'b0), .irq_7(1'b0),
    .ser_tx(ser_tx), .ser_rx(1'b1)
  );

  soc_matvec_feed #(.LANES(16)) feed (
    .clk(clk), .resetn(resetn),
    .iomem_valid(iomem_valid), .iomem_ready(iomem_ready),
    .iomem_wstrb(iomem_wstrb), .iomem_addr(iomem_addr),
    .iomem_wdata(iomem_wdata), .iomem_rdata(iomem_rdata),
    .out_captured(out_captured), .out_mant_o(out_mant), .out_exp_o(out_exp)
  );

endmodule

`default_nettype wire
