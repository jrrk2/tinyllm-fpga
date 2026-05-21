// tb_soc_regmap.sv — Stage-1 PicoSoC integration proof.
//
// Instantiates picosoc_noflash (PicoRV32 + progmem firmware + UART) and a
// 256-word register-map STUB on the iomem bus at 0x1000_0000.  The stage-1
// firmware (fw_stage1.c) writes a known value, reads it back through this
// bridge, echoes it, and raises a done sentinel.  The C++ driver checks the
// readback — proving the SoC can drive (and read) the engine register map.
//
// This is the control bridge that will later replace microgpt_eth_ctrl's HW
// frame-type parsing: the SoC mediates engine control instead of raw packets.

`default_nettype none

module tb_soc_regmap (
  input  wire        clk,
  input  wire        resetn,
  output wire        dbg_done,
  output wire [31:0] dbg_r1,
  output wire [31:0] dbg_r2,
  output wire [31:0] dbg_r3
);

  wire        iomem_valid;
  reg         iomem_ready;
  wire [3:0]  iomem_wstrb;
  wire [31:0] iomem_addr;
  wire [31:0] iomem_wdata;
  reg  [31:0] iomem_rdata;
  wire        ser_tx;

  picosoc_noflash soc (
    .clk        (clk),
    .resetn     (resetn),
    .iomem_valid(iomem_valid),
    .iomem_ready(iomem_ready),
    .iomem_wstrb(iomem_wstrb),
    .iomem_addr (iomem_addr),
    .iomem_wdata(iomem_wdata),
    .iomem_rdata(iomem_rdata),
    .irq_5      (1'b0),
    .irq_6      (1'b0),
    .irq_7      (1'b0),
    .ser_tx     (ser_tx),
    .ser_rx     (1'b1)
  );

  // ---- iomem regmap stub: 256 words at 0x10xx_xxxx -------------------------
  reg [31:0] regs [0:255];
  wire        sel  = iomem_valid && (iomem_addr[31:24] == 8'h10);
  wire [7:0]  widx = iomem_addr[9:2];

  always_ff @(posedge clk) begin
    if (!resetn) begin
      iomem_ready <= 1'b0;
      iomem_rdata <= 32'h0;
    end else begin
      iomem_ready <= 1'b0;
      // Single-cycle handshake: respond once per transaction.
      if (iomem_valid && !iomem_ready) begin
        if (sel) begin
          if (iomem_wstrb[0]) regs[widx][ 7: 0] <= iomem_wdata[ 7: 0];
          if (iomem_wstrb[1]) regs[widx][15: 8] <= iomem_wdata[15: 8];
          if (iomem_wstrb[2]) regs[widx][23:16] <= iomem_wdata[23:16];
          if (iomem_wstrb[3]) regs[widx][31:24] <= iomem_wdata[31:24];
          iomem_rdata <= regs[widx];
        end else begin
          iomem_rdata <= 32'h0;   // ack stray iomem (e.g. LEDs) so CPU advances
        end
        iomem_ready <= 1'b1;
      end
    end
  end

  assign dbg_done = (regs[255] == 32'h600DCAFE);
  assign dbg_r1   = regs[1];
  assign dbg_r2   = regs[2];
  assign dbg_r3   = regs[3];

endmodule

`default_nettype wire
