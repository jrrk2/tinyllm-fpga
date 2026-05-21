// vc707_picosoc_shell.sv — PicoSoC board bring-up shell for the VC707.
//
// Goal: get the PicoRV32 SoC running on real hardware with a DUMMY engine so we
// can iterate fast on the SoC + ethernet + DDR plumbing before re-attaching the
// real inference engine.  Reuses the proven VC707 infrastructure:
//   - framing_top_sgmii : 125 MHz eth_clk + the LSU frame-buffer bus (ethernet)
//   - xlnx_mig_7_ddr3   : DDR3 (512-bit AXI @ ui_clk 200 MHz)
//   - MMCM/reset, constraints/microgpt_eth.xdc
// The SoC runs on eth_clk (ethernet is zero-CDC; DDR goes through a CDC bridge).
//
// Build staging (this file grows per increment):
//   A (now): SoC boots, UART debug console, LED blink, PicoRV32 trace -> ILA.
//   B      : iomem 0x2000_0000 -> LSU frame bus (SoC sends/receives ethernet).
//   C      : iomem 0x3000_0000 -> MIG AXI (SoC reads/writes DDR3, CDC bridge).
//   D      : iomem 0x1000_0000 dummy engine register file (host-pokeable).
//
// iomem map (PicoSoC, addr[31:24] > 0x01):
//   0x0300_0000  LEDs
//   0x1000_0000  dummy engine regs (stub)
//   0x2000_0000  ethernet LSU frame bus (stub until B)
//   0x3000_0000  DDR3 access window (stub until C)

module vc707_picosoc_shell (
  input  wire         sys_clk_p,
  input  wire         sys_clk_n,
  input  wire         cpu_reset,        // active-high VC707 button

  // SGMII ethernet (Marvell PHY)
  input  wire         sgmii_rxp,
  input  wire         sgmii_rxn,
  output wire         sgmii_txp,
  output wire         sgmii_txn,
  input  wire         sgmii_refclk_p,
  input  wire         sgmii_refclk_n,
  output wire         eth_rst_n,
  inout  wire         eth_mdio,
  output wire         eth_mdc,

  // USB-UART debug console (CP2103) — see XDC for pins
  output wire         usb_uart_tx,      // FPGA -> host
  input  wire         usb_uart_rx,      // host -> FPGA

  output wire [7:0]   led,
  input  wire [7:0]   sw,
  output wire         fan_pwm,

  // DDR3 SODIMM
  inout  wire [63:0]  ddr3_dq,
  inout  wire [7:0]   ddr3_dqs_p,
  inout  wire [7:0]   ddr3_dqs_n,
  output wire [13:0]  ddr3_addr,
  output wire [2:0]   ddr3_ba,
  output wire         ddr3_ras_n,
  output wire         ddr3_cas_n,
  output wire         ddr3_we_n,
  output wire         ddr3_reset_n,
  output wire [0:0]   ddr3_ck_p,
  output wire [0:0]   ddr3_ck_n,
  output wire [0:0]   ddr3_cke,
  output wire [0:0]   ddr3_cs_n,
  output wire [7:0]   ddr3_dm,
  output wire [0:0]   ddr3_odt
);

  assign fan_pwm = 1'b1;   // run the fan

  // ----------------------------------------------------------------------
  // MIG DDR3 — provides ui_clk (200 MHz) which framing uses as clk_int.
  // ----------------------------------------------------------------------
  wire        ui_clk;
  wire        ui_clk_sync_rst;
  wire        mmcm_locked_mig;
  wire        init_calib_complete;
  wire        sys_clk_bufg = ui_clk;

  logic [3:0] rst_sr = 4'hF;
  wire        rst_sys = rst_sr[3];
  always_ff @(posedge ui_clk or posedge cpu_reset) begin
    if (cpu_reset) rst_sr <= 4'hF;
    else           rst_sr <= {rst_sr[2:0], cpu_reset | ui_clk_sync_rst};
  end

  // DDR3 AXI master — idle in Build A (DDR bridge arrives in increment C).
  wire [4:0]   m_axi_arid;   wire [29:0] m_axi_araddr;  wire [7:0] m_axi_arlen;
  wire [2:0]   m_axi_arsize; wire [1:0]  m_axi_arburst; wire [0:0] m_axi_arlock;
  wire [3:0]   m_axi_arcache;wire [2:0]  m_axi_arprot;  wire [3:0] m_axi_arqos;
  wire         m_axi_arvalid;wire        m_axi_arready;
  wire         m_axi_rready; wire [4:0]  m_axi_rid;     wire [511:0] m_axi_rdata;
  wire [1:0]   m_axi_rresp;  wire        m_axi_rlast;   wire         m_axi_rvalid;
  wire [4:0]   m_axi_awid;   wire [29:0] m_axi_awaddr;  wire [7:0]  m_axi_awlen;
  wire [2:0]   m_axi_awsize; wire [1:0]  m_axi_awburst; wire [0:0]  m_axi_awlock;
  wire [3:0]   m_axi_awcache;wire [2:0]  m_axi_awprot;  wire [3:0]  m_axi_awqos;
  wire         m_axi_awvalid;wire        m_axi_awready;
  wire [511:0] m_axi_wdata;  wire [63:0] m_axi_wstrb;   wire        m_axi_wlast;
  wire         m_axi_wvalid; wire        m_axi_wready;
  wire         m_axi_bready; wire [4:0]  m_axi_bid;     wire [1:0]  m_axi_bresp;
  wire         m_axi_bvalid;

  // Build C: DDR data window via the CDC AXI bridge.  The SoC issues 32-bit
  // requests on eth_clk; the bridge runs single 512-bit beats on ui_clk.
  reg         ddr_req;
  reg  [29:0] ddr_addr;     // byte address = {ddr_base, window_offset[23:0]}
  reg  [31:0] ddr_wdata;
  reg  [3:0]  ddr_wstrb;
  wire        ddr_done;
  wire [31:0] ddr_rdata;
  reg  [5:0]  ddr_base;     // bank select (addr[29:24]) -> reaches the full 1 GB

  soc_ddr_bridge i_ddr_bridge (
    .clk_soc(eth_clk), .rst_soc(~soc_resetn),
    .req(ddr_req), .addr(ddr_addr), .wdata(ddr_wdata), .wstrb(ddr_wstrb),
    .done(ddr_done), .rdata(ddr_rdata),
    .clk_axi(ui_clk), .rst_axi(ui_clk_sync_rst),
    .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
    .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst), .m_axi_arlock(m_axi_arlock),
    .m_axi_arcache(m_axi_arcache), .m_axi_arprot(m_axi_arprot), .m_axi_arqos(m_axi_arqos),
    .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
    .m_axi_rready(m_axi_rready), .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata),
    .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid),
    .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
    .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst), .m_axi_awlock(m_axi_awlock),
    .m_axi_awcache(m_axi_awcache), .m_axi_awprot(m_axi_awprot), .m_axi_awqos(m_axi_awqos),
    .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
    .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
    .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
    .m_axi_bready(m_axi_bready), .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp),
    .m_axi_bvalid(m_axi_bvalid)
  );

  xlnx_mig_7_ddr3 i_ddr (
    .ddr3_dq(ddr3_dq), .ddr3_dqs_n(ddr3_dqs_n), .ddr3_dqs_p(ddr3_dqs_p),
    .ddr3_addr(ddr3_addr), .ddr3_ba(ddr3_ba), .ddr3_ras_n(ddr3_ras_n),
    .ddr3_cas_n(ddr3_cas_n), .ddr3_we_n(ddr3_we_n), .ddr3_reset_n(ddr3_reset_n),
    .ddr3_ck_p(ddr3_ck_p), .ddr3_ck_n(ddr3_ck_n), .ddr3_cke(ddr3_cke),
    .ddr3_cs_n(ddr3_cs_n), .ddr3_dm(ddr3_dm), .ddr3_odt(ddr3_odt),
    .sys_clk_p(sys_clk_p), .sys_clk_n(sys_clk_n),
    .ui_clk(ui_clk), .ui_clk_sync_rst(ui_clk_sync_rst), .mmcm_locked(mmcm_locked_mig),
    .aresetn(~ui_clk_sync_rst),
    .app_sr_req(1'b0), .app_ref_req(1'b0), .app_zq_req(1'b0),
    .app_sr_active(), .app_ref_ack(), .app_zq_ack(),
    .s_axi_awid(m_axi_awid), .s_axi_awaddr(m_axi_awaddr), .s_axi_awlen(m_axi_awlen),
    .s_axi_awsize(m_axi_awsize), .s_axi_awburst(m_axi_awburst), .s_axi_awlock(m_axi_awlock),
    .s_axi_awcache(m_axi_awcache), .s_axi_awprot(m_axi_awprot), .s_axi_awqos(m_axi_awqos),
    .s_axi_awvalid(m_axi_awvalid), .s_axi_awready(m_axi_awready),
    .s_axi_wdata(m_axi_wdata), .s_axi_wstrb(m_axi_wstrb), .s_axi_wlast(m_axi_wlast),
    .s_axi_wvalid(m_axi_wvalid), .s_axi_wready(m_axi_wready),
    .s_axi_bready(m_axi_bready), .s_axi_bid(m_axi_bid), .s_axi_bresp(m_axi_bresp),
    .s_axi_bvalid(m_axi_bvalid),
    .s_axi_arid(m_axi_arid), .s_axi_araddr(m_axi_araddr), .s_axi_arlen(m_axi_arlen),
    .s_axi_arsize(m_axi_arsize), .s_axi_arburst(m_axi_arburst), .s_axi_arlock(m_axi_arlock),
    .s_axi_arcache(m_axi_arcache), .s_axi_arprot(m_axi_arprot), .s_axi_arqos(m_axi_arqos),
    .s_axi_arvalid(m_axi_arvalid), .s_axi_arready(m_axi_arready),
    .s_axi_rready(m_axi_rready), .s_axi_rid(m_axi_rid), .s_axi_rdata(m_axi_rdata),
    .s_axi_rresp(m_axi_rresp), .s_axi_rlast(m_axi_rlast), .s_axi_rvalid(m_axi_rvalid),
    .init_calib_complete(init_calib_complete), .device_temp(),
    .sys_rst(~cpu_reset)
  );

  // ----------------------------------------------------------------------
  // MDIO tristate + framing_top_sgmii (eth_clk + LSU frame bus).
  // ----------------------------------------------------------------------
  wire        eth_clk;
  logic [16:0] lsu_addr;  logic [63:0] lsu_wdata; logic [7:0] lsu_be;
  logic       lsu_ce;     logic       lsu_we;     logic      lsu_sel;
  wire [63:0] lsu_rdata;  wire        eth_irq;
  wire        phy_mdio_i; logic phy_mdio_o; logic phy_mdio_oe;

  IOBUF i_mdio_iobuf (.IO(eth_mdio), .I(phy_mdio_o), .O(phy_mdio_i), .T(~phy_mdio_oe));

  framing_top_sgmii i_framing (
    .msoc_clk(eth_clk),
    .core_lsu_addr(lsu_addr), .core_lsu_wdata(lsu_wdata), .core_lsu_be(lsu_be),
    .ce_d(lsu_ce), .we_d(lsu_we), .framing_sel(lsu_sel), .framing_rdata(lsu_rdata),
    .clk_int(sys_clk_bufg), .rst_int(rst_sys),
    .sgmii_rxp(sgmii_rxp), .sgmii_rxn(sgmii_rxn), .sgmii_txp(sgmii_txp), .sgmii_txn(sgmii_txn),
    .sgmii_refclk_p(sgmii_refclk_p), .sgmii_refclk_n(sgmii_refclk_n),
    .phy_reset_n(eth_rst_n), .phy_mdio_i(phy_mdio_i), .phy_mdio_o(phy_mdio_o),
    .phy_mdio_oe(phy_mdio_oe), .phy_mdc(eth_mdc), .eth_irq(eth_irq), .eth_clk_o(eth_clk),
    .dbg_firstbuf(), .dbg_nextbuf(), .dbg_lastbuf()
  );

  // ----- Reset sync into eth_clk domain (active-low resetn for the SoC)
  logic [3:0] eth_rst_sr = 4'hF;
  wire        soc_resetn = ~eth_rst_sr[3];
  always_ff @(posedge eth_clk or posedge rst_sys) begin
    if (rst_sys) eth_rst_sr <= 4'hF;
    else         eth_rst_sr <= {eth_rst_sr[2:0], 1'b0};
  end

  // ----------------------------------------------------------------------
  // PicoSoC (runs on eth_clk).
  // ----------------------------------------------------------------------
  wire        iomem_valid;
  reg         iomem_ready;
  wire [3:0]  iomem_wstrb;
  wire [31:0] iomem_addr;
  wire [31:0] iomem_wdata;
  reg  [31:0] iomem_rdata;
  (* mark_debug = "true" *) wire        trace_valid;
  (* mark_debug = "true" *) wire [35:0] trace_data;

  picosoc_noflash soc (
    .clk(eth_clk), .resetn(soc_resetn),
    .iomem_valid(iomem_valid), .iomem_ready(iomem_ready),
    .iomem_wstrb(iomem_wstrb), .iomem_addr(iomem_addr),
    .iomem_wdata(iomem_wdata), .iomem_rdata(iomem_rdata),
    .irq_5(1'b0), .irq_6(eth_irq), .irq_7(1'b0),
    .ser_tx(usb_uart_tx), .ser_rx(usb_uart_rx),
    .trace_valid(trace_valid), .trace_data(trace_data)
  );

  // ----------------------------------------------------------------------
  // iomem decode.  Build A: LED register live; eth/DDR/dummy ranges ack with
  // 0 so firmware can probe without hanging (filled in B/C/D).
  // ----------------------------------------------------------------------
  (* mark_debug = "true" *) reg [7:0] led_reg;
  wire sel_led   = (iomem_addr[31:24] == 8'h03);
  wire sel_dummy = (iomem_addr[31:24] == 8'h10);
  wire sel_eth   = (iomem_addr[31:24] == 8'h20);
  wire sel_ddr   = (iomem_addr[31:24] == 8'h30);   // DDR data window (16 MB)
  wire sel_dbase = (iomem_addr[31:24] == 8'h31);   // DDR bank select (addr[29:24])
  // Ethernet LSU bridge FSM: each 32-bit SoC access -> a 64-bit LSU access,
  // with core_lsu_be selecting the addressed 32-bit half (no holding regs).
  // The frame BRAM has 1-cycle read latency, so reads take a few cycles.
  // LSU byte map (SoC base 0x2000_0000 + lsu_byte_addr):
  //   0x0800 MACLO  0x0808 MACHI|enable  0x0810 TPLR(tx trigger {tbuf,len})
  //   0x0828 rx-enable(=31)  0x0830 RSR(rd: [15]=ready,[4:0]=buf; wr: buf+1=free)
  //   0x0C00+buf*8 rx length   0x1000+(tbuf<<11)+(w<<3) TX   0x10000+(buf<<11)+(w<<3) RX
  typedef enum logic [2:0] { E_IDLE, E_WAIT, E_DONE, E_WR, D_WAIT } eph_t;
  eph_t eph;
  logic eth_half;

  always_ff @(posedge eth_clk) begin
    if (!soc_resetn) begin
      iomem_ready <= 1'b0; iomem_rdata <= 32'h0; led_reg <= 8'h01;
      lsu_ce <= 1'b0; lsu_we <= 1'b0; lsu_sel <= 1'b1; lsu_addr <= '0;
      lsu_wdata <= '0; lsu_be <= 8'hFF; eph <= E_IDLE; eth_half <= 1'b0;
      ddr_req <= 1'b0; ddr_base <= 6'd0; ddr_addr <= '0; ddr_wdata <= '0; ddr_wstrb <= 4'd0;
    end else begin
      iomem_ready <= 1'b0;
      lsu_sel     <= 1'b1;
      ddr_req     <= 1'b0;   // 1-cycle pulse

      // Single-cycle peripherals (LED / dummy regs / DDR bank register).
      if (iomem_valid && !iomem_ready && (sel_led || sel_dummy || sel_dbase)) begin
        iomem_ready <= 1'b1;
        if (sel_led   && iomem_wstrb[0]) led_reg  <= iomem_wdata[7:0];
        if (sel_dbase && iomem_wstrb[0]) ddr_base <= iomem_wdata[5:0];
        iomem_rdata <= sel_led   ? {24'h0, led_reg} :
                       sel_dummy ? 32'hD00D_0000 :
                       sel_dbase ? {26'h0, ddr_base} : 32'h0;
      end

      // Ethernet LSU bridge + DDR data window.
      case (eph)
        E_IDLE: begin
          if (iomem_valid && !iomem_ready && sel_eth) begin
            lsu_addr <= {iomem_addr[16:3], 3'b000};
            lsu_ce   <= 1'b1;
            eth_half <= iomem_addr[2];
            if (|iomem_wstrb) begin
              lsu_we    <= 1'b1;
              lsu_wdata <= {iomem_wdata, iomem_wdata};
              lsu_be    <= iomem_addr[2] ? 8'hF0 : 8'h0F;
              eph       <= E_WR;
            end else begin
              lsu_we    <= 1'b0;
              eph       <= E_WAIT;
            end
          end else if (iomem_valid && !iomem_ready && sel_ddr) begin
            ddr_addr  <= {ddr_base, iomem_addr[23:0]};   // {bank, window offset}
            ddr_wdata <= iomem_wdata;
            ddr_wstrb <= iomem_wstrb;                    // 0 => read
            ddr_req   <= 1'b1;
            eph       <= D_WAIT;
          end
        end
        D_WAIT: if (ddr_done) begin
          iomem_rdata <= ddr_rdata;
          iomem_ready <= 1'b1;
          eph         <= E_IDLE;
        end
        E_WAIT: eph <= E_DONE;                 // ce held; framing_rdata next cycle
        E_DONE: begin
          lsu_ce      <= 1'b0;
          iomem_rdata <= eth_half ? lsu_rdata[63:32] : lsu_rdata[31:0];
          iomem_ready <= 1'b1;
          eph         <= E_IDLE;
        end
        E_WR: begin
          lsu_ce <= 1'b0; lsu_we <= 1'b0;
          iomem_ready <= 1'b1;
          eph         <= E_IDLE;
        end
        default: eph <= E_IDLE;
      endcase
    end
  end

  // ----------------------------------------------------------------------
  // LEDs: led[7] = hardware heartbeat (proof the eth_clk is alive even if the
  // CPU wedges); led[6:0] = firmware LED register (proof firmware runs).
  // ----------------------------------------------------------------------
  reg [25:0] hb;
  always_ff @(posedge eth_clk) hb <= hb + 1'b1;
  assign led = {hb[25], led_reg[6:0]};

  // ----------------------------------------------------------------------
  // On-chip ILA on the PicoRV32 trace bus + iomem/DDR activity (eth_clk).
  // IP (picosoc_ila) is created by run.tcl; gated by PICOSOC_ILA so the file
  // still elaborates without it.
  // ----------------------------------------------------------------------
`ifdef PICOSOC_ILA
  picosoc_ila i_ila (
    .clk    (eth_clk),
    .probe0 (trace_valid),   // 1
    .probe1 (trace_data),    // 36 — PicoRV32 execution trace
    .probe2 (iomem_valid),   // 1
    .probe3 (iomem_ready),   // 1
    .probe4 (iomem_wstrb),   // 4  — 0 = read, else write byte-enables
    .probe5 (iomem_addr),    // 32 — 0x03 LED / 0x10 dummy / 0x20 eth / 0x30 DDR
    .probe6 (iomem_wdata),   // 32
    .probe7 (iomem_rdata)    // 32
  );
`endif

endmodule
