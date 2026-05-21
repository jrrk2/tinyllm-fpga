// vc707_microgpt_eth.sv — VC707 top for the microgpt inference engine
// driven over raw Ethernet (replaces de1_soc_microgpt_rtl.sv on the DE1-SoC).
//
// Ethernet frame transport, MAC, RX/TX BRAMs and SGMII PCS/PMA come from
// the cva6 kaspad accelerator (framing_top_sgmii + sgmii_soc + eth_mac_1g
// + gig_ethernet_pcs_pma_0 IP). microgpt_eth_ctrl translates those frames
// into Avalon-MM master transactions, exactly as the Altera JTAG bridge
// did on the DE1-SoC, so the entire register-map / FSM / output_mem
// logic is reused unchanged.

`default_nettype none

module vc707_microgpt_eth (
  // VC707 200 MHz LVDS system clock (PCS/PMA independent_clock_bufg)
  input  wire         sys_clk_p,
  input  wire         sys_clk_n,
  input  wire         cpu_reset,        // active-high VC707 button

  // SGMII Ethernet
  input  wire         sgmii_rxp,
  input  wire         sgmii_rxn,
  output wire         sgmii_txp,
  output wire         sgmii_txn,
  input  wire         sgmii_refclk_p,
  input  wire         sgmii_refclk_n,

  // Marvell 88E1111 PHY control
  output wire         eth_rst_n,
  inout  wire         eth_mdio,
  output wire         eth_mdc,

  // VC707 user IO
  output logic [7:0]  led,
  input  wire  [7:0]  sw,
  output wire         fan_pwm,

  // DDR3 SODIMM (MT8JTF12864HZ-1G6, 1 GB DDR3-1600). Pin LOCs come from the
  // MIG IP's internal XDC; we just expose the ports here.
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

  localparam [7:0]  BOS_TOKEN     = 8'd26;
  localparam [2:0]  ST_READY      = 3'd0;
  localparam [2:0]  ST_WAIT_CORE  = 3'd1;
  localparam [2:0]  ST_DONE       = 3'd2;

  // ================================================================
  //  System clock + reset
  //
  //  The MIG owns sys_clk_p/n (instantiates its own IBUFDS internally).
  //  We use the MIG's ui_clk output (200 MHz) as our system clock —
  //  same role sys_clk_bufg played before. This means SGMII can't begin
  //  PCS/PMA bring-up until the MIG MMCM locks (~ms after power-on),
  //  which is fine.
  // ================================================================
  wire ui_clk;
  wire ui_clk_sync_rst;
  wire mmcm_locked_mig;
  wire init_calib_complete;
  wire sys_clk_bufg = ui_clk;     // alias; keep downstream code unchanged

  logic [3:0] rst_sr = 4'hF;
  wire        rst_sys = rst_sr[3];

  always_ff @(posedge sys_clk_bufg)
    rst_sr <= {rst_sr[2:0], cpu_reset | ui_clk_sync_rst};

  // ================================================================
  //  PHY MDIO tri-state
  // ================================================================
  wire  phy_mdio_i;
  logic phy_mdio_o;
  logic phy_mdio_oe;

  IOBUF i_mdio_iobuf (
    .IO ( eth_mdio     ),
    .I  ( phy_mdio_o   ),
    .O  ( phy_mdio_i   ),
    .T  ( ~phy_mdio_oe )
  );

  // ================================================================
  //  framing_top_sgmii — gives us 125 MHz eth_clk and the LSU bus
  // ================================================================
  wire        eth_clk;
  wire [16:0] lsu_addr;
  wire [63:0] lsu_wdata;
  wire [7:0]  lsu_be;
  wire        lsu_ce;
  wire        lsu_we;
  wire        lsu_sel;
  wire [63:0] lsu_rdata;
  wire        eth_irq;

  framing_top_sgmii i_framing (
    .msoc_clk       ( eth_clk           ),
    .core_lsu_addr  ( lsu_addr          ),
    .core_lsu_wdata ( lsu_wdata         ),
    .core_lsu_be    ( lsu_be            ),
    .ce_d           ( lsu_ce            ),
    .we_d           ( lsu_we            ),
    .framing_sel    ( lsu_sel           ),
    .framing_rdata  ( lsu_rdata         ),
    .clk_int        ( sys_clk_bufg      ),
    .rst_int        ( rst_sys           ),
    .sgmii_rxp      ( sgmii_rxp         ),
    .sgmii_rxn      ( sgmii_rxn         ),
    .sgmii_txp      ( sgmii_txp         ),
    .sgmii_txn      ( sgmii_txn         ),
    .sgmii_refclk_p ( sgmii_refclk_p    ),
    .sgmii_refclk_n ( sgmii_refclk_n    ),
    .phy_reset_n    ( eth_rst_n         ),
    .phy_mdio_i     ( phy_mdio_i        ),
    .phy_mdio_o     ( phy_mdio_o        ),
    .phy_mdio_oe    ( phy_mdio_oe       ),
    .phy_mdc        ( eth_mdc           ),
    .eth_irq        ( eth_irq           ),
    .eth_clk_o      ( eth_clk           ),
    .dbg_firstbuf   ( fr_firstbuf       ),
    .dbg_nextbuf    ( fr_nextbuf        ),
    .dbg_lastbuf    ( fr_lastbuf        )
  );
  wire [4:0] fr_firstbuf;
  wire [4:0] fr_nextbuf;
  wire [4:0] fr_lastbuf;
  // No CDC: framing_top's msoc_clk IS eth_clk_o, and the JTAG regmap
  // dispatch already runs in eth_clk, so fr_*_buf are sampled in-domain.

  // ----- Reset sync into eth_clk (125 MHz) domain
  logic [3:0] eth_rst_sr = 4'hF;
  wire        eth_rst_n_int = ~eth_rst_sr[3];

  always_ff @(posedge eth_clk or posedge rst_sys) begin
    if (rst_sys) eth_rst_sr <= 4'hF;
    else         eth_rst_sr <= {eth_rst_sr[2:0], 1'b0};
  end

  // ================================================================
  //  DDR3 MIG IP — 1 GB DDR3-1600 SODIMM, 512-bit AXI4 slave at 200 MHz.
  //  Owns sys_clk_p/n. Outputs ui_clk used as our sys_clk_bufg.
  // ================================================================
  wire [4:0]   m_axi_arid;
  wire [29:0]  m_axi_araddr;
  wire [7:0]   m_axi_arlen;
  wire [2:0]   m_axi_arsize;
  wire [1:0]   m_axi_arburst;
  wire [0:0]   m_axi_arlock;
  wire [3:0]   m_axi_arcache;
  wire [2:0]   m_axi_arprot;
  wire [3:0]   m_axi_arqos;
  wire         m_axi_arvalid;
  wire         m_axi_arready;
  wire         m_axi_rready;
  wire [4:0]   m_axi_rid;
  wire [511:0] m_axi_rdata;
  wire [1:0]   m_axi_rresp;
  wire         m_axi_rlast;
  wire         m_axi_rvalid;

  // AXI write channel — from ddr_write_master
  wire [4:0]   m_axi_awid;
  wire [29:0]  m_axi_awaddr;
  wire [7:0]   m_axi_awlen;
  wire [2:0]   m_axi_awsize;
  wire [1:0]   m_axi_awburst;
  wire [0:0]   m_axi_awlock;
  wire [3:0]   m_axi_awcache;
  wire [2:0]   m_axi_awprot;
  wire [3:0]   m_axi_awqos;
  wire         m_axi_awvalid;
  wire         m_axi_awready;
  wire [511:0] m_axi_wdata;
  wire [63:0]  m_axi_wstrb;
  wire         m_axi_wlast;
  wire         m_axi_wvalid;
  wire         m_axi_wready;
  wire         m_axi_bready;
  wire [4:0]   m_axi_bid;
  wire [1:0]   m_axi_bresp;
  wire         m_axi_bvalid;

  xlnx_mig_7_ddr3 i_ddr (
    .ddr3_dq         ( ddr3_dq         ),
    .ddr3_dqs_n      ( ddr3_dqs_n      ),
    .ddr3_dqs_p      ( ddr3_dqs_p      ),
    .ddr3_addr       ( ddr3_addr       ),
    .ddr3_ba         ( ddr3_ba         ),
    .ddr3_ras_n      ( ddr3_ras_n      ),
    .ddr3_cas_n      ( ddr3_cas_n      ),
    .ddr3_we_n       ( ddr3_we_n       ),
    .ddr3_reset_n    ( ddr3_reset_n    ),
    .ddr3_ck_p       ( ddr3_ck_p       ),
    .ddr3_ck_n       ( ddr3_ck_n       ),
    .ddr3_cke        ( ddr3_cke        ),
    .ddr3_cs_n       ( ddr3_cs_n       ),
    .ddr3_dm         ( ddr3_dm         ),
    .ddr3_odt        ( ddr3_odt        ),
    .sys_clk_p       ( sys_clk_p       ),
    .sys_clk_n       ( sys_clk_n       ),
    .ui_clk          ( ui_clk          ),
    .ui_clk_sync_rst ( ui_clk_sync_rst ),
    .mmcm_locked     ( mmcm_locked_mig ),
    .aresetn         ( ~ui_clk_sync_rst),     // active-low AXI reset
    .app_sr_req      ( 1'b0            ),
    .app_ref_req     ( 1'b0            ),
    .app_zq_req      ( 1'b0            ),
    .app_sr_active   (                 ),
    .app_ref_ack     (                 ),
    .app_zq_ack      (                 ),
    // Write channel — driven by ddr_write_master
    .s_axi_awid      ( m_axi_awid      ),
    .s_axi_awaddr    ( m_axi_awaddr    ),
    .s_axi_awlen     ( m_axi_awlen     ),
    .s_axi_awsize    ( m_axi_awsize    ),
    .s_axi_awburst   ( m_axi_awburst   ),
    .s_axi_awlock    ( m_axi_awlock    ),
    .s_axi_awcache   ( m_axi_awcache   ),
    .s_axi_awprot    ( m_axi_awprot    ),
    .s_axi_awqos     ( m_axi_awqos     ),
    .s_axi_awvalid   ( m_axi_awvalid   ),
    .s_axi_awready   ( m_axi_awready   ),
    .s_axi_wdata     ( m_axi_wdata     ),
    .s_axi_wstrb     ( m_axi_wstrb     ),
    .s_axi_wlast     ( m_axi_wlast     ),
    .s_axi_wvalid    ( m_axi_wvalid    ),
    .s_axi_wready    ( m_axi_wready    ),
    .s_axi_bready    ( m_axi_bready    ),
    .s_axi_bid       ( m_axi_bid       ),
    .s_axi_bresp     ( m_axi_bresp     ),
    .s_axi_bvalid    ( m_axi_bvalid    ),
    // Read channel — driven by weight_stream_axi
    .s_axi_arid      ( m_axi_arid      ),
    .s_axi_araddr    ( m_axi_araddr    ),
    .s_axi_arlen     ( m_axi_arlen     ),
    .s_axi_arsize    ( m_axi_arsize    ),
    .s_axi_arburst   ( m_axi_arburst   ),
    .s_axi_arlock    ( m_axi_arlock    ),
    .s_axi_arcache   ( m_axi_arcache   ),
    .s_axi_arprot    ( m_axi_arprot    ),
    .s_axi_arqos     ( m_axi_arqos     ),
    .s_axi_arvalid   ( m_axi_arvalid   ),
    .s_axi_arready   ( m_axi_arready   ),
    .s_axi_rready    ( m_axi_rready    ),
    .s_axi_rid       ( m_axi_rid       ),
    .s_axi_rdata     ( m_axi_rdata     ),
    .s_axi_rresp     ( m_axi_rresp     ),
    .s_axi_rlast     ( m_axi_rlast     ),
    .s_axi_rvalid    ( m_axi_rvalid    ),
    .init_calib_complete ( init_calib_complete ),
    .device_temp     (                 ),
    .sys_rst         ( ~cpu_reset      )    // MIG sys_rst is active-low
  );

  // ================================================================
  //  Weight tile cache (200 MHz ui_clk domain)
  //
  //  Ping-pong 8 KiB BRAM banks driven by an AXI master to MIG.
  //
  //  Control via UDP MMIO:
  //    0x010 [0] : start_load  (toggle CDC -> 1-cycle pulse in ui_clk)
  //    0x010 [1] : consumer_swap (toggle CDC; flips active bank)
  //    0x011     : load_addr (DDR3 byte address, 30 bits)
  //  Status read-only:
  //    0x012 [0]=busy, [1]=load_done_latched, [2]=init_calib_complete,
  //          [3]=active_bank
  //    0x013 = bank0[0]   low 32 bits  (raw sample)
  //    0x014 = bank0[127] low 32 bits
  //    0x015 = bank1[0]   low 32 bits
  //    0x016 = bank1[127] low 32 bits
  // ================================================================
  reg        ddr_load_toggle_eth = 1'b0;
  reg        ddr_swap_toggle_eth = 1'b0;
  reg [29:0] ddr_load_addr_eth   = 30'd0;

  // CDC: eth_clk → ui_clk for start_load
  reg ddr_load_sync0 = 1'b0, ddr_load_sync1 = 1'b0, ddr_load_seen = 1'b0;
  reg ddr_swap_sync0 = 1'b0, ddr_swap_sync1 = 1'b0, ddr_swap_seen = 1'b0;
  reg ddr_load_pulse_ui = 1'b0;
  reg ddr_swap_pulse_ui = 1'b0;
  always_ff @(posedge ui_clk) begin
    if (ui_clk_sync_rst) begin
      ddr_load_sync0 <= 1'b0; ddr_load_sync1 <= 1'b0; ddr_load_seen <= 1'b0;
      ddr_swap_sync0 <= 1'b0; ddr_swap_sync1 <= 1'b0; ddr_swap_seen <= 1'b0;
      ddr_load_pulse_ui <= 1'b0;
      ddr_swap_pulse_ui <= 1'b0;
    end else begin
      ddr_load_sync0 <= ddr_load_toggle_eth; ddr_load_sync1 <= ddr_load_sync0;
      ddr_swap_sync0 <= ddr_swap_toggle_eth; ddr_swap_sync1 <= ddr_swap_sync0;
      ddr_load_pulse_ui <= 1'b0;
      ddr_swap_pulse_ui <= 1'b0;
      if (ddr_load_sync1 != ddr_load_seen) begin
        ddr_load_seen     <= ddr_load_sync1;
        ddr_load_pulse_ui <= 1'b1;
      end
      if (ddr_swap_sync1 != ddr_swap_seen) begin
        ddr_swap_seen     <= ddr_swap_sync1;
        ddr_swap_pulse_ui <= 1'b1;
      end
    end
  end

  // Status (async-read from eth_clk; values are stable when busy=0)
  wire         tile_busy;
  wire         tile_load_done;
  wire         tile_active_bank;
  // Latch load_done so the host can poll it at its leisure.
  reg          tile_load_done_latched = 1'b0;
  always_ff @(posedge ui_clk) begin
    if (ui_clk_sync_rst) tile_load_done_latched <= 1'b0;
    else if (ddr_load_pulse_ui) tile_load_done_latched <= 1'b0;
    else if (tile_load_done)    tile_load_done_latched <= 1'b1;
  end

  // Sample taps (32-bit each — return 4 distinct words for diagnostic)
  // Implemented as direct array reads in ui_clk; eth_clk reads them async.
  // Tile cache exposes a consumer port; we drive rd_addr=0 and grab the
  // low half.  We snapshot 4 specific BRAM entries by adding two extra
  // read ports — easier in this case to just expose individual entries
  // through a pair of small status muxes inside the cache.
  //
  // Implementation: keep the consumer port at rd_addr=0 (active bank's
  // first word), and replicate a second cache instance signal for entry
  // 127.  But a cleaner approach is to widen the cache's debug surface;
  // for now we simply read the first 32 bits of the *active* and
  // *inactive* banks at indices 0 and 127 by twice-instantiating a tap
  // reading inside the cache.  We do that by exposing dbg_bank0_w0 etc.
  wire [31:0] dbg_bank0_w0, dbg_bank0_w127, dbg_bank1_w0, dbg_bank1_w127;

  // weight_tile_cache removed — smollm_layer_selftest now owns the AXI
  // read master to MIG (it streams its own weights from DDR3 via
  // weight_streamer_mt under `MICROGPT_DDR3_WEIGHTS).
  //
  // Tile diagnostic taps tied off (host registers 0x012-0x016 still
  // exist but read 0).
  assign tile_busy        = 1'b0;
  assign tile_load_done   = 1'b0;
  assign tile_active_bank = 1'b0;

  // Diagnostic taps removed — hierarchical references into
  // weight_streamer_mt.bank0/1 prevent Vivado from inferring those
  // arrays as BRAM (forces LUT-RAM, blew the LUT budget).  The
  // host-visible 0x013-0x016 registers now read 0.
  assign dbg_bank0_w0   = 32'd0;
  assign dbg_bank0_w127 = 32'd0;
  assign dbg_bank1_w0   = 32'd0;
  assign dbg_bank1_w127 = 32'd0;

  // ================================================================
  //  56.25 MHz core clock from 125 MHz eth_clk
  //  VCO = 125 * 9 = 1125 MHz, /20 = 56.25 MHz exactly.
  // ================================================================
  wire core_clk_unbuf, core_clk;
  wire core_mmcm_fb_unbuf, core_mmcm_fb;
  wire core_mmcm_locked;

  // core_clk = 40 MHz (CLKIN 125 × 8 / 25 = 1000 MHz VCO / 25).
  // Dropped from 50 → 40 MHz to give the BFP-streaming path an extra
  // 5 ns slack per cycle while the remaining long combinational paths
  // get further pipelined.  Revisit upward once the worst offenders
  // are split.
  MMCME2_BASE #(
    .BANDWIDTH        ( "OPTIMIZED" ),
    .CLKIN1_PERIOD    ( 8.000       ),  // 125 MHz
    .CLKFBOUT_MULT_F  ( 8.000       ),  // VCO 1000 MHz
    .DIVCLK_DIVIDE    ( 1           ),
    .CLKOUT0_DIVIDE_F ( 25.000      ),  // 40 MHz
    .STARTUP_WAIT     ( "FALSE"     )
  ) i_core_mmcm (
    .CLKIN1   ( eth_clk            ),
    .CLKFBIN  ( core_mmcm_fb       ),
    .CLKFBOUT ( core_mmcm_fb_unbuf ),
    .CLKOUT0  ( core_clk_unbuf     ),
    .CLKOUT1  (                    ),
    .CLKOUT2  (                    ),
    .CLKOUT3  (                    ),
    .CLKOUT4  (                    ),
    .CLKOUT5  (                    ),
    .CLKOUT6  (                    ),
    .LOCKED   ( core_mmcm_locked   ),
    .PWRDWN   ( 1'b0               ),
    .RST      ( ~eth_rst_n_int     )
  );

  BUFG i_core_clk_bufg ( .I( core_clk_unbuf      ), .O( core_clk      ) );
  BUFG i_core_fb_bufg  ( .I( core_mmcm_fb_unbuf  ), .O( core_mmcm_fb  ) );

  // ================================================================
  //  Internal Avalon-MM master signals (driven by the eth bridge,
  //  consumed by the register-map slave below — same names and
  //  semantics as on the DE1-SoC for the JTAG bridge)
  // ================================================================
  wire [31:0] jtag_master_address;
  wire        jtag_master_read;
  reg  [31:0] jtag_master_readdata = 32'd0;
  wire        jtag_master_write;
  wire [31:0] jtag_master_writedata;
  wire        jtag_master_waitrequest;
  reg         jtag_master_readdatavalid = 1'b0;
  wire [3:0]  jtag_master_byteenable;

  assign jtag_master_waitrequest = 1'b0;

  // ================================================================
  //  Per-spec registers and status snapshots (see DE1-SoC top)
  // ================================================================
  wire enable     = sw[0];
  wire core_resetn = eth_rst_n_int & core_mmcm_locked & ~sw[1];

  // Build version constant — auto-generated by host/gen_build_version.py
  // (re-run by scripts/run.tcl on every Vivado build).
`include "build_version.svh"

  reg [2:0]  state_reg = ST_READY;
  reg [7:0]  token_reg = BOS_TOKEN;
  reg [7:0]  pos_reg = 8'd0;
  reg [7:0]  out_len_reg = 8'd0;
  reg [31:0] rng_reg = 32'h00000001;
  reg [15:0] temperature_reg = 16'h0080;
  reg [7:0]  max_gen_reg = 8'd15;
  reg start_core_reg = 1'b0;
  reg clear_cache_reg = 1'b0;
  reg done_latched_reg = 1'b0;
  reg [7:0]  last_token_reg = 8'd0;
  reg [15:0] cycle_blink_reg = 16'd0;
  reg [7:0]  output_mem [0:15];
  reg [31:0] perf_cycles_reg = 32'd0;
  reg [31:0] tokens_per_sec_reg = 32'd0;
  reg host_toggle_reg = 1'b0;
  reg error_reg = 1'b0;
  reg host_start_req = 1'b0;
  reg host_clear_req = 1'b0;
  reg read_pending_reg = 1'b0;
  reg host_run_reg = 1'b0;
  reg host_start_toggle_eth = 1'b0;
  reg host_clear_toggle_eth = 1'b0;
  reg host_step_toggle_eth = 1'b0;
  reg [31:0] host_seed_reg = 32'h00000001;
  reg [15:0] host_temperature_reg = 16'h0080;
  reg [7:0]  host_max_gen_reg = 8'd15;
  reg host_direct_mode_eth = 1'b0;
  reg host_step_clear_eth = 1'b0;
  reg [7:0] host_step_token_eth = BOS_TOKEN;
  reg [7:0] host_step_pos_eth = 8'd0;
  // Per-layer hidden-state snapshot select.  Routed to multilayer_tm via
  // selftest.  Default = 30 (NL) → live final hidden_state (legacy path).
  // 0..29 → that layer's captured output, for per-layer divergence diff.
  // Hold the default through ~rst_sys; explicit always_ff prevents the
  // "ROM too sparse" misclassification Vivado does on `reg ... = init`
  // with a single conditional write.
  reg [4:0] snapshot_layer_sel_reg;
  always @(posedge eth_clk) begin
    if (rst_sys)
      snapshot_layer_sel_reg <= 5'd30;
    else if (jtag_master_write && jtag_word_addr == 10'h00A)
      snapshot_layer_sel_reg <= jtag_master_writedata[4:0];
  end

  // Companion token-step select for the per-layer snapshot.  The snapshot
  // latches the chosen layer's hidden_out at token position == this value.
  // Default 0 (first prompt token).  Host writes 0x00B.
  reg [10:0] snapshot_step_sel_reg;
  always @(posedge eth_clk) begin
    if (rst_sys)
      snapshot_step_sel_reg <= 11'd0;
    else if (jtag_master_write && jtag_word_addr == 10'h064)
      snapshot_step_sel_reg <= jtag_master_writedata[10:0];
  end

  // Freeze-enable: halt the engine at the programmed (snapshot_layer_sel,
  // snapshot_step_sel) so the inner layer's hout can be inspected (wr_kind
  // 10/11) or ILA-captured.  Host writes 0x00C bit0; default 0 (off).
  reg freeze_en_reg;
  always @(posedge eth_clk) begin
    if (rst_sys)
      freeze_en_reg <= 1'b0;
    else if (jtag_master_write && jtag_word_addr == 10'h065)
      freeze_en_reg <= jtag_master_writedata[0];
  end

  // Logic-analyser absolute-cycle freeze trigger.  trig_cyc = clock count
  // from start; trig_cyc_en arms it.  Host: 0x066 = count, 0x067 bit0 = enable.
  reg [31:0] trig_cyc_reg;
  reg        trig_cyc_en_reg;
  always @(posedge eth_clk) begin
    if (rst_sys) begin
      trig_cyc_reg    <= 32'd0;
      trig_cyc_en_reg <= 1'b0;
    end else if (jtag_master_write && jtag_word_addr == 10'h066)
      trig_cyc_reg    <= jtag_master_writedata;
    else if (jtag_master_write && jtag_word_addr == 10'h067)
      trig_cyc_en_reg <= jtag_master_writedata[0];
  end

  // Per-stage exponent read-out select (0..13 → n1/q/k/v/attn/o/h1/n2/g/u/mlp/
  // d/hout/hin).  Host writes 0x070; read the stage's exponents via wr_kind 16.
  reg [4:0] dbg_stage_sel_reg;
  always @(posedge eth_clk) begin
    if (rst_sys)
      dbg_stage_sel_reg <= 5'd0;
    else if (jtag_master_write && jtag_word_addr == 10'h070)
      dbg_stage_sel_reg <= jtag_master_writedata[4:0];
  end

  // Eth-side shadow regs for runtime factor-override CDC.  Reset + write
  // logic both live in the main jtag_master_write always block below; the
  // declaration here keeps the signal visible across the file.
  reg [4:0]  factor_wr_layer_eth;
  reg [31:0] factor_wr_data_eth;
  reg [1:0]  factor_wr_kind_eth;
  reg        factor_wr_toggle_eth;
  // Readback select reg — host writes {kind, layer} here, then reads
  // factor_rd_data at the next address.  Forces Vivado to keep the
  // override-RAM observable so the entire write path can't be DCE'd.
  reg [6:0]  factor_rd_sel_eth;

  // Per-row scale brom override regs.  Host writes 0x01C with
  // {kind[3:0], addr[15:0]}, then 0x01D with the 16-bit data — that
  // second write fires a one-cycle scale_wr_en pulse on the core side.
  reg [3:0]  scale_wr_kind_eth;
  reg [15:0] scale_wr_addr_eth;
  reg [15:0] scale_wr_data_eth;
  reg        scale_wr_toggle_eth;
  reg start_sync0 = 1'b0, start_sync1 = 1'b0, start_seen_reg = 1'b0;
  reg clear_sync0 = 1'b0, clear_sync1 = 1'b0, clear_seen_reg = 1'b0;
  reg step_sync0 = 1'b0, step_sync1 = 1'b0, step_seen_reg = 1'b0;
  reg host_step_req = 1'b0;
  reg direct_mode_reg = 1'b0;
  reg step_clear_reg = 1'b0;
  reg [7:0] step_token_reg = BOS_TOKEN;
  reg [7:0] step_pos_reg = 8'd0;

  wire core_busy;
  wire core_done;
  wire [7:0]  core_next_token;
  wire [7:0]  core_argmax_token;
  wire [31:0] core_rng_state;
  wire signed [15:0]      core_top_logit;
  wire signed [(27*16)-1:0] core_logits_flat;
  wire [9:0]  jtag_word_addr;
  reg  [31:0] read_data_comb;
  integer out_i;

  assign jtag_word_addr = jtag_master_address[11:2];

  // ================================================================
  //  Ethernet bridge (drives the Avalon-MM master signals above)
  // ================================================================
  // DDR3 write-path stage counters from eth_ctrl (read via regmap 0x019..0x01C)
  wire [31:0] ddr_wr_rx_count;
  wire [31:0] ddr_wr_done_count;
  wire [31:0] ddr_wr_ack_count;
  wire [31:0] ddr_wr_tx_count;

`ifdef PICOSOC_ENGINE
  // PicoRV32 SoC front-end (drop-in for microgpt_eth_ctrl: same ports).  Drives
  // the engine regmap via the Avalon master; LSU/ddr_wr tied off in this lean
  // base-fit build.  Selected by the PICOSOC=ENGINE / make picosoc-engine flow.
  picosoc_eth_bridge i_eth_ctrl (
`else
  microgpt_eth_ctrl i_eth_ctrl (
`endif
    .clk                  ( eth_clk                 ),
    .rst_n                ( eth_rst_n_int           ),
    .core_lsu_addr        ( lsu_addr                ),
    .core_lsu_wdata       ( lsu_wdata               ),
    .core_lsu_be          ( lsu_be                  ),
    .ce_d                 ( lsu_ce                  ),
    .we_d                 ( lsu_we                  ),
    .framing_sel          ( lsu_sel                 ),
    .framing_rdata        ( lsu_rdata               ),
    .master_address       ( jtag_master_address     ),
    .master_read          ( jtag_master_read        ),
    .master_write         ( jtag_master_write       ),
    .master_writedata     ( jtag_master_writedata   ),
    .master_byteenable    ( jtag_master_byteenable  ),
    .master_readdata      ( jtag_master_readdata    ),
    .master_readdatavalid ( jtag_master_readdatavalid ),
    .master_waitrequest   ( jtag_master_waitrequest ),
    .hb_state             ( {5'd0, state_reg}       ),
    .hb_last_token        ( last_token_reg          ),
    .hb_out_len           ( out_len_reg             ),
    .hb_done_flags        ( {6'd0, done_latched_reg, error_reg} ),
    .rx_activity          ( eth_ctrl_rx_activity     ),
    .ddr_wr_req           ( ddr_wr_req_eth           ),
    .ddr_wr_ack           ( ddr_wr_ack_eth_sync      ),
    .ddr_wr_addr          ( ddr_wr_addr_eth          ),
    .ddr_wr_data          ( ddr_wr_data_eth          ),
    .dbg_state            (                          ),
    .dbg_frame_type       (                          ),
    .dbg_wcnt             (                          ),
    .dbg_cur_buf          (                          ),
    .dbg_n_remaining      (                          ),
    .dbg_fpga_ip          ( eth_ctrl_fpga_ip         ),
    .ddr_wr_rx_count      ( ddr_wr_rx_count          ),
    .ddr_wr_done_count    ( ddr_wr_done_count        ),
    .ddr_wr_ack_count     ( ddr_wr_ack_count         ),
    .ddr_wr_tx_count      ( ddr_wr_tx_count          )
  );

  wire [31:0] eth_ctrl_fpga_ip;

  // ================================================================
  //  DDR3 write master (ui_clk domain) + 4-phase handshake CDC
  // ================================================================
  // ================================================================
  //  matvec self-test — runs once at boot with hardcoded data, exposes
  //  16-lane Q1.15 result through register map at 0x100-0x107.  Can be
  //  re-triggered from the host by writing a 1 to register 0x109[0].
  // ================================================================
  // ----- selftest CDC: restart pulses (eth_clk → core_clk via toggle),
  //                     done flags (core_clk → eth_clk via 2FF).
  //       result buses are wide and stable once done is asserted, so we
  //       read them directly in the eth_clk domain without per-bit sync.
  reg mvst_rst_tog_eth = 1'b0;
  reg rms_rst_tog_eth  = 1'b0;
  reg rope_rst_tog_eth = 1'b0;
  reg sg_rst_tog_eth   = 1'b0;
  reg sm_rst_tog_eth   = 1'b0;

  // Pulse signals (one core_clk cycle wide) regenerated in the core domain.
  reg [2:0] mvst_rst_sync_core = 3'd0;
  reg [2:0] rms_rst_sync_core  = 3'd0;
  reg [2:0] rope_rst_sync_core = 3'd0;
  reg [2:0] sg_rst_sync_core   = 3'd0;
  reg [2:0] sm_rst_sync_core   = 3'd0;
  always_ff @(posedge core_clk) begin
    mvst_rst_sync_core <= {mvst_rst_sync_core[1:0], mvst_rst_tog_eth};
    rms_rst_sync_core  <= {rms_rst_sync_core [1:0], rms_rst_tog_eth };
    rope_rst_sync_core <= {rope_rst_sync_core[1:0], rope_rst_tog_eth};
    sg_rst_sync_core   <= {sg_rst_sync_core  [1:0], sg_rst_tog_eth  };
    sm_rst_sync_core   <= {sm_rst_sync_core  [1:0], sm_rst_tog_eth  };
  end
  wire mvst_restart_core = mvst_rst_sync_core[2] ^ mvst_rst_sync_core[1];
  wire rms_restart_core  = rms_rst_sync_core [2] ^ rms_rst_sync_core [1];
  wire rope_restart_core = rope_rst_sync_core[2] ^ rope_rst_sync_core[1];
  wire sg_restart_core   = sg_rst_sync_core  [2] ^ sg_rst_sync_core  [1];
  wire sm_restart_core   = sm_rst_sync_core  [2] ^ sm_rst_sync_core  [1];

  // ================================================================
  //  matvec self-test (core_clk domain) — registers @ 0x100..0x108, restart 0x109
  // ================================================================
  wire [255:0] mvst_result;
  wire         mvst_done_core;
`ifdef MICROGPT_NO_OP_TESTS
  assign mvst_result    = '0;
  assign mvst_done_core = 1'b0;
`else
  matvec_selftest #(.IN_DIM(64)) i_mvst (
    .clk     ( core_clk          ),
    .rst     ( ~core_resetn      ),
    .restart ( mvst_restart_core ),
    .result  ( mvst_result       ),
    .done    ( mvst_done_core    )
  );
`endif
  reg [1:0] mvst_done_sync = 2'd0;
  always_ff @(posedge eth_clk) mvst_done_sync <= {mvst_done_sync[0], mvst_done_core};
  wire mvst_done = mvst_done_sync[1];

  // ================================================================
  //  rmsnorm self-test (core_clk) — result @ 0x110..0x12F, done 0x130, restart 0x131
  // ================================================================
  wire [1023:0] rms_result;
  wire          rms_done_core;
`ifdef MICROGPT_NO_OP_TESTS
  assign rms_result    = '0;
  assign rms_done_core = 1'b0;
`else
  rmsnorm_selftest #(.D(64)) i_rms_st (
    .clk     ( core_clk         ),
    .rst     ( ~core_resetn     ),
    .restart ( rms_restart_core ),
    .result  ( rms_result       ),
    .done    ( rms_done_core    )
  );
`endif
  reg [1:0] rms_done_sync = 2'd0;
  always_ff @(posedge eth_clk) rms_done_sync <= {rms_done_sync[0], rms_done_core};
  wire rms_done = rms_done_sync[1];

  // ================================================================
  //  rope self-test (core_clk) — result @ 0x140..0x15F, done 0x160, restart 0x161
  // ================================================================
  wire [1023:0] rope_result;
  wire          rope_done_core;
`ifdef MICROGPT_NO_OP_TESTS
  assign rope_result    = '0;
  assign rope_done_core = 1'b0;
`else
  rope_selftest #(.HEAD_DIM(64), .MAX_CTX(2048)) i_rope_st (
    .clk     ( core_clk          ),
    .rst     ( ~core_resetn      ),
    .restart ( rope_restart_core ),
    .result  ( rope_result       ),
    .done    ( rope_done_core    )
  );
`endif
  reg [1:0] rope_done_sync = 2'd0;
  always_ff @(posedge eth_clk) rope_done_sync <= {rope_done_sync[0], rope_done_core};
  wire rope_done = rope_done_sync[1];

  // ================================================================
  //  swiglu self-test (core_clk) — result @ 0x170..0x18F, done 0x190, restart 0x191
  // ================================================================
  wire [1023:0] sg_result;
  wire          sg_done_core;
`ifdef MICROGPT_NO_OP_TESTS
  assign sg_result    = '0;
  assign sg_done_core = 1'b0;
`else
  swiglu_selftest #(.N(64)) i_sg_st (
    .clk     ( core_clk        ),
    .rst     ( ~core_resetn    ),
    .restart ( sg_restart_core ),
    .result  ( sg_result       ),
    .done    ( sg_done_core    )
  );
`endif
  reg [1:0] sg_done_sync = 2'd0;
  always_ff @(posedge eth_clk) sg_done_sync <= {sg_done_sync[0], sg_done_core};
  wire sg_done = sg_done_sync[1];

  // ================================================================
  //  softmax self-test (core_clk) — result @ 0x1A0..0x1BF, done 0x1C0, restart 0x1C1
  // ================================================================
  wire [1023:0] sm_result;
  wire          sm_done_core;
`ifdef MICROGPT_NO_OP_TESTS
  assign sm_result    = '0;
  assign sm_done_core = 1'b0;
`else
  softmax_selftest #(.N(64)) i_sm_st (
    .clk     ( core_clk        ),
    .rst     ( ~core_resetn    ),
    .restart ( sm_restart_core ),
    .result  ( sm_result       ),
    .done    ( sm_done_core    )
  );
`endif
  reg [1:0] sm_done_sync = 2'd0;
  always_ff @(posedge eth_clk) sm_done_sync <= {sm_done_sync[0], sm_done_core};
  wire sm_done = sm_done_sync[1];

  // ================================================================
  //  Full transformer-layer self-test (core_clk) — hidden_out @ 0x1D0..0x1EF
  //  done @ 0x1F0[0], restart @ 0x1F1[0]
  // ================================================================
  // smollm_layer_selftest at small (D=64) dims emits the full 64-lane
  // hidden_out as a 1024-bit packed bus, served via 32 32-bit regmap
  // words at 0x1D0..0x1EF (two lanes per word).
  // Full hidden_out: 576 lanes × 16-bit = 9216 bits.  Exposed via two
  // regmap windows: legacy 32-word at 0x1D0..0x1EF (first 64 lanes,
  // back-compat with selftest_verify.py) + new 288-word window at
  // 0x200..0x31F (all 576 lanes for full-resolution lm_head decode).
  wire [9215:0] lay_result;
  wire          lay_done_core;
  reg           lay_rst_tog_eth = 1'b0;
  reg [2:0]     lay_rst_sync_core = 3'd0;
  always_ff @(posedge core_clk)
    lay_rst_sync_core <= {lay_rst_sync_core[1:0], lay_rst_tog_eth};
  wire lay_restart_core = lay_rst_sync_core[2] ^ lay_rst_sync_core[1];

  // -------------------------------------------------------------------
  // Host BRAM write protocol (Phase 1: per-model gammas + norm_w +
  // prompt).  Two regmap entries on the host side:
  //   0x060 (write): {bram_inc[31], 8'd0, bram_kind[22:18], bram_addr[17:0]}
  //                  — set target.  bram_inc=1 enables auto-increment
  //                  of bram_addr after each 0x061 write.
  //   0x061 (write): bram_data[15:0]  — pulse a write into the targeted
  //                  BRAM at current bram_addr.
  // All signals stay in eth_clk: they are wired directly into the
  // autoregress BRAMs' write port (port A) while the read port (port B)
  // remains on core_clk.  Vivado infers each BRAM as a true-dual-port
  // RAMB36E1 with asymmetric clocks — no CDC, no risk of dropped pulses.
  // -------------------------------------------------------------------
  reg [4:0]  bram_kind_eth    = 5'd0;
  reg [17:0] bram_addr_eth    = 18'd0;   // next-write pointer (also read addr)
  reg [17:0] bram_wr_addr_eth = 18'd0;   // address used by the current pulse
  reg [15:0] bram_data_eth    = 16'd0;
  reg        bram_inc_eth     = 1'b0;
  reg        bram_wr_en_eth   = 1'b0;
  // For writes, drive bram_wr_addr_eth (pre-increment snapshot so the
  // BRAM writes the address the host actually targeted).  For reads
  // (wr_en=0), drive the live bram_addr_eth so the read returns the
  // contents at the host-selected address.
  wire [17:0] bram_port_addr_eth = bram_wr_en_eth ? bram_wr_addr_eth : bram_addr_eth;
  wire [15:0] bram_rdata_eth;

  // Diagnostic snapshot signals — read by host via 0x012..0x044.
  wire [29:0]  dbg_first_araddr;
  wire [511:0] dbg_first_rdata;
  wire         dbg_first_ar_seen, dbg_first_r_seen;
  wire [511:0] dbg_eng_w_packed;
  wire [511:0] dbg_wd_packed;
  wire [63:0]  dbg_in_value_packed;
  wire         dbg_snap_done;

  // Named ILA probe wires — visible at module scope so .probeN connections
  // below show signal selection at a glance.
  wire [4:0]   ila_state;
  wire [2:0]   ila_mv_phase;
  wire [10:0]  ila_cnt;
  wire [6:0]   ila_chunk;
  wire [2:0]   ila_ws_matvec_id;
  wire         ila_ws_load_req;
  wire         ila_ws_ready;
  wire [10:0]  ila_ws_rd_addr;
  wire [127:0] ila_ws_weight_data;
  wire [127:0] ila_eng_w;
  wire [15:0]  ila_eng_in_value;
  wire         ila_eng_in_valid;
  wire         ila_eng_in_last;
  wire         ila_eng_acc_clear;
  wire         ila_eng_out_valid;
  wire [2:0]   ila_ws_state_axi;
  wire         ila_start_load_axi;
  wire [1:0]   ila_tile_idx;
  wire [6:0]   ila_beat_idx;

  // Outer time-mux FSM visibility (added with MICROGPT_LAYER_DEBUG).
  wire [2:0] ila_ml_state;
  wire [4:0] ila_ml_layer_idx;

`ifndef MICROGPT_LAYER_DEBUG
  // Re-armable BFP AXI-master sampler.  Watches the top-level AXI
  // master (which carries embed_lookup + smollm_layer + decode_head
  // reads, all merged); captures the first AR address and the first
  // R data after the arm flop toggles.  Host arms via regmap *write*
  // to 0x048 (any value) and reads back at 0x013 (addr) and
  // 0x014..0x023 (16 × 32-bit words = 512-bit rdata).  Status flags
  // appear at 0x012 [1] = ar_seen, [2] = r_seen.
  reg axi_snap_arm_eth = 1'b0;
  always_ff @(posedge eth_clk) begin
    if (~core_resetn) begin
      axi_snap_arm_eth <= 1'b0;
    end else if (jtag_master_write && jtag_word_addr == 10'h048) begin
      // Edge-pulse: any write to 0x048 toggles the arm flop, which
      // crosses CDC into ui_clk and clears the snapshot.  Pick an
      // address outside the dbg_eng_w / dbg_wd packed-read ranges
      // (0x024-0x043 are taken on the read side).
      axi_snap_arm_eth <= ~axi_snap_arm_eth;
    end
  end
  // CDC: toggle-based sync into ui_clk (where the BFP AXI master is).
  (* ASYNC_REG = "TRUE" *) reg [2:0] arm_sync_ui;
  always_ff @(posedge ui_clk) arm_sync_ui <= {arm_sync_ui[1:0], axi_snap_arm_eth};
  wire arm_edge_ui = arm_sync_ui[2] ^ arm_sync_ui[1];

  // Same pattern for the restart toggle (lay_rst_tog_eth) so both the
  // snapshot and the CRC also reset at the start of a new autoregress
  // run (host write to 0x1F1).
  (* ASYNC_REG = "TRUE" *) reg [2:0] restart_sync_ui;
  always_ff @(posedge ui_clk) restart_sync_ui <= {restart_sync_ui[1:0], lay_rst_tog_eth};
  wire restart_edge_ui = restart_sync_ui[2] ^ restart_sync_ui[1];

  wire crc_reset_ui = arm_edge_ui | restart_edge_ui;

  // Sampler proper: lives in ui_clk domain alongside the AXI bus.
  reg [29:0]  snap_araddr_ui = '0;
  reg [511:0] snap_rdata_ui  = '0;
  reg         snap_ar_seen_ui = 1'b0;
  reg         snap_r_seen_ui  = 1'b0;
  always_ff @(posedge ui_clk) begin
    if (ui_clk_sync_rst || crc_reset_ui) begin
      snap_ar_seen_ui <= 1'b0;
      snap_r_seen_ui  <= 1'b0;
    end else begin
      if (m_axi_arvalid && m_axi_arready && !snap_ar_seen_ui) begin
        snap_araddr_ui  <= m_axi_araddr;
        snap_ar_seen_ui <= 1'b1;
      end
      if (m_axi_rvalid && m_axi_rready && !snap_r_seen_ui) begin
        snap_rdata_ui  <= m_axi_rdata;
        snap_r_seen_ui <= 1'b1;
      end
    end
  end
  // CDC the captured values back to eth_clk for the regmap.  Address
  // + data are quasi-static (only change once per arm cycle), so 2FF
  // sync is safe; the seen-flags are sticky until next arm, so toggle
  // detection isn't needed.
  (* ASYNC_REG = "TRUE" *) reg [29:0]  snap_araddr_sync1, snap_araddr_sync2;
  (* ASYNC_REG = "TRUE" *) reg [511:0] snap_rdata_sync1,  snap_rdata_sync2;
  (* ASYNC_REG = "TRUE" *) reg [1:0]   snap_ar_seen_sync, snap_r_seen_sync;
  always_ff @(posedge eth_clk) begin
    snap_araddr_sync1 <= snap_araddr_ui;
    snap_araddr_sync2 <= snap_araddr_sync1;
    snap_rdata_sync1  <= snap_rdata_ui;
    snap_rdata_sync2  <= snap_rdata_sync1;
    snap_ar_seen_sync <= {snap_ar_seen_sync[0], snap_ar_seen_ui};
    snap_r_seen_sync  <= {snap_r_seen_sync[0],  snap_r_seen_ui};
  end
  assign dbg_first_araddr   = snap_araddr_sync2;
  assign dbg_first_rdata    = snap_rdata_sync2;
  assign dbg_first_ar_seen  = snap_ar_seen_sync[1];
  assign dbg_first_r_seen   = snap_r_seen_sync[1];

  // -------------------------------------------------------------------
  // Comparative hash: now lives in core_clk inside the autoregress
  // layer, watching mv_w_m_eff || mv_w_e_eff (the weight bus into the
  // matvec engine).  Same hash function (rolling XOR with rotate)
  // but at 40 MHz / 25 ns budget instead of 200 MHz / 5 ns — easy
  // timing and closer to "what the LLM actually consumes".  Only the
  // CDC into eth_clk for regmap exposure lives here at the top.
  // -------------------------------------------------------------------
  wire [31:0] bfp_weight_hash_core;
  (* ASYNC_REG = "TRUE" *) reg [31:0] crc_sync1, crc_sync2;
  always_ff @(posedge eth_clk) begin
    crc_sync1 <= bfp_weight_hash_core;
    crc_sync2 <= crc_sync1;
  end
  wire [31:0] bfp_rdata_crc_eth = crc_sync2;

  // has_run is in core_clk; sync to eth_clk for the regmap status bit.
  (* ASYNC_REG = "TRUE" *) reg [1:0] has_run_sync;
  always_ff @(posedge eth_clk) has_run_sync <= {has_run_sync[0], has_run};
  wire has_run_eth = has_run_sync[1];
  assign dbg_eng_w_packed   = '0;
  assign dbg_wd_packed      = '0;
  assign dbg_in_value_packed= '0;
  assign dbg_snap_done      = 1'b0;
  assign ila_state          = '0;
  assign ila_mv_phase       = '0;
  assign ila_cnt            = '0;
  assign ila_chunk          = '0;
  assign ila_ws_matvec_id   = '0;
  assign ila_ws_load_req    = 1'b0;
  assign ila_ws_ready       = 1'b0;
  assign ila_ws_rd_addr     = '0;
  assign ila_ws_weight_data = '0;
  assign ila_eng_w          = '0;
  assign ila_eng_in_value   = '0;
  assign ila_eng_in_valid   = 1'b0;
  assign ila_eng_in_last    = 1'b0;
  assign ila_eng_acc_clear  = 1'b0;
  assign ila_eng_out_valid  = 1'b0;
  assign ila_ws_state_axi   = '0;
  assign ila_start_load_axi = 1'b0;
  assign ila_tile_idx       = '0;
  assign ila_beat_idx       = '0;
  assign ila_ml_state       = '0;
  assign ila_ml_layer_idx   = '0;
`endif

  // CDC: snapshot_layer_sel from eth-clk regfile to core_clk-domain selftest.
  // Slow-changing host write; double-flop is sufficient.  Reset to 30 (= NL,
  // legacy bypass) on core_resetn so an uninitialised host doesn't drive X.
  reg [4:0] snap_sel_core_s0, snap_sel_core_s1;
  always @(posedge core_clk or negedge core_resetn) begin
    if (!core_resetn) begin
      snap_sel_core_s0 <= 5'd30;
      snap_sel_core_s1 <= 5'd30;
    end else begin
      snap_sel_core_s0 <= snapshot_layer_sel_reg;
      snap_sel_core_s1 <= snap_sel_core_s0;
    end
  end

  // CDC: snapshot_step_sel from eth-clk regfile to core_clk (double-flop).
  reg [10:0] step_sel_core_s0, step_sel_core_s1;
  always @(posedge core_clk or negedge core_resetn) begin
    if (!core_resetn) begin
      step_sel_core_s0 <= 11'd0;
      step_sel_core_s1 <= 11'd0;
    end else begin
      step_sel_core_s0 <= snapshot_step_sel_reg;
      step_sel_core_s1 <= step_sel_core_s0;
    end
  end

  // CDC: freeze-enable eth-clk → core_clk (double-flop).
  reg [1:0] freeze_en_core_s;
  always @(posedge core_clk or negedge core_resetn) begin
    if (!core_resetn) freeze_en_core_s <= 2'b00;
    else              freeze_en_core_s <= {freeze_en_core_s[0], freeze_en_reg};
  end
  wire freeze_en_core = freeze_en_core_s[1];

  // CDC: per-stage exponent select eth → core (double-flop; quasi-static).
  reg [4:0] dbg_stage_sel_core_s0, dbg_stage_sel_core_s1;
  always @(posedge core_clk or negedge core_resetn) begin
    if (!core_resetn) begin
      dbg_stage_sel_core_s0 <= 5'd0;
      dbg_stage_sel_core_s1 <= 5'd0;
    end else begin
      dbg_stage_sel_core_s0 <= dbg_stage_sel_reg;
      dbg_stage_sel_core_s1 <= dbg_stage_sel_core_s0;
    end
  end

  // CDC: cycle-trigger config eth → core.  Quasi-static (host programs it
  // before pulsing restart and leaves it stable), so a plain multi-FF bus
  // sync is safe — the value isn't changing when the core samples it.
  (* ASYNC_REG = "TRUE" *) reg [31:0] trig_cyc_core_s1, trig_cyc_core_s2;
  (* ASYNC_REG = "TRUE" *) reg [1:0]  trig_cyc_en_core_s;
  always @(posedge core_clk) begin
    trig_cyc_core_s1   <= trig_cyc_reg;
    trig_cyc_core_s2   <= trig_cyc_core_s1;
    trig_cyc_en_core_s <= {trig_cyc_en_core_s[0], trig_cyc_en_reg};
  end
  wire [31:0] trig_cyc_core    = trig_cyc_core_s2;
  wire        trig_cyc_en_core = trig_cyc_en_core_s[1];

  // Status from the multilayer (core_clk) → eth for read-back.  Quasi-static
  // once frozen=1 (host only reads the count/layer after observing frozen).
  wire [31:0] dbg_cyc_core;
  wire [4:0]  dbg_cur_layer_core;
  wire        dbg_frozen_core;
  (* ASYNC_REG = "TRUE" *) reg [31:0] dbg_cyc_eth_s1, dbg_cyc_eth_s2;
  (* ASYNC_REG = "TRUE" *) reg [4:0]  dbg_cur_layer_eth_s1, dbg_cur_layer_eth_s2;
  (* ASYNC_REG = "TRUE" *) reg [1:0]  dbg_frozen_eth_s;
  always @(posedge eth_clk) begin
    dbg_cyc_eth_s1       <= dbg_cyc_core;
    dbg_cyc_eth_s2       <= dbg_cyc_eth_s1;
    dbg_cur_layer_eth_s1 <= dbg_cur_layer_core;
    dbg_cur_layer_eth_s2 <= dbg_cur_layer_eth_s1;
    dbg_frozen_eth_s     <= {dbg_frozen_eth_s[0], dbg_frozen_core};
  end

  // Runtime factor-override CDC.  Host writes to 0x100..0x12F (per-layer
  // SwiGLU lo+mlp+attn factors) latch into eth-clk shadow regs + a toggle.
  // Core-clk side detects toggle edge, captures data, asserts the one-cycle
  // write-enable into the multilayer factor RAMs.
  // (factor_wr_layer_eth, factor_wr_data_eth, factor_wr_kind_eth,
  //  factor_wr_toggle_eth declared earlier near the snapshot regs.)

  reg        factor_wr_tog_s0, factor_wr_tog_s1, factor_wr_tog_s2;
  reg [4:0]  factor_wr_layer_core;
  reg [31:0] factor_wr_data_core;
  reg [1:0]  factor_wr_kind_core;
  wire       factor_wr_edge_core = factor_wr_tog_s1 ^ factor_wr_tog_s2;

  always @(posedge core_clk or negedge core_resetn) begin
    if (!core_resetn) begin
      factor_wr_tog_s0 <= 1'b0;
      factor_wr_tog_s1 <= 1'b0;
      factor_wr_tog_s2 <= 1'b0;
    end else begin
      factor_wr_tog_s0 <= factor_wr_toggle_eth;
      factor_wr_tog_s1 <= factor_wr_tog_s0;
      factor_wr_tog_s2 <= factor_wr_tog_s1;
      // Snapshot data on the edge — sender held them stable since the toggle.
      if (factor_wr_edge_core) begin
        factor_wr_layer_core <= factor_wr_layer_eth;
        factor_wr_data_core  <= factor_wr_data_eth;
        factor_wr_kind_core  <= factor_wr_kind_eth;
      end
    end
  end
  // Delay the write pulse by one core cycle so the latched data above is
  // stable when the multilayer FSM samples it.
  reg factor_wr_pulse_core;
  always @(posedge core_clk or negedge core_resetn) begin
    if (!core_resetn) factor_wr_pulse_core <= 1'b0;
    else              factor_wr_pulse_core <= factor_wr_edge_core;
  end
  wire factor_wr_en_swiglu_lo_core  = factor_wr_pulse_core && (factor_wr_kind_core == 2'd0);
  wire factor_wr_en_swiglu_mlp_core = factor_wr_pulse_core && (factor_wr_kind_core == 2'd1);
  wire factor_wr_en_attn_core       = factor_wr_pulse_core && (factor_wr_kind_core == 2'd2);

  // Readback CDC: select sync eth→core, data sync core→eth.
  // Both are slow-changing (host sets sel, sleeps ~ms, reads result) so
  // a simple double-flop per direction is sufficient.
  reg [6:0]  factor_rd_sel_core_s0, factor_rd_sel_core;
  wire [31:0] factor_rd_data_core;
  always @(posedge core_clk or negedge core_resetn) begin
    if (!core_resetn) begin
      factor_rd_sel_core_s0 <= '0;
      factor_rd_sel_core    <= '0;
    end else begin
      factor_rd_sel_core_s0 <= factor_rd_sel_eth;
      factor_rd_sel_core    <= factor_rd_sel_core_s0;
    end
  end
  reg [31:0] factor_rd_data_eth_s0, factor_rd_data_eth_s1;
  always @(posedge eth_clk) begin
    factor_rd_data_eth_s0 <= factor_rd_data_core;
    factor_rd_data_eth_s1 <= factor_rd_data_eth_s0;
  end

  // Per-row scale write CDC — same toggle-handshake pattern as factor_wr.
  reg        scale_wr_tog_s0, scale_wr_tog_s1, scale_wr_tog_s2;
  reg [3:0]  scale_wr_kind_core;
  reg [15:0] scale_wr_addr_core, scale_wr_data_core;
  wire       scale_wr_edge_core = scale_wr_tog_s1 ^ scale_wr_tog_s2;
  always @(posedge core_clk or negedge core_resetn) begin
    if (!core_resetn) begin
      scale_wr_tog_s0 <= 1'b0; scale_wr_tog_s1 <= 1'b0; scale_wr_tog_s2 <= 1'b0;
    end else begin
      scale_wr_tog_s0 <= scale_wr_toggle_eth;
      scale_wr_tog_s1 <= scale_wr_tog_s0;
      scale_wr_tog_s2 <= scale_wr_tog_s1;
      if (scale_wr_edge_core) begin
        scale_wr_kind_core <= scale_wr_kind_eth;
        scale_wr_addr_core <= scale_wr_addr_eth;
        scale_wr_data_core <= scale_wr_data_eth;
      end
    end
  end
  reg scale_wr_pulse_core;
  always @(posedge core_clk or negedge core_resetn) begin
    if (!core_resetn) scale_wr_pulse_core <= 1'b0;
    else              scale_wr_pulse_core <= scale_wr_edge_core;
  end

// Int8 path retired — see git history (commit f6a5e9a-era and earlier)
// for the smollm_multilayer_tm_selftest instantiation that lived here.
// Now the BFP autoregress stack is the only model path.
`include "lbfp_full_cfg.svh"   // LBFP_FULL_{D,HQ,HKV,HD,FFN,NL,MAX_CTX,VOCAB,NPROMPT,NGEN}
  // NB: scripts/run.tcl removes MICROGPT_LAYER_DEBUG / MICROGPT_DDR3_WEIGHTS /
  // MICROGPT_ILA from the define set when USE_BFP=1 (those defines drive
  // int8-only ILA probes + AXI weight streamer paths the BFP single-layer
  // selftest doesn't expose).
  // ----------------------------------------------------------------------
  // Block-FP path: full SmolLM2-135M autoregressive token generator.
  //
  // One-shot FSM (autoregress_bfp_top.sv) drives 19 token-step inferences
  // through embed_lookup → multilayer × NL=30 → decode_head, with the
  // 4-token prompt baked into a BRAM ROM (lbfp_full_PROMPT.hex) and
  // dec_token fed back as the next step's token_in.  All weights /
  // embedding / norm gammas are $readmemh-loaded BRAMs — no DDR3
  // streaming, so the m_axi_* read master is tied off below.
  //
  // result_tokens is 19 × 16 bits = 304 bits.  Pack into the lower bits
  // of lay_result (9216-bit legacy bus); host reads via the existing
  // 0x1D0 regmap window (32 32-bit words, 19 tokens occupy the first ~10).
  // ----------------------------------------------------------------------
  // Result-token buffer sized for the maximum prompt the bitstream
  // supports (NPROMPT_MAX) + the fixed N_GEN.  The active prompt
  // length is host-settable at runtime via regmap 0x063, so the
  // first `n_prompt_active` slots hold the prompt tokens and the
  // next N_GEN slots hold the generated tokens.
  localparam int LBFP_NSTEPS = `LBFP_FULL_NPROMPT_MAX + `LBFP_FULL_NGEN;
  wire [LBFP_NSTEPS*16-1:0] bfp_result_tokens;

  // Active prompt length register.  Eth_clk side latched on regmap
  // write to 0x063; 2-FF synced into core_clk for the autoregress
  // FSM.  Boot default = LBFP_FULL_NPROMPT (the value that came out
  // of the packer for the prompt baked into the .hex set).
  localparam int NPA_W = $clog2(`LBFP_FULL_NPROMPT_MAX + 1);
  reg [NPA_W-1:0] n_prompt_active_eth;
  (* ASYNC_REG = "TRUE" *) reg [NPA_W-1:0] npa_sync1_core, npa_sync2_core;
  always_ff @(posedge core_clk) begin
    npa_sync1_core <= n_prompt_active_eth;
    npa_sync2_core <= npa_sync1_core;
  end
  wire [NPA_W-1:0] n_prompt_active_core = npa_sync2_core;

  // Single-shot start: exactly one autoregress run per (reset OR restart
  // pulse).  Earlier perpetual auto-restart hammered MIG with reads
  // during uploads, contended with ddr_wr_master, and made it hard to
  // know whether a given run's result_tokens reflected fully-uploaded
  // weights or a mid-upload snapshot.  has_run latches at lay_done and
  // suppresses re-fire until the next restart pulse clears it.
  reg bfp_start_r;
  reg has_run;
  always_ff @(posedge core_clk) begin
    if (~core_resetn) begin
      bfp_start_r <= 1'b0;
      has_run     <= 1'b0;
    end else if (lay_restart_core) begin
      bfp_start_r <= 1'b0;
      has_run     <= 1'b0;       // re-arm for next run
    end else if (lay_done_core) begin
      bfp_start_r <= 1'b0;
      has_run     <= 1'b1;       // run complete — stay parked
    end else if (!has_run) begin
      bfp_start_r <= 1'b1;       // pre-completion: start signal held high
    end else begin
      bfp_start_r <= 1'b0;       // post-completion: idle
    end
  end

  // ----------------------------------------------------------------------
  // Weights stream from the lbfp_full_DDR3.bin image that the host
  // must have uploaded to DDR3 before pulsing `start`.  STREAM_LOOKUP
  // (EMBED row-fetch via AXI) follows the same path.
  // ----------------------------------------------------------------------
  localparam bit LBFP_STREAM_LOOKUP  = 1'b1;

`include "smollm/lbfp_ddr3.svh"

  autoregress_bfp_top #(
    .D       (`LBFP_FULL_D),
    .H_Q     (`LBFP_FULL_HQ),
    .H_KV    (`LBFP_FULL_HKV),
    .HD      (`LBFP_FULL_HD),
    .FFN     (`LBFP_FULL_FFN),
    .NL      (`LBFP_FULL_NL),
    .MAX_CTX (`LBFP_FULL_MAX_CTX),
    .VOCAB   (`LBFP_FULL_VOCAB),
    .NPROMPT_MAX(`LBFP_FULL_NPROMPT_MAX),
    .N_PROMPT(`LBFP_FULL_NPROMPT),
    .N_GEN   (`LBFP_FULL_NGEN),
    .PREFIX  ("lbfp_full_"),
    .STREAM_LOOKUP  (LBFP_STREAM_LOOKUP),
    .AXI_ADDR_WIDTH (30),
    .AXI_ID_WIDTH   (5)
  ) i_lay_st (
    .clk          ( core_clk          ),
    .rst          ( ~core_resetn | lay_restart_core ),
    .start        ( bfp_start_r       ),
    .n_prompt_active( n_prompt_active_core ),
    .done         ( lay_done_core     ),
    .result_tokens( bfp_result_tokens ),
    .weight_hash  ( bfp_weight_hash_core ),
    // DDR3 region base offsets (lbfp_ddr3.svh).
    .ws_base_WQ_m       ( `LBFP_BASE_WQ_M     ),
    .ws_base_WQ_e       ( `LBFP_BASE_WQ_E     ),
    .ws_base_WK_m       ( `LBFP_BASE_WK_M     ),
    .ws_base_WK_e       ( `LBFP_BASE_WK_E     ),
    .ws_base_WV_m       ( `LBFP_BASE_WV_M     ),
    .ws_base_WV_e       ( `LBFP_BASE_WV_E     ),
    .ws_base_WO_m       ( `LBFP_BASE_WO_M     ),
    .ws_base_WO_e       ( `LBFP_BASE_WO_E     ),
    .ws_base_WG_m       ( `LBFP_BASE_WG_M     ),
    .ws_base_WG_e       ( `LBFP_BASE_WG_E     ),
    .ws_base_WU_m       ( `LBFP_BASE_WU_M     ),
    .ws_base_WU_e       ( `LBFP_BASE_WU_E     ),
    .ws_base_WDN_m      ( `LBFP_BASE_WDN_M    ),
    .ws_base_WDN_e      ( `LBFP_BASE_WDN_E    ),
    .ws_base_EMBED_m    ( `LBFP_BASE_EMBED_M  ),
    .ws_base_EMBED_e    ( `LBFP_BASE_EMBED_E  ),
    .ws_base_EMBED_LU_m ( `LBFP_BASE_EMBED_LU_M ),
    .ws_base_EMBED_LU_e ( `LBFP_BASE_EMBED_LU_E ),
    // AXI master to MIG (ui_clk domain).
    .clk_axi      ( ui_clk            ),
    .rst_axi      ( ui_clk_sync_rst   ),
    .m_axi_arvalid( m_axi_arvalid     ),
    .m_axi_arready( m_axi_arready     ),
    .m_axi_arid   ( m_axi_arid        ),
    .m_axi_araddr ( m_axi_araddr      ),
    .m_axi_arlen  ( m_axi_arlen       ),
    .m_axi_arsize ( m_axi_arsize      ),
    .m_axi_arburst( m_axi_arburst     ),
    .m_axi_arlock ( m_axi_arlock      ),
    .m_axi_arcache( m_axi_arcache     ),
    .m_axi_arprot ( m_axi_arprot      ),
    .m_axi_arqos  ( m_axi_arqos       ),
    .m_axi_rvalid ( m_axi_rvalid      ),
    .m_axi_rready ( m_axi_rready      ),
    .m_axi_rid    ( m_axi_rid         ),
    .m_axi_rdata  ( m_axi_rdata       ),
    .m_axi_rresp  ( m_axi_rresp       ),
    .m_axi_rlast  ( m_axi_rlast       ),
    .wr_kind      ( bram_kind_eth      ),
    .wr_addr      ( bram_port_addr_eth ),
    .wr_data      ( bram_data_eth      ),
    .wr_en        ( bram_wr_en_eth     ),
    .clk_wr       ( eth_clk            ),
    .wr_rdata     ( bram_rdata_eth     ),
    .snap_layer_sel ( snap_sel_core_s1 ),
    .snap_step_sel  ( step_sel_core_s1 ),
    .dbg_stage_sel  ( dbg_stage_sel_core_s1 ),
    .freeze_en      ( freeze_en_core   ),
    .trig_cyc_en    ( trig_cyc_en_core ),
    .trig_cyc       ( trig_cyc_core    ),
    .dbg_cyc        ( dbg_cyc_core     ),
    .dbg_cur_layer  ( dbg_cur_layer_core ),
    .dbg_frozen     ( dbg_frozen_core  )
  );
  assign lay_result = {{(9216 - LBFP_NSTEPS*16){1'b0}}, bfp_result_tokens};

  // Factor read port returns zero in the BFP path (no swiglu/attn factors).
  // (factor_rd_data_core is wire-declared at module scope; drive to '0.)
  assign factor_rd_data_core = '0;

  reg [1:0] lay_done_sync = 2'd0;
  always_ff @(posedge eth_clk) lay_done_sync <= {lay_done_sync[0], lay_done_core};
  wire lay_done = lay_done_sync[1];

  // ----------------------------------------------------------------------
  // Latched lay_done — autoregress_bfp_top.done is a 1-cycle pulse so a
  // 200 ms host poll always misses it; latch in core_clk and clear on
  // the next restart.  Readable at 0x1F0[1].
  // ----------------------------------------------------------------------
  reg lay_done_latched_core = 1'b0;
  always_ff @(posedge core_clk or negedge core_resetn) begin
    if (!core_resetn) begin
      lay_done_latched_core <= 1'b0;
    end else if (lay_restart_core) begin
      lay_done_latched_core <= 1'b0;
    end else if (lay_done_core) begin
      lay_done_latched_core <= 1'b1;
    end
  end
  reg [1:0] lay_done_latched_sync = 2'd0;
  always_ff @(posedge eth_clk) lay_done_latched_sync <= {lay_done_latched_sync[0], lay_done_latched_core};
  wire lay_done_latched = lay_done_latched_sync[1];

  // ----------------------------------------------------------------------
  // BFP AXI-master activity diagnostics (ui_clk domain).  These tap the
  // module-scope m_axi_* wires that the BFP autoregress drives via the
  // 3-way mux in autoregress_bfp_top.  In the int8 path the same wires
  // are driven by weight_streamer_mt — these counters work for both.
  //   0x050 : AR-accept count   (arvalid && arready)
  //   0x051 : R-beat   count    (rvalid && rready)
  //   0x052 : last accepted araddr
  // Cross-clock: ui_clk→eth_clk via a single FF stage.  Values change
  // slowly relative to the eth poll rate, so missing one increment is
  // not a correctness issue for this diagnostic; read twice + check
  // consistency if you need a clean snapshot.
  // ----------------------------------------------------------------------
  reg [31:0] bfp_ar_cnt_ui      = 32'd0;
  reg [31:0] bfp_r_cnt_ui       = 32'd0;
  reg [29:0] bfp_last_araddr_ui = 30'd0;
  always_ff @(posedge ui_clk) begin
    if (ui_clk_sync_rst) begin
      bfp_ar_cnt_ui      <= 32'd0;
      bfp_r_cnt_ui       <= 32'd0;
      bfp_last_araddr_ui <= 30'd0;
    end else begin
      if (m_axi_arvalid && m_axi_arready) begin
        bfp_ar_cnt_ui      <= bfp_ar_cnt_ui + 32'd1;
        bfp_last_araddr_ui <= m_axi_araddr;
      end
      if (m_axi_rvalid && m_axi_rready)
        bfp_r_cnt_ui       <= bfp_r_cnt_ui + 32'd1;
    end
  end
  // Single-FF resample into eth_clk for the regfile read mux.
  reg [31:0] bfp_ar_cnt_eth      = 32'd0;
  reg [31:0] bfp_r_cnt_eth       = 32'd0;
  reg [29:0] bfp_last_araddr_eth = 30'd0;
  always_ff @(posedge eth_clk) begin
    bfp_ar_cnt_eth      <= bfp_ar_cnt_ui;
    bfp_r_cnt_eth       <= bfp_r_cnt_ui;
    bfp_last_araddr_eth <= bfp_last_araddr_ui;
  end

  wire         ddr_wr_req_eth;
  wire [29:0]  ddr_wr_addr_eth;
  wire [511:0] ddr_wr_data_eth;
  wire         ddr_wr_ack_ui;
  // ack returns to eth_ctrl directly (it does its own 2FF sync)
  wire         ddr_wr_ack_eth_sync = ddr_wr_ack_ui;

  // 2FF sync of req from eth_clk to ui_clk
  reg ddr_wr_req_sync0 = 1'b0;
  reg ddr_wr_req_sync1 = 1'b0;
  always_ff @(posedge ui_clk) begin
    if (ui_clk_sync_rst) begin
      ddr_wr_req_sync0 <= 1'b0;
      ddr_wr_req_sync1 <= 1'b0;
    end else begin
      ddr_wr_req_sync0 <= ddr_wr_req_eth;
      ddr_wr_req_sync1 <= ddr_wr_req_sync0;
    end
  end

  ddr_write_master #(
    .AXI_DATA_WIDTH ( 512 ),
    .AXI_ADDR_WIDTH ( 30  ),
    .AXI_ID_WIDTH   ( 5   )
  ) i_ddr_wr (
    .clk           ( ui_clk                                  ),
    .rst           ( ui_clk_sync_rst | ~init_calib_complete  ),
    .write_req     ( ddr_wr_req_sync1                        ),
    .write_ack     ( ddr_wr_ack_ui                           ),
    .write_addr    ( ddr_wr_addr_eth                         ),
    .write_data    ( ddr_wr_data_eth                         ),
    .m_axi_awvalid ( m_axi_awvalid ),
    .m_axi_awready ( m_axi_awready ),
    .m_axi_awid    ( m_axi_awid    ),
    .m_axi_awaddr  ( m_axi_awaddr  ),
    .m_axi_awlen   ( m_axi_awlen   ),
    .m_axi_awsize  ( m_axi_awsize  ),
    .m_axi_awburst ( m_axi_awburst ),
    .m_axi_awlock  ( m_axi_awlock  ),
    .m_axi_awcache ( m_axi_awcache ),
    .m_axi_awprot  ( m_axi_awprot  ),
    .m_axi_awqos   ( m_axi_awqos   ),
    .m_axi_wvalid  ( m_axi_wvalid  ),
    .m_axi_wready  ( m_axi_wready  ),
    .m_axi_wdata   ( m_axi_wdata   ),
    .m_axi_wstrb   ( m_axi_wstrb   ),
    .m_axi_wlast   ( m_axi_wlast   ),
    .m_axi_bvalid  ( m_axi_bvalid  ),
    .m_axi_bready  ( m_axi_bready  ),
    .m_axi_bid     ( m_axi_bid     ),
    .m_axi_bresp   ( m_axi_bresp   )
  );

  // Stretch the 1-cycle rx_activity pulse to ~50 ms so a human eye can see
  // each accepted ARP / UDP frame land.
  wire eth_ctrl_rx_activity;
  reg [22:0] rx_stretch_cnt = 23'd0;
  always_ff @(posedge eth_clk or negedge eth_rst_n_int) begin
    if (!eth_rst_n_int) rx_stretch_cnt <= 23'd0;
    else if (eth_ctrl_rx_activity) rx_stretch_cnt <= 23'h7FFFFF;
    else if (rx_stretch_cnt != 23'd0) rx_stretch_cnt <= rx_stretch_cnt - 23'd1;
  end
  wire rx_active_stretched = (rx_stretch_cnt != 23'd0);

  // ================================================================
  //  Avalon-MM slave: register-map writes / reads (eth_clk domain).
  //  This is the same logic as on the DE1-SoC, with CLOCK_50 -> eth_clk.
  // ================================================================
  always @(posedge eth_clk) begin
    jtag_master_readdatavalid <= 1'b0;
    // Default-clear the 1-cycle BRAM write pulse; the 0x061 case below
    // re-asserts it for exactly the cycle the data is latched.
    bram_wr_en_eth <= 1'b0;

    // Reset only the new factor-override shadow regs (others rely on
    // inline init that pre-dates the no-inline-init rule).
    if (rst_sys) begin
      factor_wr_layer_eth  <= '0;
      factor_wr_data_eth   <= '0;
      factor_wr_kind_eth   <= '0;
      factor_wr_toggle_eth <= 1'b0;
      factor_rd_sel_eth    <= '0;
      scale_wr_kind_eth    <= '0;
      scale_wr_addr_eth    <= '0;
      scale_wr_data_eth    <= '0;
      scale_wr_toggle_eth  <= 1'b0;
      n_prompt_active_eth  <= NPA_W'(`LBFP_FULL_NPROMPT);
    end

    if (read_pending_reg) begin
      jtag_master_readdatavalid <= 1'b1;
      read_pending_reg <= 1'b0;
    end else if (jtag_master_read) begin
      jtag_master_readdata <= read_data_comb;
      read_pending_reg <= 1'b1;
      host_toggle_reg <= ~host_toggle_reg;
    end

    if (jtag_master_write) begin
      host_toggle_reg <= ~host_toggle_reg;
      case (jtag_word_addr)
        10'h002: begin
          if (jtag_master_writedata[0])
            host_start_toggle_eth <= ~host_start_toggle_eth;
          if (jtag_master_writedata[1])
            host_clear_toggle_eth <= ~host_clear_toggle_eth;
        end
        10'h004: begin
          host_max_gen_reg     <= jtag_master_writedata[15:8];
          host_temperature_reg <= jtag_master_writedata[31:16];
        end
        10'h005: host_seed_reg <= jtag_master_writedata;
        // 0x00A handled in its own always_ff above (avoids Vivado ROM
        // misclassification when the inline case-write coexists with
        // an inline initial value).
        10'h008: begin
          host_direct_mode_eth <= jtag_master_writedata[0];
          host_step_clear_eth  <= jtag_master_writedata[1];
          host_step_pos_eth    <= jtag_master_writedata[15:8];
          host_step_token_eth  <= jtag_master_writedata[23:16];
        end
        10'h009: begin
          if (jtag_master_writedata[0])
            host_step_toggle_eth <= ~host_step_toggle_eth;
        end
        10'h010: begin
          // bit 0 = load trigger (toggle), bit 1 = consumer_swap (toggle)
          if (jtag_master_writedata[0])
            ddr_load_toggle_eth <= ~ddr_load_toggle_eth;
          if (jtag_master_writedata[1])
            ddr_swap_toggle_eth <= ~ddr_swap_toggle_eth;
        end
        10'h011: ddr_load_addr_eth <= jtag_master_writedata[29:0];
        // Selftest re-trigger: each writes a single bit which toggles a
        // per-selftest signal in eth_clk; the CDC at the top of the file
        // 2FF-syncs into core_clk and edge-detects to produce a one-cycle
        // pulse there.
        10'h109: if (jtag_master_writedata[0]) mvst_rst_tog_eth <= ~mvst_rst_tog_eth;
        10'h131: if (jtag_master_writedata[0]) rms_rst_tog_eth  <= ~rms_rst_tog_eth;
        10'h161: if (jtag_master_writedata[0]) rope_rst_tog_eth <= ~rope_rst_tog_eth;
        10'h191: if (jtag_master_writedata[0]) sg_rst_tog_eth   <= ~sg_rst_tog_eth;
        10'h1C1: if (jtag_master_writedata[0]) sm_rst_tog_eth   <= ~sm_rst_tog_eth;
        10'h1F1: if (jtag_master_writedata[0]) lay_rst_tog_eth  <= ~lay_rst_tog_eth;
        // 0x060 — set BRAM target.  Format:
        //   [31]    auto-increment enable
        //   [22:18] bram_kind  (0..6 in Phase 1, 7..31 reserved)
        //   [17:0]  base address inside the destination BRAM
        10'h060: begin
          bram_inc_eth  <= jtag_master_writedata[31];
          bram_kind_eth <= jtag_master_writedata[22:18];
          bram_addr_eth <= jtag_master_writedata[17:0];
        end
        // 0x061 — one BRAM write.  bram_data = writedata[15:0]; pulse
        // bram_wr_en_eth high for exactly the cycle the data is latched
        // so the dual-port BRAM (clk_wr = eth_clk) captures it on the
        // next edge.  If bram_inc_eth is set, increment bram_addr_eth
        // after the pulse so a stream of writes to 0x061 fills the
        // BRAM linearly.
        10'h061: begin
          bram_data_eth    <= jtag_master_writedata[15:0];
          bram_wr_addr_eth <= bram_addr_eth;       // pre-increment snapshot
          bram_wr_en_eth   <= 1'b1;
          if (bram_inc_eth) bram_addr_eth <= bram_addr_eth + 18'd1;
        end
        // 0x063 — set the active prompt length (number of prompt_rom
        // entries the autoregress FSM consumes before switching to
        // generation).  Must be in [1, LBFP_FULL_NPROMPT_MAX].  Boot
        // default is `LBFP_FULL_NPROMPT (the value baked into the
        // .hex prompt file at packer-time).  Host writes this once
        // per fine-tune after load-roms.
        10'h063: n_prompt_active_eth <= jtag_master_writedata[NPA_W-1:0];
        default: ;
      endcase
      // Runtime SwiGLU/Attn factor overrides — eth-clk shadow + toggle CDC.
      // 0x100+L (L=0..29): {up_in[31:16], gate_in[15:0]}
      // 0x140+L         : mlp_out_factor[23:0]
      // 0x180+L         : attn_factor[23:0]
      if (jtag_word_addr >= 10'h100 && jtag_word_addr < 10'h11E) begin
        factor_wr_layer_eth  <= jtag_word_addr[4:0];
        factor_wr_data_eth   <= jtag_master_writedata;
        factor_wr_kind_eth   <= 2'd0;
        factor_wr_toggle_eth <= ~factor_wr_toggle_eth;
      end else if (jtag_word_addr >= 10'h140 && jtag_word_addr < 10'h15E) begin
        factor_wr_layer_eth  <= jtag_word_addr[4:0];
        factor_wr_data_eth   <= jtag_master_writedata;
        factor_wr_kind_eth   <= 2'd1;
        factor_wr_toggle_eth <= ~factor_wr_toggle_eth;
      end else if (jtag_word_addr >= 10'h180 && jtag_word_addr < 10'h19E) begin
        factor_wr_layer_eth  <= jtag_word_addr[4:0];
        factor_wr_data_eth   <= jtag_master_writedata;
        factor_wr_kind_eth   <= 2'd2;
        factor_wr_toggle_eth <= ~factor_wr_toggle_eth;
      end else if (jtag_word_addr == 10'h00B) begin
        // Factor readback select.  Host writes {kind[1:0], layer[4:0]} here,
        // then reads 0x00F to get the factor value back.
        factor_rd_sel_eth <= jtag_master_writedata[6:0];
      end else if (jtag_word_addr == 10'h01C) begin
        // Scale brom write: stage {kind[3:0], addr[15:0]}.
        scale_wr_addr_eth <= jtag_master_writedata[15:0];
        scale_wr_kind_eth <= jtag_master_writedata[19:16];
      end else if (jtag_word_addr == 10'h01D) begin
        // Scale brom write trigger: capture data + toggle handshake.
        scale_wr_data_eth   <= jtag_master_writedata[15:0];
        scale_wr_toggle_eth <= ~scale_wr_toggle_eth;
      end
    end
  end

  // ================================================================
  //  microgpt core (56.25 MHz core_clk domain)
  // ================================================================
`ifdef MICROGPT_USE_BFP
  // BabyGPT exact-core retired in BFP builds — the host never pulses
  // start_core_reg in USE_BFP=1, so its BabyGPT FSM is dormant.  Tying
  // its outputs to constants lets Vivado constant-propagate through
  // the (unreachable) BabyGPT control FSM and prune everything it
  // touches; saves ~30k Slice LUTs on smollm360 (which otherwise
  // overflows the XC7VX485T's 303 600-LUT budget at place_design).
  assign core_busy         = 1'b0;
  assign core_done         = 1'b0;
  assign core_next_token   = '0;
  assign core_argmax_token = '0;
  assign core_rng_state    = '0;
  assign core_top_logit    = '0;
  assign core_logits_flat  = '0;
`else
  microgpt_exact_core core_inst (
    .clk(core_clk),
    .resetn(core_resetn),
    .start(start_core_reg),
    .clear_cache(clear_cache_reg),
    .sample_mode(~direct_mode_reg),
    .temperature_q8_8(temperature_reg),
    .rng_state_in(rng_reg),
    .token_in(token_reg),
    .pos_in(pos_reg),
    .busy(core_busy),
    .done(core_done),
    .next_token(core_next_token),
    .argmax_token(core_argmax_token),
    .rng_state_out(core_rng_state),
    .top_logit_q12(core_top_logit),
    .logits_flat(core_logits_flat)
  );
`endif

  // ----- Core control FSM (core_clk domain) — copied verbatim from
  // de1_soc_microgpt_rtl.sv, with toggle CDC sources renamed for eth_clk.
  always @(posedge core_clk) begin
    if (!core_resetn) begin
      state_reg <= ST_READY;
      token_reg <= BOS_TOKEN;
      pos_reg <= 8'd0;
      out_len_reg <= 8'd0;
      rng_reg <= host_seed_reg;
      temperature_reg <= host_temperature_reg;
      max_gen_reg <= host_max_gen_reg;
      start_core_reg <= 1'b0;
      clear_cache_reg <= 1'b0;
      done_latched_reg <= 1'b0;
      last_token_reg <= 8'd0;
      cycle_blink_reg <= 16'd0;
      perf_cycles_reg <= 32'd0;
      tokens_per_sec_reg <= 32'd0;
      error_reg <= 1'b0;
      host_start_req <= 1'b0;
      host_clear_req <= 1'b0;
      host_run_reg <= 1'b0;
      start_sync0 <= host_start_toggle_eth;
      start_sync1 <= host_start_toggle_eth;
      start_seen_reg <= host_start_toggle_eth;
      clear_sync0 <= host_clear_toggle_eth;
      clear_sync1 <= host_clear_toggle_eth;
      clear_seen_reg <= host_clear_toggle_eth;
      step_sync0 <= host_step_toggle_eth;
      step_sync1 <= host_step_toggle_eth;
      step_seen_reg <= host_step_toggle_eth;
      host_step_req <= 1'b0;
      direct_mode_reg <= 1'b0;
      step_clear_reg <= 1'b0;
      step_token_reg <= BOS_TOKEN;
      step_pos_reg <= 8'd0;
      for (out_i = 0; out_i < 16; out_i = out_i + 1) output_mem[out_i] <= 8'd0;
    end else begin
      start_core_reg <= 1'b0;
      clear_cache_reg <= 1'b0;
      cycle_blink_reg <= cycle_blink_reg + 16'd1;

      start_sync0 <= host_start_toggle_eth; start_sync1 <= start_sync0;
      clear_sync0 <= host_clear_toggle_eth; clear_sync1 <= clear_sync0;
      step_sync0  <= host_step_toggle_eth;  step_sync1  <= step_sync0;

      if (start_sync1 != start_seen_reg) begin
        host_start_req <= 1'b1;
        host_run_reg   <= 1'b1;
        start_seen_reg <= start_sync1;
        max_gen_reg     <= host_max_gen_reg;
        temperature_reg <= host_temperature_reg;
        rng_reg         <= host_seed_reg;
        direct_mode_reg <= 1'b0;
      end

      if (clear_sync1 != clear_seen_reg) begin
        host_clear_req <= 1'b1;
        clear_seen_reg <= clear_sync1;
      end

      if (step_sync1 != step_seen_reg) begin
        host_step_req   <= 1'b1;
        step_seen_reg   <= step_sync1;
        direct_mode_reg <= host_direct_mode_eth;
        step_clear_reg  <= host_step_clear_eth;
        step_token_reg  <= host_step_token_eth;
        step_pos_reg    <= host_step_pos_eth;
      end

      if (state_reg == ST_WAIT_CORE) perf_cycles_reg <= perf_cycles_reg + 32'd1;

      if (host_clear_req) begin
        state_reg <= ST_READY;
        token_reg <= BOS_TOKEN;
        pos_reg <= 8'd0;
        out_len_reg <= 8'd0;
        done_latched_reg <= 1'b0;
        last_token_reg <= 8'd0;
        perf_cycles_reg <= 32'd0;
        tokens_per_sec_reg <= 32'd0;
        error_reg <= 1'b0;
        host_clear_req <= 1'b0;
        host_run_reg <= 1'b0;
        host_step_req <= 1'b0;
        direct_mode_reg <= 1'b0;
        for (out_i = 0; out_i < 16; out_i = out_i + 1) output_mem[out_i] <= 8'd0;
      end else if (!enable && !host_run_reg) begin
        state_reg <= ST_READY;
        token_reg <= BOS_TOKEN;
        pos_reg <= 8'd0;
        out_len_reg <= 8'd0;
        done_latched_reg <= 1'b0;
        last_token_reg <= 8'd0;
      end else begin
        case (state_reg)
          ST_READY: begin
            if (host_step_req && direct_mode_reg) begin
              token_reg <= step_token_reg;
              pos_reg <= step_pos_reg;
              out_len_reg <= 8'd0;
              done_latched_reg <= 1'b0;
              last_token_reg <= 8'd0;
              perf_cycles_reg <= 32'd0;
              tokens_per_sec_reg <= 32'd0;
              error_reg <= 1'b0;
              if (step_clear_reg) rng_reg <= host_seed_reg;
              clear_cache_reg <= step_clear_reg;
              start_core_reg <= 1'b1;
              state_reg <= ST_WAIT_CORE;
              host_step_req <= 1'b0;
            end else if (host_start_req) begin
              token_reg <= BOS_TOKEN;
              pos_reg <= 8'd0;
              out_len_reg <= 8'd0;
              done_latched_reg <= 1'b0;
              last_token_reg <= 8'd0;
              perf_cycles_reg <= 32'd0;
              tokens_per_sec_reg <= 32'd0;
              error_reg <= 1'b0;
              for (out_i = 0; out_i < 16; out_i = out_i + 1) output_mem[out_i] <= 8'd0;
              clear_cache_reg <= 1'b1;
              if (max_gen_reg == 8'd0 || max_gen_reg > 8'd15) begin
                error_reg <= 1'b1;
                done_latched_reg <= 1'b1;
                state_reg <= ST_DONE;
              end else begin
                start_core_reg <= 1'b1;
                state_reg <= ST_WAIT_CORE;
              end
              host_start_req <= 1'b0;
            end
          end
          ST_WAIT_CORE: begin
            if (core_done) begin
              rng_reg <= core_rng_state;
              last_token_reg <= core_next_token;
              if (direct_mode_reg) begin
                done_latched_reg <= 1'b1;
                state_reg <= ST_DONE;
              end else if ((core_next_token == BOS_TOKEN) || (pos_reg == 8'd15)) begin
                done_latched_reg <= 1'b1;
                state_reg <= ST_DONE;
              end else begin
                output_mem[out_len_reg] <= core_next_token;
                token_reg <= core_next_token;
                pos_reg <= pos_reg + 8'd1;
                out_len_reg <= out_len_reg + 8'd1;
                if ((out_len_reg + 8'd1) >= max_gen_reg) begin
                  done_latched_reg <= 1'b1;
                  state_reg <= ST_DONE;
                end else begin
                  start_core_reg <= 1'b1;
                  state_reg <= ST_WAIT_CORE;
                end
              end
            end
          end
          ST_DONE: begin
            if (host_step_req && direct_mode_reg) begin
              token_reg <= step_token_reg;
              pos_reg <= step_pos_reg;
              out_len_reg <= 8'd0;
              done_latched_reg <= 1'b0;
              last_token_reg <= 8'd0;
              perf_cycles_reg <= 32'd0;
              tokens_per_sec_reg <= 32'd0;
              error_reg <= 1'b0;
              if (step_clear_reg) rng_reg <= host_seed_reg;
              clear_cache_reg <= step_clear_reg;
              start_core_reg <= 1'b1;
              state_reg <= ST_WAIT_CORE;
              host_step_req <= 1'b0;
            end else if (host_start_req) begin
              token_reg <= BOS_TOKEN;
              pos_reg <= 8'd0;
              out_len_reg <= 8'd0;
              done_latched_reg <= 1'b0;
              last_token_reg <= 8'd0;
              perf_cycles_reg <= 32'd0;
              tokens_per_sec_reg <= 32'd0;
              error_reg <= 1'b0;
              for (out_i = 0; out_i < 16; out_i = out_i + 1) output_mem[out_i] <= 8'd0;
              clear_cache_reg <= 1'b1;
              if (max_gen_reg == 8'd0 || max_gen_reg > 8'd15) begin
                error_reg <= 1'b1;
                done_latched_reg <= 1'b1;
                state_reg <= ST_DONE;
              end else begin
                start_core_reg <= 1'b1;
                state_reg <= ST_WAIT_CORE;
              end
              host_start_req <= 1'b0;
            end
          end
          default: state_reg <= ST_READY;
        endcase
      end
    end
  end

  // ================================================================
  //  Read mux — same MMIO map as on the DE1-SoC.
  // ================================================================
  always @(*) begin
    read_data_comb = 32'd0;
    out_i = 0;
    case (jtag_word_addr)
      10'h000: read_data_comb = 32'h4D475254;          // 'MGRT'
      10'h001: read_data_comb = 32'h00020001;
      10'h002: read_data_comb = 32'd0;
      10'h003: read_data_comb = {
        pos_reg, out_len_reg, 8'd0,
        2'd0,
        direct_mode_reg, host_toggle_reg,
        error_reg, done_latched_reg,
        (state_reg == ST_WAIT_CORE),
        (state_reg == ST_READY)
      };
      10'h004: read_data_comb = {temperature_reg, max_gen_reg, 8'd0};
      10'h005: read_data_comb = rng_reg;
      10'h00A: read_data_comb = {27'd0, snapshot_layer_sel_reg};
      10'h064: read_data_comb = {21'd0, snapshot_step_sel_reg};
      10'h065: read_data_comb = {31'd0, freeze_en_reg};
      10'h066: read_data_comb = trig_cyc_reg;
      10'h067: read_data_comb = {31'd0, trig_cyc_en_reg};
      10'h070: read_data_comb = {27'd0, dbg_stage_sel_reg};
      10'h071: read_data_comb = dbg_cyc_eth_s2;                       // cycles @ freeze
      10'h072: read_data_comb = {26'd0, dbg_frozen_eth_s[1], dbg_cur_layer_eth_s2}; // {frozen, layer}
      10'h00B: read_data_comb = {25'd0, factor_rd_sel_eth};
      10'h00F: read_data_comb = factor_rd_data_eth_s1;
      10'h008: read_data_comb = {8'd0, step_token_reg, step_pos_reg, step_clear_reg, direct_mode_reg};
      10'h006: read_data_comb = {core_top_logit[15:0], core_argmax_token, last_token_reg};
      10'h007: read_data_comb = {16'd0, 8'd0, BOS_TOKEN};
      10'h036: read_data_comb = perf_cycles_reg;
      10'h037: read_data_comb = tokens_per_sec_reg;
      // Diagnostic taps:
      // 0x00C: eth_clk-domain shadow registers + toggles
      10'h00C: read_data_comb = {
        host_temperature_reg,           // [31:16]
        host_max_gen_reg,                // [15:8]
        2'd0,
        host_step_toggle_eth,            // [5]
        host_clear_toggle_eth,           // [4]
        host_start_toggle_eth,           // [3]
        host_direct_mode_eth,            // [2]
        host_step_clear_eth,             // [1]
        1'b0
      };
      // 0x00E: DHCP-assigned FPGA IP (from eth_ctrl)
      10'h00E: read_data_comb = eth_ctrl_fpga_ip;
      // 0x010..0x016: DDR3 tile cache control + status
      10'h010: read_data_comb = {30'd0, ddr_swap_toggle_eth, ddr_load_toggle_eth};
      10'h011: read_data_comb = {2'd0, ddr_load_addr_eth};
      // 0x012 status flags + 0x013..0x044 diagnostic snapshots from the
      // smollm_layer streamer / matvec engine path.  Read after triggering
      // the layer selftest (host: write 0x1F1[0]=1, then read).
      //   0x012 [0]   init_calib_complete
      //         [1]   dbg_first_ar_seen   (streamer issued AR)
      //         [2]   dbg_first_r_seen    (streamer received R)
      //         [3]   dbg_snap_done       (4 eng_w samples captured)
      //   0x013       dbg_first_araddr (low 30 bits)
      //   0x014..0x023 dbg_first_rdata (16 words, 512 bits)
      //   0x024..0x033 dbg_eng_w_packed (4 × 128b)
      //   0x034..0x043 dbg_wd_packed    (4 × 128b)
      //   0x068..0x069 dbg_in_value_packed (4 × 16b lane values, 2 words)
      //                (relocated from 0x044 — that range belongs to core_logits_flat)
      10'h012: read_data_comb = {28'd0, dbg_snap_done, dbg_first_r_seen,
                                 dbg_first_ar_seen, init_calib_complete};
      10'h013: read_data_comb = {2'd0, dbg_first_araddr};
      // 0x017 [2:0]   ila_ml_state  (outer time-mux FSM:
      //                  0=LR_IDLE 1=LR_START 2=LR_LATCH 3=LR_GAP 4=LR_DONE)
      //       [7:3]   ila_ml_layer_idx (current layer 0..NL-1)
      //       [8]     lay_done_core (multilayer overall done)
      //       [13:9]  ila_state (inner smollm_layer FSM)
      //       [16:14] ila_mv_phase
      //       [27:17] ila_cnt
      10'h017: read_data_comb = {4'd0, ila_cnt, ila_mv_phase, ila_state,
                                 lay_done_core, ila_ml_layer_idx, ila_ml_state};
      // 0x018: framing_top ring-buffer pointers between MAC and parser.
      //   [4:0]   firstbuf  (read pointer, advanced after parse)
      //   [9:5]   nextbuf   (write pointer, advanced by MAC)
      //   [14:10] lastbuf   (configured max depth)
      // Depth = (nextbuf - firstbuf) & 31; saturation at lastbuf means
      // the eth FIFO has stalled and incoming frames are being refused.
      10'h018: read_data_comb = {17'd0, fr_lastbuf, fr_nextbuf, fr_firstbuf};
      // 0x049 [0] has_run — set when the single-shot autoregress run
      //              has completed since last restart pulse; cleared
      //              by writing to 0x1F1 (regular restart).
      // 0x04A [31:0] bfp_rdata_hash — rolling XOR-with-rotate hash over
      //              every 512-bit AXI rdata beat seen by the BFP
      //              master since the last restart pulse.  Cheaper than
      //              CRC32 (one rotate + 17 XORs, fits 5 ns at 200 MHz);
      //              adequate for the comparative diagnostic of
      //              hash(Python upload) vs hash(C++ upload).
      10'h049: read_data_comb = {31'd0, has_run_eth};
      10'h04A: read_data_comb = bfp_rdata_crc_eth;
      // 0x062: BRAM read-back at the address last set via 0x060.  Returns
      //        the 16-bit contents (zero-extended exponents) of the BRAM
      //        selected by bram_kind_eth.  Read latency is satisfied by
      //        the regmap's own read_pending_reg cycle plus the BRAM
      //        port-A output register that ran 1 eth_clk after 0x060.
      10'h062: read_data_comb = {16'd0, bram_rdata_eth};
      10'h063: read_data_comb = {{(32-NPA_W){1'b0}}, n_prompt_active_eth};
      // 0x019..0x01C: DDR3 write-path stage counters (eth_clk domain).
      //   0x019 ddr_wr_rx_count   = FT_DDR_WRITE frames accepted by parser
      //   0x01A ddr_wr_done_count = ddr_wr_req toggles (write dispatched)
      //   0x01B ddr_wr_ack_count  = ddr_ack_sync1 toggles seen (MIG done)
      //   0x01C ddr_wr_tx_count   = FT_ACK frames dispatched for those writes
      // During an upload, host can poll these and compare against its
      // `sent`/`acked` counts to localise where the pipeline stalls.
      10'h019: read_data_comb = ddr_wr_rx_count;
      10'h01A: read_data_comb = ddr_wr_done_count;
      10'h01B: read_data_comb = ddr_wr_ack_count;
      10'h01C: read_data_comb = ddr_wr_tx_count;
      // 0x100..0x107: matvec_selftest result (16 lanes × Q1.15 packed two
      //              per 32-bit word — lane 2k in [15:0], lane 2k+1 in [31:16]).
      // 0x108: matvec_selftest done flag (bit 0).
      // 0x109: write any value with bit[0]=1 to retrigger the selftest
      //        (read returns 0; write side is decoded in the write block below).
      // 0x10F: 32-bit BUILD_VERSION (low 32 bits of build epoch).
      10'h100: read_data_comb = mvst_result[ 31:  0];
      10'h101: read_data_comb = mvst_result[ 63: 32];
      10'h102: read_data_comb = mvst_result[ 95: 64];
      10'h103: read_data_comb = mvst_result[127: 96];
      10'h104: read_data_comb = mvst_result[159:128];
      10'h105: read_data_comb = mvst_result[191:160];
      10'h106: read_data_comb = mvst_result[223:192];
      10'h107: read_data_comb = mvst_result[255:224];
      10'h108: read_data_comb = {31'd0, mvst_done};
      10'h10F: read_data_comb = BUILD_VERSION;

      // 0x110..0x12F: rmsnorm_selftest result (64 lanes, two per 32-bit word).
      // 0x130[0]: rmsnorm_done.  0x131[0]: write 1 to retrigger.
      10'h130: read_data_comb = {31'd0, rms_done};
      // 0x160[0]: rope_done    (result at 0x140-0x15F via default-range)
      // 0x190[0]: swiglu_done  (result at 0x170-0x18F via default-range)
      // 0x1C0[0]: softmax_done (result at 0x1A0-0x1BF via default-range)
      10'h160: read_data_comb = {31'd0, rope_done};
      10'h190: read_data_comb = {31'd0, sg_done};
      10'h1C0: read_data_comb = {31'd0, sm_done};
      // 0x1D0..0x1EF: smollm_layer hidden_out (64 lanes Q1.15, two per word)
      // 0x1F0[0]: layer_done    0x1F1[0]: layer_restart
      10'h1F0: read_data_comb = {30'd0, lay_done_latched, lay_done};
      10'h050: read_data_comb = bfp_ar_cnt_eth;
      10'h051: read_data_comb = bfp_r_cnt_eth;
      10'h052: read_data_comb = {2'd0, bfp_last_araddr_eth};

      // 0x00D: core_clk-domain FSM observation (single-cycle snapshot)
      10'h00D: read_data_comb = {
        16'd0,
        host_seed_reg[7:0],              // [15:8]
        2'd0,                            // [7:6]
        step_seen_reg,                   // [5]
        clear_seen_reg,                  // [4]
        start_seen_reg,                  // [3]
        host_run_reg,                    // [2]
        host_start_req,                  // [1]
        host_clear_req                   // [0]
      };
      default: begin
        if ((jtag_word_addr >= 10'h014) && (jtag_word_addr < 10'h024)) begin
          // dbg_first_rdata: 16 words × 32 bits = 512 bits
          out_i = jtag_word_addr - 10'h014;
          read_data_comb = dbg_first_rdata[out_i*32 +: 32];
        end else if ((jtag_word_addr >= 10'h024) && (jtag_word_addr < 10'h034)) begin
          // dbg_eng_w_packed
          out_i = jtag_word_addr - 10'h024;
          read_data_comb = dbg_eng_w_packed[out_i*32 +: 32];
        end else if ((jtag_word_addr >= 10'h034) && (jtag_word_addr < 10'h044)) begin
          // dbg_wd_packed
          out_i = jtag_word_addr - 10'h034;
          read_data_comb = dbg_wd_packed[out_i*32 +: 32];
        end else if ((jtag_word_addr >= 10'h068) && (jtag_word_addr < 10'h06A)) begin
          // dbg_in_value_packed: 64 bits → 2 words (0x040..0x05A is taken by
          // core_logits_flat, so this lives at 0x068..0x069)
          out_i = jtag_word_addr - 10'h068;
          read_data_comb = dbg_in_value_packed[out_i*32 +: 32];
        end else if ((jtag_word_addr >= 10'h040) && (jtag_word_addr < (10'h040 + 10'd27))) begin
          out_i = jtag_word_addr - 10'h040;
          read_data_comb = {{16{core_logits_flat[(out_i*16)+15]}}, core_logits_flat[(out_i*16) +: 16]};
        end else if ((jtag_word_addr >= 10'h018) && (jtag_word_addr < 10'h028)) begin
          read_data_comb = {24'd0, output_mem[jtag_word_addr - 10'h018]};
        end else if ((jtag_word_addr >= 10'h110) && (jtag_word_addr < 10'h130)) begin
          // rmsnorm_selftest result: 32 × 32-bit words = 64 Q1.15 lanes
          out_i = jtag_word_addr - 10'h110;
          read_data_comb = rms_result[out_i*32 +: 32];
        end else if ((jtag_word_addr >= 10'h140) && (jtag_word_addr < 10'h160)) begin
          out_i = jtag_word_addr - 10'h140;
          read_data_comb = rope_result[out_i*32 +: 32];
        end else if ((jtag_word_addr >= 10'h170) && (jtag_word_addr < 10'h190)) begin
          out_i = jtag_word_addr - 10'h170;
          read_data_comb = sg_result[out_i*32 +: 32];
        end else if ((jtag_word_addr >= 10'h1A0) && (jtag_word_addr < 10'h1C0)) begin
          out_i = jtag_word_addr - 10'h1A0;
          read_data_comb = sm_result[out_i*32 +: 32];
        end else if ((jtag_word_addr >= 10'h1D0) && (jtag_word_addr < 10'h1F0)) begin
          // Legacy: first 64 Q1.15 lanes of smollm_layer hidden_out (2 per word).
          out_i = jtag_word_addr - 10'h1D0;
          read_data_comb = lay_result[out_i*32 +: 32];
        end else if ((jtag_word_addr >= 10'h200) && (jtag_word_addr < 10'h320)) begin
          // FULL hidden_out: 288 words × 32-bit = 576 Q1.15 lanes (2 per word).
          out_i = jtag_word_addr - 10'h200;
          read_data_comb = lay_result[out_i*32 +: 32];
        end else begin
          read_data_comb = 32'd0;
        end
      end
    endcase
  end

  // ================================================================
  //  LEDs (8 on VC707): coarse status only — no HEX displays.
  //   [0] eth_clk alive (~7.5 Hz blink)
  //   [1] enable switch
  //   [2] ARP/UDP RX activity (stretched to ~50 ms per frame)
  //   [3] generation done
  //   [4] eth_irq (raw MAC RX strobe)
  //   [5] core_mmcm_locked
  //   [6] eth_rst_n_int
  //   [7] cycle_blink
  // ================================================================
  logic [23:0] blink_cnt;
  always_ff @(posedge eth_clk or negedge eth_rst_n_int)
    if (!eth_rst_n_int) blink_cnt <= '0;
    else                blink_cnt <= blink_cnt + 1;

  assign led[0] = blink_cnt[23];
  assign led[1] = enable;
  assign led[2] = rx_active_stretched;
  assign led[3] = done_latched_reg;
  assign led[4] = eth_irq;
  assign led[5] = core_mmcm_locked;
  assign led[6] = eth_rst_n_int;
  assign led[7] = cycle_blink_reg[15];

  // Fan always on
  assign fan_pwm = 1'b1;

`ifdef MICROGPT_ILA
  // ================================================================
  //  ILA-CORE — layer matvec datapath, clk_core (50 MHz), 8192 samples.
  //  Suggested HW manager triggers:
  //    state[4:0] == 5'd2  && mv_phase[2:0] == 3'd4  (S_M_Q && MV_DRIVE) — first matvec
  //    eng_in_valid && !eng_acc_clear                                    — every real MAC
  //    ws_load_req                                                        — chunk transitions
  // ================================================================
  microgpt_ila_core i_ila_core (
    .clk    ( core_clk            ),
    .probe0 ( ila_state           ),  // 5  layer FSM state
    .probe1 ( ila_mv_phase        ),  // 3  matvec sub-FSM
    .probe2 ( ila_cnt             ),  // 11 MV_DRIVE drive index
    .probe3 ( ila_chunk           ),  // 7  chunk index in matvec
    .probe4 ( ila_ws_matvec_id    ),  // 3  matrix id (0=Q..6=DOWN)
    .probe5 ( ila_ws_load_req     ),  // 1  to streamer
    .probe6 ( ila_ws_ready        ),  // 1  from streamer (CDC'd)
    .probe7 ( ila_ws_rd_addr      ),  // 11 bank read address
    .probe8 ( ila_ws_weight_data  ),  // 128 weights from streamer
    .probe9 ( ila_eng_w           ),  // 128 weights into engine
    .probe10( ila_eng_in_value    ),  // 16 src element into engine
    .probe11( ila_eng_in_valid    ),  // 1  engine MAC enable
    .probe12( ila_eng_in_last     ),  // 1  last MAC of chunk
    .probe13( ila_eng_acc_clear   ),  // 1  accumulator reset
    .probe14( ila_eng_out_valid   )   // 1  result valid pulse
  );

  // ================================================================
  //  ILA-AXI — streamer ↔ MIG read channel, ui_clk (200 MHz), 8192 samples.
  //  Suggested HW manager triggers:
  //    m_axi_arvalid && m_axi_arready             — capture each AR
  //    m_axi_rvalid && m_axi_rlast                — capture last beat per burst
  //    start_load_axi                             — capture each load dispatch
  //    m_axi_araddr == 30'h0000_0000              — first chunk of Q (BASE_Q)
  // ================================================================
  microgpt_ila_axi i_ila_axi (
    .clk    ( ui_clk               ),
    .probe0 ( m_axi_arvalid        ),  // 1
    .probe1 ( m_axi_arready        ),  // 1
    .probe2 ( m_axi_araddr         ),  // 30
    .probe3 ( m_axi_rvalid         ),  // 1
    .probe4 ( m_axi_rlast          ),  // 1
    .probe5 ( m_axi_rdata[127:0]   ),  // 128 (low slice = entry 0 of 4 in beat)
    .probe6 ( ila_ws_state_axi     ),  // 3  streamer FSM state
    .probe7 ( ila_start_load_axi   ),  // 1
    .probe8 ( ila_tile_idx         ),  // 2
    .probe9 ( ila_beat_idx         )   // 7
  );
`endif

endmodule

`default_nettype wire
