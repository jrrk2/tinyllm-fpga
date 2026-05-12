// tb_xsim_top.sv — pure-SV testbench for xsim with unisim primitives.
// Drives smollm_multilayer_tm + mock_axi_slave, captures hidden_out
// (576 lanes × 16-bit Q1.15) into xsim_out.hex.  Compare to Verilator
// dump and FPGA readout to localise any RTL-vs-stub divergence.

`include "tm_layer_ddr3_bases.svh"
`include "tm_layer_data.svh"

`default_nettype none

module tb_xsim_top;

  // --- SmolLM2 dims ---
  localparam int D       = 576;
  localparam int H_Q     = 9;
  localparam int H_KV    = 3;
  localparam int HD      = 64;
  localparam int FFN     = 1536;
  localparam int MAX_CTX = 4;
  localparam int NL      = 1;   // bisect: 1 layer is enough to catch RTL divergence

  // --- clocks ---
  logic clk = 0;
  always #5 clk = ~clk;       // 100 MHz model clock (timebase only)

  logic rst = 1;

  // --- DUT inputs ---
  logic         start  = 1'b0;
  logic [10:0]  pos    = 11'd3;
  logic [4:0]   kv_pos = 5'd3;
  logic [4:0]   snap_sel = 5'd31;   // start in bypass (live hidden_state)

  // hidden_in baked in from the case-statement ROM (same .svh file the
  // selftest wrapper uses).  Real SmolLM2 embed of token id 655 (' time')
  // at layer-0 h_in_p2 scale.
`include "layer_hidden_in_packed.svh"
  logic signed [D*16-1:0] hidden_in;
  genvar gh;
  generate
    for (gh = 0; gh < D; gh++)
      assign hidden_in[gh*16 +: 16] = layer_hidden_in_lut(gh);
  endgenerate

  // --- DUT outputs ---
  wire signed [D*16-1:0] hidden_out;
  wire                   done;

  // AXI between layer streamer and mock memory
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
    .snapshot_layer_sel(snap_sel),
    .factor_wr_layer         (5'd0),
    .factor_wr_data          (32'd0),
    .factor_wr_en_swiglu_lo  (1'b0),
    .factor_wr_en_swiglu_mlp (1'b0),
    .factor_wr_en_attn       (1'b0),
    .factor_rd_sel           (7'b0),
    .factor_rd_data          (/* unused */),
    .factor_ram_por_init     (rst),

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
    .INIT_HEX_FILE("../generated/tm_layer_DDR3.hex")
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

  // --- VCD dump for X tracing ---
  // depth=1 only — just the multilayer wrapper signals (cur_resid2_factor,
  // cur_sh_h1_to_h_out, layer_idx, lr_state, hidden_state).  Full hierarchy
  // explodes to GBs.  Specific deeper signals dumped explicitly.
  initial begin
    $dumpfile("/tmp/xsim.vcd");
    $dumpvars(1, i_ml);
    $dumpvars(1, tb_xsim_top);
    // Layer-internal probes for X tracing
    $dumpvars(0, i_ml.i_layer.state);
    $dumpvars(0, i_ml.i_layer.cnt);
    $dumpvars(0, i_ml.i_layer.layer_idx);
    $dumpvars(0, i_ml.i_layer.resid2_factor);
    $dumpvars(0, i_ml.i_layer.sh_h1_to_h_out);
  end

  // --- run ---
  initial begin
    integer fd, i;
    automatic longint cycle_count = 0;
    $display("xsim: start (NL=%0d D=%0d)", NL, D);
    repeat (10) @(posedge clk);
    rst   <= 1'b0;
    start <= 1'b1;

    // Per-layer ~250K cycles; NL=30 ≈ 8M cycles.  Cap at 30M for safety.
    while (!done && cycle_count < 30_000_000) begin
      @(posedge clk);
      cycle_count = cycle_count + 1;
    end

    if (!done) begin
      $display("xsim: FAIL — done never asserted (cycle=%0d)", cycle_count);
      $finish(1);
    end
    $display("xsim: done at cycle %0d (NL=%0d layers)", cycle_count, NL);

    // Dump 576 lanes × signed-int decimal — same format as
    // Verilator's tm_layer_dut_out.txt for direct diff.
    fd = $fopen("/tmp/xsim_out.txt", "w");
    if (fd == 0) begin
      $display("xsim: FAIL — cannot open /tmp/xsim_out.txt");
      $finish(1);
    end
    $fdisplay(fd, "# xsim hidden_out (bypass, snap_sel=%0d), %0d lanes, Q1.15", snap_sel, D);
    for (i = 0; i < D; i++) begin
      automatic logic signed [15:0] v = hidden_out[i*16 +: 16];
      $fdisplay(fd, "%0d", v);
    end
    $fclose(fd);

    // Now exercise the snapshot path: switch snap_sel to 0 (layer 0's
    // captured snapshot), wait for the refresh FSM to fully repopulate
    // refreshed_snap_bus (~D/2 = 288 cycles), then dump.  With NL=1,
    // layer 0's snapshot should equal the bypass dump above (bit-exact).
    snap_sel <= 5'd0;
    repeat (1024) @(posedge clk);    // > 2 × refresh period for safety
    fd = $fopen("/tmp/xsim_out_snap0.txt", "w");
    $fdisplay(fd, "# xsim hidden_out (snap_sel=0 = layer 0 snapshot), %0d lanes, Q1.15", D);
    for (i = 0; i < D; i++) begin
      automatic logic signed [15:0] v = hidden_out[i*16 +: 16];
      $fdisplay(fd, "%0d", v);
    end
    $fclose(fd);
    $display("xsim: wrote /tmp/xsim_out_snap0.txt (snap_sel=0)");
    // Snap-path diagnostic probes
    $display("snap probes:");
    $display("  snap_per_layer_dout[0]   = %h", i_ml.snap_per_layer_dout[0]);
    $display("  snap_rd_data             = %h", i_ml.snap_rd_data);
    $display("  refresh_pair             = %0d", i_ml.refresh_pair);
    $display("  refresh_pair_q1          = %0d", i_ml.refresh_pair_q1);
    $display("  refreshed_snap_bus[31:0] = %h", i_ml.refreshed_snap_bus[31:0]);
    $display("  refreshed_snap_bus[63:32]= %h", i_ml.refreshed_snap_bus[63:32]);
    $display("  cap_pair                 = %0d", i_ml.cap_pair);
    $display("  captured_layer_idx       = %0d", i_ml.captured_layer_idx);
    $display("  snap_wr_data             = %h", i_ml.snap_wr_data);
    $display("  snap_wr_layer            = %0d  snap_wr_pair = %0d",
              i_ml.snap_wr_layer, i_ml.snap_wr_pair);

    // Diagnostic: probe internal buffers for X propagation
    $display("--- internal probes ---");
    $display("layer.state = %0d  layer_idx=%0d  done=%b",
              i_ml.i_layer.state, i_ml.layer_idx, done);
    $display("hidden1_buf[0..3] = %h %h %h %h",
              i_ml.i_layer.hidden1_buf[0], i_ml.i_layer.hidden1_buf[1],
              i_ml.i_layer.hidden1_buf[2], i_ml.i_layer.hidden1_buf[3]);
    $display("down_buf[0..3]    = %h %h %h %h",
              i_ml.i_layer.down_buf[0], i_ml.i_layer.down_buf[1],
              i_ml.i_layer.down_buf[2], i_ml.i_layer.down_buf[3]);
    $display("norm1_buf[0..3]   = %h %h %h %h",
              i_ml.i_layer.norm1_buf[0], i_ml.i_layer.norm1_buf[1],
              i_ml.i_layer.norm1_buf[2], i_ml.i_layer.norm1_buf[3]);
    $display("q_buf[0..3]       = %h %h %h %h",
              i_ml.i_layer.q_buf[0], i_ml.i_layer.q_buf[1],
              i_ml.i_layer.q_buf[2], i_ml.i_layer.q_buf[3]);
    $display("attn_buf[0..3]    = %h %h %h %h",
              i_ml.i_layer.attn_buf[0], i_ml.i_layer.attn_buf[1],
              i_ml.i_layer.attn_buf[2], i_ml.i_layer.attn_buf[3]);
    $display("resid2_factor=%h sh_h1_to_h_out=%h hidden_state[0]=%h",
              i_ml.cur_resid2_factor, i_ml.cur_sh_h1_to_h_out,
              i_ml.hidden_state[15:0]);
    $display("TM_RESCALE[0] = %h  TM_RESCALE[1] = %h",
              i_ml.TM_RESCALE[0], i_ml.TM_RESCALE[1]);
    $display("layer_idx = %b  lr_state=%0d  layer_done=%b",
              i_ml.layer_idx, i_ml.lr_state, i_ml.layer_done);
    $display("layer_hidden_out[0..2] = %h %h %h",
              i_ml.layer_hidden_out[15:0],
              i_ml.layer_hidden_out[31:16],
              i_ml.layer_hidden_out[47:32]);
    // Probe internal r2_prod / h1_aligned for lane 0 to localise where X enters
    $display("g_hidden_out[0]: r2_prod=%h r2_delta=%h h1_aligned=%h r2_sum=%h",
              i_ml.i_layer.g_hidden_out[0].r2_prod,
              i_ml.i_layer.g_hidden_out[0].r2_delta,
              i_ml.i_layer.g_hidden_out[0].h1_aligned,
              i_ml.i_layer.g_hidden_out[0].r2_sum);
    $display("layer.resid2_factor=%h layer.sh_h1_to_h_out=%h",
              i_ml.i_layer.resid2_factor,
              i_ml.i_layer.sh_h1_to_h_out);
    $display("xsim: wrote /tmp/xsim_out.txt (%0d lanes)", D);
    $finish(0);
  end

endmodule

`default_nettype wire
