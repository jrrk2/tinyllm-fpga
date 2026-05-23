// picosoc_eth_bridge.sv — drop-in replacement for microgpt_eth_ctrl that puts a
// PicoRV32 SoC in front of the engine.  Same port list as microgpt_eth_ctrl, so
// vc707_microgpt_eth instantiates it under `ifdef PICOSOC_ENGINE.
//
// LEAN base-fit scope: the SoC drives the Avalon-MM master into the engine
// regmap (iomem 0x10) and can read engine status (iomem 0x40).  The LSU frame
// bus and ddr_wr upload path are TIED OFF for now (ethernet + uploads come back
// once base fit/timing is confirmed).  No UART pin in this top, so ser_tx
// dangles (firmware UART still consumes logic, just unrouted).
//
// Regmap is an Avalon slave with waitrequest=0 and word addr = master_address
// [11:2]; reads pulse readdatavalid.  iomem 0x10 maps reg N at 0x1000_0000+N*4.

`default_nettype none

module picosoc_eth_bridge (
  input  wire           clk,
  input  wire           rst_n,
  // framing LSU bus (TIED OFF in this lean build)
  output logic [16:0]   core_lsu_addr,
  output logic [63:0]   core_lsu_wdata,
  output logic [7:0]    core_lsu_be,
  output logic          ce_d,
  output logic          we_d,
  output logic          framing_sel,
  input  wire  [63:0]   framing_rdata,
  // Avalon-MM master -> engine regmap (DRIVEN by the SoC)
  output logic [31:0]   master_address,
  output logic          master_read,
  output logic          master_write,
  output logic [31:0]   master_writedata,
  output logic [3:0]    master_byteenable,
  input  wire  [31:0]   master_readdata,
  input  wire           master_readdatavalid,
  input  wire           master_waitrequest,
  // engine status snapshot (readable by firmware at iomem 0x40)
  input  wire  [7:0]    hb_state,
  input  wire  [7:0]    hb_last_token,
  input  wire  [7:0]    hb_out_len,
  input  wire  [7:0]    hb_done_flags,
  output logic          rx_activity,
  // DDR3 upload path (TIED OFF in this lean build)
  output logic          ddr_wr_req,
  input  wire           ddr_wr_ack,
  output logic [29:0]   ddr_wr_addr,
  output logic [511:0]  ddr_wr_data,
  // debug (tied off)
  output logic [5:0]    dbg_state,
  output logic [7:0]    dbg_frame_type,
  output logic [7:0]    dbg_wcnt,
  output logic [4:0]    dbg_cur_buf,
  output logic [7:0]    dbg_n_remaining,
  output wire  [31:0]   dbg_fpga_ip,
  output logic [31:0]   ddr_wr_rx_count,
  output logic [31:0]   ddr_wr_done_count,
  output logic [31:0]   ddr_wr_ack_count,
  output logic [31:0]   ddr_wr_tx_count,
  // UART console — the engine top routes these to the usb_uart pins.  This is
  // the ONLY SoC-visible channel in this top (LEDs belong to the engine), so it
  // carries the CPU boot banner + heartbeat.
  output wire           ser_tx,
  input  wire           ser_rx
);

  // ---- tie-offs ----
  // core_lsu_*, ce_d, we_d, framing_sel are now driven by the SoC eth window below.
  assign rx_activity = 1'b0;
  // ddr_wr_req/addr/data are now driven by the SoC DDR3 write window (below).
  assign dbg_state = '0;  assign dbg_frame_type = '0;  assign dbg_wcnt = '0;
  assign dbg_cur_buf = '0;  assign dbg_n_remaining = '0;  assign dbg_fpga_ip = '0;
  assign ddr_wr_rx_count = '0;  assign ddr_wr_tx_count = '0;
  // ddr_wr_done_count (beats issued) / ddr_wr_ack_count (beats completed) driven below.

  // ---- PicoSoC ----
  wire        iomem_valid;
  reg         iomem_ready;
  wire [3:0]  iomem_wstrb;
  wire [31:0] iomem_addr;
  wire [31:0] iomem_wdata;
  reg  [31:0] iomem_rdata;
  (* mark_debug = "true" *) wire        trace_valid;
  (* mark_debug = "true" *) wire [35:0] trace_data;

  picosoc_noflash soc (
    .clk(clk), .resetn(rst_n),
    .iomem_valid(iomem_valid), .iomem_ready(iomem_ready),
    .iomem_wstrb(iomem_wstrb), .iomem_addr(iomem_addr),
    .iomem_wdata(iomem_wdata), .iomem_rdata(iomem_rdata),
    .irq_5(1'b0), .irq_6(1'b0), .irq_7(1'b0),
    .ser_tx(ser_tx), .ser_rx(ser_rx),
    .trace_valid(trace_valid), .trace_data(trace_data)
  );

  // ---- iomem decode: 0x10 -> Avalon regmap, 0x40 -> engine status ----
  wire sel_reg = (iomem_addr[31:24] == 8'h10);   // engine regmap (Avalon master)
  wire sel_hb  = (iomem_addr[31:24] == 8'h40);   // engine status snapshot
  // picosoc_noflash routes EVERYTHING with addr[31:24] > 0x01 to the iomem bus,
  // INCLUDING its own spimemio/UART at 0x0200_xxxx (addr[31:24]==0x02).  Those are
  // handled internally and gated by the UART's reg_dat_wait back-pressure.  The
  // catch-all below must NOT ack 0x02, or it satisfies mem_ready early and the CPU
  // never stalls on the busy UART -> every byte after the first is dropped (the
  // "echo + '>' only" symptom).  The shell never had this because its decode only
  // acks 0x03/0x10/0x20/0x30/0x31, never 0x02.
  wire sel_internal = (iomem_addr[31:24] == 8'h02);   // spimemio/UART — leave to internal back-pressure
  wire sel_prog     = (iomem_addr[31:24] == 8'h50);   // progmem self-write — owned by picosoc_noflash (port B)

  // ---- SoC -> DDR3 write window ----------------------------------------
  // The engine's weight store lives in DDR3.  This lean build had no upload
  // path (eth tied off), so the SoC now feeds DDR3 directly: a 32-bit data
  // port (0x30) fills a 512-bit beat one word at a time; a control reg (0x31)
  // sets the 64-byte-aligned byte address.  The 16th word toggles ddr_wr_req
  // (the BL=1 64-byte AXI write handled by ddr_write_master in the top) and
  // STALLS the CPU (iomem_ready held low) until the 2FF-synced ddr_wr_ack
  // toggle matches, then auto-advances the address by 64 for the next beat.
  // Back-pressure means firmware just streams 16 words/beat with no polling.
  wire sel_ddrd = (iomem_addr[31:24] == 8'h30);   // DDR write data port (push 32b)
  wire sel_ddra = (iomem_addr[31:24] == 8'h31);   // DDR write addr/control
  reg  [3:0] dw_idx;                               // word fill index 0..15
  reg        dw_wait;                              // beat issued, awaiting ack
  reg        ddr_ack_s0, ddr_ack_s1;              // 2FF sync of ui-clk ddr_wr_ack

  // ---- SoC <-> ethernet MAC (framing_top_sgmii) over the LSU frame bus -------
  // Stage 2: the SoC parses eth frames in FIRMWARE (bfp_client protocol) instead
  // of the old microgpt_eth_ctrl HW FSM.  iomem 0x20 + lsu_byte_addr maps each
  // 32-bit access to a 64-bit LSU access (which 32-bit half via addr[2]) — the
  // same window the bring-up shell proved (vc707_picosoc_shell.sv).  A read takes
  // 2 cycles (the frame BRAM has 1-cycle latency); a write is 1 cycle.
  wire sel_eth = (iomem_addr[31:24] == 8'h20);
  typedef enum logic [1:0] { E_IDLE, E_WAIT, E_DONE, E_WR } eph_t;
  eph_t eph;
  reg   eth_half;

  typedef enum logic [1:0] { B_IDLE, B_WR, B_RD, B_RD_WAIT } bst_t;
  bst_t bst;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      iomem_ready <= 1'b0; iomem_rdata <= 32'h0;
      master_address <= 32'h0; master_read <= 1'b0; master_write <= 1'b0;
      master_writedata <= 32'h0; master_byteenable <= 4'h0;
      bst <= B_IDLE;
      ddr_wr_req <= 1'b0; ddr_wr_addr <= 30'h0; ddr_wr_data <= 512'h0;
      dw_idx <= 4'h0; dw_wait <= 1'b0; ddr_ack_s0 <= 1'b0; ddr_ack_s1 <= 1'b0;
      ddr_wr_done_count <= 32'h0; ddr_wr_ack_count <= 32'h0;
      eph <= E_IDLE; eth_half <= 1'b0;
      core_lsu_addr <= 17'h0; core_lsu_wdata <= 64'h0; core_lsu_be <= 8'hFF;
      ce_d <= 1'b0; we_d <= 1'b0; framing_sel <= 1'b1;
    end else begin
      iomem_ready <= 1'b0;
      master_read <= 1'b0; master_write <= 1'b0;
      framing_sel <= 1'b1;            // LSU select held high (matches the shell)
      // 2FF sync of the ui-clk write-ack toggle into eth_clk.
      ddr_ack_s0 <= ddr_wr_ack;
      ddr_ack_s1 <= ddr_ack_s0;

      // single-cycle: engine status snapshot
      if (iomem_valid && !iomem_ready && sel_hb) begin
        iomem_ready <= 1'b1;
        iomem_rdata <= {hb_done_flags, hb_out_len, hb_last_token, hb_state};
      end

      // catch-all: any iomem address NOT mapped here (LED 0x03 — not wired) is
      // acked immediately with 0 so a stray firmware access can never stall the
      // PicoRV32 forever.  EXCLUDE sel_internal (0x02 spimemio/UART — governed by
      // the UART's own back-pressure), the DDR write window (0x30/0x31), and the
      // eth window (0x20) — all handled by their own FSMs below.
      if (iomem_valid && !iomem_ready && !sel_reg && !sel_hb && !sel_internal
          && !sel_ddrd && !sel_ddra && !sel_eth && !sel_prog) begin
        iomem_ready <= 1'b1;
        iomem_rdata <= 32'h0;
      end

      // ---- SoC -> DDR3 write window ----
      if (dw_wait) begin
        // 512-bit beat issued; wait for the AXI write to complete (ack toggle
        // mirrors req), then release the stalled 16th-word write + advance addr.
        if (ddr_ack_s1 == ddr_wr_req) begin
          dw_wait          <= 1'b0;
          dw_idx           <= 4'h0;
          ddr_wr_addr      <= ddr_wr_addr + 30'd64;   // next 64-byte beat
          ddr_wr_ack_count <= ddr_wr_ack_count + 32'd1;
          iomem_ready      <= 1'b1;                    // ack the pending 0x30 write
        end
      end else begin
        // 0x31: set 64-byte-aligned byte address (write) / read back {busy,addr}.
        if (iomem_valid && !iomem_ready && sel_ddra) begin
          if (|iomem_wstrb) begin
            ddr_wr_addr <= iomem_wdata[29:0];
            dw_idx      <= 4'h0;
          end
          iomem_rdata <= {dw_wait, 1'b0, ddr_wr_addr};
          iomem_ready <= 1'b1;
        end
        // 0x30: push next 32-bit word into the 512-bit beat (LSW first); a read
        // returns the current fill index (so a stray read can't hang the CPU).
        if (iomem_valid && !iomem_ready && sel_ddrd) begin
          if (|iomem_wstrb) begin
            ddr_wr_data[dw_idx*32 +: 32] <= iomem_wdata;
            if (dw_idx == 4'd15) begin
              ddr_wr_req        <= ~ddr_wr_req;          // issue the beat
              dw_wait           <= 1'b1;                 // stall until ack
              ddr_wr_done_count <= ddr_wr_done_count + 32'd1;
              // iomem_ready stays low -> CPU blocks on this write until ack
            end else begin
              dw_idx      <= dw_idx + 4'd1;
              iomem_ready <= 1'b1;
            end
          end else begin
            iomem_rdata <= {28'h0, dw_idx};
            iomem_ready <= 1'b1;
          end
        end
      end

      // ---- SoC <-> eth MAC: 32-bit iomem (0x20) <-> 64-bit LSU access ----
      case (eph)
        E_IDLE:
          if (iomem_valid && !iomem_ready && sel_eth) begin
            core_lsu_addr <= {iomem_addr[16:3], 3'b000};
            ce_d          <= 1'b1;
            eth_half      <= iomem_addr[2];
            if (|iomem_wstrb) begin
              we_d           <= 1'b1;
              core_lsu_wdata <= {iomem_wdata, iomem_wdata};
              core_lsu_be    <= iomem_addr[2] ? 8'hF0 : 8'h0F;
              eph            <= E_WR;
            end else begin
              we_d <= 1'b0;
              eph  <= E_WAIT;
            end
          end
        E_WAIT: eph <= E_DONE;                  // ce held; framing_rdata next cycle
        E_DONE: begin
          ce_d        <= 1'b0;
          iomem_rdata <= eth_half ? framing_rdata[63:32] : framing_rdata[31:0];
          iomem_ready <= 1'b1;
          eph         <= E_IDLE;
        end
        E_WR: begin
          ce_d <= 1'b0; we_d <= 1'b0;
          iomem_ready <= 1'b1;
          eph         <= E_IDLE;
        end
        default: eph <= E_IDLE;
      endcase

      // Avalon-MM bridge to the engine regmap (waitrequest=0; reads pulse
      // readdatavalid).  reg N at iomem 0x1000_0000 + N*4 -> master_address.
      case (bst)
        B_IDLE:
          if (iomem_valid && !iomem_ready && sel_reg) begin
            master_address <= {12'h0, iomem_addr[19:0]};
            if (|iomem_wstrb) begin
              master_write      <= 1'b1;
              master_writedata  <= iomem_wdata;
              master_byteenable <= iomem_wstrb;
              bst <= B_WR;
            end else begin
              master_read <= 1'b1;
              bst <= B_RD;
            end
          end
        B_WR: begin            // waitrequest=0 -> write accepted last cycle
          iomem_ready <= 1'b1;
          bst <= B_IDLE;
        end
        B_RD: bst <= B_RD_WAIT;             // read issued; wait for data
        B_RD_WAIT:
          if (master_readdatavalid) begin
            iomem_rdata <= master_readdata;
            iomem_ready <= 1'b1;
            bst <= B_IDLE;
          end
        default: bst <= B_IDLE;
      endcase
    end
  end

  // ----------------------------------------------------------------------
  // On-chip ILA on the PicoRV32 trace bus + iomem activity (eth_clk domain).
  // IP picosoc_ila is read by run.tcl; gated by PICOSOC_ILA so the bridge
  // still elaborates without it.  This is how we see whether the CPU is
  // fetching/executing and whether it ever reaches the UART when nothing
  // appears on the wire.
  // ----------------------------------------------------------------------
`ifdef PICOSOC_ILA
  picosoc_ila i_ila (
    .clk    (clk),
    .probe0 (trace_valid),   // 1  — CPU retired-instruction trace strobe
    .probe1 (trace_data),    // 36 — trace payload (PC / writeback)
    .probe2 (iomem_valid),   // 1
    .probe3 (iomem_ready),   // 1
    .probe4 (iomem_wstrb),   // 4  — 0 = read, else write byte-enables
    .probe5 (iomem_addr),    // 32 — 0x02 uart(internal) / 0x10 regmap / 0x40 hb
    .probe6 (iomem_wdata),   // 32
    .probe7 (iomem_rdata)    // 32
  );
`endif

endmodule

`default_nettype wire
