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
  output logic [31:0]   ddr_wr_tx_count
);

  // ---- tie-offs (lean build: no eth/upload yet) ----
  assign core_lsu_addr = '0;  assign core_lsu_wdata = '0;  assign core_lsu_be = '0;
  assign ce_d = 1'b0;  assign we_d = 1'b0;  assign framing_sel = 1'b0;
  assign rx_activity = 1'b0;
  assign ddr_wr_req = 1'b0;  assign ddr_wr_addr = '0;  assign ddr_wr_data = '0;
  assign dbg_state = '0;  assign dbg_frame_type = '0;  assign dbg_wcnt = '0;
  assign dbg_cur_buf = '0;  assign dbg_n_remaining = '0;  assign dbg_fpga_ip = '0;
  assign ddr_wr_rx_count = '0;  assign ddr_wr_done_count = '0;
  assign ddr_wr_ack_count = '0;  assign ddr_wr_tx_count = '0;

  // ---- PicoSoC ----
  wire        iomem_valid;
  reg         iomem_ready;
  wire [3:0]  iomem_wstrb;
  wire [31:0] iomem_addr;
  wire [31:0] iomem_wdata;
  reg  [31:0] iomem_rdata;
  wire        ser_tx_unused;
  wire        trace_valid_unused;
  wire [35:0] trace_data_unused;

  picosoc_noflash soc (
    .clk(clk), .resetn(rst_n),
    .iomem_valid(iomem_valid), .iomem_ready(iomem_ready),
    .iomem_wstrb(iomem_wstrb), .iomem_addr(iomem_addr),
    .iomem_wdata(iomem_wdata), .iomem_rdata(iomem_rdata),
    .irq_5(1'b0), .irq_6(1'b0), .irq_7(1'b0),
    .ser_tx(ser_tx_unused), .ser_rx(1'b1),
    .trace_valid(trace_valid_unused), .trace_data(trace_data_unused)
  );

  // ---- iomem decode: 0x10 -> Avalon regmap, 0x40 -> engine status ----
  wire sel_reg = (iomem_addr[31:24] == 8'h10);   // engine regmap (Avalon master)
  wire sel_hb  = (iomem_addr[31:24] == 8'h40);   // engine status snapshot

  typedef enum logic [1:0] { B_IDLE, B_WR, B_RD, B_RD_WAIT } bst_t;
  bst_t bst;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      iomem_ready <= 1'b0; iomem_rdata <= 32'h0;
      master_address <= 32'h0; master_read <= 1'b0; master_write <= 1'b0;
      master_writedata <= 32'h0; master_byteenable <= 4'h0;
      bst <= B_IDLE;
    end else begin
      iomem_ready <= 1'b0;
      master_read <= 1'b0; master_write <= 1'b0;

      // single-cycle: engine status snapshot
      if (iomem_valid && !iomem_ready && sel_hb) begin
        iomem_ready <= 1'b1;
        iomem_rdata <= {hb_done_flags, hb_out_len, hb_last_token, hb_state};
      end

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

endmodule

`default_nettype wire
