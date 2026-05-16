// See LICENSE for license details.
`default_nettype none

module framing_top_sgmii
  (
  input wire          msoc_clk,
  input wire [16:0]   core_lsu_addr,
  input wire [63:0]   core_lsu_wdata,
  input wire [7:0]    core_lsu_be,
  input wire          ce_d,
  input wire          we_d,
  input wire          framing_sel,
  output logic [63:0] framing_rdata,

    // Internal clock
  input wire          clk_int,
  input wire          rst_int,

    /*
     * Ethernet: 1000BASE-X SGMII
     */
  input wire          sgmii_rxp,
  input wire          sgmii_rxn,
  output wire         sgmii_txp,
  output wire         sgmii_txn,
  input wire          sgmii_refclk_p,
  input wire          sgmii_refclk_n,
  output wire         phy_reset_n,

  input wire          phy_mdio_i,
  output reg          phy_mdio_o,
  output reg          phy_mdio_oe,
  output wire         phy_mdc,

  output reg          eth_irq,

  // 125 MHz MAC clock output (from internal sgmii_soc / PCS/PMA)
  output wire         eth_clk_o,

  // Ring-buffer diagnostic taps (msoc_clk domain).  Depth =
  // (nextbuf - firstbuf) & 31; max = lastbuf.  Used by host to
  // monitor the upstream eth FIFO between the MAC and the parser.
  output wire [4:0]   dbg_firstbuf,
  output wire [4:0]   dbg_nextbuf,
  output wire [4:0]   dbg_lastbuf
   );

logic       phy_mdclk;
assign phy_mdc = phy_mdclk;

// 125 MHz MAC clock from PCS/PMA IP (via sgmii_soc)
wire        eth_clk;
assign      eth_clk_o = eth_clk;

logic [16:0] core_lsu_addr_dly;

logic tx_enable_i, mac_gmii_tx_en;
logic [47:0] mac_address, rx_dest_mac;
logic [10:0] tx_frame_addr, tx_packet_length;
logic [11:0] rx_length_axis[0:31];  // [11]=FCS error, [10:0]=byte count
logic [12:0] axis_tx_frame_size;
logic        ce_d_dly, avail;
logic [63:0] framing_rdata_pkt, framing_wdata_pkt;
logic [3:0] tx_enable_dly;
logic [4:0] firstbuf, nextbuf, lastbuf;
assign dbg_firstbuf = firstbuf;
assign dbg_nextbuf  = nextbuf;
assign dbg_lastbuf  = lastbuf;
logic [2:0] last;
   
reg        byte_sync, sync, irq_en, tx_busy, tx_buf_sel;

   wire [7:0] m_enb = (we_d ? core_lsu_be : 8'hFF);
   logic cooked, tx_enable_old, loopback, promiscuous;
   logic [3:0] spare;
   logic [10:0] rx_addr_axis;

   // ================================================================
   //  CDC synchronizers (rx_clk 125 MHz <-> msoc_clk 50 MHz)
   // ================================================================

   // --- Buffer pointers: msoc_clk → rx_clk (2FF sync) ---
   // These are quasi-static during packet reception (only change
   // between frames), so 2FF is sufficient.
   (* ASYNC_REG = "TRUE" *) reg [4:0] nextbuf_sync1, nextbuf_sync2;
   (* ASYNC_REG = "TRUE" *) reg [4:0] firstbuf_sync1, firstbuf_sync2;
   (* ASYNC_REG = "TRUE" *) reg [4:0] lastbuf_sync1, lastbuf_sync2;

   always @(posedge rx_clk) begin
      nextbuf_sync1  <= nextbuf;   nextbuf_sync2  <= nextbuf_sync1;
      firstbuf_sync1 <= firstbuf;  firstbuf_sync2 <= firstbuf_sync1;
      lastbuf_sync1  <= lastbuf;   lastbuf_sync2  <= lastbuf_sync1;
   end

   // --- Frame-done: rx_clk → msoc_clk (toggle + 2FF edge detect) ---
   // Toggled when last transitions 0→non-zero (end of frame).
   reg frame_done_toggle;
   (* ASYNC_REG = "TRUE" *) reg frame_done_sync1, frame_done_sync2;
   reg frame_done_sync3;
   wire frame_done_pulse = frame_done_sync2 ^ frame_done_sync3;

   always @(posedge rx_clk or posedge rst_int)
      if (rst_int) frame_done_toggle <= 1'b0;
      else if (rx_axis_tlast && byte_sync) frame_done_toggle <= ~frame_done_toggle;

   always @(posedge msoc_clk or posedge rst_int)
      if (rst_int) begin
         frame_done_sync1 <= 1'b0;
         frame_done_sync2 <= 1'b0;
         frame_done_sync3 <= 1'b0;
      end else begin
         frame_done_sync1 <= frame_done_toggle;
         frame_done_sync2 <= frame_done_sync1;
         frame_done_sync3 <= frame_done_sync2;
      end

   // --- rx_dest_mac snapshot in msoc_clk domain ---
   // Captured on frame_done_pulse; stable by then (written during first 6 bytes).
   reg [47:0] rx_dest_mac_s;
   reg [10:0] rx_length_s;    // length snapshot for current frame
   reg        frame_pending;  // a frame is waiting for MAC check

   always @(posedge msoc_clk or posedge rst_int)
      if (rst_int) begin
         rx_dest_mac_s <= '0;
         rx_length_s   <= '0;
         frame_pending <= 1'b0;
      end else begin
         if (frame_done_pulse) begin
            rx_dest_mac_s <= rx_dest_mac;
            frame_pending <= 1'b1;
         end
         if (frame_pending && !frame_done_pulse) begin
            // Process in next cycle after capture
            frame_pending <= 1'b0;
         end
      end

   // --- TX enable: msoc_clk → eth_clk (single-bit 2FF) ---
   wire tx_enable_rdy = &tx_enable_dly;  // all-ones detect in msoc_clk
   (* ASYNC_REG = "TRUE" *) reg tx_enable_rdy_sync1, tx_enable_rdy_sync2;

   always @(posedge eth_clk or posedge rst_int)
      if (rst_int) begin
         tx_enable_rdy_sync1 <= 1'b0;
         tx_enable_rdy_sync2 <= 1'b0;
      end else begin
         tx_enable_rdy_sync1 <= tx_enable_rdy;
         tx_enable_rdy_sync2 <= tx_enable_rdy_sync1;
      end

   // --- TX status: eth_clk → msoc_clk (single-bit 2FF) ---
   (* ASYNC_REG = "TRUE" *) reg tx_enable_i_sync1, tx_enable_i_sync2;
   (* ASYNC_REG = "TRUE" *) reg mac_gmii_tx_en_sync1, mac_gmii_tx_en_sync2;

   always @(posedge msoc_clk or posedge rst_int)
      if (rst_int) begin
         tx_enable_i_sync1    <= 1'b0; tx_enable_i_sync2    <= 1'b0;
         mac_gmii_tx_en_sync1 <= 1'b0; mac_gmii_tx_en_sync2 <= 1'b0;
      end else begin
         tx_enable_i_sync1    <= tx_enable_i;     tx_enable_i_sync2    <= tx_enable_i_sync1;
         mac_gmii_tx_en_sync1 <= mac_gmii_tx_en;  mac_gmii_tx_en_sync2 <= mac_gmii_tx_en_sync1;
      end
   
       /*
        * AXI input
        */
        reg         tx_axis_tvalid;
        reg         tx_axis_tvalid_dly;
        reg 	    tx_axis_tlast;
        wire [7:0]  tx_axis_tdata;
        wire        tx_axis_tready;
        wire        tx_axis_tuser = 1'b0;
   
       /*
        * AXI output
        */
       wire       rx_clk;
       wire [7:0] rx_axis_tdata;
       wire       rx_axis_tvalid;
       wire       rx_axis_tlast;
       wire       rx_axis_tuser;
   
      /*
        * AXIS Status
        */
         wire [31:0] tx_fcs_reg_rev, rx_fcs_reg_rev;
   
   always @(posedge rx_clk)
     if (rst_int == 1'b1)
       begin
	  byte_sync <= 1'b0;
       end
     else
       begin
          // Use synchronized buffer pointers for buffer-full check
	  if (rx_axis_tvalid && (byte_sync == 0) && (nextbuf_sync2 != (firstbuf_sync2+lastbuf_sync2)&31))
            begin
               byte_sync <= 1'b1;
            end
	  if (rx_axis_tlast && byte_sync)
            begin
               last <= 1'b1;
            end
          else if ((last > 0) && (last < 7))
            begin
	       byte_sync <= 1'b0;
               last <= last + 3'b1;
            end
          else
            begin
               last <= 3'b0;
            end
       end

   wire  [1:0] rx_wr = rx_axis_tvalid << rx_addr_axis[2];
   logic [15:0] douta;
   assign tx_axis_tdata = douta >> {tx_frame_addr[2],3'b000};
   
   dualmem_widen8 #(
                     .ADDR_A_WIDTH(15),   // 5-bit buf select + 8-bit word + 2-bit byte = 15
                     .ADDR_B_WIDTH(13)    // 64KB / 8 bytes = 8K words → 13 bits
   ) RAMB16_inst_rx (
                                    .clka(rx_clk),                // Port A Clock
                                    .clkb(msoc_clk),              // Port B Clock
                                    .douta(),                     // Port A 8-bit Data Output
                                    .addra({nextbuf_sync2[4:0],rx_addr_axis[10:3],rx_addr_axis[1:0]}),    // Port A 15-bit Address Input (sync'd)
                                    .dina({rx_axis_tdata,rx_axis_tdata}), // Port A 8-bit Data Input
                                    .ena(rx_axis_tvalid),         // Port A RAM Enable Input
                                    .wea(rx_wr),                  // Port A Write Enable Input
                                    .doutb(framing_rdata_pkt),    // Port B 64-bit Data Output
                                    .addrb(core_lsu_addr[15:3]),  // Port B 13-bit Address Input
                                    .dinb(core_lsu_wdata),        // Port B 64-bit Data Input
                                    .enb(ce_d & framing_sel & core_lsu_addr[16]),
                                                                  // Port B RAM Enable Input (RX at 0x10000)
                                    .web(we_d ? {(|core_lsu_be[7:4]),(|core_lsu_be[3:0])} : 2'b0) // Port B Write Enable Input
                                    );

    dualmem_widen RAMB16_inst_tx (
                                   .clka(~eth_clk),             // Port A Clock (125 MHz MAC domain)
                                   .clkb(msoc_clk),              // Port A Clock
                                   .douta(douta),                // Port A 8-bit Data Output
                                   .addra({tx_buf_sel,tx_frame_addr[10:3],tx_frame_addr[1:0]}),  // Port A 11-bit Address Input
                                   .dina(16'b0),                 // Port A 8-bit Data Input
                                   .ena(tx_axis_tvalid),         // Port A RAM Enable Input
                                   .wea(2'b0),                  // Port A Write Enable Input
                                   .doutb(framing_wdata_pkt),    // Port B 32-bit Data Output
                                   .addrb(core_lsu_addr[11:3]),  // Port B 9-bit Address Input
                                   .dinb(core_lsu_wdata), // Port B 32-bit Data Input
                                   .enb(ce_d & framing_sel & (core_lsu_addr[16:12]==5'b00001)),
				                                 // Port B RAM Enable Input (TX at 0x1000)
                                   .web(we_d ? {(|core_lsu_be[7:4]),(|core_lsu_be[3:0])} : 2'b0) // Port B Write Enable Input
                                   );

always @(posedge msoc_clk)
  if (rst_int)
    begin
    core_lsu_addr_dly <= 0;
    mac_address <= 48'H230100890702;
    tx_packet_length <= 0;
    tx_enable_dly <= 0;
    cooked <= 1'b0;
    loopback <= 1'b0;
    spare <= 4'b0;
    promiscuous <= 1'b0;
    phy_mdio_oe <= 1'b0;
    phy_mdio_o <= 1'b0;
    phy_mdclk <= 1'b0;
    sync <= 1'b0;
    firstbuf <= 5'b0;
    lastbuf <= 5'b0;
    nextbuf <= 5'b0;
    eth_irq <= 1'b0;
    irq_en <= 1'b0;
    ce_d_dly <= 1'b0;
    tx_busy <= 1'b0;
    tx_buf_sel <= 1'b0;
    avail = 1'b0;
    end
  else
    begin
    core_lsu_addr_dly <= core_lsu_addr;
    ce_d_dly <= ce_d;
    avail = nextbuf != firstbuf;
    eth_irq <= avail & irq_en; // make eth_irq go away immediately if irq_en is low
    if (framing_sel&we_d&(&core_lsu_be[3:0])&(core_lsu_addr[16:11]==6'b000001))
      case(core_lsu_addr[6:3])
        0: mac_address[31:0] <= core_lsu_wdata;
        1: {irq_en,promiscuous,spare,loopback,cooked,mac_address[47:32]} <= core_lsu_wdata;
        2: begin tx_enable_dly <= 10; tx_packet_length <= core_lsu_wdata; tx_buf_sel <= core_lsu_wdata[11]; end /* tx payload size + buf sel */
        3: begin tx_enable_dly <= 0; tx_packet_length <= 0; end
        4: begin {phy_mdio_oe,phy_mdio_o,phy_mdclk} <= core_lsu_wdata; end
        5: begin lastbuf <= core_lsu_wdata[4:0]; end
        6: begin firstbuf <= core_lsu_wdata[4:0]; end
        default:;
      endcase
       // Use synchronized frame_done pulse instead of raw `last` from rx_clk
       if (frame_pending && ~sync)
         begin
         // check broadcast/multicast address using synchronized rx_dest_mac snapshot
	     sync <= (rx_dest_mac_s[47:24]==24'h01005E) | (&rx_dest_mac_s) | (mac_address == rx_dest_mac_s) | promiscuous;
         end
       else if (!frame_pending && sync)
         begin
            nextbuf <= nextbuf + 1'b1;
            sync <= 1'b0;
         end
       if (mac_gmii_tx_en_sync2 && tx_enable_i_sync2)  // synchronized from eth_clk
         begin
            tx_enable_dly <= 0;
         end
       else if (1'b1 == |tx_enable_dly)
         begin
         tx_busy <= 1'b1;
         tx_enable_dly <= tx_enable_dly + !(&tx_enable_dly);
         end
       else if (~mac_gmii_tx_en_sync2)  // synchronized from eth_clk
         tx_busy <= 1'b0;         
    end

always @(posedge eth_clk)
  if (rst_int)
    begin
       tx_enable_i <= 1'b0;
    end
  else
    begin
       if (mac_gmii_tx_en && tx_axis_tlast)
         begin
            tx_enable_i <= 1'b0;
         end
       else if (tx_enable_rdy_sync2)  // synchronized from msoc_clk
         tx_enable_i <= 1'b1;
    end
   
   always @* casez({ce_d_dly,core_lsu_addr_dly[16:3]})
    15'b100_0001_0???_0000 : framing_rdata = mac_address[31:0];                                        // 0x0800
    15'b100_0001_0???_0001 : framing_rdata = {irq_en, promiscuous, spare, loopback, cooked, mac_address[47:32]}; // 0x0808
    15'b100_00??_????_0010 : framing_rdata = {tx_busy, 4'b0, tx_frame_addr, 5'b0, tx_packet_length};  // 0x0810
    15'b100_0001_0???_0011 : framing_rdata = tx_fcs_reg_rev;                                           // 0x0818
    15'b100_0001_0???_0100 : framing_rdata = {phy_mdio_i,phy_mdio_oe,phy_mdio_o,phy_mdclk};           // 0x0820
    15'b100_0001_0???_0101 : framing_rdata = rx_fcs_reg_rev;                                           // 0x0828
    15'b100_0001_0???_0110 : framing_rdata = {eth_irq, avail, lastbuf, nextbuf, firstbuf};             // 0x0830
    15'b100_0001_1??_????? : framing_rdata = rx_length_axis[core_lsu_addr_dly[7:3]];                   // 0x0C00 RPLR (32 entries)
    15'b100_001?_???????? : framing_rdata = framing_wdata_pkt;                                           // 0x1000 TX buffer
    15'b1_1?_????????????  : framing_rdata = framing_rdata_pkt;                                         // 0x10000 RX buffers (64KB)
    default: framing_rdata = 'h0;
    endcase

   parameter dly = 0;
   
   wire [31:0] 	    tx_fcs_reg, rx_fcs_reg;
   wire [15:0]      pcspma_status;
   assign 	    tx_fcs_reg_rev = {tx_fcs_reg[0],tx_fcs_reg[1],tx_fcs_reg[2],tx_fcs_reg[3],
                                          tx_fcs_reg[4],tx_fcs_reg[5],tx_fcs_reg[6],tx_fcs_reg[7],
                                          tx_fcs_reg[8],tx_fcs_reg[9],tx_fcs_reg[10],tx_fcs_reg[11],
                                          tx_fcs_reg[12],tx_fcs_reg[13],tx_fcs_reg[14],tx_fcs_reg[15],
                                          tx_fcs_reg[16],tx_fcs_reg[17],tx_fcs_reg[18],tx_fcs_reg[19],
                                          tx_fcs_reg[20],tx_fcs_reg[21],tx_fcs_reg[22],tx_fcs_reg[23],
                                          tx_fcs_reg[24],tx_fcs_reg[25],tx_fcs_reg[26],tx_fcs_reg[27],
                                          tx_fcs_reg[28],tx_fcs_reg[29],tx_fcs_reg[30],tx_fcs_reg[31]};
   assign 	    rx_fcs_reg_rev = {rx_fcs_reg[0],rx_fcs_reg[1],rx_fcs_reg[2],rx_fcs_reg[3],
                                          rx_fcs_reg[4],rx_fcs_reg[5],rx_fcs_reg[6],rx_fcs_reg[7],
                                          rx_fcs_reg[8],rx_fcs_reg[9],rx_fcs_reg[10],rx_fcs_reg[11],
                                          rx_fcs_reg[12],rx_fcs_reg[13],rx_fcs_reg[14],rx_fcs_reg[15],
                                          rx_fcs_reg[16],rx_fcs_reg[17],rx_fcs_reg[18],rx_fcs_reg[19],
                                          rx_fcs_reg[20],rx_fcs_reg[21],rx_fcs_reg[22],rx_fcs_reg[23],
                                          rx_fcs_reg[24],rx_fcs_reg[25],rx_fcs_reg[26],rx_fcs_reg[27],
                                          rx_fcs_reg[28],rx_fcs_reg[29],rx_fcs_reg[30],rx_fcs_reg[31]};
   
   always @(posedge eth_clk)
     if (rst_int)
       begin
          tx_axis_tvalid <= 'b0;
	  tx_axis_tvalid_dly <= 'b0;
	  tx_frame_addr <= 'b0;
	  tx_axis_tlast <= 'b0;
       end
     else
       begin
          tx_enable_old <= tx_enable_i;
	  if (tx_enable_i & (tx_enable_old == 0))
	    begin
	       tx_frame_addr <= 'b0;
	    end
	  if (tx_axis_tready)
	    begin
	       tx_frame_addr <= tx_frame_addr + 1;
	       tx_axis_tlast <= (tx_frame_addr == tx_packet_length-2) & tx_axis_tvalid_dly;
	    end
          tx_axis_tvalid <= tx_axis_tvalid_dly;
	  if (tx_enable_old)
	      tx_axis_tvalid_dly <= 1'b1;
	  else if (~tx_axis_tlast)
	      tx_axis_tvalid_dly <= 1'b0;
      end
 
   always @(posedge rx_clk)
     if (rst_int)
       begin
          rx_addr_axis <= 'b0;
          rx_dest_mac <= 'b0;
       end
     else
       begin
	  if (rx_axis_tvalid)
            begin
            rx_addr_axis <= rx_addr_axis + 1;
            if (rx_addr_axis < 6)
              rx_dest_mac <= {rx_dest_mac[39:0],rx_axis_tdata};
            end
	  if (rx_axis_tlast)
            begin
	        rx_length_axis[nextbuf_sync2] <= {rx_axis_tuser, rx_addr_axis + 1};  // [11]=FCS err
	        rx_addr_axis <= 'b0;
            end
      end
 
sgmii_soc sgmii_soc1
  (
   .clk_int(clk_int),
   .rst_int(rst_int),
   .eth_clk(eth_clk),
   .sgmii_rxp(sgmii_rxp),
   .sgmii_rxn(sgmii_rxn),
   .sgmii_txp(sgmii_txp),
   .sgmii_txn(sgmii_txn),
   .sgmii_refclk_p(sgmii_refclk_p),
   .sgmii_refclk_n(sgmii_refclk_n),
   .phy_reset_n(phy_reset_n),
   .mac_gmii_tx_en(mac_gmii_tx_en),
   .rx_clk(rx_clk),
   .tx_axis_tdata(tx_axis_tdata),
   .tx_axis_tvalid(tx_axis_tvalid),
   .tx_axis_tready(tx_axis_tready),
   .tx_axis_tlast(tx_axis_tlast),
   .tx_axis_tuser(tx_axis_tuser),
   .rx_axis_tdata(rx_axis_tdata),
   .rx_axis_tvalid(rx_axis_tvalid),
   .rx_axis_tlast(rx_axis_tlast),
   .rx_axis_tuser(rx_axis_tuser),
   .rx_fcs_reg(rx_fcs_reg),
   .tx_fcs_reg(tx_fcs_reg),
   .pcspma_status(pcspma_status)
);

/* ILA removed to free resources for 16-lane mining
xlnx_ila eth_ila (
    .clk(eth_clk),
    .probe0(rx_axis_tdata),
    .probe1(tx_axis_tdata),
    .probe2({rx_axis_tvalid, rx_axis_tlast, rx_axis_tuser,
             tx_axis_tvalid, tx_axis_tready, tx_axis_tlast,
             mac_gmii_tx_en, tx_enable_i}),
    .probe3({byte_sync, sync, avail, tx_busy}),
    .probe4(nextbuf[3:0]),
    .probe5(firstbuf[3:0]),
    .probe6(tx_frame_addr),
    .probe7(pcspma_status),
    .probe8(rx_addr_axis)
);
*/

endmodule // framing_top_sgmii
`default_nettype wire
