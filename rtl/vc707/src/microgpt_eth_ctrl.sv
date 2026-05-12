// microgpt_eth_ctrl.sv — DHCP/UDP/IPv4 transport for the microgpt MMIO bridge.
//
// Speaks the framing_top_sgmii LSU bus and presents an Avalon-MM master to
// the microgpt register-map slave. On boot, runs a minimal DHCP client
// (DISCOVER → wait OFFER → REQUEST → wait ACK → BOUND). Once bound, accepts
// raw-Ethernet ARP and UDP/IPv4 traffic addressed to the assigned IP and
// FPGA_PORT (19783, 0x4D47).
//
// DHCP scope:
//   - DISCOVER and REQUEST broadcasts (no unicast renewal).
//   - First-option-is-msg-type assumption for OFFER/ACK parsing (works
//     against ISC dhcpd, dnsmasq, most consumer routers).
//   - No lease-renewal, no NAK handling — once BOUND, the IP is held until
//     reset.
//   - Server identifier inferred from the OFFER's IP src address.
//
// Frame protocol on the bound UDP port (unchanged from earlier revision):
//   FT_REG_WRITE  payload[2]=n_writes (1..16), [8..]=n*{addr16, data32, pad16}
//   FT_REG_READ   payload[2:3]=start_addr LE, [4]=nwords (1..MAX_REG_READS)
//   FT_REG_RSP    same hdr as REG_READ, then nwords*32-bit LE data values
//   FT_HEARTBEAT  periodic FPGA→host
//   FT_ACK / FT_NAK   single-byte responses

`default_nettype none

module microgpt_eth_ctrl (
  input  wire           clk,         // 125 MHz eth_clk (userclk2)
  input  wire           rst_n,

  // framing_top_sgmii bus (master side)
  output logic [16:0]   core_lsu_addr,
  output logic [63:0]   core_lsu_wdata,
  output logic [7:0]    core_lsu_be,
  output logic          ce_d,
  output logic          we_d,
  output logic          framing_sel,
  input  wire  [63:0]   framing_rdata,

  // Avalon-MM master (drives microgpt register-map slave inside the top)
  output logic [31:0]   master_address,
  output logic          master_read,
  output logic          master_write,
  output logic [31:0]   master_writedata,
  output logic [3:0]    master_byteenable,
  input  wire  [31:0]   master_readdata,
  input  wire           master_readdatavalid,
  input  wire           master_waitrequest,

  // Status snapshot bytes for HEARTBEAT frames
  input  wire  [7:0]    hb_state,
  input  wire  [7:0]    hb_last_token,
  input  wire  [7:0]    hb_out_len,
  input  wire  [7:0]    hb_done_flags,

  // Activity flicker — pulses high for ~1 eth_clk on each accepted frame
  output logic          rx_activity,

  // DDR3 write request (4-phase handshake, eth_clk → ui_clk).
  output logic          ddr_wr_req,        // toggle: new write
  input  wire           ddr_wr_ack,        // matches when complete
  output logic [29:0]   ddr_wr_addr,
  output logic [511:0]  ddr_wr_data,

  // Debug
  output logic [5:0]    dbg_state,
  output logic [7:0]    dbg_frame_type,
  output logic [7:0]    dbg_wcnt,
  output logic [4:0]    dbg_cur_buf,
  output logic [7:0]    dbg_n_remaining,
  // DHCP-side observation
  output wire  [31:0]   dbg_fpga_ip,

  // DDR3 write-path debug counters (free-running; clear on reset).
  // Used to localise upload stalls — host polls these via regmap and
  // compares against ddr_write.py's `sent`/`acked` counts.
  //   ddr_wr_rx_count   = FT_DDR_WRITE frames accepted by the parser
  //   ddr_wr_done_count = ddr_wr_req toggles (= writes dispatched to MIG)
  //   ddr_wr_ack_count  = ddr_ack_sync1 toggles seen (= MIG completed)
  //   ddr_wr_tx_count   = FT_ACK frames dispatched for those writes
  output logic [31:0]   ddr_wr_rx_count,
  output logic [31:0]   ddr_wr_done_count,
  output logic [31:0]   ddr_wr_ack_count,
  output logic [31:0]   ddr_wr_tx_count
);

  // ================================================================
  //  Network parameters
  // ================================================================
  localparam [47:0] FPGA_MAC  = 48'h02_00_00_4D_47_31;
  localparam [15:0] FPGA_PORT = 16'd19783;              // 0x4D47 ("MG")

  localparam [15:0] ETH_TYPE_IPV4 = 16'h0800;
  localparam [15:0] ETH_TYPE_ARP  = 16'h0806;
  localparam [7:0]  IP_PROTO_UDP  = 8'h11;
  localparam [7:0]  IP_VER_IHL    = 8'h45;
  localparam [7:0]  IP_TTL        = 8'h40;
  localparam [15:0] IP_FLAGS_FRAG = 16'h4000;
  localparam [15:0] ARP_HW_TYPE   = 16'h0001;
  localparam [15:0] ARP_OP_REPLY  = 16'h0002;

  // DHCP
  localparam [15:0] DHCP_PORT_SERVER = 16'd67;
  localparam [15:0] DHCP_PORT_CLIENT = 16'd68;
  localparam [7:0]  DHCP_OP_REPLY    = 8'd2;
  localparam [7:0]  DHCP_MSG_OFFER   = 8'd2;
  localparam [7:0]  DHCP_MSG_ACK     = 8'd5;

  // tx_type values
  localparam [7:0] FT_REG_WRITE     = 8'h01;
  localparam [7:0] FT_REG_READ      = 8'h02;
  localparam [7:0] FT_REG_RSP       = 8'h03;
  localparam [7:0] FT_HEARTBEAT     = 8'h04;
  localparam [7:0] FT_NAK           = 8'h05;
  localparam [7:0] FT_ACK           = 8'h06;
  localparam [7:0] FT_ARP_REPLY     = 8'h07;            // internal
  localparam [7:0] FT_DHCP_DISCOVER = 8'h08;            // internal
  localparam [7:0] FT_DHCP_REQUEST  = 8'h09;            // internal
  localparam [7:0] FT_DDR_WRITE     = 8'h0A;            // host->FPGA bulk write

  // Heartbeat interval ~100 ms at 125 MHz.
  localparam [23:0] HB_INTERVAL = 24'd12_500_000;

  // Per-frame TX BRAM is 256 64-bit words (2 KB) per buffer. DHCP DISCOVER
  // = 286 bytes (36 words), DHCP REQUEST = 298 bytes (38 words).
  localparam [7:0] MAX_REG_WRITES = 8'd16;
  localparam [7:0] MAX_REG_READS  = 8'd19;

  // DHCP retry deadline (uptime_sec increments at 1 Hz)
  localparam [31:0] DHCP_TIMEOUT_SEC = 32'd4;

  // ================================================================
  //  States
  // ================================================================
  typedef enum logic [5:0] {
    S_INIT,
    S_IDLE,
    S_POLL_WAIT,    S_POLL_DONE,
    S_LEN_WAIT,     S_LEN_DONE,
    S_HDR0_WAIT,    S_HDR0_PROC,
    S_HDR1_WAIT,    S_HDR1_PROC,

    // ARP path
    S_ARP_W2_WAIT,  S_ARP_W2_PROC,
    S_ARP_W3_WAIT,  S_ARP_W3_PROC,
    S_ARP_W4_WAIT,  S_ARP_W4_PROC,
    S_ARP_W5_WAIT,  S_ARP_W5_PROC,

    // IPv4 / UDP path
    S_IPV4_W2_WAIT, S_IPV4_W2_PROC,
    S_IPV4_W3_WAIT, S_IPV4_W3_PROC,
    S_IPV4_W4_WAIT, S_IPV4_W4_PROC,
    S_IPV4_W5_WAIT, S_IPV4_W5_PROC,

    // DHCP RX (after IPv4_W4 dispatches to DHCP)
    S_DHCP_RX_W5_WAIT,  S_DHCP_RX_W5_PROC,
    S_DHCP_RX_W6_WAIT,  S_DHCP_RX_W6_PROC,
    S_DHCP_RX_W7_WAIT,  S_DHCP_RX_W7_PROC,
    S_DHCP_RX_OPT_WAIT, S_DHCP_RX_OPT_PROC,

    // DHCP wait-for-server states (poll RX while a deadline counts down)
    S_DHCP_WAIT_OFFER,
    S_DHCP_WAIT_ACK,

    // DHCP TX prep states (set up tx_type, then fall through to S_TX_WR)
    S_DHCP_DISCOVER_PREP,
    S_DHCP_REQUEST_PREP,

    // REG_WRITE entries
    S_RW_ENTRY_WAIT, S_RW_ENTRY_PROC,
    S_RW_DRIVE,

    // DDR_WRITE: read 8 RX BRAM words (6..13) into a 512-bit chunk,
    // then issue a 64-byte AXI write via the ddr_wr_req handshake.
    S_DDRW_LOAD_WAIT, S_DDRW_LOAD_PROC,
    S_DDRW_REQ,       S_DDRW_WAIT,

    // REG_READ master-bus loop
    S_RR_DRIVE, S_RR_RECV,

    // TX
    S_TX_WR, S_TX_WAIT, S_TX_WAIT_DONE, S_TX_GO,

    S_ADV_BUF, S_DONE
  } state_t;

  state_t state;

  // ================================================================
  //  Working registers
  // ================================================================
  logic [1:0]   init_step;
  logic [4:0]   cur_buf;
  logic [10:0]  frame_len;
  logic [7:0]   frame_type;
  logic [7:0]   frame_seq;
  logic [7:0]   wcnt;
  logic [63:0]  saved_w0;

  logic [7:0]   n_remaining;
  logic [15:0]  rw_addr;
  logic [31:0]  rw_data;

  logic [15:0]  rr_addr;
  logic [7:0]   rr_total;
  logic [7:0]   rr_done_count;
  logic [7:0]   rr_seq;
  logic [31:0]  rr_pending [0:18];

  logic [47:0]  req_mac;
  logic [31:0]  req_ip;
  logic [15:0]  req_src_port;
  logic [31:0]  arp_target_ip;

  logic [7:0]   tx_type;
  logic [7:0]   tx_seq;
  logic [7:0]   tx_total_words;
  logic         tx_buf;
  logic [3:0]   tx_return;

  logic [23:0]  hb_timer;
  logic         hb_pending;
  logic [26:0]  sec_prescaler;
  logic [31:0]  uptime_sec;

  // DHCP
  logic [31:0]  fpga_ip;            // 0 until DHCP-bound
  logic [31:0]  dhcp_xid;
  logic [31:0]  dhcp_offered_ip;
  logic [31:0]  dhcp_server_ip;
  logic [31:0]  dhcp_deadline_sec;
  logic [7:0]   dhcp_msg_type;
  logic [31:0]  dhcp_rx_xid;
  logic [3:0]   dhcp_retries;
  logic         dhcp_in_wait_ack;       // 0=expecting OFFER, 1=expecting ACK
  logic         dhcp_offer_pending;     // pulse: OFFER seen, schedule REQUEST
  logic         dhcp_ack_pending;       // pulse: ACK seen, schedule BOUND

  // DDR3 write staging
  logic [2:0]   ddr_chunk_idx;          // 0..7 which 64-bit slot of chunk we're loading
  logic         ddr_ack_sync0, ddr_ack_sync1;

  integer ri;

  // ================================================================
  //  Uptime + heartbeat timer
  // ================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sec_prescaler <= '0;
      uptime_sec    <= '0;
      hb_timer      <= '0;
      hb_pending    <= 1'b0;
    end else begin
      if (sec_prescaler >= 27'd124_999_999) begin
        sec_prescaler <= '0;
        uptime_sec    <= uptime_sec + 1;
      end else
        sec_prescaler <= sec_prescaler + 1;

      if (hb_timer >= HB_INTERVAL) begin
        hb_timer   <= '0;
        hb_pending <= 1'b1;
      end else
        hb_timer <= hb_timer + 1;

      if (state == S_TX_WR && tx_type == FT_HEARTBEAT && wcnt == 0)
        hb_pending <= 1'b0;
    end
  end

  // ================================================================
  //  IP / UDP length computed from tx_type (combinational)
  // ================================================================
  logic [15:0] ip_total_length;
  logic [15:0] udp_length;

  always_comb begin
    case (tx_type)
      FT_ACK:           begin ip_total_length = 16'd30;  udp_length = 16'd10; end  // type+seq
      FT_NAK:           begin ip_total_length = 16'd31;  udp_length = 16'd11; end  // type+seq+err
      FT_HEARTBEAT:     begin ip_total_length = 16'd40;  udp_length = 16'd20; end
      FT_REG_RSP: begin
        ip_total_length = 16'd36 + {6'd0, rr_total, 2'd0};
        udp_length      = 16'd16 + {6'd0, rr_total, 2'd0};
      end
      FT_DHCP_DISCOVER: begin ip_total_length = 16'd272; udp_length = 16'd252; end
      FT_DHCP_REQUEST:  begin ip_total_length = 16'd284; udp_length = 16'd264; end
      default:          begin ip_total_length = 16'd28;  udp_length = 16'd8;  end
    endcase
  end

  // Source / destination IP for the OUTGOING frame
  logic [31:0] tx_src_ip;
  logic [31:0] tx_dst_ip;
  always_comb begin
    if (tx_type == FT_DHCP_DISCOVER || tx_type == FT_DHCP_REQUEST) begin
      tx_src_ip = 32'd0;
      tx_dst_ip = 32'hFFFF_FFFF;
    end else begin
      tx_src_ip = fpga_ip;
      tx_dst_ip = req_ip;
    end
  end

  // ================================================================
  //  IP header checksum (combinational)
  //   sum = 0x4500 + total_length + 0 (ID) + 0x4000 + 0x4011
  //       + tx_src_ip[31:16] + tx_src_ip[15:0]
  //       + tx_dst_ip[31:16] + tx_dst_ip[15:0]
  // ================================================================
  logic [19:0] ip_sum_a;
  logic [16:0] ip_sum_b;
  logic [15:0] ip_sum_c;
  logic [15:0] ip_checksum;

  always_comb begin
    ip_sum_a = 20'h0C511                         // 0x4500 + 0x4000 + 0x4011
             + {4'd0, ip_total_length}
             + {4'd0, tx_src_ip[31:16]}
             + {4'd0, tx_src_ip[15:0]}
             + {4'd0, tx_dst_ip[31:16]}
             + {4'd0, tx_dst_ip[15:0]};
    ip_sum_b    = {1'b0, ip_sum_a[15:0]} + {13'd0, ip_sum_a[19:16]};
    ip_sum_c    = ip_sum_b[15:0] + {15'd0, ip_sum_b[16]};
    ip_checksum = ~ip_sum_c;
  end

  // UDP src/dst ports based on tx_type
  logic [15:0] tx_udp_src_port;
  logic [15:0] tx_udp_dst_port;
  always_comb begin
    if (tx_type == FT_DHCP_DISCOVER || tx_type == FT_DHCP_REQUEST) begin
      tx_udp_src_port = DHCP_PORT_CLIENT;
      tx_udp_dst_port = DHCP_PORT_SERVER;
    end else begin
      tx_udp_src_port = FPGA_PORT;
      tx_udp_dst_port = req_src_port;
    end
  end

  // Eth-frame dst MAC: broadcast for DHCP, requester's MAC otherwise
  logic [47:0] tx_dst_mac;
  always_comb begin
    if (tx_type == FT_DHCP_DISCOVER || tx_type == FT_DHCP_REQUEST)
      tx_dst_mac = 48'hFF_FF_FF_FF_FF_FF;
    else
      tx_dst_mac = req_mac;
  end

  // ================================================================
  //  Common UDP-payload byte slots (used by REG_RSP / HEARTBEAT / ACK / NAK).
  // ================================================================
  logic [7:0] pay0, pay1, pay2, pay3, pay4, pay5;
  logic [7:0] pay6, pay7, pay8, pay9, pay10, pay11, pay12, pay13;

  always_comb begin
    pay0  = tx_type;
    pay1  = tx_seq;
    pay2  = 8'h00; pay3  = 8'h00; pay4  = 8'h00; pay5  = 8'h00;
    pay6  = 8'h00; pay7  = 8'h00; pay8  = 8'h00; pay9  = 8'h00;
    pay10 = 8'h00; pay11 = 8'h00; pay12 = 8'h00; pay13 = 8'h00;

    case (tx_type)
      FT_NAK: pay2 = 8'h01;
      FT_HEARTBEAT: begin
        pay1  = 8'h00;
        pay2  = hb_state;
        pay3  = hb_last_token;
        pay4  = hb_out_len;
        pay5  = hb_done_flags;
        pay8  = uptime_sec[7:0];
        pay9  = uptime_sec[15:8];
        pay10 = uptime_sec[23:16];
        pay11 = uptime_sec[31:24];
      end
      FT_REG_RSP: begin
        pay2 = rr_addr[7:0];
        pay3 = rr_addr[15:8];
        pay4 = rr_total;
      end
      default: ;
    endcase
  end

  // ================================================================
  //  REG_RSP word K (K>=6) handled by misalignment-aware function.
  // ================================================================
  function automatic logic [31:0] rr_at(input logic signed [9:0] v);
    if (v < 0 || v >= $signed({2'd0, rr_total})) rr_at = 32'd0;
    else                                         rr_at = rr_pending[v[4:0]];
  endfunction

  function automatic logic [63:0] reg_rsp_word(input logic [7:0] K);
    logic signed [9:0] vmid, vlo, vhi;
    logic [31:0] vlo_val, vmid_val, vhi_val;
    begin
      vmid     = $signed({2'd0, K}) - 10'sd6;
      vmid     = vmid <<< 1;
      vlo      = vmid - 10'sd1;
      vhi      = vmid + 10'sd1;
      vlo_val  = rr_at(vlo);
      vmid_val = rr_at(vmid);
      vhi_val  = rr_at(vhi);
      reg_rsp_word = {vhi_val[15:0], vmid_val, vlo_val[31:16]};
    end
  endfunction

  // ================================================================
  //  TX word generator
  //
  //  BRAM byte-N -> word N/8, bits[(N%8)*8 +: 8].  All 16-bit NBO fields
  //  are written {LSB-byte, MSB-byte} in concat (rightmost = byte 16).
  // ================================================================
  logic [63:0] tx_word;
  logic        is_dhcp_tx;
  assign is_dhcp_tx = (tx_type == FT_DHCP_DISCOVER) ||
                      (tx_type == FT_DHCP_REQUEST);

  // Last option byte for word 35 of DISCOVER/REQUEST
  // DISCOVER: byte285=0xFF (END), bytes 286-287 pad 0.
  // REQUEST:  byte285=0x32 (opt 50), byte286=0x04 (length), byte287=offered IP[31:24].
  logic [7:0] dhcp_w35_b5, dhcp_w35_b6, dhcp_w35_b7;
  always_comb begin
    if (tx_type == FT_DHCP_REQUEST) begin
      dhcp_w35_b5 = 8'h32;                       // option 50
      dhcp_w35_b6 = 8'h04;                       // length
      dhcp_w35_b7 = dhcp_offered_ip[31:24];
    end else begin
      dhcp_w35_b5 = 8'hFF;                       // END
      dhcp_w35_b6 = 8'h00;
      dhcp_w35_b7 = 8'h00;
    end
  end

  // DHCP message type byte at word5 byte 2 (= frame byte 42 = DHCP op)
  logic [7:0] dhcp_msg_byte;
  always_comb begin
    case (tx_type)
      FT_DHCP_DISCOVER: dhcp_msg_byte = 8'd1;
      FT_DHCP_REQUEST:  dhcp_msg_byte = 8'd3;
      default:          dhcp_msg_byte = 8'd0;
    endcase
  end

  always_comb begin
    tx_word = 64'd0;

    if (tx_type == FT_ARP_REPLY) begin
      // ARP reply: 6 words, 42 bytes. (unchanged)
      case (wcnt)
        8'd0: tx_word = {FPGA_MAC[39:32], FPGA_MAC[47:40],
                         req_mac[7:0],  req_mac[15:8],
                         req_mac[23:16], req_mac[31:24],
                         req_mac[39:32], req_mac[47:40]};
        8'd1: tx_word = {ARP_HW_TYPE[7:0], ARP_HW_TYPE[15:8],
                         ETH_TYPE_ARP[7:0], ETH_TYPE_ARP[15:8],
                         FPGA_MAC[7:0],  FPGA_MAC[15:8],
                         FPGA_MAC[23:16], FPGA_MAC[31:24]};
        8'd2: tx_word = {FPGA_MAC[39:32], FPGA_MAC[47:40],
                         ARP_OP_REPLY[7:0], ARP_OP_REPLY[15:8],
                         8'h04, 8'h06,
                         ETH_TYPE_IPV4[7:0], ETH_TYPE_IPV4[15:8]};
        8'd3: tx_word = {fpga_ip[7:0], fpga_ip[15:8],
                         fpga_ip[23:16], fpga_ip[31:24],
                         FPGA_MAC[7:0],  FPGA_MAC[15:8],
                         FPGA_MAC[23:16], FPGA_MAC[31:24]};
        8'd4: tx_word = {req_ip[23:16], req_ip[31:24],
                         req_mac[7:0],  req_mac[15:8],
                         req_mac[23:16], req_mac[31:24],
                         req_mac[39:32], req_mac[47:40]};
        8'd5: tx_word = {16'h0000, 16'h0000, 16'h0000,
                         req_ip[7:0],  req_ip[15:8]};
        default: tx_word = 64'd0;
      endcase
    end else begin
      // UDP/IPv4 frame (also covers DHCP — uses the broadcast/zero source
      // selections via tx_dst_mac / tx_src_ip / tx_dst_ip / tx_udp_*_port).
      case (wcnt)
        8'd0: tx_word = {FPGA_MAC[39:32], FPGA_MAC[47:40],
                         tx_dst_mac[7:0],  tx_dst_mac[15:8],
                         tx_dst_mac[23:16], tx_dst_mac[31:24],
                         tx_dst_mac[39:32], tx_dst_mac[47:40]};
        8'd1: tx_word = {8'h00,
                         IP_VER_IHL,
                         ETH_TYPE_IPV4[7:0], ETH_TYPE_IPV4[15:8],
                         FPGA_MAC[7:0],  FPGA_MAC[15:8],
                         FPGA_MAC[23:16], FPGA_MAC[31:24]};
        8'd2: tx_word = {IP_PROTO_UDP, IP_TTL,
                         IP_FLAGS_FRAG[7:0], IP_FLAGS_FRAG[15:8],
                         8'h00, 8'h00,
                         ip_total_length[7:0], ip_total_length[15:8]};
        8'd3: tx_word = {tx_dst_ip[23:16], tx_dst_ip[31:24],
                         tx_src_ip[7:0], tx_src_ip[15:8],
                         tx_src_ip[23:16], tx_src_ip[31:24],
                         ip_checksum[7:0], ip_checksum[15:8]};
        8'd4: tx_word = {udp_length[7:0], udp_length[15:8],
                         tx_udp_dst_port[7:0], tx_udp_dst_port[15:8],
                         tx_udp_src_port[7:0], tx_udp_src_port[15:8],
                         tx_dst_ip[7:0], tx_dst_ip[15:8]};
        8'd5: begin
          if (is_dhcp_tx) begin
            // DHCP word 5: udp_cksum=0 (bytes 40-41), then BOOTP op/htype/
            // hlen/hops (bytes 42-45) + xid hi 2 bytes (46-47, NBO).
            tx_word = {dhcp_xid[23:16],          // byte 47
                       dhcp_xid[31:24],          // byte 46
                       8'h00,                     // byte 45 = hops
                       8'h06,                     // byte 44 = hlen
                       8'h01,                     // byte 43 = htype
                       8'h01,                     // byte 42 = op = REQUEST
                       8'h00,                     // byte 41 udp cksum lo
                       8'h00};                    // byte 40 udp cksum hi
          end else begin
            tx_word = {pay5, pay4, pay3, pay2,
                       pay1, pay0,
                       8'h00, 8'h00};
          end
        end
        8'd6: begin
          if (is_dhcp_tx) begin
            // DHCP word 6: xid lo + secs + flags=0x8000 (broadcast) + ciaddr hi
            tx_word = {8'h00, 8'h00,             // ciaddr[31:16] = 0
                       8'h00, 8'h80,             // flags NBO: byte52=0x80, byte53=0x00 → bits[39:32]=0x80, [47:40]=0x00
                       8'h00, 8'h00,             // secs
                       dhcp_xid[7:0],            // byte 49 = xid LSB
                       dhcp_xid[15:8]};          // byte 48 = xid byte 2
          end else if (tx_type == FT_HEARTBEAT) begin
            tx_word = {pay13, pay12, pay11, pay10, pay9, pay8, pay7, pay6};
          end else if (tx_type == FT_REG_RSP) begin
            tx_word = reg_rsp_word(8'd6);
          end else begin
            tx_word = 64'd0;
          end
        end
        8'd7: begin
          if (is_dhcp_tx) tx_word = 64'd0;       // ciaddr lo + yiaddr + siaddr hi
          else            tx_word = (tx_type == FT_REG_RSP) ? reg_rsp_word(8'd7) : 64'd0;
        end
        8'd8: begin
          if (is_dhcp_tx)
            // siaddr lo (54-55) + giaddr (56-59... wait giaddr is bytes 66-69? let me recheck)
            // bytes 64-65 = siaddr[15:0] (continuing from word 7)
            // bytes 66-69 = giaddr (4 bytes)
            // bytes 70-71 = chaddr[0:1] = MAC[47:40], MAC[39:32]
            tx_word = {FPGA_MAC[39:32],          // byte 71
                       FPGA_MAC[47:40],          // byte 70
                       8'h00, 8'h00, 8'h00, 8'h00,  // giaddr 66-69
                       8'h00, 8'h00};             // siaddr lo 64-65
          else if (tx_type == FT_REG_RSP) tx_word = reg_rsp_word(8'd8);
          else                            tx_word = 64'd0;
        end
        8'd9: begin
          if (is_dhcp_tx)
            // chaddr[2:9] — bytes 72-79
            tx_word = {8'h00, 8'h00, 8'h00, 8'h00,    // bytes 76-79 = chaddr[6:9] = 0
                       FPGA_MAC[7:0],                  // byte 75 = chaddr[5]
                       FPGA_MAC[15:8],                 // byte 74
                       FPGA_MAC[23:16],                // byte 73
                       FPGA_MAC[31:24]};               // byte 72
          else if (tx_type == FT_REG_RSP) tx_word = reg_rsp_word(8'd9);
          else                            tx_word = 64'd0;
        end
        // Words 10..33 are all-zero for DHCP (chaddr tail + sname + middle of file).
        // For REG_RSP they continue the data stream.
        8'd34: begin
          if (is_dhcp_tx)
            // file end (bytes 272-277 = 0) + magic cookie hi 2 bytes (278-279 = 0x63, 0x82)
            tx_word = {8'h82, 8'h63,             // bytes 279, 278
                       8'h00, 8'h00, 8'h00, 8'h00,
                       8'h00, 8'h00};
          else if (tx_type == FT_REG_RSP) tx_word = reg_rsp_word(8'd34);
          else                            tx_word = 64'd0;
        end
        8'd35: begin
          if (is_dhcp_tx)
            // bytes 280-281 = cookie lo (0x53, 0x63)
            // byte 282 = 0x35 (option 53), byte 283 = 0x01 (length)
            // byte 284 = msg type (1 or 3)
            // byte 285+ = either END (DISCOVER) or option 50 hdr (REQUEST)
            tx_word = {dhcp_w35_b7,              // byte 287
                       dhcp_w35_b6,              // byte 286
                       dhcp_w35_b5,              // byte 285
                       dhcp_msg_byte,            // byte 284
                       8'h01,                     // byte 283 = length 1
                       8'h35,                     // byte 282 = option 53
                       8'h63,                     // byte 281 = cookie byte 3
                       8'h53};                    // byte 280 = cookie byte 2
          else if (tx_type == FT_REG_RSP) tx_word = reg_rsp_word(8'd35);
          else                            tx_word = 64'd0;
        end
        8'd36: begin
          // REQUEST only: req IP[23:0] + opt 54 + server IP[31:8]
          if (tx_type == FT_DHCP_REQUEST)
            tx_word = {dhcp_server_ip[15:8],         // byte 295
                       dhcp_server_ip[23:16],        // byte 294
                       dhcp_server_ip[31:24],        // byte 293
                       8'h04,                         // byte 292 = length
                       8'h36,                         // byte 291 = option 54
                       dhcp_offered_ip[7:0],         // byte 290
                       dhcp_offered_ip[15:8],        // byte 289
                       dhcp_offered_ip[23:16]};      // byte 288
          else if (tx_type == FT_REG_RSP) tx_word = reg_rsp_word(8'd36);
          else                            tx_word = 64'd0;
        end
        8'd37: begin
          // REQUEST only: server IP[7:0] + END + pad
          if (tx_type == FT_DHCP_REQUEST)
            tx_word = {8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
                       8'hFF,                         // byte 297 = END
                       dhcp_server_ip[7:0]};         // byte 296
          else if (tx_type == FT_REG_RSP) tx_word = reg_rsp_word(8'd37);
          else                            tx_word = 64'd0;
        end
        default: begin
          if (tx_type == FT_REG_RSP) tx_word = reg_rsp_word(wcnt);
          else                       tx_word = 64'd0;
        end
      endcase
    end
  end

  // Combinational TX byte-length / word-count
  function automatic logic [10:0] tx_byte_len(input logic [7:0] ftype,
                                              input logic [7:0] ndata);
    case (ftype)
      FT_ARP_REPLY:     return 11'd42;
      FT_ACK:           return 11'd44;     // 14+20+8+2
      FT_NAK:           return 11'd45;     // 14+20+8+3
      FT_HEARTBEAT:     return 11'd54;
      FT_REG_RSP:       return 11'd64 + {3'd0, ndata, 2'd0};
      FT_DHCP_DISCOVER: return 11'd286;
      FT_DHCP_REQUEST:  return 11'd298;
      default:          return 11'd42;
    endcase
  endfunction

  function automatic logic [7:0] tx_word_count(input logic [7:0] ftype,
                                               input logic [7:0] ndata);
    case (ftype)
      FT_ARP_REPLY:     return 8'd6;
      FT_ACK:           return 8'd6;
      FT_NAK:           return 8'd6;
      FT_HEARTBEAT:     return 8'd7;
      FT_REG_RSP:       return 8'd6 + ((ndata + 8'd2) >> 1);
      FT_DHCP_DISCOVER: return 8'd36;
      FT_DHCP_REQUEST:  return 8'd38;
      default:          return 8'd6;
    endcase
  endfunction

  logic [10:0] tx_pkt_len;
  always_comb tx_pkt_len = tx_byte_len(tx_type, rr_total);

  // ================================================================
  //  Bus address helpers
  // ================================================================
  function automatic logic [16:0] rx_addr(logic [4:0] buf_id, logic [7:0] word);
    return {1'b1, buf_id, word, 3'b000};
  endfunction

  // TX BRAM: bit 12=TX region, bit 11=buf select, bits[10:3]=word (0..255).
  function automatic logic [16:0] tx_addr(logic toggle_buf, logic [7:0] word);
    return {4'b0, 1'b1, toggle_buf, word[7:0], 3'b000};
  endfunction

  function automatic logic [16:0] len_addr(logic [4:0] buf_id);
    return {9'b0_0000_1100, buf_id, 3'b000};
  endfunction

  // ================================================================
  //  Main FSM
  // ================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state             <= S_INIT;
      init_step         <= 2'd0;
      ce_d              <= 1'b0;
      we_d              <= 1'b0;
      framing_sel       <= 1'b1;
      core_lsu_addr     <= '0;
      core_lsu_wdata    <= '0;
      core_lsu_be       <= 8'hFF;
      master_address    <= '0;
      master_read       <= 1'b0;
      master_write      <= 1'b0;
      master_writedata  <= '0;
      master_byteenable <= 4'hF;
      cur_buf           <= '0;
      frame_len         <= '0;
      frame_type        <= '0;
      frame_seq         <= '0;
      wcnt              <= '0;
      saved_w0          <= '0;
      n_remaining       <= '0;
      rw_addr           <= '0;
      rw_data           <= '0;
      rr_addr           <= '0;
      rr_total          <= '0;
      rr_done_count     <= '0;
      rr_seq            <= '0;
      req_mac           <= 48'hFF_FF_FF_FF_FF_FF;
      req_ip            <= 32'hFFFFFFFF;
      req_src_port      <= 16'd0;
      arp_target_ip     <= 32'd0;
      tx_type           <= '0;
      tx_seq            <= '0;
      tx_total_words    <= '0;
      tx_buf            <= 1'b0;
      tx_return         <= '0;
      rx_activity       <= 1'b0;
      fpga_ip           <= 32'd0;
      dhcp_xid          <= 32'h4D470001;
      dhcp_offered_ip   <= 32'd0;
      dhcp_server_ip    <= 32'd0;
      dhcp_deadline_sec <= '0;
      dhcp_msg_type     <= 8'd0;
      dhcp_rx_xid       <= 32'd0;
      dhcp_retries      <= 4'd0;
      dhcp_in_wait_ack  <= 1'b0;
      dhcp_offer_pending<= 1'b0;
      dhcp_ack_pending  <= 1'b0;
      ddr_wr_req        <= 1'b0;
      ddr_wr_addr       <= '0;
      ddr_wr_data       <= '0;
      ddr_chunk_idx     <= '0;
      ddr_wr_rx_count   <= '0;
      ddr_wr_done_count <= '0;
      ddr_wr_ack_count  <= '0;
      ddr_wr_tx_count   <= '0;
      ddr_ack_sync0     <= 1'b0;
      ddr_ack_sync1     <= 1'b0;
      for (ri = 0; ri < 19; ri = ri + 1) rr_pending[ri] <= '0;
    end else begin
      master_read  <= 1'b0;
      master_write <= 1'b0;
      rx_activity  <= 1'b0;

      // 2FF sync of the ddr_wr_ack toggle from ui_clk
      ddr_ack_sync0 <= ddr_wr_ack;
      ddr_ack_sync1 <= ddr_ack_sync0;

      case (state)
        // ----------------------------------------------------------------
        S_INIT: begin
          ce_d <= 1'b1; we_d <= 1'b1;
          case (init_step)
            2'd0: begin
              core_lsu_addr  <= 17'h00800;
              core_lsu_wdata <= {32'd0, FPGA_MAC[31:0]};
              init_step      <= 2'd1;
            end
            2'd1: begin
              core_lsu_addr  <= 17'h00808;
              core_lsu_wdata <= {40'd0, 1'b0, 1'b1, 4'b0, 1'b0, 1'b0, FPGA_MAC[47:32]};
              init_step      <= 2'd2;
            end
            2'd2: begin
              core_lsu_addr  <= 17'h00828;
              core_lsu_wdata <= {59'd0, 5'd31};
              init_step      <= 2'd3;
            end
            default: begin
              ce_d  <= 1'b0; we_d <= 1'b0;
              state <= S_DHCP_DISCOVER_PREP;
            end
          endcase
        end

        // ----------------------------------------------------------------
        //  DHCP TX prep
        // ----------------------------------------------------------------
        S_DHCP_DISCOVER_PREP: begin
          ce_d <= 1'b0; we_d <= 1'b0;
          tx_type        <= FT_DHCP_DISCOVER;
          tx_seq         <= 8'd0;
          tx_total_words <= tx_word_count(FT_DHCP_DISCOVER, 8'd0);
          tx_return      <= 4'd2;            // 2 = after TX_GO go to DHCP_WAIT_OFFER
          wcnt           <= '0;
          state          <= S_TX_WR;
        end

        S_DHCP_REQUEST_PREP: begin
          ce_d <= 1'b0; we_d <= 1'b0;
          tx_type        <= FT_DHCP_REQUEST;
          tx_seq         <= 8'd0;
          tx_total_words <= tx_word_count(FT_DHCP_REQUEST, 8'd0);
          tx_return      <= 4'd3;            // 3 = after TX_GO go to DHCP_WAIT_ACK
          wcnt           <= '0;
          state          <= S_TX_WR;
        end

        // ----------------------------------------------------------------
        //  DHCP wait-for-server
        // ----------------------------------------------------------------
        S_DHCP_WAIT_OFFER, S_DHCP_WAIT_ACK: begin
          ce_d <= 1'b0; we_d <= 1'b0;
          if (uptime_sec >= dhcp_deadline_sec) begin
            // Timeout — re-send DISCOVER and start over.
            dhcp_xid     <= dhcp_xid + 32'd1;
            dhcp_retries <= dhcp_retries + 4'd1;
            state        <= S_DHCP_DISCOVER_PREP;
          end else begin
            // Poll RX
            ce_d          <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= 17'h00830;
            state         <= S_POLL_WAIT;
          end
        end

        // ----------------------------------------------------------------
        S_IDLE: begin
          ce_d <= 1'b0; we_d <= 1'b0;
          if (hb_pending && req_ip != 32'hFFFFFFFF && fpga_ip != 32'd0) begin
            tx_type        <= FT_HEARTBEAT;
            tx_seq         <= 8'd0;
            tx_total_words <= tx_word_count(FT_HEARTBEAT, 8'd0);
            tx_return      <= 4'd0;
            wcnt           <= '0;
            state          <= S_TX_WR;
          end else begin
            ce_d          <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= 17'h00830;
            state         <= S_POLL_WAIT;
          end
        end

        S_POLL_WAIT: state <= S_POLL_DONE;

        S_POLL_DONE: begin
          if (framing_rdata[15]) begin
            cur_buf       <= framing_rdata[4:0];
            ce_d          <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= len_addr(framing_rdata[4:0]);
            state         <= S_LEN_WAIT;
          end else begin
            // No frame queued. Return to whichever steady state we were in.
            if (fpga_ip == 32'd0)
              state <= dhcp_in_wait_ack ? S_DHCP_WAIT_ACK : S_DHCP_WAIT_OFFER;
            else
              state <= S_IDLE;
          end
        end

        S_LEN_WAIT: state <= S_LEN_DONE;

        S_LEN_DONE: begin
          frame_len <= framing_rdata[10:0];
          if (framing_rdata[11]) begin
            state <= S_ADV_BUF;
          end else begin
            wcnt          <= '0;
            ce_d          <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= rx_addr(cur_buf, 8'd0);
            state         <= S_HDR0_WAIT;
          end
        end

        S_HDR0_WAIT: state <= S_HDR0_PROC;

        S_HDR0_PROC: begin
          saved_w0      <= framing_rdata;
          ce_d          <= 1'b1; we_d <= 1'b0;
          core_lsu_addr <= rx_addr(cur_buf, 8'd1);
          state         <= S_HDR1_WAIT;
        end

        S_HDR1_WAIT: state <= S_HDR1_PROC;

        S_HDR1_PROC: begin
          req_mac[47:40] <= saved_w0[55:48];
          req_mac[39:32] <= saved_w0[63:56];
          req_mac[31:24] <= framing_rdata[7:0];
          req_mac[23:16] <= framing_rdata[15:8];
          req_mac[15:8]  <= framing_rdata[23:16];
          req_mac[7:0]   <= framing_rdata[31:24];

          if ({framing_rdata[39:32], framing_rdata[47:40]} == ETH_TYPE_ARP) begin
            ce_d <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= rx_addr(cur_buf, 8'd2);
            state <= S_ARP_W2_WAIT;
          end else if ({framing_rdata[39:32], framing_rdata[47:40]} == ETH_TYPE_IPV4) begin
            ce_d <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= rx_addr(cur_buf, 8'd2);
            state <= S_IPV4_W2_WAIT;
          end else begin
            state <= S_ADV_BUF;
          end
        end

        // ----------------------------------------------------------------
        //  ARP request parsing
        // ----------------------------------------------------------------
        S_ARP_W2_WAIT: state <= S_ARP_W2_PROC;
        S_ARP_W2_PROC: begin
          ce_d <= 1'b1; we_d <= 1'b0;
          core_lsu_addr <= rx_addr(cur_buf, 8'd3);
          state <= S_ARP_W3_WAIT;
        end

        S_ARP_W3_WAIT: state <= S_ARP_W3_PROC;
        S_ARP_W3_PROC: begin
          req_ip[31:24] <= framing_rdata[39:32];
          req_ip[23:16] <= framing_rdata[47:40];
          req_ip[15:8]  <= framing_rdata[55:48];
          req_ip[7:0]   <= framing_rdata[63:56];
          ce_d <= 1'b1; we_d <= 1'b0;
          core_lsu_addr <= rx_addr(cur_buf, 8'd4);
          state <= S_ARP_W4_WAIT;
        end

        S_ARP_W4_WAIT: state <= S_ARP_W4_PROC;
        S_ARP_W4_PROC: begin
          arp_target_ip[31:24] <= framing_rdata[55:48];
          arp_target_ip[23:16] <= framing_rdata[63:56];
          ce_d <= 1'b1; we_d <= 1'b0;
          core_lsu_addr <= rx_addr(cur_buf, 8'd5);
          state <= S_ARP_W5_WAIT;
        end

        S_ARP_W5_WAIT: state <= S_ARP_W5_PROC;
        S_ARP_W5_PROC: begin
          if (fpga_ip != 32'd0 &&
              {arp_target_ip[31:16], framing_rdata[7:0], framing_rdata[15:8]}
              == fpga_ip) begin
            rx_activity         <= 1'b1;
            arp_target_ip[15:8] <= framing_rdata[7:0];
            arp_target_ip[7:0]  <= framing_rdata[15:8];
            tx_type             <= FT_ARP_REPLY;
            tx_seq              <= 8'd0;
            tx_total_words      <= tx_word_count(FT_ARP_REPLY, 8'd0);
            tx_return           <= 4'd1;
            wcnt                <= '0;
            state               <= S_TX_WR;
          end else begin
            state <= S_ADV_BUF;
          end
        end

        // ----------------------------------------------------------------
        //  IPv4 / UDP request parsing
        // ----------------------------------------------------------------
        S_IPV4_W2_WAIT: state <= S_IPV4_W2_PROC;
        S_IPV4_W2_PROC: begin
          ce_d <= 1'b1; we_d <= 1'b0;
          core_lsu_addr <= rx_addr(cur_buf, 8'd3);
          state <= S_IPV4_W3_WAIT;
        end

        S_IPV4_W3_WAIT: state <= S_IPV4_W3_PROC;
        S_IPV4_W3_PROC: begin
          req_ip[31:24] <= framing_rdata[23:16];
          req_ip[23:16] <= framing_rdata[31:24];
          req_ip[15:8]  <= framing_rdata[39:32];
          req_ip[7:0]   <= framing_rdata[47:40];
          arp_target_ip[31:24] <= framing_rdata[55:48];
          arp_target_ip[23:16] <= framing_rdata[63:56];
          ce_d <= 1'b1; we_d <= 1'b0;
          core_lsu_addr <= rx_addr(cur_buf, 8'd4);
          state <= S_IPV4_W4_WAIT;
        end

        S_IPV4_W4_WAIT: state <= S_IPV4_W4_PROC;
        S_IPV4_W4_PROC: begin
          // Capture UDP src port + decide MicroGPT vs DHCP path.
          req_src_port[15:8] <= framing_rdata[23:16];
          req_src_port[7:0]  <= framing_rdata[31:24];
          // dst port (NBO) bytes 36-37 = framing_rdata[39:32], [47:40]
          if ({framing_rdata[39:32], framing_rdata[47:40]} == DHCP_PORT_CLIENT) begin
            // DHCP reply (port 68). Accept regardless of dst IP — DHCP
            // servers may unicast to yiaddr, broadcast to 0xFFFFFFFF, or
            // address us by our pre-bound state. xid validation done later.
            ce_d <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= rx_addr(cur_buf, 8'd5);
            state <= S_DHCP_RX_W5_WAIT;
          end else if ({arp_target_ip[31:16],
                        framing_rdata[7:0], framing_rdata[15:8]} == fpga_ip
                       && fpga_ip != 32'd0
                       && {framing_rdata[39:32], framing_rdata[47:40]} == FPGA_PORT) begin
            // MicroGPT UDP — only after DHCP-bound.
            ce_d <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= rx_addr(cur_buf, 8'd5);
            state <= S_IPV4_W5_WAIT;
          end else begin
            state <= S_ADV_BUF;
          end
        end

        S_IPV4_W5_WAIT: state <= S_IPV4_W5_PROC;
        S_IPV4_W5_PROC: begin
          rx_activity <= 1'b1;
          frame_type  <= framing_rdata[23:16];
          frame_seq   <= framing_rdata[31:24];

          case (framing_rdata[23:16])
            FT_REG_WRITE: begin
              n_remaining <= (framing_rdata[39:32] > MAX_REG_WRITES)
                              ? MAX_REG_WRITES : framing_rdata[39:32];
              wcnt          <= 8'd6;
              ce_d <= 1'b1; we_d <= 1'b0;
              core_lsu_addr <= rx_addr(cur_buf, 8'd6);
              state <= S_RW_ENTRY_WAIT;
            end
            FT_REG_READ: begin
              rr_addr  <= {framing_rdata[47:40], framing_rdata[39:32]};
              rr_total <= (framing_rdata[55:48] > MAX_REG_READS)
                            ? MAX_REG_READS : framing_rdata[55:48];
              rr_seq        <= framing_rdata[31:24];
              rr_done_count <= '0;
              state         <= S_RR_DRIVE;
            end
            FT_DDR_WRITE: begin
              // payload bytes 2..5 = ddr_addr LE (frame bytes 44..47 =
              // word 5 [39:32]..[63:56]). Reconstruct the 32-bit LE byte
              // address and keep the low 30 bits (DDR3 = 1 GB; high 2
              // bits assumed 0). Caller must ensure 64-byte alignment.
              ddr_wr_addr <= {framing_rdata[61:56], framing_rdata[55:48],
                              framing_rdata[47:40], framing_rdata[39:32]};
              // Begin reading word 6 (chunk start, word-aligned).
              ddr_chunk_idx <= '0;
              wcnt          <= 8'd6;
              ce_d <= 1'b1; we_d <= 1'b0;
              core_lsu_addr <= rx_addr(cur_buf, 8'd6);
              state <= S_DDRW_LOAD_WAIT;
              ddr_wr_rx_count <= ddr_wr_rx_count + 32'd1;
            end
            default: begin
              tx_type        <= FT_NAK;
              tx_seq         <= framing_rdata[31:24];
              tx_total_words <= tx_word_count(FT_NAK, 8'd0);
              tx_return      <= 4'd1;
              wcnt           <= '0;
              state          <= S_TX_WR;
            end
          endcase
        end

        // ----------------------------------------------------------------
        //  DHCP RX
        // ----------------------------------------------------------------
        // Word 5 (frame bytes 40-47) — UDP cksum + DHCP op/htype/hlen/hops + xid hi
        S_DHCP_RX_W5_WAIT: state <= S_DHCP_RX_W5_PROC;

        S_DHCP_RX_W5_PROC: begin
          // op at bits[23:16] = byte 42; xid hi 16 bits at [55:48], [63:56]
          // (NBO byte 46 = MSB).
          if (framing_rdata[23:16] != DHCP_OP_REPLY) begin
            // Not a server reply — ignore.
            state <= S_ADV_BUF;
          end else begin
            dhcp_rx_xid[31:24] <= framing_rdata[55:48];
            dhcp_rx_xid[23:16] <= framing_rdata[63:56];
            ce_d <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= rx_addr(cur_buf, 8'd6);
            state <= S_DHCP_RX_W6_WAIT;
          end
        end

        // Word 6 — xid lo + secs + flags + ciaddr hi
        S_DHCP_RX_W6_WAIT: state <= S_DHCP_RX_W6_PROC;

        S_DHCP_RX_W6_PROC: begin
          // xid lo at bytes 48-49 = framing_rdata[7:0], [15:8] (NBO byte 48 = MSB of xid lo).
          dhcp_rx_xid[15:8] <= framing_rdata[7:0];
          dhcp_rx_xid[7:0]  <= framing_rdata[15:8];
          // Check xid match (against our sent xid)
          if ({dhcp_rx_xid[31:16], framing_rdata[7:0], framing_rdata[15:8]}
              != dhcp_xid) begin
            state <= S_ADV_BUF;
          end else begin
            ce_d <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= rx_addr(cur_buf, 8'd7);
            state <= S_DHCP_RX_W7_WAIT;
          end
        end

        // Word 7 — ciaddr lo (56-57) + yiaddr (58-61) + siaddr hi (62-63)
        S_DHCP_RX_W7_WAIT: state <= S_DHCP_RX_W7_PROC;

        S_DHCP_RX_W7_PROC: begin
          // yiaddr at bytes 58-61 (NBO).
          //   byte 58 = bits[23:16] = MSB
          dhcp_offered_ip[31:24] <= framing_rdata[23:16];
          dhcp_offered_ip[23:16] <= framing_rdata[31:24];
          dhcp_offered_ip[15:8]  <= framing_rdata[39:32];
          dhcp_offered_ip[7:0]   <= framing_rdata[47:40];
          // Capture the server IP from req_ip (already set in S_IPV4_W3_PROC
          // — that was the IP src of the OFFER/ACK).
          dhcp_server_ip <= req_ip;
          // Now jump to word 35 to read the first option (assumed to be msg type).
          ce_d <= 1'b1; we_d <= 1'b0;
          core_lsu_addr <= rx_addr(cur_buf, 8'd35);
          state <= S_DHCP_RX_OPT_WAIT;
        end

        // Word 35 — magic cookie tail (bytes 280-281) + first option (282 onward)
        S_DHCP_RX_OPT_WAIT: state <= S_DHCP_RX_OPT_PROC;

        S_DHCP_RX_OPT_PROC: begin
          // Assume first option (byte 282 = framing_rdata[23:16]) is msg
          // type (53), length at byte 283 = [31:24], value at byte 284 =
          // [39:32]. Match against current expectation.
          dhcp_msg_type <= framing_rdata[39:32];
          if (!dhcp_in_wait_ack && framing_rdata[39:32] == DHCP_MSG_OFFER) begin
            rx_activity        <= 1'b1;
            dhcp_offer_pending <= 1'b1;
          end else if (dhcp_in_wait_ack && framing_rdata[39:32] == DHCP_MSG_ACK) begin
            rx_activity      <= 1'b1;
            dhcp_ack_pending <= 1'b1;
          end
          // Always advance buf regardless of match.
          state <= S_ADV_BUF;
        end

        // ----------------------------------------------------------------
        //  REG_WRITE entries
        // ----------------------------------------------------------------
        S_RW_ENTRY_WAIT: state <= S_RW_ENTRY_PROC;
        S_RW_ENTRY_PROC: begin
          rw_addr <= {framing_rdata[31:24], framing_rdata[23:16]};
          rw_data <= {framing_rdata[63:56], framing_rdata[55:48],
                      framing_rdata[47:40], framing_rdata[39:32]};
          state   <= S_RW_DRIVE;
        end

        S_RW_DRIVE: begin
          ce_d <= 1'b0; we_d <= 1'b0;
          master_address   <= {{14{1'b0}}, rw_addr, 2'b00};
          master_writedata <= rw_data;
          master_write     <= 1'b1;
          if (n_remaining > 8'd1) begin
            n_remaining   <= n_remaining - 8'd1;
            wcnt          <= wcnt + 8'd1;
            ce_d          <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= rx_addr(cur_buf, wcnt + 8'd1);
            state         <= S_RW_ENTRY_WAIT;
          end else begin
            tx_type        <= FT_ACK;
            tx_seq         <= frame_seq;
            tx_total_words <= tx_word_count(FT_ACK, 8'd0);
            tx_return      <= 4'd1;
            wcnt           <= '0;
            state          <= S_TX_WR;
          end
        end

        // ----------------------------------------------------------------
        //  DDR_WRITE: load 8 RX BRAM words into a 512-bit chunk, then
        //  toggle ddr_wr_req for the ui_clk-side write master.
        // ----------------------------------------------------------------
        S_DDRW_LOAD_WAIT: state <= S_DDRW_LOAD_PROC;

        S_DDRW_LOAD_PROC: begin
          // Capture word K = chunk_idx into chunk[chunk_idx*64 +: 64]
          ddr_wr_data[ddr_chunk_idx*64 +: 64] <= framing_rdata;
          if (ddr_chunk_idx == 3'd7) begin
            state <= S_DDRW_REQ;
          end else begin
            ddr_chunk_idx <= ddr_chunk_idx + 3'd1;
            wcnt          <= wcnt + 8'd1;
            ce_d          <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= rx_addr(cur_buf, wcnt + 8'd1);
            state         <= S_DDRW_LOAD_WAIT;
          end
        end

        S_DDRW_REQ: begin
          // Toggle the request — write master will pick up addr+data
          // (held stable since LOAD_PROC).
          ce_d <= 1'b0; we_d <= 1'b0;
          ddr_wr_req <= ~ddr_wr_req;
          state      <= S_DDRW_WAIT;
          ddr_wr_done_count <= ddr_wr_done_count + 32'd1;
        end

        S_DDRW_WAIT: begin
          ce_d <= 1'b0; we_d <= 1'b0;
          // Wait until the write master mirrors our toggle (write done).
          if (ddr_ack_sync1 == ddr_wr_req) begin
            tx_type        <= FT_ACK;
            tx_seq         <= frame_seq;
            tx_total_words <= tx_word_count(FT_ACK, 8'd0);
            tx_return      <= 4'd1;
            wcnt           <= '0;
            state          <= S_TX_WR;
            ddr_wr_ack_count <= ddr_wr_ack_count + 32'd1;
            ddr_wr_tx_count  <= ddr_wr_tx_count  + 32'd1;
          end
        end

        // ----------------------------------------------------------------
        //  REG_READ
        // ----------------------------------------------------------------
        S_RR_DRIVE: begin
          ce_d <= 1'b0; we_d <= 1'b0;
          if (rr_done_count >= rr_total) begin
            tx_type        <= FT_REG_RSP;
            tx_seq         <= rr_seq;
            tx_total_words <= tx_word_count(FT_REG_RSP, rr_total);
            tx_return      <= 4'd1;
            wcnt           <= '0;
            state          <= S_TX_WR;
          end else begin
            master_address <= {{14{1'b0}}, rr_addr + {8'd0, rr_done_count}, 2'b00};
            master_read    <= 1'b1;
            state          <= S_RR_RECV;
          end
        end

        S_RR_RECV: begin
          ce_d <= 1'b0; we_d <= 1'b0;
          if (master_readdatavalid) begin
            rr_pending[rr_done_count] <= master_readdata;
            rr_done_count             <= rr_done_count + 8'd1;
            state                     <= S_RR_DRIVE;
          end
        end

        // ----------------------------------------------------------------
        //  TX
        // ----------------------------------------------------------------
        S_TX_WR: begin
          ce_d           <= 1'b1; we_d <= 1'b1;
          core_lsu_addr  <= tx_addr(tx_buf, wcnt);
          core_lsu_wdata <= tx_word;
          if (wcnt + 1 >= tx_total_words) begin
            wcnt  <= '0;
            state <= S_TX_WAIT;
          end else begin
            wcnt  <= wcnt + 8'd1;
            state <= S_TX_WR;
          end
        end

        S_TX_WAIT: begin
          ce_d          <= 1'b1; we_d <= 1'b0;
          core_lsu_addr <= 17'h00810;
          state         <= S_TX_WAIT_DONE;
        end

        S_TX_WAIT_DONE: begin
          if (framing_rdata[31]) begin
            ce_d          <= 1'b1; we_d <= 1'b0;
            core_lsu_addr <= 17'h00810;
            state         <= S_TX_WAIT;
          end else begin
            state <= S_TX_GO;
          end
        end

        S_TX_GO: begin
          ce_d           <= 1'b1; we_d <= 1'b1;
          core_lsu_addr  <= 17'h00810;
          core_lsu_wdata <= {52'd0, tx_buf, tx_pkt_len};
          tx_buf         <= ~tx_buf;
          // Dispatch by tx_return:
          //   0 = back to S_IDLE (heartbeat)
          //   1 = advance buf (most replies)
          //   2 = into DHCP_WAIT_OFFER (after DISCOVER tx)
          //   3 = into DHCP_WAIT_ACK   (after REQUEST tx)
          if (tx_return == 4'd2) begin
            // After DISCOVER tx → wait for OFFER.
            dhcp_deadline_sec  <= uptime_sec + DHCP_TIMEOUT_SEC;
            dhcp_in_wait_ack   <= 1'b0;
            dhcp_offer_pending <= 1'b0;
            dhcp_ack_pending   <= 1'b0;
            state              <= S_DHCP_WAIT_OFFER;
          end else if (tx_return == 4'd3) begin
            // After REQUEST tx → wait for ACK.
            dhcp_deadline_sec  <= uptime_sec + DHCP_TIMEOUT_SEC;
            dhcp_in_wait_ack   <= 1'b1;
            state              <= S_DHCP_WAIT_ACK;
          end else if (tx_return == 4'd1) begin
            state <= S_ADV_BUF;
          end else begin
            state <= S_DONE;
          end
        end

        S_ADV_BUF: begin
          ce_d           <= 1'b1; we_d <= 1'b1;
          core_lsu_addr  <= 17'h00830;
          core_lsu_wdata <= {59'd0, cur_buf + 5'd1};
          frame_type     <= '0;
          state          <= S_DONE;
        end

        S_DONE: begin
          ce_d  <= 1'b0; we_d <= 1'b0;
          if (dhcp_ack_pending) begin
            // ACK received — bind and proceed to normal operation.
            dhcp_ack_pending <= 1'b0;
            fpga_ip          <= dhcp_offered_ip;
            state            <= S_IDLE;
          end else if (dhcp_offer_pending) begin
            // OFFER received — proceed to REQUEST phase.
            dhcp_offer_pending <= 1'b0;
            state              <= S_DHCP_REQUEST_PREP;
          end else if (fpga_ip != 32'd0) begin
            state <= S_IDLE;
          end else begin
            state <= dhcp_in_wait_ack ? S_DHCP_WAIT_ACK : S_DHCP_WAIT_OFFER;
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  // Debug taps
  assign dbg_state       = state;
  assign dbg_frame_type  = frame_type;
  assign dbg_wcnt        = wcnt;
  assign dbg_cur_buf     = cur_buf;
  assign dbg_n_remaining = n_remaining;
  assign dbg_fpga_ip     = fpga_ip;

endmodule

`default_nettype wire
