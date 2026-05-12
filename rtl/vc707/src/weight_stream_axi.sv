// weight_stream_axi.sv — bandwidth-test AXI4 read master for the MIG DDR3.
//
// Sits on the MIG's 512-bit AXI slave at 200 MHz ui_clk and issues
// back-to-back INCR bursts with up to MAX_OUTSTANDING transactions in
// flight. Counts beats consumed and cycles elapsed to measure sustained
// bandwidth.
//
// Control:
//   start_pulse (1-cycle): begin a test of NUM_BURSTS bursts starting at
//                          BASE_ADDR. Latches the parameters.
//   busy / done            self-explanatory.
//
// Counters frozen when busy=0; safe to read across clock domain after
// done has been set (each counter is flopped + held).

`default_nettype none

module weight_stream_axi #(
  parameter int AXI_DATA_WIDTH    = 512,
  parameter int AXI_ADDR_WIDTH    = 30,
  parameter int AXI_ID_WIDTH      = 5,
  parameter int BURST_LEN_LOG2    = 7,        // 128-beat bursts -> 8 KiB (full DDR3 row)
  parameter int MAX_OUTSTANDING   = 16        // deep pipeline; bandwidth-latency product is small
                                              // but extras hide refresh / row-change stalls
)(
  input  wire                       clk,      // ui_clk (200 MHz)
  input  wire                       rst,      // active high

  // Control / status (single-cycle pulses cross from eth_clk via CDC at top)
  input  wire                       start_pulse,
  output logic                      busy,
  output logic                      done,
  input  wire  [31:0]               num_bursts,
  input  wire  [AXI_ADDR_WIDTH-1:0] base_addr,

  output logic [31:0]               bytes_read,
  output logic [31:0]               cycles_elapsed,
  output logic [31:0]               bursts_issued,
  output logic [31:0]               bursts_received,

  // AXI4 read address channel
  output logic                      m_axi_arvalid,
  input  wire                       m_axi_arready,
  output logic [AXI_ID_WIDTH-1:0]   m_axi_arid,
  output logic [AXI_ADDR_WIDTH-1:0] m_axi_araddr,
  output logic [7:0]                m_axi_arlen,
  output logic [2:0]                m_axi_arsize,
  output logic [1:0]                m_axi_arburst,
  output logic                      m_axi_arlock,
  output logic [3:0]                m_axi_arcache,
  output logic [2:0]                m_axi_arprot,
  output logic [3:0]                m_axi_arqos,

  // AXI4 read data channel
  input  wire                       m_axi_rvalid,
  output wire                       m_axi_rready,
  input  wire  [AXI_ID_WIDTH-1:0]   m_axi_rid,
  input  wire  [AXI_DATA_WIDTH-1:0] m_axi_rdata,
  input  wire  [1:0]                m_axi_rresp,
  input  wire                       m_axi_rlast
);

  localparam int BEAT_BYTES   = AXI_DATA_WIDTH / 8;     // 64
  localparam int BURST_BEATS  = 1 << BURST_LEN_LOG2;    // 16
  localparam int BURST_BYTES  = BURST_BEATS * BEAT_BYTES; // 1024

  // Constant AR channel attributes — INCR, no protection, bufferable.
  assign m_axi_arid    = '0;
  assign m_axi_arlen   = BURST_BEATS - 1;       // AXI4 length = beats - 1
  assign m_axi_arsize  = 3'd6;                  // 2^6 = 64 bytes per beat
  assign m_axi_arburst = 2'b01;                 // INCR
  assign m_axi_arlock  = 1'b0;
  assign m_axi_arcache = 4'b0011;               // bufferable
  assign m_axi_arprot  = 3'b000;
  assign m_axi_arqos   = 4'b0000;
  assign m_axi_rready  = 1'b1;                  // always sink

  // Working registers
  logic [4:0]                       outstanding;        // up to 16 in flight
  logic [31:0]                      num_bursts_latched;
  logic [AXI_ADDR_WIDTH-1:0]        cur_addr;

  // Edge detect on start (caller holds it for >= 2 ui_clk cycles)
  logic start_d;

  always_ff @(posedge clk) begin
    if (rst) begin
      busy               <= 1'b0;
      done               <= 1'b0;
      outstanding        <= '0;
      bursts_issued      <= '0;
      bursts_received    <= '0;
      num_bursts_latched <= '0;
      cur_addr           <= '0;
      bytes_read         <= '0;
      cycles_elapsed     <= '0;
      m_axi_arvalid      <= 1'b0;
      m_axi_araddr       <= '0;
      start_d            <= 1'b0;
    end else begin
      start_d <= start_pulse;

      // Cycle counter ticks while we have a run going
      if (busy) cycles_elapsed <= cycles_elapsed + 1;

      // Count beats received (each rvalid handshake = one BEAT_BYTES)
      if (m_axi_rvalid) begin
        bytes_read <= bytes_read + BEAT_BYTES;
        if (m_axi_rlast) begin
          // Burst complete
          bursts_received <= bursts_received + 1;
          // Decrement outstanding only here (one rlast per burst)
          // Suppressed if simultaneously issuing (handled below).
        end
      end

      // Issue AR — keep N in flight, advance address
      if (busy && bursts_issued < num_bursts_latched
              && outstanding < MAX_OUTSTANDING
              && (!m_axi_arvalid || m_axi_arready)) begin
        m_axi_arvalid <= 1'b1;
        m_axi_araddr  <= cur_addr;
        cur_addr      <= cur_addr + BURST_BYTES;
        bursts_issued <= bursts_issued + 1;
      end else if (m_axi_arvalid && m_axi_arready) begin
        m_axi_arvalid <= 1'b0;
      end

      // Maintain `outstanding`: +1 on AR handshake, -1 on rlast
      case ({m_axi_arvalid && m_axi_arready,
             m_axi_rvalid  && m_axi_rlast})
        2'b10: outstanding <= outstanding + 1;
        2'b01: outstanding <= outstanding - 1;
        // 2'b11: net change zero
        default: ;
      endcase

      // Completion: all bursts received
      if (busy && (bursts_received +
                   ((m_axi_rvalid && m_axi_rlast) ? 1 : 0)) >= num_bursts_latched) begin
        busy <= 1'b0;
        done <= 1'b1;
      end

      // Latch start
      if (start_pulse && !start_d && !busy) begin
        busy               <= 1'b1;
        done               <= 1'b0;
        bursts_issued      <= '0;
        bursts_received    <= '0;
        outstanding        <= '0;
        bytes_read         <= '0;
        cycles_elapsed     <= '0;
        cur_addr           <= base_addr;
        num_bursts_latched <= (num_bursts == 0) ? 32'd1 : num_bursts;
      end
    end
  end

endmodule

`default_nettype wire
