// ddr_write_master.sv — single-burst AXI4 write master for DDR3.
//
// One AXI write per request: BL=1 (single 64-byte beat at AXI 512-bit
// data width). Driven via a 2-flop "request" toggle from eth_clk; the
// caller holds (write_addr, write_data) stable while toggling
// write_req. Master flips write_ack when the B response arrives.
//
// Designed for the SmolLM2 weight-upload flow: ~2M frames over the
// course of a weights load, but still simple enough for first-cut.
// Pipelining (multiple in-flight writes) can be added later.

`default_nettype none

module ddr_write_master #(
  parameter int AXI_DATA_WIDTH = 512,
  parameter int AXI_ADDR_WIDTH = 30,
  parameter int AXI_ID_WIDTH   = 5
)(
  input  wire                       clk,        // ui_clk (200 MHz)
  input  wire                       rst,

  // Request side (eth_clk → ui_clk via 2FF sync of write_req)
  input  wire                       write_req,  // toggle: signal new write
  output logic                      write_ack,  // toggle: matches when done
  input  wire  [AXI_ADDR_WIDTH-1:0] write_addr, // stable when req toggles
  input  wire  [AXI_DATA_WIDTH-1:0] write_data, // stable when req toggles

  // AXI4 write channels
  output logic                       m_axi_awvalid,
  input  wire                        m_axi_awready,
  output logic [AXI_ID_WIDTH-1:0]    m_axi_awid,
  output logic [AXI_ADDR_WIDTH-1:0]  m_axi_awaddr,
  output logic [7:0]                 m_axi_awlen,
  output logic [2:0]                 m_axi_awsize,
  output logic [1:0]                 m_axi_awburst,
  output logic                       m_axi_awlock,
  output logic [3:0]                 m_axi_awcache,
  output logic [2:0]                 m_axi_awprot,
  output logic [3:0]                 m_axi_awqos,

  output logic                       m_axi_wvalid,
  input  wire                        m_axi_wready,
  output logic [AXI_DATA_WIDTH-1:0]  m_axi_wdata,
  output logic [AXI_DATA_WIDTH/8-1:0]m_axi_wstrb,
  output logic                       m_axi_wlast,

  input  wire                        m_axi_bvalid,
  output wire                        m_axi_bready,
  input  wire  [AXI_ID_WIDTH-1:0]    m_axi_bid,
  input  wire  [1:0]                 m_axi_bresp
);

  // Constant AW signals
  assign m_axi_awid    = '0;
  assign m_axi_awlen   = 8'd0;        // BL = 1 (single beat)
  assign m_axi_awsize  = 3'd6;        // 64 bytes per beat
  assign m_axi_awburst = 2'b01;
  assign m_axi_awlock  = 1'b0;
  assign m_axi_awcache = 4'b0011;
  assign m_axi_awprot  = 3'b000;
  assign m_axi_awqos   = 4'b0000;
  assign m_axi_bready  = 1'b1;
  assign m_axi_wstrb   = '1;          // all 64 bytes

  // 2FF sync of the request toggle
  logic req_sync0, req_sync1, req_seen;

  typedef enum logic [1:0] { S_IDLE, S_WRITE, S_WAIT_B } state_t;
  state_t state;

  always_ff @(posedge clk) begin
    if (rst) begin
      req_sync0     <= 1'b0;
      req_sync1     <= 1'b0;
      req_seen      <= 1'b0;
      write_ack     <= 1'b0;
      m_axi_awvalid <= 1'b0;
      m_axi_awaddr  <= '0;
      m_axi_wvalid  <= 1'b0;
      m_axi_wdata   <= '0;
      m_axi_wlast   <= 1'b0;
      state         <= S_IDLE;
    end else begin
      req_sync0 <= write_req;
      req_sync1 <= req_sync0;

      case (state)
        S_IDLE: begin
          if (req_sync1 != req_seen) begin
            m_axi_awvalid <= 1'b1;
            m_axi_awaddr  <= write_addr;
            m_axi_wvalid  <= 1'b1;
            m_axi_wdata   <= write_data;
            m_axi_wlast   <= 1'b1;     // BL=1 — only beat
            state         <= S_WRITE;
          end
        end

        // AW and W are independent channels — handle them concurrently.
        // The MIG can ack either first.  Drop each *valid* the cycle its
        // handshake fires; transition once both have been accepted.
        S_WRITE: begin
          if (m_axi_awvalid && m_axi_awready) m_axi_awvalid <= 1'b0;
          if (m_axi_wvalid  && m_axi_wready ) begin
            m_axi_wvalid <= 1'b0;
            m_axi_wlast  <= 1'b0;
          end
          if (((m_axi_awvalid && m_axi_awready) || !m_axi_awvalid)
           && ((m_axi_wvalid  && m_axi_wready ) || !m_axi_wvalid)) begin
            state <= S_WAIT_B;
          end
        end

        S_WAIT_B: begin
          if (m_axi_bvalid) begin
            req_seen  <= req_sync1;
            write_ack <= req_sync1;
            state     <= S_IDLE;
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
