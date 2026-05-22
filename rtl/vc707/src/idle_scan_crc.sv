// idle_scan_crc.sv — standalone AXI read master that streams a DDR3 region and
// rolling-hashes it, so the SoC can VERIFY an upload without running inference.
//
// The engine's own weight hash (0x04A) is taken over the *unpacked* BFP weight
// bus as the matvec engine consumes it, so it can only be produced by a full
// run and is not a function of the raw bytes.  This scanner instead hashes the
// RAW 512-bit DDR3 beats over an arbitrary [base, base+len*64) region, using the
// same rotate-xor style so a host can compute the expected value:
//
//   crc = 0xFFFFFFFF
//   for each 64-byte beat:
//       fold = XOR of the sixteen 32-bit words of the beat
//       crc  = rotl(crc,1) ^ fold
//
// One BL=1 (single 64-byte) read per beat — simple and correct; an idle CRC is
// not latency-critical.  Runs in ui_clk alongside the MIG AXI bus; the AR/R
// channels are muxed in the top against the engine's weight reader (the scan is
// only triggered when the engine is idle, so there is no real contention).

`default_nettype none

module idle_scan_crc #(
  parameter integer AXI_ADDR_WIDTH = 30,
  parameter integer AXI_DATA_WIDTH = 512,
  parameter integer AXI_ID_WIDTH   = 5
)(
  input  wire                          clk,        // ui_clk (200 MHz)
  input  wire                          rst,
  // Control — already in ui_clk domain (regmap CDCs base/len; trig is a pulse).
  input  wire  [AXI_ADDR_WIDTH-1:0]    base,       // 64-byte-aligned byte addr
  input  wire  [31:0]                  len,        // number of 512-bit beats
  input  wire                          trig,       // 1-cycle start pulse
  output logic                         busy,
  output logic                         done,       // sticky high until next trig
  output logic [31:0]                  crc,
  // AXI read address channel
  output logic                         m_axi_arvalid,
  input  wire                          m_axi_arready,
  output logic [AXI_ID_WIDTH-1:0]      m_axi_arid,
  output logic [AXI_ADDR_WIDTH-1:0]    m_axi_araddr,
  output logic [7:0]                   m_axi_arlen,
  output logic [2:0]                   m_axi_arsize,
  output logic [1:0]                   m_axi_arburst,
  output logic                         m_axi_arlock,
  output logic [3:0]                   m_axi_arcache,
  output logic [2:0]                   m_axi_arprot,
  output logic [3:0]                   m_axi_arqos,
  // AXI read data channel
  input  wire                          m_axi_rvalid,
  output logic                         m_axi_rready,
  input  wire  [AXI_ID_WIDTH-1:0]      m_axi_rid,
  input  wire  [AXI_DATA_WIDTH-1:0]    m_axi_rdata,
  input  wire  [1:0]                   m_axi_rresp,
  input  wire                          m_axi_rlast
);

  // Fixed AR attributes: single 64-byte beat per request.
  assign m_axi_arid    = '0;
  assign m_axi_arlen   = 8'd0;        // BL = 1
  assign m_axi_arsize  = 3'd6;        // 2^6 = 64 bytes per beat
  assign m_axi_arburst = 2'b01;       // INCR
  assign m_axi_arlock  = 1'b0;
  assign m_axi_arcache = 4'b0011;
  assign m_axi_arprot  = 3'b000;
  assign m_axi_arqos   = 4'b0000;

  // Fold a 512-bit beat down to 32 bits by XORing its sixteen words.
  function automatic [31:0] beat_fold(input [AXI_DATA_WIDTH-1:0] d);
    integer i;
    reg [31:0] x;
    begin
      x = 32'h0;
      for (i = 0; i < AXI_DATA_WIDTH/32; i = i + 1)
        x = x ^ d[i*32 +: 32];
      beat_fold = x;
    end
  endfunction

  typedef enum logic [1:0] { S_IDLE, S_AR, S_R } st_t;
  st_t st;
  logic [AXI_ADDR_WIDTH-1:0] cur_addr;
  logic [31:0]               rem;     // beats remaining

  always_ff @(posedge clk) begin
    if (rst) begin
      st            <= S_IDLE;
      busy          <= 1'b0;
      done          <= 1'b0;
      crc           <= 32'hFFFFFFFF;
      m_axi_arvalid <= 1'b0;
      m_axi_araddr  <= '0;
      m_axi_rready  <= 1'b0;
      cur_addr      <= '0;
      rem           <= 32'h0;
    end else begin
      case (st)
        S_IDLE: begin
          if (trig) begin
            done <= 1'b0;
            crc  <= 32'hFFFFFFFF;
            if (len == 32'h0) begin
              busy <= 1'b0;
              done <= 1'b1;                 // empty scan completes immediately
            end else begin
              busy          <= 1'b1;
              cur_addr      <= base;
              rem           <= len;
              m_axi_araddr  <= base;
              m_axi_arvalid <= 1'b1;
              st            <= S_AR;
            end
          end
        end

        S_AR: begin
          if (m_axi_arvalid && m_axi_arready) begin
            m_axi_arvalid <= 1'b0;
            m_axi_rready  <= 1'b1;
            st            <= S_R;
          end
        end

        S_R: begin
          if (m_axi_rvalid && m_axi_rready) begin
            crc          <= {crc[30:0], crc[31]} ^ beat_fold(m_axi_rdata);
            m_axi_rready <= 1'b0;
            if (rem == 32'h1) begin
              busy <= 1'b0;
              done <= 1'b1;
              st   <= S_IDLE;
            end else begin
              rem           <= rem - 32'h1;
              cur_addr      <= cur_addr + AXI_ADDR_WIDTH'(64);
              m_axi_araddr  <= cur_addr + AXI_ADDR_WIDTH'(64);
              m_axi_arvalid <= 1'b1;
              st            <= S_AR;
            end
          end
        end

        default: st <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
