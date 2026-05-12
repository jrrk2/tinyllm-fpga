// mock_axi_slave.sv — minimal AXI4 read slave for sim-time validation of
// weight_streamer.sv.  Serves a 512-bit-per-beat read interface backed by
// an in-memory array of 128-bit entries.  Burst length / size are accepted
// as-given (we honour AXI_BURST_INCR with arsize=6 → 64-byte beats).
//
// Initialisation pattern: each 128-bit entry equals its linear index
// (mem[i] = i).  This makes the testbench self-checking — for any read
// at byte-address A, the high 4-bit nibble of the 128-bit word is the
// 128-bit-entry-index = A/16.
//
// Memory size MEM_ENTRIES is sized for the test (default 8192 × 16 B = 128 KiB).

`default_nettype none

module mock_axi_slave #(
  parameter int AXI_DATA_WIDTH = 512,
  parameter int AXI_ADDR_WIDTH = 30,
  parameter int AXI_ID_WIDTH   = 5,
  parameter int MEM_ENTRIES    = 8192,      // # of 128-bit entries
  parameter     INIT_HEX_FILE  = ""         // if non-empty, $readmemh from here
)(
  input  wire                       clk,
  input  wire                       rst,

  // AR
  input  wire                       m_axi_arvalid,
  output logic                      m_axi_arready,
  input  wire [AXI_ID_WIDTH-1:0]    m_axi_arid,
  input  wire [AXI_ADDR_WIDTH-1:0]  m_axi_araddr,
  input  wire [7:0]                 m_axi_arlen,
  input  wire [2:0]                 m_axi_arsize,
  input  wire [1:0]                 m_axi_arburst,
  input  wire                       m_axi_arlock,
  input  wire [3:0]                 m_axi_arcache,
  input  wire [2:0]                 m_axi_arprot,
  input  wire [3:0]                 m_axi_arqos,

  // R
  output logic                      m_axi_rvalid,
  input  wire                       m_axi_rready,
  output logic [AXI_ID_WIDTH-1:0]   m_axi_rid,
  output logic [AXI_DATA_WIDTH-1:0] m_axi_rdata,
  output logic [1:0]                m_axi_rresp,
  output logic                      m_axi_rlast
);

  // 128-bit-entry-indexed memory.  Default: mem[i] = i for self-test.
  // Override via INIT_HEX_FILE to populate from a packed-weight image.
  logic [127:0] mem [0:MEM_ENTRIES-1];

  initial begin
    if (INIT_HEX_FILE != "") begin
      // Pre-zero — $readmemh leaves un-initialised entries unspecified
      // and we want determinism for areas the test never reads.
      for (int i = 0; i < MEM_ENTRIES; i++) mem[i] = '0;
      $readmemh(INIT_HEX_FILE, mem);
    end else begin
      for (int i = 0; i < MEM_ENTRIES; i++) mem[i] = 128'(i);
    end
  end

  // ------------------------------------------------------------------
  //  FSM
  // ------------------------------------------------------------------
  typedef enum logic [1:0] {AR_IDLE, R_BURST} st_t;
  st_t st;

  logic [AXI_ID_WIDTH-1:0]   cur_id;
  logic [AXI_ADDR_WIDTH-1:0] cur_addr;     // byte address, 64-byte aligned
  logic [7:0]                cur_beat;
  logic [7:0]                total_beats;

  // 128-bit entry index of the next beat's first lane.
  // (cur_addr/16) + cur_beat*4
  logic [AXI_ADDR_WIDTH-1:0] beat_entry_base;
  always_comb beat_entry_base = (cur_addr >> 4) + ({22'd0, cur_beat} << 2);

  // arready: ready to accept a request whenever idle
  always_comb m_axi_arready = (st == AR_IDLE);

  always_ff @(posedge clk) begin
    if (rst) begin
      st           <= AR_IDLE;
      m_axi_rvalid <= 1'b0;
      m_axi_rid    <= '0;
      m_axi_rdata  <= '0;
      m_axi_rresp  <= 2'b00;
      m_axi_rlast  <= 1'b0;
      cur_id       <= '0;
      cur_addr     <= '0;
      cur_beat     <= '0;
      total_beats  <= '0;
    end else begin
      case (st)
        AR_IDLE: begin
          m_axi_rvalid <= 1'b0;
          m_axi_rlast  <= 1'b0;
          if (m_axi_arvalid) begin
            cur_id      <= m_axi_arid;
            cur_addr    <= m_axi_araddr;
            cur_beat    <= '0;
            total_beats <= m_axi_arlen + 8'd1;
            st          <= R_BURST;
          end
        end
        R_BURST: begin
          if (!m_axi_rvalid || m_axi_rready) begin
            // 4 × 128-bit entries per 512-bit beat (lane 0 in low 128 bits)
            m_axi_rdata <= {
              mem[beat_entry_base + 3],
              mem[beat_entry_base + 2],
              mem[beat_entry_base + 1],
              mem[beat_entry_base + 0]
            };
            m_axi_rid    <= cur_id;
            m_axi_rresp  <= 2'b00;
            m_axi_rvalid <= 1'b1;
            m_axi_rlast  <= (cur_beat == total_beats - 8'd1);
            cur_beat     <= cur_beat + 8'd1;
            if (cur_beat == total_beats - 8'd1) st <= AR_IDLE;
          end
        end
        default: st <= AR_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
