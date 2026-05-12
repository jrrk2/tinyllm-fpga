// weight_tile_cache.sv — ping-pong BRAM tile cache fed from DDR3 via AXI4.
//
// Two 8 KiB BRAM banks (one DDR3 row each). At any moment one bank is
// the "active" read source for the consumer (matvec lanes), and the other
// is being filled by an AXI master from DDR3. When the consumer reaches
// the end of the active bank AND the loader has finished filling the
// inactive bank, the banks swap and the loader is dispatched to fetch
// the next tile from DDR3.
//
// In this initial integration the consumer port is exposed for testing /
// future use; real microgpt-style matvec engines plug into rd_addr/rd_data.
//
// Tile layout assumption: a tile is up to 128 × 64-byte beats = 8 KiB.
// AXI burst length = 128 (BL=128). One AXI transaction = one tile load.
//
// Control / status (single-cycle pulses cross from eth_clk via toggle CDC
// at the top level — driven from the FSM here in ui_clk):
//   start_load   : pulse to dispatch a load of the inactive bank.
//   load_addr    : 30-bit DDR3 byte address (must be 64-byte aligned).
//   busy         : 1 while a load is in flight.
//   load_done    : pulse on rlast.
//   active_bank  : 0 or 1, indicates which bank is the active read source.
//   consumer_swap: pulse to swap active/inactive banks (e.g. when consumer
//                  finishes the current tile). Caller must have a fresh
//                  load already complete on the inactive side, or accept
//                  unloaded data.

`default_nettype none

module weight_tile_cache #(
  parameter int AXI_DATA_WIDTH = 512,
  parameter int AXI_ADDR_WIDTH = 30,
  parameter int AXI_ID_WIDTH   = 5,
  parameter int BURST_LEN_LOG2 = 7,         // 128 beats × 64 B = 8 KiB
  parameter int BANK_ENTRIES   = 128         // = 1 << BURST_LEN_LOG2
)(
  input  wire                       clk,     // ui_clk (200 MHz)
  input  wire                       rst,

  // ------------------------------------------------------------------
  //  Loader control (host-driven through CDC)
  // ------------------------------------------------------------------
  input  wire                       start_load,
  input  wire  [AXI_ADDR_WIDTH-1:0] load_addr,
  output logic                      busy,
  output logic                      load_done,    // 1-cycle pulse on rlast

  // ------------------------------------------------------------------
  //  Consumer port (combinational read of the *active* bank)
  // ------------------------------------------------------------------
  input  wire  [$clog2(BANK_ENTRIES)-1:0]   rd_addr,
  output logic [AXI_DATA_WIDTH-1:0]         rd_data,
  input  wire                               consumer_swap,
  output logic                              active_bank,

  // ------------------------------------------------------------------
  //  AXI4 read master to MIG
  // ------------------------------------------------------------------
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

  input  wire                       m_axi_rvalid,
  output wire                       m_axi_rready,
  input  wire  [AXI_ID_WIDTH-1:0]   m_axi_rid,
  input  wire  [AXI_DATA_WIDTH-1:0] m_axi_rdata,
  input  wire  [1:0]                m_axi_rresp,
  input  wire                       m_axi_rlast
);

  localparam int BEAT_BYTES   = AXI_DATA_WIDTH / 8;
  localparam int BANK_AW      = $clog2(BANK_ENTRIES);

  // Constant AR signals
  assign m_axi_arid    = '0;
  assign m_axi_arlen   = (1 << BURST_LEN_LOG2) - 1;
  assign m_axi_arsize  = 3'd6;          // 64 B per beat
  assign m_axi_arburst = 2'b01;         // INCR
  assign m_axi_arlock  = 1'b0;
  assign m_axi_arcache = 4'b0011;
  assign m_axi_arprot  = 3'b000;
  assign m_axi_arqos   = 4'b0000;
  assign m_axi_rready  = 1'b1;          // always sink

  // ------------------------------------------------------------------
  //  Two ping-pong BRAM banks
  //  Each: 128 × 512-bit words = 8 KiB. Synthesized as 2 RAMB36 each.
  // ------------------------------------------------------------------
  logic [AXI_DATA_WIDTH-1:0] bank0 [0:BANK_ENTRIES-1];
  logic [AXI_DATA_WIDTH-1:0] bank1 [0:BANK_ENTRIES-1];

  // Active bank for consumer, inactive bank for loader.
  // Swap on consumer_swap (when caller signals "done with current tile").
  always_ff @(posedge clk) begin
    if (rst) active_bank <= 1'b0;
    else if (consumer_swap) active_bank <= ~active_bank;
  end

  // Combinational read of the active bank.
  always_comb begin
    rd_data = active_bank ? bank1[rd_addr] : bank0[rd_addr];
  end

  // ------------------------------------------------------------------
  //  Loader: writes the *inactive* bank as AXI beats arrive
  // ------------------------------------------------------------------
  logic [BANK_AW-1:0]  beat_idx;       // counts within the burst
  logic                target_bank;    // = ~active_bank at start of load

  always_ff @(posedge clk) begin
    if (rst) begin
      busy            <= 1'b0;
      load_done       <= 1'b0;
      m_axi_arvalid   <= 1'b0;
      m_axi_araddr    <= '0;
      beat_idx        <= '0;
      target_bank     <= 1'b0;
    end else begin
      load_done <= 1'b0;

      // Issue AR
      if (start_load && !busy) begin
        m_axi_arvalid <= 1'b1;
        m_axi_araddr  <= load_addr;
        target_bank   <= ~active_bank;
        beat_idx      <= '0;
        busy          <= 1'b1;
      end else if (m_axi_arvalid && m_axi_arready) begin
        m_axi_arvalid <= 1'b0;
      end

      // Capture beats into the target bank
      if (m_axi_rvalid) begin
        if (target_bank == 1'b0) bank0[beat_idx] <= m_axi_rdata;
        else                     bank1[beat_idx] <= m_axi_rdata;
        beat_idx <= beat_idx + 1;
        if (m_axi_rlast) begin
          busy      <= 1'b0;
          load_done <= 1'b1;
        end
      end
    end
  end

endmodule

`default_nettype wire
