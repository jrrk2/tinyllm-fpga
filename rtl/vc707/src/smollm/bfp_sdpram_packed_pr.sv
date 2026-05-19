// bfp_sdpram_packed_pr.sv — packed write, parallel read.
//
// Both write side and read side access LANES contiguous entries at a
// LANES-aligned logical base.  Storage is a single bfp_sdpram word of
// LANES × WIDTH bits per tile (= LOGICAL_DEPTH / LANES tiles), so one
// BRAM access serves all LANES entries.
//
// Use this for arrays whose access pattern is:
//   - Writes: LANES entries packed side-by-side at a LANES-aligned base
//             (caller drives wr_addr_tile = base/LANES).
//   - Reads:  LANES contiguous entries at a LANES-aligned base, consumed
//             as a wide rd_data_packed bus (caller indexes lane*WIDTH +: WIDTH).
//
// Latency: 1 cycle from rd_addr_tile to rd_data_packed.  Matches the OLD
// inferred-LUTRAM async-read latency only after a 1-cycle prefetch — the
// FSM driver must issue rd_addr_tile one cycle BEFORE the first cycle
// that consumes rd_data_packed.
//
// This is the symmetric mirror of bfp_sdpram_striped: striped takes a
// single wr_data + lane-decoded write but presents LANES on read, while
// this wrapper takes LANES-packed write AND presents LANES on read in
// one BRAM tile-row.

`default_nettype none

module bfp_sdpram_packed_pr #(
  parameter int LANES         = 16,
  parameter int LOGICAL_DEPTH = 320,
  parameter int WIDTH         = 16,
  parameter int TILE_DEPTH    = (LOGICAL_DEPTH + LANES - 1) / LANES,
  parameter int TILE_AW       = $clog2(TILE_DEPTH > 1 ? TILE_DEPTH : 2)
)(
  input  wire                                clk,
  input  wire                                rst,

  // Write port — LANES entries packed side-by-side.  Address is the
  // tile index (base/LANES), 0..TILE_DEPTH-1.
  input  wire                                we,
  input  wire [TILE_AW-1:0]                  wr_addr_tile,
  input  wire [LANES*WIDTH-1:0]              wr_data_packed,

  // Read port — same packed shape.  rd_addr_tile is the same flavour
  // (base/LANES).  rd_data_packed appears 1 cycle later.
  input  wire [TILE_AW-1:0]                  rd_addr_tile,
  output wire [LANES*WIDTH-1:0]              rd_data_packed
);

  bfp_sdpram #(.DEPTH(TILE_DEPTH), .WIDTH(LANES*WIDTH))
    i_word_ram (
      .clk     (clk),
      .rst     (rst),
      .we      (we),
      .wr_addr (wr_addr_tile),
      .wr_data (wr_data_packed),
      .rd_addr (rd_addr_tile),
      .rd_data (rd_data_packed)
    );

endmodule

`default_nettype wire
