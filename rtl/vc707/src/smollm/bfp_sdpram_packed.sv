// bfp_sdpram_packed.sv — single BRAM word per LANES-chunk, the
// symmetric mirror of bfp_sdpram_striped.
//
// Use this for arrays whose access pattern is:
//   - Writes: LANES entries written *together* per cycle at LANES-aligned
//             base.  Natural for matvec-output requant for-loops:
//               for (ii=0..LANES-1) arr[base+ii] <= requant_mant(...);
//             Caller packs the LANES requant outputs into one wr_data bus.
//   - Reads:  single element at logical address `i`.  The wrapper splits
//             i into tile-addr (high bits) + lane-sel (low bits),
//             reads the wide BRAM word, registers the lane select to
//             align with the BRAM's 1-cycle latency, and muxes the
//             corresponding entry out.  Net read latency: 1 cycle —
//             matches the OLD `data_r <= arr[i]` pattern.
//
// Storage: one bfp_sdpram, DEPTH = LOGICAL_DEPTH/LANES, WIDTH = LANES*WIDTH.
// Total bit storage and BRAM tile count is the same as a flat LUTRAM of
// the original shape, but packed into BRAM primitives instead of LUTs.

`default_nettype none

module bfp_sdpram_packed #(
  parameter int LANES         = 16,
  parameter int LOGICAL_DEPTH = 960,
  parameter int WIDTH         = 16,
  parameter int TILE_DEPTH    = (LOGICAL_DEPTH + LANES - 1) / LANES,
  parameter int TILE_AW       = $clog2(TILE_DEPTH > 1 ? TILE_DEPTH : 2),
  parameter int FULL_AW       = $clog2(LOGICAL_DEPTH > 1 ? LOGICAL_DEPTH : 2),
  parameter int LANE_SEL_BITS = $clog2(LANES > 1 ? LANES : 2)
)(
  input  wire                                clk,
  input  wire                                rst,

  // Write port — LANES entries packed side-by-side.  Address is the
  // TILE index (= base/LANES), 0..TILE_DEPTH-1.  Caller must ensure
  // `base` is LANES-aligned (true for the matvec/requant call sites
  // that have `chunk*LANES + ii` or `av_row_base + chunk*LANES + ii`).
  input  wire                                we,
  input  wire [TILE_AW-1:0]                  wr_addr_tile,
  input  wire [LANES*WIDTH-1:0]              wr_data_packed,

  // Read port — single-element logical address.  rd_data appears
  // 1 cycle after rd_addr (same edge timing as `data_r <= arr[i]`).
  input  wire [FULL_AW-1:0]                  rd_addr,
  output wire [WIDTH-1:0]                    rd_data
);

  wire [LANE_SEL_BITS-1:0] rd_lane = rd_addr[LANE_SEL_BITS-1:0];
  wire [TILE_AW-1:0]       rd_tile = rd_addr[FULL_AW-1:LANE_SEL_BITS];

  // Lane select must lag with BRAM latency — register at the read clock
  // so the output mux selects the right slice of rd_data_word.
  reg [LANE_SEL_BITS-1:0] rd_lane_reg;
  always_ff @(posedge clk) rd_lane_reg <= rd_lane;

  wire [LANES*WIDTH-1:0] rd_data_word;
  bfp_sdpram #(.DEPTH(TILE_DEPTH), .WIDTH(LANES*WIDTH))
    i_word_ram (
      .clk     (clk),
      .rst     (rst),
      .we      (we),
      .wr_addr (wr_addr_tile),
      .wr_data (wr_data_packed),
      .rd_addr (rd_tile),
      .rd_data (rd_data_word)
    );

  assign rd_data = rd_data_word[rd_lane_reg * WIDTH +: WIDTH];

endmodule

`default_nettype wire
