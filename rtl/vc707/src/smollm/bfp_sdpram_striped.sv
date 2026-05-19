// bfp_sdpram_striped.sv — lane-striped simple-dual-port memory.
//
// Wraps LANES instances of bfp_sdpram so a logical array of
// LOGICAL_DEPTH entries × WIDTH bits can be written one entry per
// cycle AND read LANES entries in parallel.
//
// Layout: entry index `i` lives in lane `i % LANES` at address
// `i / LANES`.  Each lane is its own BRAM tile; writes use a lane
// demux on idx[$clog2(LANES)-1:0], reads of LANES_aligned base
// access all lanes at the same `base/LANES` address.
//
// Use this for arrays whose access pattern is:
//   - Writes: 1 entry/cycle at index `i`.
//   - Reads:  `arr[base + 0 .. base + LANES-1]` consumed in parallel,
//             where `base` is a multiple of LANES.
//
// The bfp_sdpram inner instance contributes the 1-cycle read latency
// the OLD inferred array pattern (`data_r <= arr[idx]`) had — no
// extra register, no lookahead.
//
// Output: rd_data_packed is a wide bus of LANES × WIDTH bits.
// Consumers index it as
//   rd_data_packed[lane*WIDTH +: WIDTH]
// to recover lane `lane`.  Equivalent to the OLD `q_rot_m[base+lane]`
// pattern (rd_addr having been driven from `base/LANES` one cycle earlier).

`default_nettype none

module bfp_sdpram_striped #(
  parameter int LANES         = 16,
  parameter int LOGICAL_DEPTH = 960,                     // entries in the logical array
  parameter int WIDTH         = 16,                      // bits per entry
  parameter int LANE_DEPTH    = (LOGICAL_DEPTH + LANES - 1) / LANES,
  parameter int LANE_AW       = $clog2(LANE_DEPTH > 1 ? LANE_DEPTH : 2),
  parameter int FULL_AW       = $clog2(LOGICAL_DEPTH > 1 ? LOGICAL_DEPTH : 2),
  parameter int LANE_SEL_BITS = $clog2(LANES > 1 ? LANES : 2)
)(
  input  wire                                clk,
  input  wire                                rst,

  // Write port: 1 entry/cycle.  wr_addr is the FULL logical index;
  // lane select and per-lane address are derived internally.
  input  wire                                we,
  input  wire [FULL_AW-1:0]                  wr_addr,
  input  wire [WIDTH-1:0]                    wr_data,

  // Read port: rd_addr is the TILE index (base/LANES), 0..LANE_DEPTH-1.
  // All lanes are read at the same tile-address; rd_data_packed holds
  // LANES entries side-by-side.
  input  wire [LANE_AW-1:0]                  rd_addr,
  output wire [LANES*WIDTH-1:0]              rd_data_packed
);

  // Lane select for writes — low bits of wr_addr.
  wire [LANE_SEL_BITS-1:0] wr_lane = wr_addr[LANE_SEL_BITS-1:0];
  // Within-lane write address — high bits of wr_addr.
  wire [LANE_AW-1:0]       wr_lane_addr = wr_addr[FULL_AW-1:LANE_SEL_BITS];

  genvar lane;
  generate
    for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
      wire we_lane = we && (wr_lane == LANE_SEL_BITS'(lane));
      bfp_sdpram #(.DEPTH(LANE_DEPTH), .WIDTH(WIDTH))
        i_lane (
          .clk     (clk),
          .rst     (rst),
          .we      (we_lane),
          .wr_addr (wr_lane_addr),
          .wr_data (wr_data),
          .rd_addr (rd_addr),
          .rd_data (rd_data_packed[lane*WIDTH +: WIDTH])
        );
    end
  endgenerate

endmodule

`default_nettype wire
