// bfp_sdpram.sv — simple-dual-port BRAM built from explicit RAMB36E1
// primitives in a generate loop.  No inference, no XPM macro: each
// tile is named directly, address-decoded by hand.
//
// Tile geometry: RAMB36E1 in TDP mode, aspect chosen from WIDTH so narrow
// memories don't waste half of every tile:
//   WIDTH <= 8   4096 deep x 9  (8 data bits)
//   WIDTH <= 16  2048 deep x 18 (16 data bits)
//   otherwise    1024 deep x 36 (32 data bits per column)
// Port A is write-only (parity tied 0), port B is read-only.  For
// (DEPTH=23040, WIDTH=256) the array is 23 rows x 8 cols = 184 tiles; the
// 16-bit K-mantissa KV cache needs half the tiles of the old 1024x32-only
// geometry, which is what lets smollm360 run at MAX_CTX=64.
//
// Latency contract: 1 cycle from rd_addr to rd_data, matching the
// previous XPM_MEMORY_SDPRAM wrapper.  The BRAM core register provides
// that cycle; the output mux is combinational on the registered tile
// outputs and selected by a 1-cycle-delayed rd_row.
//
// Unused address LSBs are tied HIGH (UG473 rule for sub-36-bit-aligned
// modes — though at TDP-36 width=36 the low 5 bits ARE unused and per
// UG473 must be 1'b1).  ADDRARDADDR[15] is the cascade-extension bit
// and is tied LOW since RAM_EXTENSION_A/B = "NONE".

`default_nettype none

module bfp_sdpram #(
  parameter int DEPTH        = 23040,
  parameter int WIDTH        = 256,
  parameter int ADDR_WIDTH   = $clog2(DEPTH > 1 ? DEPTH : 2),
  parameter     PRIMITIVE    = "block"   // accepted for API back-compat, ignored
)(
  input  wire                    clk,
  input  wire                    rst,
  input  wire                    we,
  input  wire [ADDR_WIDTH-1:0]   wr_addr,
  input  wire [WIDTH-1:0]        wr_data,
  input  wire [ADDR_WIDTH-1:0]   rd_addr,
  output wire [WIDTH-1:0]        rd_data
);

`ifdef VERILATOR
  // Behavioural simple-dual-port BRAM for Verilator (RAMB36E1 is
  // Vivado-only).  Same 1-cycle read latency the primitive gives.
  logic [WIDTH-1:0] mem [0:DEPTH-1];
  logic [WIDTH-1:0] rd_data_r;
  always_ff @(posedge clk) begin
    if (we) mem[wr_addr] <= wr_data;
    rd_data_r <= mem[rd_addr];
  end
  assign rd_data = rd_data_r;
`else
  localparam int TILE_WIDTH = (WIDTH <= 8) ? 8 : (WIDTH <= 16) ? 16 : 32;
  localparam int TILE_DEPTH = 32768 / TILE_WIDTH;          // 4096 / 2048 / 1024
  localparam int OFF_BITS   = $clog2(TILE_DEPTH);          // 12 / 11 / 10
  localparam int LSB_TIE    = 15 - OFF_BITS;               // 3 / 4 / 5 unused LSBs
  localparam int PRIM_W     = TILE_WIDTH + TILE_WIDTH / 8; // 9 / 18 / 36
  localparam int N_ROWS     = (DEPTH + TILE_DEPTH - 1) / TILE_DEPTH;
  localparam int N_COLS     = (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH;
  localparam int ROW_BITS   = (N_ROWS > 1) ? $clog2(N_ROWS) : 1;

  // Address split: low OFF_BITS = within-tile row; remaining MSBs = tile row select.
  // When DEPTH ≤ TILE_DEPTH (N_ROWS=1) there are no row-select bits — the
  // part-select [ADDR_WIDTH-1:OFF_BITS] would be malformed (e.g. [9:10] for D=960
  // → ADDR_WIDTH=10).  Generate-if elides the slice for the single-row case.
  wire [OFF_BITS-1:0]   rd_off = OFF_BITS'(rd_addr);
  wire [OFF_BITS-1:0]   wr_off = OFF_BITS'(wr_addr);
  wire [ROW_BITS-1:0]   rd_row, wr_row;
  if (N_ROWS > 1) begin : g_row_select
    assign rd_row = rd_addr[ADDR_WIDTH-1:OFF_BITS];
    assign wr_row = wr_addr[ADDR_WIDTH-1:OFF_BITS];
  end else begin : g_no_row_select
    assign rd_row = '0;
    assign wr_row = '0;
  end

  // 1-cycle delay on rd_row so the output mux selects the right tile
  // the cycle the BRAM core register presents its data.
  reg [ROW_BITS-1:0] rd_row_d;
  always_ff @(posedge clk) rd_row_d <= rd_row;

  // Pad write data with zeros up to N_COLS*TILE_WIDTH (when WIDTH isn't
  // a clean multiple of the tile width).
  wire [N_COLS*TILE_WIDTH-1:0] wr_data_pad = {{(N_COLS*TILE_WIDTH-WIDTH){1'b0}}, wr_data};

  // Per-row reassembled data outputs (one entry per N_ROWS row group).
  wire [N_COLS*TILE_WIDTH-1:0] row_dout [N_ROWS];

  genvar r, c;
  generate
    for (r = 0; r < N_ROWS; r = r + 1) begin : g_row
      wire row_we = we && (wr_row == ROW_BITS'(r));

      for (c = 0; c < N_COLS; c = c + 1) begin : g_col
        wire [31:0] tile_dout32;
        wire [31:0] tile_din32 = 32'(wr_data_pad[c*TILE_WIDTH +: TILE_WIDTH]);
        assign row_dout[r][c*TILE_WIDTH +: TILE_WIDTH] = tile_dout32[TILE_WIDTH-1:0];

        RAMB36E1 #(
          .RAM_MODE                  ("TDP"),
          .READ_WIDTH_A              (PRIM_W),
          .WRITE_WIDTH_A             (PRIM_W),
          .READ_WIDTH_B              (PRIM_W),
          .WRITE_WIDTH_B             (PRIM_W),
          .WRITE_MODE_A              ("WRITE_FIRST"),
          .WRITE_MODE_B              ("WRITE_FIRST"),
          .DOA_REG                   (0),
          .DOB_REG                   (0),
          .EN_ECC_READ               ("FALSE"),
          .EN_ECC_WRITE              ("FALSE"),
          .RAM_EXTENSION_A           ("NONE"),
          .RAM_EXTENSION_B           ("NONE"),
          .RDADDR_COLLISION_HWCONFIG ("DELAYED_WRITE"),
          .SIM_COLLISION_CHECK       ("ALL"),
          .SIM_DEVICE                ("7SERIES"),
          .INIT_A                    (36'h0),
          .INIT_B                    (36'h0),
          .SRVAL_A                   (36'h0),
          .SRVAL_B                   (36'h0)
        ) i_ramb (
          // Port A — write only.  ADDR[14:LSB_TIE] used, the LSBs below
          // the port width tied high (UG473), ADDR[15] low (no cascade).
          .ADDRARDADDR  ({1'b0, wr_off, {LSB_TIE{1'b1}}}),
          .CLKARDCLK    (clk),
          .DIADI        (tile_din32),
          .DIPADIP      (4'b0000),
          .ENARDEN      (row_we),
          .REGCEAREGCE  (1'b0),
          .RSTRAMARSTRAM(1'b0),
          .RSTREGARSTREG(1'b0),
          .WEA          (4'b1111),
          .DOADO        (),
          .DOPADOP      (),
          // Port B — read only.  Read address always presented, ENB
          // always high (data is multiplexed downstream by rd_row_d).
          .ADDRBWRADDR  ({1'b0, rd_off, {LSB_TIE{1'b1}}}),
          .CLKBWRCLK    (clk),
          .DIBDI        (32'h0),
          .DIPBDIP      (4'b0000),
          .ENBWREN      (1'b1),
          .REGCEB       (1'b0),
          .RSTRAMB      (1'b0),
          .RSTREGB      (1'b0),
          .WEBWE        (8'h00),
          .DOBDO        (tile_dout32),
          .DOPBDOP      (),
          // No cascade extension.
          .CASCADEINA   (1'b0),
          .CASCADEINB   (1'b0),
          .INJECTDBITERR(1'b0),
          .INJECTSBITERR(1'b0)
        );
      end
    end
  endgenerate

  // Single combinational mux on registered tile outputs, selected by
  // the 1-cycle-delayed rd_row.  Total latency = 1 cycle.
  assign rd_data = row_dout[rd_row_d][WIDTH-1:0];
`endif

endmodule

`default_nettype wire
