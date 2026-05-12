// weight_streamer_mt.sv — multi-tile DDR3-backed weight feeder, dual-clock.
//
// Loader FSM + AXI master live on `clk_axi` (= MIG's ui_clk on FPGA, 200 MHz).
// Consumer port (rd_addr → weight_data) lives on `clk_core` (= the smollm_layer's
// 50 MHz clock).  Each bank is a true dual-port BRAM: AXI port writes, consumer
// port reads.  Control CDC:
//   - load_req (core→axi): toggle synchroniser; payload (matrix_base, chunk_idx,
//     in_dim) held stable on the core side until the load completes.
//   - ready (axi→core): 2FF level synchroniser.
//
// For Verilator sim we tie clk_axi = clk_core externally (single-clock
// effectively); the toggle/2FF logic still works correctly under that.

`default_nettype none

module weight_streamer_mt #(
  parameter int AXI_DATA_WIDTH = 512,
  parameter int AXI_ADDR_WIDTH = 30,
  parameter int AXI_ID_WIDTH   = 5,
  parameter int IN_DIM_BITS    = 11,
  parameter int CHUNK_BITS     = 7,
  parameter int MAX_TILES      = 3,
  parameter int TILE_ENTRIES   = 512,
  parameter int BURST_LEN_LOG2 = 7
)(
  // ------------------------------------------------------------------
  //  Consumer side (clk_core)
  // ------------------------------------------------------------------
  input  wire                       clk_core,
  input  wire                       rst_core,
  input  wire [AXI_ADDR_WIDTH-1:0]  matrix_base,
  input  wire [CHUNK_BITS-1:0]      chunk_idx,
  input  wire [IN_DIM_BITS-1:0]     in_dim,
  input  wire                       load_req,        // 1-cycle pulse on clk_core
  output logic                      ready,
  output logic                      busy,
  input  wire [IN_DIM_BITS-1:0]     rd_addr,
  output logic [127:0]              weight_data,

  // ------------------------------------------------------------------
  //  AXI side (clk_axi) — read master to MIG
  // ------------------------------------------------------------------
  input  wire                       clk_axi,
  input  wire                       rst_axi,
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
  input  wire                       m_axi_rlast,

  // Diagnostic: latched on the FIRST AXI handshake/beat after rst_axi.
  output logic [AXI_ADDR_WIDTH-1:0]  dbg_first_araddr,
  output logic [AXI_DATA_WIDTH-1:0]  dbg_first_rdata,
  output logic                       dbg_first_ar_seen,
  output logic                       dbg_first_r_seen,
  // ILA-AXI taps (clk_axi domain, named).
  output wire  [2:0]                 ila_ws_state,
  output wire                        ila_start_load_axi,
  output wire  [1:0]                 ila_tile_idx,
  output wire  [6:0]                 ila_beat_idx
);

  localparam int BANK_ENTRIES = 1 << BURST_LEN_LOG2;
  localparam int BANK_AW      = BURST_LEN_LOG2;
  localparam int TILE_BYTES   = TILE_ENTRIES * 16;
  localparam int TILE_AW      = $clog2(TILE_ENTRIES);
  localparam int TILES_AW     = $clog2(MAX_TILES + 1);

  // Constant AR signals
  assign m_axi_arid    = '0;
  assign m_axi_arlen   = (1 << BURST_LEN_LOG2) - 1;
  assign m_axi_arsize  = 3'd6;
  assign m_axi_arburst = 2'b01;
  assign m_axi_arlock  = 1'b0;
  assign m_axi_arcache = 4'b0011;
  assign m_axi_arprot  = 3'b000;
  assign m_axi_arqos   = 4'b0000;
  assign m_axi_rready  = 1'b1;

  // ------------------------------------------------------------------
  //  Banks: dual-clock dual-port BRAM
  //    port A (write, clk_axi):  bank[tile_idx][beat_idx] <= rdata
  //    port B (read,  clk_core): registered read for the consumer
  //  Vivado infers RAMB36 in TDP mode with separate port clocks.
  // ------------------------------------------------------------------
  // We split into MAX_TILES separate single-bank arrays so each gets one BRAM.
  // (2D unpacked arrays with dual-port inference can be brittle; one BRAM
  //  per "tile" is the cleanest pattern.)
  logic [AXI_DATA_WIDTH-1:0] bank0 [0:BANK_ENTRIES-1];
  logic [AXI_DATA_WIDTH-1:0] bank1 [0:BANK_ENTRIES-1];
  logic [AXI_DATA_WIDTH-1:0] bank2 [0:BANK_ENTRIES-1];

  // ------------------------------------------------------------------
  //  Core-side: capture (matrix_base, chunk_idx, in_dim) on load_req,
  //  and toggle a 1-bit token to signal the AXI side.
  // ------------------------------------------------------------------
  logic                      core_tog;
  logic [AXI_ADDR_WIDTH-1:0] core_mb;
  logic [CHUNK_BITS-1:0]     core_ck;
  logic [IN_DIM_BITS-1:0]    core_id;

  always_ff @(posedge clk_core) begin
    if (rst_core) begin
      core_tog <= 1'b0;
      core_mb  <= '0;
      core_ck  <= '0;
      core_id  <= '0;
    end else if (load_req) begin
      core_tog <= ~core_tog;
      core_mb  <= matrix_base;
      core_ck  <= chunk_idx;
      core_id  <= in_dim;
    end
  end

  // ------------------------------------------------------------------
  //  AXI-side: 2FF-sync core_tog and the payload bus.  Edge-detect the
  //  toggle to produce a one-cycle start_load_axi pulse.  Payload is
  //  captured at the same time — assumed stable through the AXI burst
  //  (the core side won't pulse load_req again until ready returns).
  // ------------------------------------------------------------------
  logic [1:0]                tog_sync_axi;
  logic                      tog_seen_axi;
  logic                      start_load_axi;
  logic [AXI_ADDR_WIDTH-1:0] axi_mb;
  logic [CHUNK_BITS-1:0]     axi_ck;
  logic [IN_DIM_BITS-1:0]    axi_id;

  always_ff @(posedge clk_axi) begin
    if (rst_axi) begin
      tog_sync_axi   <= '0;
      tog_seen_axi   <= 1'b0;
      start_load_axi <= 1'b0;
      axi_mb <= '0; axi_ck <= '0; axi_id <= '0;
    end else begin
      tog_sync_axi   <= {tog_sync_axi[0], core_tog};
      start_load_axi <= 1'b0;
      if (tog_sync_axi[1] != tog_seen_axi) begin
        tog_seen_axi   <= tog_sync_axi[1];
        start_load_axi <= 1'b1;
        // core_mb/ck/id are synchronously stable from the toggle's
        // perspective; multi-bit MUX inputs may glitch but only matter
        // at the instant they're sampled here, which is after the
        // toggle has propagated through 2 FFs.
        axi_mb <= core_mb;
        axi_ck <= core_ck;
        axi_id <= core_id;
      end
    end
  end

  // ------------------------------------------------------------------
  //  Loader FSM (clk_axi)
  // ------------------------------------------------------------------
  typedef enum logic [2:0] {
    WS_IDLE, WS_LOAD_AR, WS_LOAD_R, WS_NEXT_TILE, WS_READY
  } ws_state_t;
  ws_state_t ws_state;

  logic [TILES_AW-1:0]       num_tiles;
  logic [TILES_AW-1:0]       tile_idx;
  logic [BANK_AW-1:0]        beat_idx;
  logic [AXI_ADDR_WIDTH-1:0] base_addr;
  logic [AXI_ADDR_WIDTH-1:0] chunk_byte_off;
  logic [AXI_ADDR_WIDTH-1:0] tile_addr;
  logic                      ready_axi;

  always_comb begin
    chunk_byte_off = (AXI_ADDR_WIDTH'(axi_ck)) *
                     (AXI_ADDR_WIDTH'(axi_id)) *
                     AXI_ADDR_WIDTH'(16);
    tile_addr      = base_addr +
                     (AXI_ADDR_WIDTH'(tile_idx) << $clog2(TILE_BYTES));
  end

  function automatic [TILES_AW-1:0] tiles_for_dim(input [IN_DIM_BITS-1:0] dim);
    tiles_for_dim = TILES_AW'((dim + TILE_ENTRIES - 1) >> TILE_AW);
  endfunction

  always_ff @(posedge clk_axi) begin
    if (rst_axi) begin
      ws_state      <= WS_IDLE;
      ready_axi     <= 1'b0;
      num_tiles     <= '0;
      tile_idx      <= '0;
      beat_idx      <= '0;
      base_addr     <= '0;
      m_axi_arvalid <= 1'b0;
      m_axi_araddr  <= '0;
    end else begin
      case (ws_state)
        WS_IDLE: begin
          ready_axi <= 1'b0;
          if (start_load_axi) begin
            base_addr <= axi_mb + chunk_byte_off;
            num_tiles <= tiles_for_dim(axi_id);
            tile_idx  <= '0;
            beat_idx  <= '0;
            ws_state  <= WS_LOAD_AR;
          end
        end
        WS_LOAD_AR: begin
          m_axi_arvalid <= 1'b1;
          m_axi_araddr  <= tile_addr;
          if (m_axi_arvalid && m_axi_arready) begin
            m_axi_arvalid <= 1'b0;
            ws_state      <= WS_LOAD_R;
          end
        end
        WS_LOAD_R: begin
          if (m_axi_rvalid) begin
            // Bank write happens in the dual-port always_ff below.
            beat_idx <= beat_idx + 1'b1;
            if (m_axi_rlast) begin
              ws_state <= WS_NEXT_TILE;
              beat_idx <= '0;
            end
          end
        end
        WS_NEXT_TILE: begin
          tile_idx <= tile_idx + 1'b1;
          if (tile_idx + 1'b1 == num_tiles) ws_state <= WS_READY;
          else                              ws_state <= WS_LOAD_AR;
        end
        WS_READY: begin
          ready_axi <= 1'b1;
          if (start_load_axi) begin
            // Re-pulse from core: start a new load.
            base_addr <= axi_mb + chunk_byte_off;
            num_tiles <= tiles_for_dim(axi_id);
            tile_idx  <= '0;
            beat_idx  <= '0;
            ws_state  <= WS_LOAD_AR;
            ready_axi <= 1'b0;
          end
        end
        default: ws_state <= WS_IDLE;
      endcase
    end
  end

  // ------------------------------------------------------------------
  //  Per-bank dedicated dual-port BRAM blocks.
  //  Pattern: write port on clk_axi (only fires on bank's tile match),
  //  read port on clk_core (registered output).  Vivado infers RAMB36
  //  TDP for each bank when each is in its own always_ff with simple
  //  enable logic — combining all banks into one always_ff with a case
  //  statement on tile_idx blocks BRAM inference.
  // ------------------------------------------------------------------
  localparam int TILE_SEL_BITS = (IN_DIM_BITS > TILE_AW) ? (IN_DIM_BITS - TILE_AW) : 1;
  logic [TILE_SEL_BITS-1:0] tile_sel;
  logic [BANK_AW-1:0]       beat_sel;
  logic [1:0]               sub_sel;

  always_comb begin
    if (IN_DIM_BITS > TILE_AW) tile_sel = rd_addr[IN_DIM_BITS-1:TILE_AW];
    else                       tile_sel = '0;
    beat_sel = rd_addr[TILE_AW-1:2];
    sub_sel  = rd_addr[1:0];
  end

  // Per-bank write enables (clk_axi domain).
  wire bank_we = (ws_state == WS_LOAD_R) && m_axi_rvalid;
  wire bank0_we = bank_we && (tile_idx == TILES_AW'(0));
  wire bank1_we = bank_we && (tile_idx == TILES_AW'(1));
  wire bank2_we = bank_we && (tile_idx == TILES_AW'(2));

  logic [AXI_DATA_WIDTH-1:0] bank0_rd, bank1_rd, bank2_rd;

  // Bank 0
  always_ff @(posedge clk_axi)  if (bank0_we) bank0[beat_idx] <= m_axi_rdata;
  always_ff @(posedge clk_core) bank0_rd <= bank0[beat_sel];

  // Bank 1
  always_ff @(posedge clk_axi)  if (bank1_we) bank1[beat_idx] <= m_axi_rdata;
  always_ff @(posedge clk_core) bank1_rd <= bank1[beat_sel];

  // Bank 2
  always_ff @(posedge clk_axi)  if (bank2_we) bank2[beat_idx] <= m_axi_rdata;
  always_ff @(posedge clk_core) bank2_rd <= bank2[beat_sel];

  // Tile + sub-beat selection registered alongside the bank read so
  // the final mux + 128-bit slice is combinational on the registered
  // beat_data — 1-cycle total read latency to the consumer.
  logic [TILE_SEL_BITS-1:0] tile_sel_r;
  logic [1:0]               sub_sel_r;
  always_ff @(posedge clk_core) begin
    tile_sel_r <= tile_sel;
    sub_sel_r  <= sub_sel;
  end

  logic [AXI_DATA_WIDTH-1:0] beat_data_r;
  always_comb begin
    case (tile_sel_r)
      TILES_AW'(0): beat_data_r = bank0_rd;
      TILES_AW'(1): beat_data_r = bank1_rd;
      TILES_AW'(2): beat_data_r = bank2_rd;
      default:      beat_data_r = '0;
    endcase
  end

  always_comb weight_data = beat_data_r[sub_sel_r * 128 +: 128];

  // ------------------------------------------------------------------
  //  Diagnostic latches (clk_axi domain — stable after first event).
  // ------------------------------------------------------------------
  always_ff @(posedge clk_axi) begin
    if (rst_axi) begin
      dbg_first_araddr  <= '0;
      dbg_first_rdata   <= '0;
      dbg_first_ar_seen <= 1'b0;
      dbg_first_r_seen  <= 1'b0;
    end else begin
      if (m_axi_arvalid && m_axi_arready && !dbg_first_ar_seen) begin
        dbg_first_araddr  <= m_axi_araddr;
        dbg_first_ar_seen <= 1'b1;
      end
      if (m_axi_rvalid && !dbg_first_r_seen) begin
        dbg_first_rdata  <= m_axi_rdata;
        dbg_first_r_seen <= 1'b1;
      end
    end
  end

  // ------------------------------------------------------------------
  //  Ready / busy CDC: ready_axi (axi domain) → ready (core domain).
  // ------------------------------------------------------------------
  logic [1:0] ready_sync_core;
  always_ff @(posedge clk_core) begin
    if (rst_core) ready_sync_core <= '0;
    else          ready_sync_core <= {ready_sync_core[0], ready_axi};
  end
  always_comb ready = ready_sync_core[1];
  always_comb busy  = ~ready;

  // Named ILA taps (clk_axi domain).
  assign ila_ws_state       = ws_state;
  assign ila_start_load_axi = start_load_axi;
  assign ila_tile_idx       = tile_idx;
  assign ila_beat_idx       = beat_idx;

endmodule

`default_nettype wire
