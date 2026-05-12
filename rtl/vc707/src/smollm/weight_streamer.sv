// weight_streamer.sv — DDR3-backed weight feeder for the matvec engine.
//
// Sits between the existing weight_tile_cache (ping-pong 8 KiB BRAM
// banks, 512-bit-per-beat consumer port, AXI4 loader from MIG) and the
// matvec_int8_engine's 128-bit-per-cycle weight bus.
//
// Per chunk the layer FSM does:
//   1. load_req with (matrix_base_addr, in_dim) — streamer dispatches
//      tile_cache.start_load and waits for tile_cache.load_done
//   2. ready = 1 → layer FSM begins MV_DRIVE
//   3. layer FSM raises rd_addr = 0..in_dim-1 (one cycle per drive cycle).
//      Streamer slices the 512-bit tile-cache read into 128-bit weight_data.
//
// AXI tile = 8 KiB = 128 beats × 64 bytes.  At 16 lanes × 8 bits = 16 B
// per matvec drive cycle, one tile holds 8192/16 = 512 drive cycles.
// SmolLM2 matvecs have in_dim ∈ {576, 1536} — 576 fits in a single tile;
// 1536 needs three.  This module currently handles single-tile chunks
// (in_dim ≤ 512).  Multi-tile (for FFN's 1536-input down-proj) is a
// follow-on enhancement: pre-fetch tile N+1 while consuming tile N via
// the existing ping-pong (already supported by the underlying cache).
//
// DDR3 layout (host-determined, baked into per-matrix base addresses):
//   For matrix W of shape [out_dim, in_dim], chunk c (16 output lanes):
//     base + c × in_dim × 16 bytes
//   So `chunk_byte_offset = chunk_idx * in_dim * 16`.
//
// Per-row scales (16 × Q1.15 = 32 bytes per chunk) live in a separate
// region — they are streamed by the same tile cache via a second load
// (or held in a small auxiliary BRAM).  This module exposes a stub
// scale interface; full integration deferred.
//
// Status: synth-clean module skeleton; Verilator end-to-end test
// against a mock AXI slave + smollm_layer integration is the next
// session's work.

`default_nettype none

module weight_streamer #(
  parameter int AXI_DATA_WIDTH = 512,
  parameter int AXI_ADDR_WIDTH = 30,
  parameter int AXI_ID_WIDTH   = 5,
  parameter int IN_DIM_BITS    = 11,         // log2 max in_dim
  parameter int CHUNK_BITS     = 7           // log2 max chunks per matrix (FFN/16=96)
)(
  input  wire                       clk,
  input  wire                       rst,

  // ------------------------------------------------------------------
  //  Layer FSM control
  // ------------------------------------------------------------------
  input  wire [AXI_ADDR_WIDTH-1:0]  matrix_base,    // DDR3 byte addr of W's chunk 0
  input  wire [CHUNK_BITS-1:0]      chunk_idx,
  input  wire [IN_DIM_BITS-1:0]     in_dim,         // # of drive cycles per chunk
  input  wire                       load_req,       // 1-cycle pulse to start a load
  output logic                      ready,          // tile data available, drive may begin
  output logic                      busy,

  // ------------------------------------------------------------------
  //  Matvec consumer port: rd_addr → weight_data, 1-cycle latency
  //  weight_data is 16 INT8 weights packed (lane 0 in low byte)
  // ------------------------------------------------------------------
  input  wire [IN_DIM_BITS-1:0]     rd_addr,
  output logic [127:0]              weight_data,

  // ------------------------------------------------------------------
  //  AXI4 read master (forwarded from underlying tile cache)
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

  // ------------------------------------------------------------------
  //  Tile cache — ping-pong 8 KiB BRAM, 512-bit consumer
  // ------------------------------------------------------------------
  localparam int BANK_AW = 7;          // 128-entry banks (8 KiB ÷ 64 B)

  logic                  tc_start_load;
  logic [AXI_ADDR_WIDTH-1:0] tc_load_addr;
  logic                  tc_busy;
  logic                  tc_load_done;
  logic [BANK_AW-1:0]    tc_rd_addr;       // 64-byte beat index
  logic [AXI_DATA_WIDTH-1:0] tc_rd_data;
  logic                  tc_consumer_swap;
  logic                  tc_active_bank;

  weight_tile_cache #(
    .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
    .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH),
    .AXI_ID_WIDTH   (AXI_ID_WIDTH),
    .BURST_LEN_LOG2 (7),                   // 128 beats → 8 KiB
    .BANK_ENTRIES   (128)
  ) i_tc (
    .clk(clk), .rst(rst),
    .start_load   (tc_start_load),
    .load_addr    (tc_load_addr),
    .busy         (tc_busy),
    .load_done    (tc_load_done),
    .rd_addr      (tc_rd_addr),
    .rd_data      (tc_rd_data),
    .consumer_swap(tc_consumer_swap),
    .active_bank  (tc_active_bank),
    .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
    .m_axi_arid   (m_axi_arid),    .m_axi_araddr (m_axi_araddr),
    .m_axi_arlen  (m_axi_arlen),   .m_axi_arsize (m_axi_arsize),
    .m_axi_arburst(m_axi_arburst), .m_axi_arlock (m_axi_arlock),
    .m_axi_arcache(m_axi_arcache), .m_axi_arprot (m_axi_arprot),
    .m_axi_arqos  (m_axi_arqos),
    .m_axi_rvalid (m_axi_rvalid),  .m_axi_rready (m_axi_rready),
    .m_axi_rid    (m_axi_rid),     .m_axi_rdata  (m_axi_rdata),
    .m_axi_rresp  (m_axi_rresp),   .m_axi_rlast  (m_axi_rlast)
  );

  // ------------------------------------------------------------------
  //  Loader FSM
  //    On load_req: dispatch tile_cache.start_load with
  //      addr = matrix_base + chunk_idx × in_dim × 16
  //    Wait for tile_cache.load_done → swap bank → ready=1
  // ------------------------------------------------------------------
  typedef enum logic [1:0] {WS_IDLE, WS_LOADING, WS_READY} ws_state_t;
  ws_state_t ws_state;

  logic [AXI_ADDR_WIDTH-1:0] chunk_byte_off;
  // chunk_byte_off = chunk_idx * in_dim * 16.  Up to 1536*16*96 ≈ 2.4 MB.
  // Use a wide multiply but the synthesiser will simplify when in_dim is
  // a constant per matrix instance.
  always_comb begin
    chunk_byte_off = (AXI_ADDR_WIDTH'(chunk_idx)) *
                     (AXI_ADDR_WIDTH'(in_dim))    *
                     AXI_ADDR_WIDTH'(16);
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      ws_state         <= WS_IDLE;
      tc_start_load    <= 1'b0;
      tc_load_addr     <= '0;
      tc_consumer_swap <= 1'b0;
      ready            <= 1'b0;
      busy             <= 1'b0;
    end else begin
      tc_start_load    <= 1'b0;
      tc_consumer_swap <= 1'b0;
      case (ws_state)
        WS_IDLE: begin
          ready <= 1'b0;
          busy  <= 1'b0;
          if (load_req) begin
            tc_load_addr  <= matrix_base + chunk_byte_off;
            tc_start_load <= 1'b1;
            ws_state      <= WS_LOADING;
            busy          <= 1'b1;
          end
        end
        WS_LOADING: begin
          if (tc_load_done) begin
            tc_consumer_swap <= 1'b1;       // swap inactive↔active
            ws_state         <= WS_READY;
          end
        end
        WS_READY: begin
          ready <= 1'b1;
          busy  <= 1'b0;
          if (load_req) begin
            // Next chunk requested; start another load
            tc_load_addr  <= matrix_base + chunk_byte_off;
            tc_start_load <= 1'b1;
            ws_state      <= WS_LOADING;
            ready         <= 1'b0;
            busy          <= 1'b1;
          end
        end
        default: ws_state <= WS_IDLE;
      endcase
    end
  end

  // ------------------------------------------------------------------
  //  Consumer slicing:
  //    rd_addr   ∈ [0, in_dim) — counts matvec drive cycles
  //    Each AXI beat (512 bits) holds 4 × 128-bit "weight_data" words.
  //    tile entry index = rd_addr >> 2
  //    sub-beat (which 128-bit slice of 512) = rd_addr[1:0]
  //
  //  rd_addr is registered upstream; tc_rd_addr is combinational; the
  //  tile cache's read returns combinationally — but per the ASIC-TODO
  //  memo we also want a registered output for clean BRAM inference,
  //  so weight_data is registered here.
  // ------------------------------------------------------------------
  always_comb begin
    // rd_addr is IN_DIM_BITS wide; we only need the upper bits to index
    // into the 128-entry tile bank.  Slice down to BANK_AW.
    tc_rd_addr = rd_addr[2 +: BANK_AW];
  end

  logic [1:0] rd_sub_r;    // delayed by 1 cycle to align with tc_rd_data

  always_ff @(posedge clk) begin
    if (rst) begin
      rd_sub_r    <= '0;
      weight_data <= '0;
    end else begin
      rd_sub_r    <= rd_addr[1:0];
      // Slice the appropriate 128-bit word from the 512-bit beat.
      weight_data <= tc_rd_data[rd_sub_r*128 +: 128];
    end
  end

endmodule

`default_nettype wire
