// weight_streamer_bfp_mt.sv — block-FP DDR3 weight feeder, dual-clock.
//
//   Same architecture as weight_streamer_mt.sv (int8 path) but adapted
//   for BFP wide-packed data:
//     - mantissa entry: 16 lanes × 16 b = 256 b per column.
//     - exponent entry: 16 lanes ×  8 b = 128 b per tile (1 tile = 16 cols).
//   Consumer reads `weight_m_out` (256 b) by col index and `weight_e_out`
//   (128 b) by tile index.
//
//   Per `load_req`, the streamer fetches one CHUNK of {mantissas, exps}
//   for the requested {matrix_base_m, matrix_base_e, chunk_idx, in_dim}
//   from DDR3 into two BRAM banks, then asserts `ready`.  The matvec
//   engine consumes the cached chunk before the next load_req.
//
//   AXI master lives on clk_axi (MIG ui_clk, ~200 MHz); consumer port
//   lives on clk_core (50 MHz).  Toggle CDC between domains.

`default_nettype none

module weight_streamer_bfp_mt #(
  parameter int AXI_DATA_WIDTH = 512,       // MIG AXI4 read data width
  parameter int AXI_ADDR_WIDTH = 30,
  parameter int AXI_ID_WIDTH   = 5,
  parameter int IN_DIM_MAX     = 1536,      // largest matvec input dim (FFN)
  parameter int IN_DIM_BITS    = 12,        // bits for in_dim (1536 fits 11)
  parameter int CHUNK_BITS     = 7,         // chunks per output dim (max 96 for FFN)
  parameter int MAX_AR_LEN     = 256,       // AXI burst max-beats (MIG default)
  // Sim-only selftest shadow sizing.  Default (1<<14) suits the
  // D=64 layer-selftest; the layer module overrides to cover the
  // largest matvec (max(out_dim) × max(in_dim) / 16) when running at
  // real-model dims.  Caller computes max(D*D, D*FFN, FFN*D) / 16.
  parameter int SIM_M_DEPTH_P  = 1 << 14
)(
  // ------------------------------------------------------------------
  //  Consumer side (clk_core)
  // ------------------------------------------------------------------
  input  wire                       clk_core,
  input  wire                       rst_core,
  input  wire [AXI_ADDR_WIDTH-1:0]  matrix_base_m,    // DDR3 byte offset of mantissa region
  input  wire [AXI_ADDR_WIDTH-1:0]  matrix_base_e,    // DDR3 byte offset of exp region
  input  wire [CHUNK_BITS-1:0]      chunk_idx,        // 0..CHUNKS_OUT-1 within layer
  input  wire [IN_DIM_BITS-1:0]     in_dim,           // matvec input dim (cols)
  input  wire [IN_DIM_BITS-1:0]     in_dim_tiles,     // NT_in = in_dim / 16
  input  wire                       load_req,         // 1-cycle pulse
  output logic                      ready,
  output logic                      busy,
  // Mantissa read port: rd_col → 256-bit weight_m_out (1-cycle latency)
  input  wire [IN_DIM_BITS-1:0]     rd_col,
  output logic [255:0]              weight_m_out,
  // Exponent read port: rd_tile → 128-bit weight_e_out (1-cycle latency)
  input  wire [IN_DIM_BITS-1:0]     rd_tile,
  output logic [127:0]              weight_e_out,

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
  input  wire                       m_axi_rlast
`ifdef VERILATOR
  ,
  // Sim-only: which of {WQ,WK,WV,WO,WG,WU,WDN} the layer is asking for,
  // so the shadow loader picks the right .hex.  In production this
  // distinction comes from matrix_base_m's DDR3 offset (per-matrix bases
  // in autoregress_bfp_top).  In selftest all bases are 0.
  input  wire [2:0]                 sim_matvec_id
`endif
);

  // ------------------------------------------------------------------
  //  Geometry — derived constants.
  //
  //  Per AXI beat (512 b):  2 mantissa col entries  (2 × 256 b)
  //                         4 exponent tile entries (4 × 128 b)
  //
  //  Bank sizes (1536-col worst case):
  //    bank_m: 1536 col entries / 2 per beat = 768 beats × 512 b = 393 Kbit
  //    bank_e:   96 tile entries / 4 per beat =  24 beats × 512 b ≈ 12 Kbit
  // ------------------------------------------------------------------
  localparam int M_BEATS_MAX = IN_DIM_MAX / 2;          // 768 for in_dim=1536
  localparam int E_BEATS_MAX = (IN_DIM_MAX / 16) / 4;   // 24 for NT_in=96
  // Round to next power of two for clean addressing.
  localparam int M_BANK_DEPTH = (M_BEATS_MAX > 512) ? 1024 : 512;   // 1024 entries
  localparam int E_BANK_DEPTH = 32;
  localparam int M_BANK_AW    = $clog2(M_BANK_DEPTH);
  localparam int E_BANK_AW    = $clog2(E_BANK_DEPTH);

  // ------------------------------------------------------------------
  //  Constant AR signals (MIG / AXI4 defaults).
  // ------------------------------------------------------------------
  assign m_axi_arid    = '0;
  assign m_axi_arsize  = 3'd6;            // 2^6 = 64 B = AXI_DATA_WIDTH/8
  assign m_axi_arburst = 2'b01;           // INCR
  assign m_axi_arlock  = 1'b0;
  assign m_axi_arcache = 4'b0011;
  assign m_axi_arprot  = 3'b000;
  assign m_axi_arqos   = 4'b0000;
  assign m_axi_rready  = 1'b1;

  // ------------------------------------------------------------------
  //  Bank BRAMs (true dual-port).
  //    port A (write, clk_axi):  bank[beat_idx] <= rdata
  //    port B (read,  clk_core): registered read for the consumer
  // ------------------------------------------------------------------
  (* ram_style = "block" *) logic [AXI_DATA_WIDTH-1:0] bank_m [0:M_BANK_DEPTH-1];
  (* ram_style = "block" *) logic [AXI_DATA_WIDTH-1:0] bank_e [0:E_BANK_DEPTH-1];

  // ------------------------------------------------------------------
  //  Core-side toggle + payload capture.
  // ------------------------------------------------------------------
  logic                      core_tog;
  logic [AXI_ADDR_WIDTH-1:0] core_mb_m, core_mb_e;
  logic [CHUNK_BITS-1:0]     core_ck;
  logic [IN_DIM_BITS-1:0]    core_id, core_idt;

  always_ff @(posedge clk_core) begin
    if (rst_core) begin
      core_tog <= 1'b0;
      core_mb_m<= '0; core_mb_e<= '0; core_ck<= '0; core_id<= '0; core_idt<= '0;
    end else if (load_req) begin
      core_tog <= ~core_tog;
      core_mb_m<= matrix_base_m;
      core_mb_e<= matrix_base_e;
      core_ck  <= chunk_idx;
      core_id  <= in_dim;
      core_idt <= in_dim_tiles;
    end
  end

  // ------------------------------------------------------------------
  //  AXI-side: 2FF-sync core_tog and capture payload on the edge.
  // ------------------------------------------------------------------
  logic [1:0]                tog_sync_axi;
  logic                      tog_seen_axi;
  logic                      start_load_axi;
  logic [AXI_ADDR_WIDTH-1:0] axi_mb_m, axi_mb_e;
  logic [CHUNK_BITS-1:0]     axi_ck;
  logic [IN_DIM_BITS-1:0]    axi_id, axi_idt;

  always_ff @(posedge clk_axi) begin
    if (rst_axi) begin
      tog_sync_axi   <= '0; tog_seen_axi <= 1'b0;
      start_load_axi <= 1'b0;
      axi_mb_m<= '0; axi_mb_e<= '0; axi_ck<= '0; axi_id<= '0; axi_idt<= '0;
    end else begin
      tog_sync_axi   <= {tog_sync_axi[0], core_tog};
      start_load_axi <= 1'b0;
      if (tog_sync_axi[1] != tog_seen_axi) begin
        tog_seen_axi   <= tog_sync_axi[1];
        start_load_axi <= 1'b1;
        axi_mb_m <= core_mb_m;
        axi_mb_e <= core_mb_e;
        axi_ck   <= core_ck;
        axi_id   <= core_id;
        axi_idt  <= core_idt;
      end
    end
  end

  // ------------------------------------------------------------------
  //  Loader FSM — fetch mantissa region then exp region.
  //
  //  Mantissa burst total beats = in_dim / 2.  Split into ceil((id/2) /
  //  MAX_AR_LEN) ARs.  Exp burst is small enough to fit one AR.
  // ------------------------------------------------------------------
  typedef enum logic [3:0] {
    WS_IDLE,
    WS_M_AR, WS_M_R, WS_M_NEXT,
    WS_E_AR, WS_E_R,
    WS_READY
  } ws_state_t;
  ws_state_t ws_state;

  logic [AXI_ADDR_WIDTH-1:0] m_base_axi;          // matrix_base_m + chunk*in_dim*32
  logic [AXI_ADDR_WIDTH-1:0] e_base_axi;          // matrix_base_e + chunk*NT_in*16
  logic [11:0]               m_total_beats;       // in_dim / 2
  logic [11:0]               m_beats_remaining;
  logic [AXI_ADDR_WIDTH-1:0] m_next_addr;
  logic [9:0]                bank_m_widx;         // write addr into bank_m
  logic [11:0]               e_total_beats;
  logic [9:0]                bank_e_widx;
  logic [8:0]                cur_burst_beats;     // 1..256
  logic                      ready_axi;

  // Per-chunk byte offsets:
  //   mantissa region:  in_dim cols × 32 B per chunk
  //   exp region:       NT_in tiles × 16 B per chunk
  logic [AXI_ADDR_WIDTH-1:0] m_chunk_off, e_chunk_off;
  always_comb begin
    m_chunk_off = (AXI_ADDR_WIDTH'(axi_ck)) * (AXI_ADDR_WIDTH'(axi_id))  * AXI_ADDR_WIDTH'(32);
    e_chunk_off = (AXI_ADDR_WIDTH'(axi_ck)) * (AXI_ADDR_WIDTH'(axi_idt)) * AXI_ADDR_WIDTH'(16);
  end

  always_ff @(posedge clk_axi) begin
    if (rst_axi) begin
      ws_state          <= WS_IDLE;
      ready_axi         <= 1'b0;
      m_axi_arvalid     <= 1'b0;
      m_axi_araddr      <= '0;
      m_axi_arlen       <= '0;
      m_total_beats     <= '0;
      m_beats_remaining <= '0;
      m_next_addr       <= '0;
      bank_m_widx       <= '0;
      bank_e_widx       <= '0;
      cur_burst_beats   <= '0;
      e_total_beats     <= '0;
      m_base_axi        <= '0;
      e_base_axi        <= '0;
    end else begin
      case (ws_state)
        WS_IDLE: begin
          ready_axi <= 1'b0;
          if (start_load_axi) begin
`ifdef VERILATOR
            // Selftest mode (Verilator only): when matrix_base_m == 0 the
            // testbench is exercising the layer with AXI tied off — the
            // burst would never make progress because arready/rvalid stay
            // low.  Skip the load and jump straight to WS_READY so the
            // layer FSM can advance through all its states; bank_m/bank_e
            // stay at their reset zeros (output will be junk but the FSM
            // path is the artefact we're testing).
            if (axi_mb_m == 0) begin
              ws_state <= WS_READY;
            end else
`endif
            begin
            // Per-chunk total beats:
            //   mantissa: in_dim columns, 2 cols per 512-b beat → in_dim/2 beats
            //   exp:      NT_in tiles,    4 tiles per beat       → NT_in/4 beats
            m_total_beats     <= 12'(axi_id  >> 1);
            m_beats_remaining <= 12'(axi_id  >> 1);
            e_total_beats     <= 12'((axi_idt + 3) >> 2);
            m_base_axi        <= axi_mb_m + m_chunk_off;
            e_base_axi        <= axi_mb_e + e_chunk_off;
            m_next_addr       <= axi_mb_m + m_chunk_off;
            bank_m_widx       <= '0;
            bank_e_widx       <= '0;
            ws_state          <= WS_M_AR;
            end
          end
        end
        WS_M_AR: begin
          // Determine burst length (max MAX_AR_LEN beats, or whatever's left).
          if (m_beats_remaining > MAX_AR_LEN) begin
            cur_burst_beats <= 9'(MAX_AR_LEN);
            m_axi_arlen     <= 8'(MAX_AR_LEN - 1);
          end else begin
            cur_burst_beats <= 9'(m_beats_remaining);
            m_axi_arlen     <= 8'(m_beats_remaining - 1);
          end
          m_axi_arvalid <= 1'b1;
          m_axi_araddr  <= m_next_addr;
          if (m_axi_arvalid && m_axi_arready) begin
            m_axi_arvalid <= 1'b0;
            ws_state      <= WS_M_R;
          end
        end
        WS_M_R: begin
          if (m_axi_rvalid) begin
            // bank_m write happens in the per-bank always_ff below.
            bank_m_widx       <= bank_m_widx + 1'b1;
            m_beats_remaining <= m_beats_remaining - 1'b1;
            if (m_axi_rlast) begin
              m_next_addr <= m_next_addr + (cur_burst_beats * (AXI_DATA_WIDTH/8));
              ws_state    <= WS_M_NEXT;
            end
          end
        end
        WS_M_NEXT: begin
          if (m_beats_remaining == '0) begin
            // Mantissa done — start exp burst.
            m_axi_araddr  <= e_base_axi;
            m_axi_arlen   <= 8'(e_total_beats - 1);
            m_axi_arvalid <= 1'b1;
            ws_state      <= WS_E_AR;
          end else begin
            ws_state <= WS_M_AR;
          end
        end
        WS_E_AR: begin
          if (m_axi_arvalid && m_axi_arready) begin
            m_axi_arvalid <= 1'b0;
            ws_state      <= WS_E_R;
          end
        end
        WS_E_R: begin
          if (m_axi_rvalid) begin
            bank_e_widx <= bank_e_widx + 1'b1;
            if (m_axi_rlast) begin
              ws_state <= WS_READY;
            end
          end
        end
        WS_READY: begin
          ready_axi <= 1'b1;
          if (start_load_axi) begin
            // New load requested — restart.
            ready_axi         <= 1'b0;
`ifdef VERILATOR
            if (axi_mb_m == 0) begin
              // Selftest: stay in WS_READY so ready_axi pulses 0 for one cycle
              // (next cycle re-asserts).  Layer's WSP_HOLD `if (!ws_ready)`
              // catches that dip and advances to WSP_WAIT → WSP_READY.
              ws_state <= WS_READY;
            end else
`endif
            begin
            m_total_beats     <= 12'(axi_id  >> 1);
            m_beats_remaining <= 12'(axi_id  >> 1);
            e_total_beats     <= 12'((axi_idt + 3) >> 2);
            m_base_axi        <= axi_mb_m + m_chunk_off;
            e_base_axi        <= axi_mb_e + e_chunk_off;
            m_next_addr       <= axi_mb_m + m_chunk_off;
            bank_m_widx       <= '0;
            bank_e_widx       <= '0;
            ws_state          <= WS_M_AR;
            end
          end
        end
        default: ws_state <= WS_IDLE;
      endcase
    end
  end

  // ------------------------------------------------------------------
  //  Per-bank dual-port BRAMs.
  // ------------------------------------------------------------------
  wire bank_m_we = (ws_state == WS_M_R) && m_axi_rvalid;
  wire bank_e_we = (ws_state == WS_E_R) && m_axi_rvalid;

  logic [AXI_DATA_WIDTH-1:0] bank_m_rd, bank_e_rd;
  // rd_col is a column index; one beat holds 2 cols, so beat_idx = rd_col/2,
  // sub_sel  = rd_col[0]   selects which 256-b half of the 512-b beat.
  logic [M_BANK_AW-1:0] m_beat_sel;
  logic                 m_sub_sel;
  always_comb begin
    m_beat_sel = rd_col[M_BANK_AW:1];
    m_sub_sel  = rd_col[0];
  end
  // rd_tile maps to 4 tiles per beat → beat_idx = rd_tile/4, sub = rd_tile[1:0].
  logic [E_BANK_AW-1:0] e_beat_sel;
  logic [1:0]           e_sub_sel;
  always_comb begin
    e_beat_sel = rd_tile[E_BANK_AW+1:2];
    e_sub_sel  = rd_tile[1:0];
  end

  always_ff @(posedge clk_axi)  if (bank_m_we) bank_m[bank_m_widx[M_BANK_AW-1:0]] <= m_axi_rdata;
  always_ff @(posedge clk_core) bank_m_rd <= bank_m[m_beat_sel];

  always_ff @(posedge clk_axi)  if (bank_e_we) bank_e[bank_e_widx[E_BANK_AW-1:0]] <= m_axi_rdata;
  always_ff @(posedge clk_core) bank_e_rd <= bank_e[e_beat_sel];

  // Register sub-select alongside the bank read.
  logic       m_sub_r;
  logic [1:0] e_sub_r;
  always_ff @(posedge clk_core) begin
    m_sub_r <= m_sub_sel;
    e_sub_r <= e_sub_sel;
  end

`ifdef VERILATOR
  // ------------------------------------------------------------------
  //  Sim-only selftest weight shadow.
  //
  //  Loaded from the per-matvec .hex files emitted by emit_W_concat in
  //  gen_smollm_blockfp_full.py (256-bit packed-lane mantissa words, one
  //  per (chunk, col) — selftest assumes NL=1 so layer_idx isn't plumbed).
  //  Activated when the testbench drives axi_mb_m==0 at the first
  //  load_req (= ws_base_*=0 selftest config); production reads still
  //  flow through bank_m / bank_e.
  // ------------------------------------------------------------------
  // Selftest shadow sizing — driven by the SIM_M_DEPTH_P parameter so the
  // smollm_layer_bfp instantiation can compute max(D*D, D*FFN, FFN*D)/16
  // and override.  E_DEPTH is shadow_lines / 16 since exponents are tile-
  // shared.
  localparam int SIM_M_DEPTH   = SIM_M_DEPTH_P;
  localparam int SIM_E_DEPTH   = (SIM_M_DEPTH + 15) / 16 + 1;
  // $readmemh needs a single-dim destination; use one array per matvec_id
  // rather than a 2-D shadow[7][...] (this is a Verilator limitation).
  logic [255:0] sim_shadow_m_Q  [0:SIM_M_DEPTH-1];
  logic [255:0] sim_shadow_m_K  [0:SIM_M_DEPTH-1];
  logic [255:0] sim_shadow_m_V  [0:SIM_M_DEPTH-1];
  logic [255:0] sim_shadow_m_O  [0:SIM_M_DEPTH-1];
  logic [255:0] sim_shadow_m_G  [0:SIM_M_DEPTH-1];
  logic [255:0] sim_shadow_m_U  [0:SIM_M_DEPTH-1];
  logic [255:0] sim_shadow_m_DN [0:SIM_M_DEPTH-1];
  logic [127:0] sim_shadow_e_Q  [0:SIM_E_DEPTH-1];
  logic [127:0] sim_shadow_e_K  [0:SIM_E_DEPTH-1];
  logic [127:0] sim_shadow_e_V  [0:SIM_E_DEPTH-1];
  logic [127:0] sim_shadow_e_O  [0:SIM_E_DEPTH-1];
  logic [127:0] sim_shadow_e_G  [0:SIM_E_DEPTH-1];
  logic [127:0] sim_shadow_e_U  [0:SIM_E_DEPTH-1];
  logic [127:0] sim_shadow_e_DN [0:SIM_E_DEPTH-1];

  // Shadow-array load fires only for the bfp-layer selftest tb, which
  // sets +define+LBFP_STREAMER_SELFTEST at Verilator-build time.  The
  // per-matvec .hex files are sized for the D=64 selftest geometry
  // (SIM_M_DEPTH=16384 lines); real-model weights (smollm135/360) far
  // exceed that and would abort with "$readmem file address beyond
  // bounds of array".  Other testbenches (tb_full_bfp, tb_autoregress_
  // bfp_stream) and FPGA synth get all-zero shadows here — which is
  // fine because their weight path is the real bank_m/bank_e fetch via
  // mock_axi_slave / DDR3, not the selftest fast-path.
`ifdef LBFP_STREAMER_SELFTEST
  initial begin
    $readmemh("../generated/lbfp_WQ_m.hex",  sim_shadow_m_Q );
    $readmemh("../generated/lbfp_WK_m.hex",  sim_shadow_m_K );
    $readmemh("../generated/lbfp_WV_m.hex",  sim_shadow_m_V );
    $readmemh("../generated/lbfp_WO_m.hex",  sim_shadow_m_O );
    $readmemh("../generated/lbfp_WG_m.hex",  sim_shadow_m_G );
    $readmemh("../generated/lbfp_WU_m.hex",  sim_shadow_m_U );
    $readmemh("../generated/lbfp_WDN_m.hex", sim_shadow_m_DN);
    $readmemh("../generated/lbfp_WQ_e.hex",  sim_shadow_e_Q );
    $readmemh("../generated/lbfp_WK_e.hex",  sim_shadow_e_K );
    $readmemh("../generated/lbfp_WV_e.hex",  sim_shadow_e_V );
    $readmemh("../generated/lbfp_WO_e.hex",  sim_shadow_e_O );
    $readmemh("../generated/lbfp_WG_e.hex",  sim_shadow_e_G );
    $readmemh("../generated/lbfp_WU_e.hex",  sim_shadow_e_U );
    $readmemh("../generated/lbfp_WDN_e.hex", sim_shadow_e_DN);
  end
`endif

  // Detect selftest mode: first load_req with axi_mb_m == 0.
  // Gated by LBFP_STREAMER_SELFTEST — full-model testbenches
  // (tb_full_bfp, tb_autoregress_bfp_stream) drive zero-offset loads
  // through the real AXI path and must NOT be reinterpreted as
  // selftest, or the streamer ignores the real weights and serves
  // all-zero shadows instead.
  logic selftest;
  initial selftest = 1'b0;
`ifdef LBFP_STREAMER_SELFTEST
  always_ff @(posedge clk_axi) begin
    if (rst_axi) selftest <= 1'b0;
    else if (ws_state == WS_IDLE && start_load_axi && axi_mb_m == 0) begin
      selftest <= 1'b1;
      $display("[ws] selftest mode activated");
    end
  end
`endif

  // (debug $displays removed — shadow path verified working.)

  // 1-cycle latency shadow read (matches bank_m_rd's pipeline).  rd_col
  // / rd_tile / chunk_idx / sim_matvec_id are all valid inputs already.
  logic [255:0] sim_m_rd;
  logic [127:0] sim_e_rd;
  logic [13:0]  sim_m_idx, sim_e_idx;
  always_comb begin
    sim_m_idx = chunk_idx * in_dim       + rd_col;
    sim_e_idx = chunk_idx * in_dim_tiles + rd_tile;
  end
  always_ff @(posedge clk_core) begin
    case (sim_matvec_id)
      3'd0: sim_m_rd <= sim_shadow_m_Q [sim_m_idx];
      3'd1: sim_m_rd <= sim_shadow_m_K [sim_m_idx];
      3'd2: sim_m_rd <= sim_shadow_m_V [sim_m_idx];
      3'd3: sim_m_rd <= sim_shadow_m_O [sim_m_idx];
      3'd4: sim_m_rd <= sim_shadow_m_G [sim_m_idx];
      3'd5: sim_m_rd <= sim_shadow_m_U [sim_m_idx];
      3'd6: sim_m_rd <= sim_shadow_m_DN[sim_m_idx];
      default: sim_m_rd <= '0;
    endcase
    case (sim_matvec_id)
      3'd0: sim_e_rd <= sim_shadow_e_Q [sim_e_idx];
      3'd1: sim_e_rd <= sim_shadow_e_K [sim_e_idx];
      3'd2: sim_e_rd <= sim_shadow_e_V [sim_e_idx];
      3'd3: sim_e_rd <= sim_shadow_e_O [sim_e_idx];
      3'd4: sim_e_rd <= sim_shadow_e_G [sim_e_idx];
      3'd5: sim_e_rd <= sim_shadow_e_U [sim_e_idx];
      3'd6: sim_e_rd <= sim_shadow_e_DN[sim_e_idx];
      default: sim_e_rd <= '0;
    endcase
  end
`endif

  always_comb weight_m_out =
`ifdef VERILATOR
                             selftest ? sim_m_rd :
`endif
                             bank_m_rd[m_sub_r * 256 +: 256];
  always_comb weight_e_out =
`ifdef VERILATOR
                             selftest ? sim_e_rd :
`endif
                             bank_e_rd[e_sub_r * 128 +: 128];

  // ------------------------------------------------------------------
  //  Ready/busy CDC: ready_axi → ready (clk_core).
  // ------------------------------------------------------------------
  logic [1:0] ready_sync_core;
  always_ff @(posedge clk_core) begin
    if (rst_core) ready_sync_core <= '0;
    else          ready_sync_core <= {ready_sync_core[0], ready_axi};
  end
  always_comb ready = ready_sync_core[1];
  always_comb busy  = ~ready;

endmodule

`default_nettype wire
