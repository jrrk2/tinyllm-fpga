// smollm_multilayer_tm.sv — TIME-MULTIPLEXED multi-layer wrapper.
//
// Runs ONE smollm_layer instance NL times sequentially.  Per iteration
// the wrapper drives `layer_idx` (0..NL-1) and `layer_base_addr` (=
// layer_idx × LAYER_BYTES) into the layer; the layer indexes its
// gamma/scale/KV-cache 2D ROMs by layer_idx and computes the streamer's
// matrix_base = layer_base_addr + per-matrix offset, so each iteration
// fetches that layer's weights from a different DDR3 region.
//
// Cascade: hidden_state register starts at the wrapper's hidden_in and
// captures the layer's hidden_out at the end of each iteration.  Final
// hidden_state after NL iterations is the wrapper's hidden_out.
//
// Compared to smollm_multilayer.sv (NL separate instances), this fits
// VC707 at SmolLM2 dims with NL=30.  Per-call overhead is the existing
// per-layer cycle count + ~3 wrapper handshake cycles per iteration.

`default_nettype none

module smollm_multilayer_tm #(
  parameter int    D       = 576,
  parameter int    H_Q     = 9,
  parameter int    H_KV    = 3,
  parameter int    HD      = 64,
  parameter int    FFN     = 1536,
  parameter int    MAX_CTX = 4,
  parameter int    NL      = 30,
  parameter        PREFIX  = "tm_layer_",
  // Bytes consumed by ONE layer's weights in DDR3.  Wrapper computes
  // layer_base_addr = layer_idx × LAYER_BYTES.  Default below is the
  // per-layer image size at SmolLM2 dims (8 KiB × 432 chunks ≈ 3.5 MB),
  // rounded up to a 64 KiB boundary for clean addressing.
  parameter [29:0] LAYER_BYTES = 30'h36_4000,
  // Per-matrix byte offset WITHIN one layer (must match host's
  // gen_layer_ddr3.py output for that matrix).  Defaults from the
  // small-config layout — overridden at instantiation for SmolLM dims.
  parameter [29:0] BASE_Q    = 30'h00_0000,
  parameter [29:0] BASE_K    = 30'h08_0000,
  parameter [29:0] BASE_V    = 30'h10_0000,
  parameter [29:0] BASE_O    = 30'h18_0000,
  parameter [29:0] BASE_GATE = 30'h20_0000,
  parameter [29:0] BASE_UP   = 30'h28_0000,
  parameter [29:0] BASE_DOWN = 30'h30_0000
)(
  input  wire                    clk,
  input  wire                    rst,
  input  wire                    start,
  input  wire [10:0]             pos,
  input  wire [4:0]              kv_pos,
  // Hidden state is wide (24-bit Q15.9) to absorb the residual stream.
  input  wire signed [D*16-1:0]  hidden_in,
  output logic signed [D*16-1:0] hidden_out,
  output logic                   done,
  // Per-layer snapshot select.  Captures hidden_state at every layer
  // boundary; host selects which layer's output appears on hidden_out.
  // Values 0..NL-1 → that layer's snapshot; NL → the live hidden_state
  // (final result after all NL layers).
  input  wire [4:0]              snapshot_layer_sel,
  // Runtime factor-override write port (CDC'd from eth_clk regmap).
  input  wire [4:0]              factor_wr_layer,
  input  wire [31:0]             factor_wr_data,
  input  wire                    factor_wr_en_swiglu_lo,
  input  wire                    factor_wr_en_swiglu_mlp,
  input  wire                    factor_wr_en_attn,
  // Factor readback — host writes {kind[1:0], layer[4:0]} to factor_rd_sel,
  // reads back factor_rd_data on next host transaction.  Forces the RAM to
  // be observable so Vivado can't optimise the write path away.
  input  wire [6:0]              factor_rd_sel,
  output wire [31:0]             factor_rd_data,
  // Power-on-only init pulse for the factor RAM (does NOT toggle on
  // user-triggered restart).  Driven from selftest by a startup SR.
  input  wire                    factor_ram_por_init
`ifdef MICROGPT_DDR3_WEIGHTS
  ,
  input  wire                        clk_axi,
  input  wire                        rst_axi,
  output wire                        m_axi_arvalid,
  input  wire                        m_axi_arready,
  output wire [4:0]                  m_axi_arid,
  output wire [29:0]                 m_axi_araddr,
  output wire [7:0]                  m_axi_arlen,
  output wire [2:0]                  m_axi_arsize,
  output wire [1:0]                  m_axi_arburst,
  output wire                        m_axi_arlock,
  output wire [3:0]                  m_axi_arcache,
  output wire [2:0]                  m_axi_arprot,
  output wire [3:0]                  m_axi_arqos,
  input  wire                        m_axi_rvalid,
  output wire                        m_axi_rready,
  input  wire  [4:0]                 m_axi_rid,
  input  wire  [511:0]               m_axi_rdata,
  input  wire  [1:0]                 m_axi_rresp,
  input  wire                        m_axi_rlast,

  // Debug taps forwarded from the inner smollm_layer + this wrapper's
  // own outer-FSM state.  Always present in the DDR3 build so a single
  // top-level `MICROGPT_LAYER_DEBUG` ifdef can wire them to the regmap
  // without touching wrapper signatures.  Vivado DCEs them when not
  // consumed (the underlying state is needed for normal operation).
  output wire  [29:0]                dbg_first_araddr,
  output wire  [511:0]               dbg_first_rdata,
  output wire                        dbg_first_ar_seen,
  output wire                        dbg_first_r_seen,
  output wire  [511:0]               dbg_eng_w_packed,
  output wire  [511:0]               dbg_wd_packed,
  output wire  [63:0]                dbg_in_value_packed,
  output wire                        dbg_snap_done_o,
  output wire  [4:0]                 ila_state,
  output wire  [2:0]                 ila_mv_phase,
  output wire  [10:0]                ila_cnt,
  output wire  [6:0]                 ila_chunk,
  output wire  [2:0]                 ila_ws_matvec_id,
  output wire                        ila_ws_load_req,
  output wire                        ila_ws_ready,
  output wire  [10:0]                ila_ws_rd_addr,
  output wire  [127:0]               ila_ws_weight_data,
  output wire  [127:0]               ila_eng_w,
  output wire  [15:0]                ila_eng_in_value,
  output wire                        ila_eng_in_valid,
  output wire                        ila_eng_in_last,
  output wire                        ila_eng_acc_clear,
  output wire                        ila_eng_out_valid,
  output wire  [2:0]                 ila_ws_state_axi,
  output wire                        ila_start_load_axi,
  output wire  [1:0]                 ila_tile_idx,
  output wire  [6:0]                 ila_beat_idx,
  // Outer time-mux FSM visibility
  output wire  [2:0]                 ila_ml_state,
  output wire  [4:0]                 ila_ml_layer_idx
`endif
);

  // ------------------------------------------------------------------
  //  Layer interface
  // ------------------------------------------------------------------
  logic [4:0]                    layer_idx;
  logic [29:0]                   layer_base_addr;
  logic                          layer_start;
  logic signed [D*16-1:0]        layer_hidden_in;
  wire  signed [D*16-1:0]        layer_hidden_out;
  wire                           layer_done;

  // Per-layer rescale data — produced by gen_smollm_calib.py and baked
  // into tm_layer_data.svh's TM_RESCALE[NL] localparam.  Each entry packs:
  //   [15:0]   resid1_factor    (multiplies o_buf into Q15.9 residual)
  //   [31:16]  resid2_factor    (multiplies down_buf into Q15.9 residual)
  //   [39:32]  rms1_shift       (right-shift hidden_in for RMSNorm 1 input)
  //   [47:40]  rms2_shift       (right-shift hidden1   for RMSNorm 2 input)
  // We mux on layer_idx so the active layer sees its own rescale data.
`include "tm_layer_data.svh"
  // Continuous assigns into wires: event-driven on RHS change, so xsim
  // updates correctly when layer_idx settles.  always_comb on a logic
  // target relies on the simulator's synthesized sensitivity list, which
  // xsim 2020.1 mis-builds for indexed unpacked-array reads — the logic
  // sits at its initial X and never updates.  Synth-identical to the
  // original procedural form.
  wire [23:0]       cur_resid1_factor  = TM_RESCALE[layer_idx][23:0];
  wire [23:0]       cur_resid2_factor  = TM_RESCALE[layer_idx][47:24];
  wire [3:0]        cur_h_in_p2        = TM_RESCALE[layer_idx][51:48];
  wire [3:0]        cur_h_out_p2       = TM_RESCALE[layer_idx][55:52];
  wire signed [3:0] cur_sh_h_in_to_h1  = $signed(TM_RESCALE[layer_idx][59:56]);
  wire signed [3:0] cur_sh_h1_to_h_out = $signed(TM_RESCALE[layer_idx][63:60]);

  // Scale-aware SwiGLU / Attn-AV factors.  Loaded at reset from the
  // calibration constants in tm_layer_swiglu_attn.svh; can be overridden
  // at runtime by the host via the eth-clk regmap (addresses 0x100/0x140/0x180
  // per layer).  Per-tensor scales can't yet be modelled bit-exactly so
  // having them tweakable from Python is a big debug win — every rebuild
  // is ~1 hour, every host write is ~milliseconds.
  wire [15:0] cur_sg_gate_in_factor;
  wire [15:0] cur_sg_up_in_factor;
  wire [23:0] cur_sg_mlp_out_factor;
  wire [23:0] cur_attn_factor;
  // factor RAM is initialised by a separate power-on signal so user
  // restarts (REG_RESTART) don't wipe runtime tweaks back to defaults.
  // por_init goes high for the first few cycles after FPGA configuration,
  // then stays low — see selftest wrapper for its derivation.
  factor_ram #(.NL(NL)) i_factor_ram (
    .clk(clk), .rst(factor_ram_por_init),
    .layer_idx                (layer_idx),
    .cur_sg_gate_in_factor    (cur_sg_gate_in_factor),
    .cur_sg_up_in_factor      (cur_sg_up_in_factor),
    .cur_sg_mlp_out_factor    (cur_sg_mlp_out_factor),
    .cur_attn_factor          (cur_attn_factor),
    .factor_wr_layer          (factor_wr_layer),
    .factor_wr_data           (factor_wr_data),
    .factor_wr_en_swiglu_lo   (factor_wr_en_swiglu_lo),
    .factor_wr_en_swiglu_mlp  (factor_wr_en_swiglu_mlp),
    .factor_wr_en_attn        (factor_wr_en_attn),
    .factor_rd_sel            (factor_rd_sel),
    .factor_rd_data           (factor_rd_data)
  );

  // layer_base_addr = layer_idx × LAYER_BYTES (small constant mult; for
  // synthesisable cleanliness we let Vivado figure this out).
  always_comb layer_base_addr = layer_idx * LAYER_BYTES;

  smollm_layer #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX),
    .NL(NL),
    .PREFIX(PREFIX),
    .BASE_Q(BASE_Q), .BASE_K(BASE_K), .BASE_V(BASE_V), .BASE_O(BASE_O),
    .BASE_GATE(BASE_GATE), .BASE_UP(BASE_UP), .BASE_DOWN(BASE_DOWN)
  ) i_layer (
    .clk(clk), .rst(rst),
    .start(layer_start),
    .pos(pos), .kv_pos(kv_pos),
    .layer_idx(layer_idx),
    .layer_base_addr(layer_base_addr),
    .hidden_in(layer_hidden_in),
    .hidden_out(layer_hidden_out),
    .done(layer_done),
    .resid1_factor(cur_resid1_factor),
    .resid2_factor(cur_resid2_factor),
    .sh_h_in_to_h1(cur_sh_h_in_to_h1),
    .sh_h1_to_h_out(cur_sh_h1_to_h_out),
    .sg_gate_in_factor(cur_sg_gate_in_factor),
    .sg_up_in_factor  (cur_sg_up_in_factor),
    .sg_mlp_out_factor(cur_sg_mlp_out_factor),
    .attn_factor      (cur_attn_factor)
`ifdef MICROGPT_DDR3_WEIGHTS
    ,
    .clk_axi(clk_axi), .rst_axi(rst_axi),
    .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
    .m_axi_arid(m_axi_arid),       .m_axi_araddr(m_axi_araddr),
    .m_axi_arlen(m_axi_arlen),     .m_axi_arsize(m_axi_arsize),
    .m_axi_arburst(m_axi_arburst), .m_axi_arlock(m_axi_arlock),
    .m_axi_arcache(m_axi_arcache), .m_axi_arprot(m_axi_arprot),
    .m_axi_arqos(m_axi_arqos),
    .m_axi_rvalid(m_axi_rvalid),   .m_axi_rready(m_axi_rready),
    .m_axi_rid(m_axi_rid),         .m_axi_rdata(m_axi_rdata),
    .m_axi_rresp(m_axi_rresp),     .m_axi_rlast(m_axi_rlast),
    // Debug forwards
    .dbg_first_araddr   (dbg_first_araddr   ),
    .dbg_first_rdata    (dbg_first_rdata    ),
    .dbg_first_ar_seen  (dbg_first_ar_seen  ),
    .dbg_first_r_seen   (dbg_first_r_seen   ),
    .dbg_eng_w_packed   (dbg_eng_w_packed   ),
    .dbg_wd_packed      (dbg_wd_packed      ),
    .dbg_in_value_packed(dbg_in_value_packed),
    .dbg_snap_done_o    (dbg_snap_done_o    ),
    .ila_state          (ila_state          ),
    .ila_mv_phase       (ila_mv_phase       ),
    .ila_cnt            (ila_cnt            ),
    .ila_chunk          (ila_chunk          ),
    .ila_ws_matvec_id   (ila_ws_matvec_id   ),
    .ila_ws_load_req    (ila_ws_load_req    ),
    .ila_ws_ready       (ila_ws_ready       ),
    .ila_ws_rd_addr     (ila_ws_rd_addr     ),
    .ila_ws_weight_data (ila_ws_weight_data ),
    .ila_eng_w          (ila_eng_w          ),
    .ila_eng_in_value   (ila_eng_in_value   ),
    .ila_eng_in_valid   (ila_eng_in_valid   ),
    .ila_eng_in_last    (ila_eng_in_last    ),
    .ila_eng_acc_clear  (ila_eng_acc_clear  ),
    .ila_eng_out_valid  (ila_eng_out_valid  ),
    .ila_ws_state_axi   (ila_ws_state_axi   ),
    .ila_start_load_axi (ila_start_load_axi ),
    .ila_tile_idx       (ila_tile_idx       ),
    .ila_beat_idx       (ila_beat_idx       )
`endif
  );

  // ------------------------------------------------------------------
  //  Outer FSM — runs the layer NL times.
  //    LR_IDLE   wait for start
  //    LR_START  raise layer.start (held until layer.done seen)
  //    LR_LATCH  layer.done observed → capture hidden_out, drop start
  //    LR_GAP    wait for layer to return to S_IDLE (done=0); advance idx
  //    LR_DONE   all NL layers run; pulse wrapper.done
  // ------------------------------------------------------------------
  typedef enum logic [2:0] {
    LR_IDLE, LR_START, LR_LATCH, LR_GAP, LR_RESCALE, LR_DONE
  } lr_state_t;
  lr_state_t lr_state;

  logic signed [D*16-1:0] hidden_state;
  logic [3:0]             prev_h_out_p2;
  // cascade_shift = cur_h_in_p2 - prev_h_out_p2  (signed; >0 = right shift)
  wire signed [4:0]       cascade_shift =
        $signed({1'b0, cur_h_in_p2}) - $signed({1'b0, prev_h_out_p2});
  // Element-wise rescale of hidden_state to the new layer's h_in scale.
  // Combinational so layer_hidden_in <= hidden_state_rescaled in LR_RESCALE
  // captures the right values at the FSM edge.
  wire signed [D*16-1:0]  hidden_state_rescaled;
  genvar gh;
  generate
    for (gh = 0; gh < D; gh++) begin : g_cascade_rescale
      wire signed [15:0] he;
      wire signed [31:0] aligned;
      assign he = hidden_state[gh*16 +: 16];
      assign aligned = (cascade_shift >= 0) ?
            ($signed({{16{he[15]}}, he}) >>> cascade_shift) :
            ($signed({{16{he[15]}}, he}) <<< (-cascade_shift));
      assign hidden_state_rescaled[gh*16 +: 16] =
            (aligned >  32'sd32767) ?  16'sh7FFF :
            (aligned < -32'sd32768) ?  16'sh8000 : aligned[15:0];
    end
  endgenerate

  always_ff @(posedge clk) begin
    if (rst) begin
      lr_state        <= LR_IDLE;
      layer_idx       <= '0;
      layer_start     <= 1'b0;
      layer_hidden_in <= '0;
      hidden_state    <= '0;
      done            <= 1'b0;
    end else begin
      case (lr_state)
        LR_IDLE: begin
          done <= 1'b0;
          if (start) begin
            // First layer's hidden_in = wrapper's hidden_in.
            hidden_state    <= hidden_in;
            layer_hidden_in <= hidden_in;
            layer_idx       <= '0;
            layer_start     <= 1'b1;
            lr_state        <= LR_START;
          end
        end
        LR_START: begin
          // layer.start held high; wait for layer.done
          if (layer_done) begin
            hidden_state  <= layer_hidden_out;
            prev_h_out_p2 <= cur_h_out_p2;     // remember for cascade rescale
            layer_start   <= 1'b0;             // drop start; layer returns to S_IDLE
            lr_state      <= LR_LATCH;
          end
        end
        LR_LATCH: begin
          // 1 cycle for layer to transition S_DONE → S_IDLE
          lr_state <= LR_GAP;
        end
        LR_GAP: begin
          if (!layer_done) begin
            if (layer_idx == NL - 1) begin
              lr_state <= LR_DONE;
            end else begin
              layer_idx <= layer_idx + 1'b1;   // bump idx → cur_h_in_p2 reflects NEW layer
              lr_state  <= LR_RESCALE;
            end
          end
        end
        LR_RESCALE: begin
          // cur_h_in_p2 reflects the new layer (we bumped layer_idx in LR_GAP).
          // hidden_state_rescaled is comb-aligned to it; capture into FF.
          layer_hidden_in <= hidden_state_rescaled;
          layer_start     <= 1'b1;
          lr_state        <= LR_START;
        end
        LR_DONE: begin
          done <= 1'b1;
        end
        default: lr_state <= LR_IDLE;
      endcase
    end
  end

  // ----------------------------------------------------------------------
  // Per-layer hidden-state snapshot store.  30 RAMB16_S36_S36 instances
  // (Vivado retarget primitive → RAMB36E1 TDP 512×36 dual port, one per
  // layer).  Pack 2 × 16-bit lanes per 32-bit entry, 288 pairs used.
  // Port A = read (WEA=0), port B = write (WEB=wr_en_this).  Per-layer
  // write enable one-hot from captured_layer_idx.  Total: 30 RAMB36E1 ≈
  // 6% of VC707.
  // ----------------------------------------------------------------------
  logic [4:0]  snap_wr_layer;
  logic [8:0]  snap_wr_pair;
  logic [31:0] snap_wr_data;
  logic [4:0]  snap_rd_layer;
  logic [8:0]  snap_rd_pair;
  logic [31:0] snap_rd_data;

  wire [NL-1:0][31:0] snap_per_layer_dout;

  genvar gnl;
  generate
    for (gnl = 0; gnl < NL; gnl++) begin : g_snap_bram
      wire wr_en_this = (snap_wr_layer == gnl[4:0]);
      // ADDR is 16 bits for RAMB36E1; in 36-bit-wide mode the lower 5
      // are within-word byte-select, upper 9 bits = the 1024-deep
      // word addr (we use only 512 of 1024 entries → 288 pairs).
      wire [15:0] addr_a = {2'b0, snap_rd_pair, 5'b00000};
      wire [15:0] addr_b = {2'b0, snap_wr_pair, 5'b00000};
      RAMB36E1 #(
        .RAM_MODE       ("TDP"),
        .READ_WIDTH_A   (36),
        .WRITE_WIDTH_A  (0),
        .READ_WIDTH_B   (0),
        .WRITE_WIDTH_B  (36),
        .DOA_REG        (0),
        .DOB_REG        (0),
        .EN_ECC_READ    ("FALSE"),
        .EN_ECC_WRITE   ("FALSE"),
        .RAM_EXTENSION_A("NONE"),
        .RAM_EXTENSION_B("NONE"),
        .SIM_DEVICE     ("7SERIES"),
        .WRITE_MODE_A   ("WRITE_FIRST"),
        .WRITE_MODE_B   ("WRITE_FIRST"),
        .INIT_A         (36'h0),
        .INIT_B         (36'h0),
        .SRVAL_A        (36'h0),
        .SRVAL_B        (36'h0)
      ) i_snap_bram (
        .CLKARDCLK     (clk),
        .CLKBWRCLK     (clk),
        .ENARDEN       (1'b1),
        .ENBWREN       (1'b1),
        .REGCEAREGCE   (1'b0),
        .REGCEB        (1'b0),
        .RSTRAMARSTRAM (1'b0),
        .RSTRAMB       (1'b0),
        .RSTREGARSTREG (1'b0),
        .RSTREGB       (1'b0),
        .CASCADEINA    (1'b0),
        .CASCADEINB    (1'b0),
        .INJECTSBITERR (1'b0),
        .INJECTDBITERR (1'b0),
        // Port A = read (32-bit data + parity ignored).
        .ADDRARDADDR   (addr_a),
        .DOADO         (snap_per_layer_dout[gnl]),
        .DOBDO         (),
        .DOPADOP       (),
        .DOPBDOP       (),
        // Port B = write.  WEBWE[3:0] = byte enables (gated by
        // wr_en_this).  Tying WEA = 4'b0 prevents any port-A writes.
        .ADDRBWRADDR   (addr_b),
        .DIADI         (32'b0),
        .DIBDI         (snap_wr_data),
        .DIPADIP       (4'b0),
        .DIPBDIP       (4'b0),
        .WEA           (4'b0),
        .WEBWE         ({4'b0, {4{wr_en_this}}}),
        // Unused outputs — tied off to silence Synth 8-7023 warnings.
        .CASCADEOUTA   (),
        .CASCADEOUTB   (),
        .DBITERR       (),
        .ECCPARITY     (),
        .RDADDRECC     (),
        .SBITERR       ()
      );
    end
  endgenerate

  // Combinational mux across the 30 per-layer outputs.
  assign snap_rd_data = snap_per_layer_dout[snap_rd_layer];

  // Capture: a free-running counter walks word_pair 0..D/2-1 continuously.
  // captured_layer_idx latched at LR_LATCH = the just-finished layer.
  // hidden_state is stable for ~250K cycles afterwards (until the next
  // layer's done), so the layer's BRAM gets repeatedly overwritten with
  // the same correct snapshot.  No FSM needed.
  localparam int D_PAIRS = D / 2;        // 288
  logic [8:0] cap_pair;
  logic [4:0] captured_layer_idx;
  always_ff @(posedge clk) begin
    if (rst) begin
      cap_pair           <= '0;
      captured_layer_idx <= '0;
    end else begin
      cap_pair <= (cap_pair == D_PAIRS-1) ? '0 : cap_pair + 1'b1;
      if (lr_state == LR_LATCH) captured_layer_idx <= layer_idx;
    end
  end
  assign snap_wr_layer = captured_layer_idx;
  assign snap_wr_pair  = cap_pair;
  assign snap_wr_data  = hidden_state[cap_pair*32 +: 32];

  // Refresh: walk word_pair 0..D/2-1 reading snap_ram into the wide bus.
  // ~D/2 = 288 cycles latency for a snap_sel change to fully propagate.
  logic [8:0] refresh_pair;
  logic [8:0] refresh_pair_q1;
  always_ff @(posedge clk) begin
    // Reset refresh to D_PAIRS/2 so it stays exactly 144 cycles offset
    // from cap_pair (which resets to 0).  Both counters increment in
    // lockstep — the offset prevents cross-port same-address BRAM
    // collisions that the unisim model treats as X (and that hardware
    // would also produce undefined results for in WRITE_FIRST mode).
    if (rst) refresh_pair <= 9'(D_PAIRS/2);
    else     refresh_pair <= (refresh_pair == D_PAIRS-1) ? '0 : refresh_pair + 1'b1;
    refresh_pair_q1 <= refresh_pair;
  end
  wire [4:0] snap_sel_clamped = (snapshot_layer_sel >= NL[4:0])
                                  ? 5'(NL-1) : snapshot_layer_sel;
  assign snap_rd_layer = snap_sel_clamped;
  assign snap_rd_pair  = refresh_pair;
  // snap_rd_data aligns with refresh_pair_q1 (1-cycle BRAM latency).
  logic [D*16-1:0] refreshed_snap_bus;
  always_ff @(posedge clk) begin
    refreshed_snap_bus[refresh_pair_q1*32 +: 32] <= snap_rd_data;
  end

  assign hidden_out = (snapshot_layer_sel >= NL[4:0])
                        ? hidden_state
                        : refreshed_snap_bus;

  // Outer FSM visibility (declared after FSM body so xsim/xvlog accept
  // the references — Verilator was lenient with forward refs).
`ifdef MICROGPT_DDR3_WEIGHTS
  assign ila_ml_state     = lr_state;
  assign ila_ml_layer_idx = layer_idx;
`endif

endmodule

`default_nettype wire
