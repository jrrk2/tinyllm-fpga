// smollm_layer_bfp.sv — block-FP version of a single SmolLM2 transformer
// layer.  Wires the 6 verified block-FP leaf ops (matvec_bfp_engine,
// rmsnorm_bfp, rope_bfp, softmax_bfp, swiglu_bfp, residual_bfp) into a
// complete pre-norm transformer block.  All hidden state, weights, and
// activations are in the block-FP format defined in bfp_format.svh:
//   value = mantissa / 2^15 * 2^exp     (TILE=16 mantissas per shared exp).
//
// FSM:
//   S_RMSNORM1 → S_QMV → S_KMV → S_VMV → S_ROPE_Q → S_ROPE_K → S_KV_WRITE
//             → for head: S_QK → S_SM → S_AV
//             → S_OMV → S_RES1
//             → S_RMSNORM2 → S_GMV → S_UMV → S_SWG → S_DMV → S_RES2
//             → S_DONE
//
// One shared matvec engine (LANES=16) handles all 7 weight matrix
// multiplies plus the per-head QK^T scores and AV; chunked by output
// dimension so the 16 output lanes fill (out_dim / 16) times per matvec.
// Weights and KV cache init from $readmemh — DDR3 streamer can replace.
//
// All inter-engine wiring is registered (single non-blocking driver per
// signal) to avoid multi-driver synthesis errors and to keep timing
// predictable for VC707 (50 MHz core clock).

`include "bfp_format.svh"

`default_nettype none

module smollm_layer_bfp #(
  parameter int    D       = 64,
  parameter int    H_Q     = 1,
  parameter int    H_KV    = 1,
  parameter int    HD      = 64,
  parameter int    FFN     = 128,
  parameter int    MAX_CTX = 4,
  // Number of layers stored in the weight / KV-cache ROMs (time-mux).
  // NL=1 → single-layer selftest (weights addressed without layer offset);
  // NL>1 → multilayer mode where layer_idx selects the active layer's
  // ROM bank.  KV cache likewise scales by NL.
  parameter int    NL      = 1,
  parameter        PREFIX  = "lbfp_",
  parameter int    AXI_ADDR_WIDTH = 30,
  parameter int    AXI_ID_WIDTH   = 5
)(
  input  wire                                clk,
  input  wire                                rst,
  input  wire                                start,
  input  wire [10:0]                         pos,
  input  wire [6:0]                          kv_pos,       // up to MAX_CTX-1
  input  wire [$clog2(NL+1)-1:0]             layer_idx,    // 0..NL-1
  input  wire signed [D*BFP_MANT_W-1:0]      hidden_in_m,
  input  wire signed [(D/BFP_TILE)*BFP_EXP_W-1:0] hidden_in_e,
  output logic signed [D*BFP_MANT_W-1:0]     hidden_out_m,
  output logic signed [(D/BFP_TILE)*BFP_EXP_W-1:0] hidden_out_e,
  output logic                               done,
  // -------------------------------------------------------------------
  // DDR3 streamer interface.  Caller (autoregress_bfp_top / multilayer_
  // tm) computes the layer-qualified byte base for each of the 7
  // matrices and presents them here; the layer applies chunk_idx *
  // in_dim_bytes internally via the streamer.  AXI master ports route
  // to MIG ui_clk domain.
  // -------------------------------------------------------------------
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WQ_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WQ_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WK_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WK_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WV_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WV_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WO_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WO_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WG_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WG_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WU_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WU_e,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WDN_m,
  input  wire [AXI_ADDR_WIDTH-1:0]           ws_base_WDN_e,
  input  wire                                clk_axi,
  input  wire                                rst_axi,
  output wire                                m_axi_arvalid,
  input  wire                                m_axi_arready,
  output wire [AXI_ID_WIDTH-1:0]             m_axi_arid,
  output wire [AXI_ADDR_WIDTH-1:0]           m_axi_araddr,
  output wire [7:0]                          m_axi_arlen,
  output wire [2:0]                          m_axi_arsize,
  output wire [1:0]                          m_axi_arburst,
  output wire                                m_axi_arlock,
  output wire [3:0]                          m_axi_arcache,
  output wire [2:0]                          m_axi_arprot,
  output wire [3:0]                          m_axi_arqos,
  input  wire                                m_axi_rvalid,
  output wire                                m_axi_rready,
  input  wire [AXI_ID_WIDTH-1:0]             m_axi_rid,
  input  wire [511:0]                        m_axi_rdata,
  input  wire [1:0]                          m_axi_rresp,
  input  wire                                m_axi_rlast,
  // Host write port for the weight / gamma / KV-init BRAMs.  Lets the host
  // patch BFP weights post-bitstream without re-synth (matches the existing
  // scale-brom flow for the int8 path).
  //   wr_kind: 0=WQ_m 1=WQ_e 2=WK_m 3=WK_e 4=WV_m 5=WV_e 6=WO_m 7=WO_e
  //            8=WG_m 9=WG_e 10=WU_m 11=WU_e 12=WDN_m 13=WDN_e
  //            14=G1_m 15=G1_e 16=G2_m 17=G2_e
  //            18=K_INIT_m 19=K_INIT_e 20=V_INIT_m 21=V_INIT_e
  //   wr_addr: per-rom entry index
  //   wr_data: 16-bit mantissa (lower 16) or 8-bit exponent (lower 8)
  //   wr_en : 1-cycle pulse
  input  wire [4:0]                          wr_kind,
  input  wire [17:0]                         wr_addr,
  input  wire [15:0]                         wr_data,
  input  wire                                wr_en,
  input  wire                                clk_wr,    // BRAM write clock (eth_clk at top)
  // Read-back of the BRAM at wr_addr — uses port A (clk_wr) of the same
  // TDP BRAM, registered output (1-cycle latency).  Muxed by wr_kind so
  // the caller can verify what was loaded.  Exponents are zero-extended
  // to 16 b.  Kinds outside this module's range return 0; the upstream
  // mux selects between this output and the decode-head/prompt outputs.
  output logic [15:0]                        wr_rdata,
  // Per-layer hidden-state snapshot select (driven from the multilayer / eth
  // top).  When layer_idx==snap_layer_sel AND pos==snap_step_sel, this layer's
  // hidden_out is mirrored into a snap_m/snap_e TDP BRAM (same pattern as hout)
  // for host read-back via wr_kind 12 (mantissa) / 13 (per-tile exponent).
  input  wire [4:0]                          snap_layer_sel,
  input  wire [10:0]                         snap_step_sel,
  // Per-stage exponent read-out select (logic analyser).  Read via wr_kind 16:
  //  0 n1 1 q 2 k 3 v 4 attn 5 o 6 h1 7 n2 8 g 9 u 10 mlp 11 d 12 hout 13 hin.
  //  Exposes each stage's per-tile BFP exponent so the host can see which
  //  stage's exponent saturates first within a frozen layer.
  input  wire [4:0]                          dbg_stage_sel,
  // Debug taps (synthesis-friendly — only used by sim testbench)
  output logic [6:0]                         dbg_state,
  output logic [11:0]                        dbg_cnt,
  output logic [6:0]                         dbg_chunk,
  // Comparative hash of every weight beat the matvec engine consumes
  // from the streamer (mv_w_m_eff || mv_w_e_eff captured each cycle
  // mv_valid && is_stream_matvec).  Resets on `start` rising edge.
  // 384-bit input per cycle, ~3 LUT levels of XOR-tree at 40 MHz — fits
  // core_clk's 25 ns budget trivially.  Useful for cross-run comparison:
  // hash(Python upload) vs hash(C++ upload).
  output logic [31:0]                        weight_hash
);

  // ---------------------------------------------------------------------------
  // Derived sizes
  // ---------------------------------------------------------------------------
  localparam int NT_D    = (D + BFP_TILE - 1) / BFP_TILE;
  localparam int NT_KV   = (H_KV*HD + BFP_TILE - 1) / BFP_TILE;
  localparam int NT_FFN  = (FFN + BFP_TILE - 1) / BFP_TILE;
  localparam int LANES   = 16;
  localparam int CHUNKS_D   = D    / LANES;
  localparam int CHUNKS_KV  = (H_KV*HD) / LANES;
  localparam int CHUNKS_FFN = FFN  / LANES;
  localparam int CHUNKS_HD  = HD   / LANES;

  // Counter widths (max of D, FFN, H_KV*HD)
  localparam int CW_D    = $clog2(D    + 1);
  localparam int CW_FFN  = $clog2(FFN  + 1);
  localparam int CW_KV   = $clog2(H_KV*HD + 1);
  localparam int CW_HD   = $clog2(HD   + 1);
  localparam int CW_CTX  = $clog2(MAX_CTX + 1);

  // Wide-packed ROM widths: each entry holds LANES mantissas (or exponents)
  // concatenated, so one BRAM read per cycle returns all 16 lanes worth
  // of data — single-port access matches what Vivado can infer / cascade
  // RAMB18/36 primitives for.  The previous per-lane unpacked array was
  // a 16-port read (one address per lane), which Vivado refuses to BRAM-
  // infer (max 2 read ports) and is too big for distributed RAM (>128 Kb).
  // Gamma / KV cache: scaled by NL for multilayer time-mux.  Indexed
  // by {layer_idx, position, lane}.
  (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] rom_G1_m [0:NL*D-1];
  (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] rom_G2_m [0:NL*D-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] rom_G1_e [0:NL*NT_D-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] rom_G2_e [0:NL*NT_D-1];

`ifdef VERILATOR
  // In production these ROMs are filled at boot via the wr_kind port
  // (host upload).  The bfp-layer testbench leaves wr_en tied to 0, so
  // without this init the RMSNorm gammas would all be zero and every
  // matvec would receive zero activations regardless of weights.  We
  // also seed the KV cache: kv_k_m / kv_k_e / kv_v_e are direct
  // single-element-per-line .hex matches for the bfp_sdpram mem arrays;
  // kv_v_chk stores LANES-packed mantissas so we load a flat shadow
  // and pack lane-by-lane below.
  logic [BFP_MANT_W-1:0] sim_vinit_flat [0:NL*MAX_CTX*H_KV*HD-1];
  initial begin
    $readmemh("../generated/lbfp_G1_m.hex", rom_G1_m);
    $readmemh("../generated/lbfp_G1_e.hex", rom_G1_e);
    $readmemh("../generated/lbfp_G2_m.hex", rom_G2_m);
    $readmemh("../generated/lbfp_G2_e.hex", rom_G2_e);
    $readmemh("../generated/lbfp_K_INIT_m.hex", i_kv_k_m_bram.mem);
    $readmemh("../generated/lbfp_K_INIT_e.hex", i_kv_k_e_bram.mem);
    $readmemh("../generated/lbfp_V_INIT_e.hex", i_kv_v_e_bram.mem);
    $readmemh("../generated/lbfp_V_INIT_m.hex", sim_vinit_flat);
    for (int e = 0; e < NL*MAX_CTX*KVV_CHUNKS_PER_KVPOS; e++)
      for (int l = 0; l < LANES; l++)
        i_kv_v_chk_bram.mem[e][l*BFP_MANT_W +: BFP_MANT_W] =
            sim_vinit_flat[e*LANES + l];
    $display("[lay] kv_k_m[0]=%h kv_k_m[1]=%h kv_k_m[63]=%h",
             i_kv_k_m_bram.mem[0], i_kv_k_m_bram.mem[1], i_kv_k_m_bram.mem[63]);
    $display("[lay] kv_k_e[0]=%h kv_v_e[0]=%h",
             i_kv_k_e_bram.mem[0], i_kv_v_e_bram.mem[0]);
    $display("[lay] kv_v_chk[0]=%h", i_kv_v_chk_bram.mem[0]);
  end

  // Snapshot q_m's packed tiles after the Q matvec + RoPE finish, so we
  // can see whether Q is being computed/stored correctly even though we
  // no longer dump q_m via $writememh (packed BRAM).
  always_ff @(posedge clk) begin
    if (state == S_ROPEQ_RQ_B && cnt == NT_D - 1) begin
      $display("[lay] post-RoPE q_m tiles: t0=%h t1=%h t2=%h t3=%h",
               i_q_m_bram.i_word_ram.mem[0],
               i_q_m_bram.i_word_ram.mem[1],
               i_q_m_bram.i_word_ram.mem[2],
               i_q_m_bram.i_word_ram.mem[3]);
      $display("[lay] post-RoPE k_m tiles: t0=%h t1=%h t2=%h t3=%h",
               i_k_m_bram.i_word_ram.mem[0],
               i_k_m_bram.i_word_ram.mem[1],
               i_k_m_bram.i_word_ram.mem[2],
               i_k_m_bram.i_word_ram.mem[3]);
    end
  end
`endif

  // kv_k_m / kv_k_e / kv_v_e / kv_v_m_chk all live in explicit RAMB36E1
  // primitives via bfp_sdpram.  Vivado's inferencer refused BRAM for the
  // three kv_k_*/kv_v_e arrays ("Infeasible attribute ram_style=block")
  // because their reads were buried in conditional muxes inside the main
  // FSM always_ff (the `if (kv_t == kv_pos) k_m else kv_k_m` pattern),
  // which the pattern-matcher couldn't unpick — falling back to LUTRAM
  // (~92K cells for kv_k_m alone, blowing the 130800 RAMD64E budget).
  //
  // Explicit primitives sidestep inference entirely.  Read latency is
  // 1 cycle (RAMB output register); FSM consumers prefetch rd_addr one
  // cycle ahead and use consume_t = cnt-1 to pick the matching xs.
  localparam int KVV_CHUNKS_PER_KVPOS = (H_KV * HD) / LANES;          // 12 for full SmolLM2
  localparam int KVV_DEPTH            = NL * MAX_CTX * KVV_CHUNKS_PER_KVPOS;
  localparam int KVV_AW               = $clog2(KVV_DEPTH);
  logic                           kv_v_chk_we;
  logic [KVV_AW-1:0]              kv_v_chk_wr_addr;
  logic [LANES*BFP_MANT_W-1:0]    kv_v_chk_wr_data;
  logic [KVV_AW-1:0]              kv_v_chk_rd_addr;
  wire  [LANES*BFP_MANT_W-1:0]    kv_v_chk_rd_data;

  bfp_sdpram #(
    .DEPTH (KVV_DEPTH),
    .WIDTH (LANES * BFP_MANT_W)
  ) i_kv_v_chk_bram (
    .clk     (clk),
    .rst     (rst),
    .we      (kv_v_chk_we),
    .wr_addr (kv_v_chk_wr_addr),
    .wr_data (kv_v_chk_wr_data),
    .rd_addr (kv_v_chk_rd_addr),
    .rd_data (kv_v_chk_rd_data)
  );

  // Comb drivers for the kv_v_m_chk BRAM ports.  Both wr and rd
  // sides are driven entirely by state + cnt + layer_idx + kv_pos +
  // head_grp + chunk + v_m so there's no FSM-internal write-decode
  // tangle for Vivado to unpack.  rd_addr is issued ahead of cnt by
  // one cycle (driven by `cnt`, which the AV_DRIVE pipeline advances
  // every cycle); rd_data arrives one cycle later, consumed via
  // consume_t = cnt - 1 in S_AV_DRIVE.
  always_comb begin
    kv_v_chk_we      = 1'b0;
    kv_v_chk_wr_addr = '0;
    kv_v_chk_wr_data = '0;
    // KVWR_M writes the wide-packed entry once per 16-lane chunk
    // (cnt[3:0]==15) — assemble lanes from v_m (small FF-backed array,
    // 16-port read is cheap).
    if (state == S_KVWR_M && cnt[3:0] == 4'd15) begin
      kv_v_chk_we      = 1'b1;
      kv_v_chk_wr_addr = KVV_AW'(layer_idx * MAX_CTX * KVV_CHUNKS_PER_KVPOS
                                 + kv_pos * KVV_CHUNKS_PER_KVPOS
                                 + cnt[CW_KV-1:4]);
      kv_v_chk_wr_data = v_m_rd_data_packed;
    end
    // Default rd_addr points at addr(cnt) using the current head_grp /
    // chunk.  The S_AV_PREFETCH and S_AV_DRIVE states drive cnt so the
    // BRAM receives the right address each cycle.
    kv_v_chk_rd_addr = KVV_AW'(layer_idx * MAX_CTX * KVV_CHUNKS_PER_KVPOS
                                + cnt[CW_CTX-1:0] * KVV_CHUNKS_PER_KVPOS
                                + head_grp * (HD/LANES) + chunk);
  end

  // Comb drivers for kv_k_m / kv_k_e / kv_v_e BRAM ports.  Writes
  // mirror the in-FSM stores during S_KVWR_M / S_KVWR_E; rd_addr is
  // driven from the current FSM cursor a cycle ahead of consumption
  // (consume_t = cnt - 1 in the DRIVE states).
  always_comb begin
    kv_k_m_we      = 1'b0;
    kv_k_m_wr_addr = '0;
    kv_k_m_wr_data = '0;
    if (state == S_KVWR_M) begin
      kv_k_m_we      = 1'b1;
      kv_k_m_wr_addr = KKM_AW'(layer_idx * MAX_CTX * H_KV * HD
                                + kv_pos * H_KV * HD
                                + cnt[CW_KV-1:0]);
      kv_k_m_wr_data = k_m_rd_data;
    end
    // Default rd_addr: QK_DRIVE pattern (kv_t × stride + head_grp × HD + cnt).
    // S_QK_PREFETCH sets cnt=0 so addr(t=kv_t, j=0) is latched; DRIVE
    // bumps cnt to 1..HD and consumes rd_data at consume_j = cnt-1.
    kv_k_m_rd_addr = KKM_AW'(layer_idx * MAX_CTX * H_KV * HD
                              + kv_t * H_KV * HD
                              + head_grp * HD
                              + cnt[CW_HD-1:0]);
  end

  always_comb begin
    kv_k_e_we      = 1'b0;
    kv_k_e_wr_addr = '0;
    kv_k_e_wr_data = '0;
    kv_v_e_we      = 1'b0;
    kv_v_e_wr_addr = '0;
    kv_v_e_wr_data = '0;
    if (state == S_KVWR_E) begin
      kv_k_e_we      = 1'b1;
      kv_k_e_wr_addr = KVE_AW'(layer_idx * MAX_CTX * NT_KV
                                + kv_pos * NT_KV + cnt[CW_KV-1:0]);
      kv_k_e_wr_data = k_e[cnt[CW_KV-1:0]];
      kv_v_e_we      = 1'b1;
      kv_v_e_wr_addr = KVE_AW'(layer_idx * MAX_CTX * NT_KV
                                + kv_pos * NT_KV + cnt[CW_KV-1:0]);
      kv_v_e_wr_data = v_e[cnt[CW_KV-1:0]];
    end
    // kv_k_e read addr: tracks QK_DRIVE lane-0 pattern.
    kv_k_e_rd_addr = KVE_AW'(layer_idx * MAX_CTX * NT_KV
                              + kv_t * NT_KV
                              + (head_grp * HD + cnt[CW_HD-1:0]) / BFP_TILE);
    // kv_v_e read addr: two contexts.
    //   S_AV_EMAX_SCAN: walk t = av_scan_cnt, looking at the tile for
    //     (head_grp*HD + chunk*LANES) / BFP_TILE.
    //   Otherwise (S_AV_PREFETCH / S_AV_DRIVE): tracks cnt (= prefetch
    //     index) for the same tile.
    if (state == S_AV_PRIME || state == S_AV_EMAX_PREFETCH
                            || state == S_AV_EMAX_SCAN) begin
      kv_v_e_rd_addr = KVE_AW'(layer_idx * MAX_CTX * NT_KV
                                + av_scan_cnt[CW_CTX-1:0] * NT_KV
                                + (head_grp * HD + chunk * LANES) / BFP_TILE);
    end else begin
      kv_v_e_rd_addr = KVE_AW'(layer_idx * MAX_CTX * NT_KV
                                + cnt[CW_CTX-1:0] * NT_KV
                                + (head_grp * HD + chunk * LANES) / BFP_TILE);
    end
  end

  // ---------------------------------------------------------------------------
  // mlp_m BRAM ports — pure wire drives (no FSM modification needed).
  // - rd_addr tracks cnt directly.  The bfp_sdpram's own internal output
  //   register provides the 1-cycle latency that the OLD `mv_x_m <=
  //   mlp_m[cnt]` had, so mlp_m_rd_data at cycle T = mlp_m[cnt at T-1].
  //   Matvec is fed via the mv_x_m_eff mux below, which selects rd_data
  //   in S_DMV_DRIVE and the existing mv_x_m register elsewhere.  No
  //   extra register, no lookahead.
  // - we asserts only in the write states; outside those states the BRAM
  //   is read-only.  wr_addr / wr_data are valid when we is high.
  // ---------------------------------------------------------------------------
  assign mlp_m_we      = (state == S_SWG || state == S_SWG_WAIT) && sg_y_valid;
  assign mlp_m_wr_addr = out_cnt[CW_FFN-1:0];
  assign mlp_m_wr_data = sg_y_m;
  assign mlp_m_rd_addr = cnt[CW_FFN-1:0];

  // n1_m: written in S_NORM1_WAIT (one cycle per cnt step), read by all
  // three of Q/K/V matvec DRIVE states using cnt as the index.
  assign n1_m_we      = (state == S_NORM1_WAIT) && rn_y_valid;
  assign n1_m_wr_addr = cnt[CW_D-1:0];
  assign n1_m_wr_data = rn_y_m;
  assign n1_m_rd_addr = cnt[CW_D-1:0];

  // n2_m: written in S_NORM2_WAIT, read by both G and U matvec DRIVE states.
  assign n2_m_we      = (state == S_NORM2_WAIT) && rn_y_valid;
  assign n2_m_wr_addr = cnt[CW_D-1:0];
  assign n2_m_wr_data = rn_y_m;
  assign n2_m_rd_addr = cnt[CW_D-1:0];

  // q_rot_m: written in S_ROPEQ_WAIT (one entry per rp_y_valid cycle),
  // read LANES-parallel in S_ROPEQ_RQ_B (rd_addr pre-driven in
  // S_ROPEQ_RQ_A so BRAM latency aligns).
  assign q_rot_m_we      = (state == S_ROPEQ_WAIT) && rp_y_valid;
  assign q_rot_m_wr_addr = CW_D'(head_idx * HD + cnt[CW_HD-1:0]);
  assign q_rot_m_wr_data = rp_y_m;
  assign q_rot_m_rd_addr = cnt[$clog2(NT_D+1)-1:0];

  // k_rot_m: same pattern, written in S_ROPEK_WAIT, read LANES-parallel
  // in S_ROPEK_RQ_B (rd_addr pre-driven in S_ROPEK_RQ_A).
  assign k_rot_m_we      = (state == S_ROPEK_WAIT) && rp_y_valid;
  assign k_rot_m_wr_addr = ($clog2(H_KV*HD))'(head_idx * HD + cnt[CW_HD-1:0]);
  assign k_rot_m_wr_data = rp_y_m;
  assign k_rot_m_rd_addr = cnt[$clog2(NT_KV+1)-1:0];

  // attn_m: packed bfp_sdpram (LANES entries per BRAM word).
  // Write side: S_AV_REQ pack-and-store of the 16 requant_mant outputs.
  //   Same combinational requant_mant calls as the OLD for-loop, just
  //   assembled into the packed wr_data bus instead of writing the
  //   inferred array.  emax_f reused combinationally.
  // Read side:  S_OMV_DRIVE single-element read; rd_addr = cnt, output
  //   feeds mv_x_m_eff mux.
  assign attn_m_we           = (state == S_AV_REQ);
  assign attn_m_wr_addr_tile = ($clog2((D+LANES-1)/LANES))'(
                                  (av_row_base + chunk * LANES) / LANES);
  assign attn_m_rd_addr      = cnt[CW_D-1:0];
  // emax_f for the matvec-output requant (shared across packed writers
  // that use mv_out_*_drain_r as the source — attn_m / o_m / d_m).
  wire signed [BFP_EXP_W-1:0] mv_drain_emax_f =
      (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
  for (genvar pack_ii = 0; pack_ii < LANES; pack_ii++) begin : g_attn_m_pack
    assign attn_m_wr_data_packed[pack_ii*BFP_MANT_W +: BFP_MANT_W] =
      requant_mant(
        mv_out_m_drain_r[pack_ii*BFP_MANT_W +: BFP_MANT_W],
        mv_out_e_drain_r[pack_ii*BFP_EXP_W  +: BFP_EXP_W ],
        mv_drain_emax_f);
  end

  // o_m: same shape as attn_m's write side (LANES-parallel requant of
  // mv_out_*_drain_r); read is combinational with cnt+1 lookahead.
  assign o_m_we           = (state == S_OMV_REQ);
  assign o_m_wr_addr_tile = ($clog2((D+LANES-1)/LANES))'(chunk);
  // Gate the +1 lookahead on rs_in_ready: cnt only advances when the
  // residual is ready (in_ready=1 && rs_valid_c).  When residual is
  // stalled (e.g. starting up from IDLE→LOAD, or during S_EMIT cycles),
  // cnt stays put and we must keep rd_addr at cnt so the BRAM doesn't
  // pre-fetch the next element and feed it as the stalled-cycle input.
  assign o_m_rd_addr      = (state == S_RES1)
                              ? cnt[CW_D-1:0] + {{(CW_D-1){1'b0}}, rs_in_ready}
                              : '0;
  for (genvar pack_ii = 0; pack_ii < LANES; pack_ii++) begin : g_o_m_pack
    assign o_m_wr_data_packed[pack_ii*BFP_MANT_W +: BFP_MANT_W] =
      requant_mant(
        mv_out_m_drain_r[pack_ii*BFP_MANT_W +: BFP_MANT_W],
        mv_out_e_drain_r[pack_ii*BFP_EXP_W  +: BFP_EXP_W ],
        mv_drain_emax_f);
  end

  // d_m: same pattern, read in S_RES2.
  assign d_m_we           = (state == S_DMV_REQ);
  assign d_m_wr_addr_tile = ($clog2((D+LANES-1)/LANES))'(chunk);
  assign d_m_rd_addr      = (state == S_RES2)
                              ? cnt[CW_D-1:0] + {{(CW_D-1){1'b0}}, rs_in_ready}
                              : '0;
  for (genvar pack_ii = 0; pack_ii < LANES; pack_ii++) begin : g_d_m_pack
    assign d_m_wr_data_packed[pack_ii*BFP_MANT_W +: BFP_MANT_W] =
      requant_mant(
        mv_out_m_drain_r[pack_ii*BFP_MANT_W +: BFP_MANT_W],
        mv_out_e_drain_r[pack_ii*BFP_EXP_W  +: BFP_EXP_W ],
        mv_drain_emax_f);
  end

  // q_m: write side multiplexes between S_QMV_REQ (matvec drain → requant)
  // and S_ROPEQ_RQ_B (q_rot tile → tile-quant requant).  Read side has two
  // sequential consumers — S_ROPEQ (per-element into rp_x_m, 1-cycle ahead
  // lookahead `head_idx*HD + cnt + 1`) and S_QK_DRIVE (per-element into
  // mv_x_m, addr = `head_idx*HD + cnt` because S_QK_PREFETCH absorbs the
  // 1-cycle lag).  S_ROPEQ_WAIT drives `(head_idx+1)*HD` so the BRAM has
  // the next head's element 0 already-fetched when S_ROPEQ resumes.
  assign q_m_we           = (state == S_QMV_REQ) || (state == S_ROPEQ_RQ_B);
  assign q_m_wr_addr_tile =
      (state == S_ROPEQ_RQ_B) ? ($clog2((D+LANES-1)/LANES))'(cnt[$clog2(NT_D+1)-1:0])
                              : ($clog2((D+LANES-1)/LANES))'(chunk);
  always_comb begin
    case (state)
      S_ROPEQ:        q_m_rd_addr = CW_D'(head_idx * HD) + cnt[CW_D-1:0] + 1'b1;
      S_ROPEQ_WAIT:   q_m_rd_addr = CW_D'((head_idx + 1) * HD);
      S_QK_PRIME,
      S_QK_PREFETCH,
      S_QK_DRIVE:     q_m_rd_addr = CW_D'(head_idx * HD) + cnt[CW_D-1:0];
      default:        q_m_rd_addr = '0;
    endcase
  end
  // wr_data_packed: per-lane combinational mux on state.
  //   S_QMV_REQ:    requant(mv_out_*_drain_r, mv_drain_emax_f)
  //   S_ROPEQ_RQ_B: requant(q_rot_m_rd_data_packed, q_rot_e[base+lane], rq_emax)
  //                 where base = cnt * BFP_TILE = cnt * LANES (BFP_TILE==LANES==16).
  wire signed [BFP_EXP_W-1:0] q_rq_emax =
      (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
  for (genvar pack_ii = 0; pack_ii < LANES; pack_ii++) begin : g_q_m_pack
    wire [BFP_MANT_W-1:0] qmv_data =
        requant_mant(
          mv_out_m_drain_r[pack_ii*BFP_MANT_W +: BFP_MANT_W],
          mv_out_e_drain_r[pack_ii*BFP_EXP_W  +: BFP_EXP_W ],
          mv_drain_emax_f);
    wire [BFP_MANT_W-1:0] qrq_data =
        requant_mant(
          $signed(q_rot_m_rd_data_packed[pack_ii*BFP_MANT_W +: BFP_MANT_W]),
          q_rot_e[cnt[$clog2(NT_D+1)-1:0] * BFP_TILE + pack_ii],
          q_rq_emax);
    assign q_m_wr_data_packed[pack_ii*BFP_MANT_W +: BFP_MANT_W] =
      (state == S_ROPEQ_RQ_B) ? qrq_data : qmv_data;
  end

  // k_m: mirror of q_m.  Writes in S_KMV_REQ (chunk-indexed, matvec drain
  // source) and S_ROPEK_RQ_B (cnt-tile-indexed, k_rot source).  Reads in
  // S_ROPEK (rp_x_m, head_idx*HD+cnt lookahead, S_ROPEK_WAIT prefetch) and
  // S_KVWR_M (cnt+1 lookahead so the comb-driver always_comb at top
  // gets k_m[cnt] = k_m_rd_data at the cycle it's needed).
  assign k_m_we           = (state == S_KMV_REQ) || (state == S_ROPEK_RQ_B);
  assign k_m_wr_addr_tile =
      (state == S_ROPEK_RQ_B) ? ($clog2((H_KV*HD+LANES-1)/LANES))'(cnt[$clog2(NT_KV+1)-1:0])
                              : ($clog2((H_KV*HD+LANES-1)/LANES))'(chunk);
  always_comb begin
    case (state)
      S_ROPEK:        k_m_rd_addr = CW_KV'(head_idx * HD) + cnt[CW_KV-1:0] + 1'b1;
      S_ROPEK_WAIT:   k_m_rd_addr = CW_KV'((head_idx + 1) * HD);
      S_KVWR_M:       k_m_rd_addr = cnt[CW_KV-1:0] + 1'b1;
      default:        k_m_rd_addr = '0;
    endcase
  end
  wire signed [BFP_EXP_W-1:0] k_rq_emax =
      (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
  for (genvar pack_ii = 0; pack_ii < LANES; pack_ii++) begin : g_k_m_pack
    wire [BFP_MANT_W-1:0] kmv_data =
        requant_mant(
          mv_out_m_drain_r[pack_ii*BFP_MANT_W +: BFP_MANT_W],
          mv_out_e_drain_r[pack_ii*BFP_EXP_W  +: BFP_EXP_W ],
          mv_drain_emax_f);
    wire [BFP_MANT_W-1:0] krq_data =
        requant_mant(
          $signed(k_rot_m_rd_data_packed[pack_ii*BFP_MANT_W +: BFP_MANT_W]),
          k_rot_e[cnt[$clog2(NT_KV+1)-1:0] * BFP_TILE + pack_ii],
          k_rq_emax);
    assign k_m_wr_data_packed[pack_ii*BFP_MANT_W +: BFP_MANT_W] =
      (state == S_ROPEK_RQ_B) ? krq_data : kmv_data;
  end

  // v_m: single LANES-parallel write (S_VMV_REQ, matvec drain → requant).
  // Reads are LANES-parallel at LANES-aligned base from TWO consumers:
  //   - always_comb @ S_KVWR_M&cnt[3:0]==15 → kv_v_chk_wr_data (tile = cnt>>4)
  //   - always_ff  @ S_AV_DRIVE&consume_t==kv_pos → mv_w_m  (tile = head_grp*(HD/LANES)+chunk)
  // S_KVWR_M reads sequence: cnt[3:0] cycles 0..15; we need rd_data_packed
  //   stable for cnt[3:0]==15, which means rd_addr_tile = cnt>>4 throughout
  //   (the BRAM re-issues the same tile address every cycle of the 16-step
  //   window).  S_AV_DRIVE reads at a single iteration (consume_t==kv_pos);
  //   driving tile from S_AV_PREFETCH onward gives the BRAM ≥1 cycle to settle.
  assign v_m_we           = (state == S_VMV_REQ);
  assign v_m_wr_addr_tile = VMT_AW'(chunk);
  always_comb begin
    case (state)
      S_KVWR_M:    v_m_rd_addr_tile = VMT_AW'(cnt[CW_KV-1:4]);
      S_AV_PREFETCH,
      S_AV_DRIVE:  v_m_rd_addr_tile =
                     VMT_AW'((head_grp * HD + chunk * LANES) / LANES);
      default:     v_m_rd_addr_tile = '0;
    endcase
  end
  for (genvar pack_ii = 0; pack_ii < LANES; pack_ii++) begin : g_v_m_pack
    assign v_m_wr_data_packed[pack_ii*BFP_MANT_W +: BFP_MANT_W] =
      requant_mant(
        mv_out_m_drain_r[pack_ii*BFP_MANT_W +: BFP_MANT_W],
        mv_out_e_drain_r[pack_ii*BFP_EXP_W  +: BFP_EXP_W ],
        mv_drain_emax_f);
  end

  // g_m / u_m: write side = S_GMV_REQ / S_UMV_REQ matvec drain.
  // Read side: comb read in always_comb (sg_g_m_c / sg_u_m_c during S_SWG).
  // rd_addr = cnt+1 during S_SWG.
  assign g_m_we           = (state == S_GMV_REQ);
  assign g_m_wr_addr_tile = ($clog2((FFN+LANES-1)/LANES))'(chunk);
  // See hin_m/o_m rd_addr above — same gated-lookahead pattern: only
  // pre-fetch when SwiGLU is ready to advance cnt next cycle.
  assign g_m_rd_addr      = (state == S_SWG)
                              ? cnt[CW_FFN-1:0]
                                + {{(CW_FFN-1){1'b0}}, sg_in_ready}
                              : '0;
  assign u_m_we           = (state == S_UMV_REQ);
  assign u_m_wr_addr_tile = ($clog2((FFN+LANES-1)/LANES))'(chunk);
  assign u_m_rd_addr      = (state == S_SWG)
                              ? cnt[CW_FFN-1:0]
                                + {{(CW_FFN-1){1'b0}}, sg_in_ready}
                              : '0;
  for (genvar pack_ii = 0; pack_ii < LANES; pack_ii++) begin : g_g_m_pack
    assign g_m_wr_data_packed[pack_ii*BFP_MANT_W +: BFP_MANT_W] =
      requant_mant(
        mv_out_m_drain_r[pack_ii*BFP_MANT_W +: BFP_MANT_W],
        mv_out_e_drain_r[pack_ii*BFP_EXP_W  +: BFP_EXP_W ],
        mv_drain_emax_f);
  end
  for (genvar pack_ii = 0; pack_ii < LANES; pack_ii++) begin : g_u_m_pack
    assign u_m_wr_data_packed[pack_ii*BFP_MANT_W +: BFP_MANT_W] =
      requant_mant(
        mv_out_m_drain_r[pack_ii*BFP_MANT_W +: BFP_MANT_W],
        mv_out_e_drain_r[pack_ii*BFP_EXP_W  +: BFP_EXP_W ],
        mv_drain_emax_f);
  end

  // hin_m: 1-entry/cycle write from hidden_in_m during S_LATCH_IN.
  // rd_addr=cnt+1 during S_NORM1 (registered) and S_RES1 (comb).
  assign hin_m_we      = (state == S_LATCH_IN);
  assign hin_m_wr_addr = cnt[CW_D-1:0];
  assign hin_m_wr_data = hidden_in_m[cnt[CW_D-1:0]*BFP_MANT_W +: BFP_MANT_W];
  always_comb begin
    case (state)
      S_NORM1: hin_m_rd_addr = cnt[CW_D-1:0] + 1'b1;
      S_RES1:  hin_m_rd_addr = cnt[CW_D-1:0]
                                + {{(CW_D-1){1'b0}}, rs_in_ready};
      default: hin_m_rd_addr = '0;
    endcase
  end

  // h1_m: 1-entry/cycle write during S_RES1 / S_RES1_WAIT when residual
  // output is valid (gated by rs_y_valid).  wr_addr = out_cnt.
  // rd_addr=cnt+1 during S_NORM2 (registered) and S_RES2 (comb).
  assign h1_m_we      = ((state == S_RES1) || (state == S_RES1_WAIT)) && rs_y_valid;
  assign h1_m_wr_addr = out_cnt[CW_D-1:0];
  assign h1_m_wr_data = rs_y_m;
  always_comb begin
    case (state)
      S_NORM2: h1_m_rd_addr = cnt[CW_D-1:0] + 1'b1;
      S_RES2:  h1_m_rd_addr = cnt[CW_D-1:0]
                                + {{(CW_D-1){1'b0}}, rs_in_ready};
      default: h1_m_rd_addr = '0;
    endcase
  end

  // ---------------------------------------------------------------------------
  // kv_k_m: per-mantissa K cache.  Writes 1 entry/cycle during S_KVWR_M,
  // reads 1 entry/cycle during S_QK_DRIVE (lane 0 of mv_w_m).
  // ---------------------------------------------------------------------------
  localparam int KKM_DEPTH = NL * MAX_CTX * H_KV * HD;
  localparam int KKM_AW    = $clog2(KKM_DEPTH);
  logic                       kv_k_m_we;
  logic [KKM_AW-1:0]          kv_k_m_wr_addr;
  logic [BFP_MANT_W-1:0]      kv_k_m_wr_data;
  logic [KKM_AW-1:0]          kv_k_m_rd_addr;
  wire  [BFP_MANT_W-1:0]      kv_k_m_rd_data;
  bfp_sdpram #(.DEPTH(KKM_DEPTH), .WIDTH(BFP_MANT_W))
    i_kv_k_m_bram (.clk(clk), .rst(rst),
                   .we(kv_k_m_we), .wr_addr(kv_k_m_wr_addr), .wr_data(kv_k_m_wr_data),
                   .rd_addr(kv_k_m_rd_addr), .rd_data(kv_k_m_rd_data));

  // ---------------------------------------------------------------------------
  // kv_k_e / kv_v_e: per-tile K/V exponent caches.  Writes 1 entry/cycle
  // during S_KVWR_E.  kv_k_e read during S_QK_DRIVE; kv_v_e read during
  // S_AV_EMAX_SCAN and S_AV_DRIVE.
  // ---------------------------------------------------------------------------
  localparam int KVE_DEPTH = NL * MAX_CTX * NT_KV;
  localparam int KVE_AW    = $clog2(KVE_DEPTH);
  logic                       kv_k_e_we;
  logic [KVE_AW-1:0]          kv_k_e_wr_addr;
  logic [BFP_EXP_W-1:0]       kv_k_e_wr_data;
  logic [KVE_AW-1:0]          kv_k_e_rd_addr;
  wire  [BFP_EXP_W-1:0]       kv_k_e_rd_data;
  bfp_sdpram #(.DEPTH(KVE_DEPTH), .WIDTH(BFP_EXP_W))
    i_kv_k_e_bram (.clk(clk), .rst(rst),
                   .we(kv_k_e_we), .wr_addr(kv_k_e_wr_addr), .wr_data(kv_k_e_wr_data),
                   .rd_addr(kv_k_e_rd_addr), .rd_data(kv_k_e_rd_data));

  logic                       kv_v_e_we;
  logic [KVE_AW-1:0]          kv_v_e_wr_addr;
  logic [BFP_EXP_W-1:0]       kv_v_e_wr_data;
  logic [KVE_AW-1:0]          kv_v_e_rd_addr;
  wire  [BFP_EXP_W-1:0]       kv_v_e_rd_data;
  bfp_sdpram #(.DEPTH(KVE_DEPTH), .WIDTH(BFP_EXP_W))
    i_kv_v_e_bram (.clk(clk), .rst(rst),
                   .we(kv_v_e_we), .wr_addr(kv_v_e_wr_addr), .wr_data(kv_v_e_wr_data),
                   .rd_addr(kv_v_e_rd_addr), .rd_data(kv_v_e_rd_data));

  // Weights come from DDR3 via the streamer below; gammas continue to
  // load from hex.  KV caches are zero-initialised by Vivado's bitstream
  // (no $readmemh) — autoregress writes each kv_pos before reading it.
  // Per-model gamma BRAMs (G1_m, G2_m, G1_e, G2_e) are host-loaded at
  // boot via the wr_* port below — no $readmemh.  Power-on state is
  // all zeros; host must call bfp_client load-roms before inference.

  // ---------------------------------------------------------------------------
  // Host write port for the gamma BRAMs — clocked by clk_wr (eth_clk
  // at the top).  Read side stays on `clk` (core_clk).  Vivado infers
  // each ROM as a true-dual-port BRAM with asymmetric clocks — no CDC
  // needed for kind/addr/data/en, no risk of pulse loss.
  //   wr_kind: 0=G1_m  1=G1_e  2=G2_m  3=G2_e   (kinds 4..6 belong to the
  //            decode head and autoregress top; this module ignores them.)
  // ---------------------------------------------------------------------------
  logic [BFP_MANT_W-1:0] rd_G1_m, rd_G2_m;
  logic [BFP_EXP_W -1:0] rd_G1_e, rd_G2_e;
  // Debug peek — reads the layer's persistent hout_m / hout_e arrays.
  // After the multilayer wrapper finishes its NL passes, these hold the
  // LAST layer's hidden_out (= the decode_head's input).  Host can read
  // via wr_kind=10 (mantissa) / wr_kind=11 (per-tile exponent) to verify
  // whether the layer chain is collapsing to zero / NaN.
  logic [BFP_MANT_W-1:0] rd_hout_m;
  logic [BFP_EXP_W -1:0] rd_hout_e;
  logic [BFP_MANT_W-1:0] rd_snap_m;
  logic [BFP_EXP_W -1:0] rd_snap_e;
  logic [BFP_EXP_W -1:0] rd_stage_e;   // per-stage exponent read-out (wr_kind 16)
  logic [4:0]            wr_kind_q;

  always_ff @(posedge clk_wr) begin
    if (wr_en) begin
      case (wr_kind)
        5'd0: rom_G1_m[wr_addr[$clog2(NL*D)-1:0]]      <= $signed(wr_data);
        5'd1: rom_G1_e[wr_addr[$clog2(NL*NT_D)-1:0]]   <= $signed(wr_data[BFP_EXP_W-1:0]);
        5'd2: rom_G2_m[wr_addr[$clog2(NL*D)-1:0]]      <= $signed(wr_data);
        5'd3: rom_G2_e[wr_addr[$clog2(NL*NT_D)-1:0]]   <= $signed(wr_data[BFP_EXP_W-1:0]);
        default: ;  // other kinds handled elsewhere in the hierarchy
      endcase
    end
    // Read-back path — registered out of each BRAM (port-A read).
    rd_G1_m   <= rom_G1_m[wr_addr[$clog2(NL*D)-1:0]];
    rd_G1_e   <= rom_G1_e[wr_addr[$clog2(NL*NT_D)-1:0]];
    rd_G2_m   <= rom_G2_m[wr_addr[$clog2(NL*D)-1:0]];
    rd_G2_e   <= rom_G2_e[wr_addr[$clog2(NL*NT_D)-1:0]];
    rd_hout_m <= hout_m[wr_addr[$clog2(D)-1:0]];
    rd_hout_e <= hout_e[wr_addr[$clog2(NT_D)-1:0]];
    rd_snap_m <= snap_m[wr_addr[$clog2(D)-1:0]];
    rd_snap_e <= snap_e[wr_addr[$clog2(NT_D)-1:0]];
    // Per-stage exponent read-out: select the stage's inline _e array.
    case (dbg_stage_sel)
      5'd0:  rd_stage_e <= n1_e  [wr_addr[$clog2(NT_D)-1:0]];
      5'd1:  rd_stage_e <= q_e   [wr_addr[$clog2(NT_D)-1:0]];
      5'd2:  rd_stage_e <= k_e   [wr_addr[$clog2(NT_KV)-1:0]];
      5'd3:  rd_stage_e <= v_e   [wr_addr[$clog2(NT_KV)-1:0]];
      5'd4:  rd_stage_e <= attn_e[wr_addr[$clog2(NT_D)-1:0]];
      5'd5:  rd_stage_e <= o_e   [wr_addr[$clog2(NT_D)-1:0]];
      5'd6:  rd_stage_e <= h1_e  [wr_addr[$clog2(NT_D)-1:0]];
      5'd7:  rd_stage_e <= n2_e  [wr_addr[$clog2(NT_D)-1:0]];
      5'd8:  rd_stage_e <= g_e   [wr_addr[$clog2(NT_FFN)-1:0]];
      5'd9:  rd_stage_e <= u_e   [wr_addr[$clog2(NT_FFN)-1:0]];
      5'd10: rd_stage_e <= mlp_e [wr_addr[$clog2(NT_FFN)-1:0]];
      5'd11: rd_stage_e <= d_e   [wr_addr[$clog2(NT_D)-1:0]];
      5'd12: rd_stage_e <= hout_e[wr_addr[$clog2(NT_D)-1:0]];
      5'd13: rd_stage_e <= hin_e [wr_addr[$clog2(NT_D)-1:0]];
      default: rd_stage_e <= '0;
    endcase
    wr_kind_q <= wr_kind;
  end

  always_comb begin
    case (wr_kind_q)
      5'd0: wr_rdata = {{(16-BFP_MANT_W){rd_G1_m[BFP_MANT_W-1]}}, rd_G1_m};
      5'd1: wr_rdata = {{(16-BFP_EXP_W ){rd_G1_e[BFP_EXP_W -1]}}, rd_G1_e};
      5'd2: wr_rdata = {{(16-BFP_MANT_W){rd_G2_m[BFP_MANT_W-1]}}, rd_G2_m};
      5'd3: wr_rdata = {{(16-BFP_EXP_W ){rd_G2_e[BFP_EXP_W -1]}}, rd_G2_e};
      5'd10: wr_rdata = {{(16-BFP_MANT_W){rd_hout_m[BFP_MANT_W-1]}}, rd_hout_m};
      5'd11: wr_rdata = {{(16-BFP_EXP_W ){rd_hout_e[BFP_EXP_W -1]}}, rd_hout_e};
      5'd12: wr_rdata = {{(16-BFP_MANT_W){rd_snap_m[BFP_MANT_W-1]}}, rd_snap_m};
      5'd13: wr_rdata = {{(16-BFP_EXP_W ){rd_snap_e[BFP_EXP_W -1]}}, rd_snap_e};
      5'd16: wr_rdata = {{(16-BFP_EXP_W ){rd_stage_e[BFP_EXP_W -1]}}, rd_stage_e};
      default: wr_rdata = 16'h0000;
    endcase
  end

  // ---------------------------------------------------------------------------
  // Working buffers
  // ---------------------------------------------------------------------------
  // hin_m: bfp_sdpram (flat, single-port).  Write 1 entry/cycle from
  // hidden_in_m during S_LATCH_IN.  Read combinationally into rs_a_m_c
  // during S_RES1 (cnt+1 lookahead) and registered into rn_x_m during
  // S_NORM1 (also cnt+1 lookahead so the OLD 1-cycle path is preserved).
  logic                       hin_m_we;
  logic [CW_D-1:0]            hin_m_wr_addr;
  logic [BFP_MANT_W-1:0]      hin_m_wr_data;
  logic [CW_D-1:0]            hin_m_rd_addr;
  wire  [BFP_MANT_W-1:0]      hin_m_rd_data;
  bfp_sdpram #(.DEPTH(D), .WIDTH(BFP_MANT_W))
    i_hin_m_bram (.clk(clk), .rst(rst),
                  .we(hin_m_we), .wr_addr(hin_m_wr_addr), .wr_data(hin_m_wr_data),
                  .rd_addr(hin_m_rd_addr), .rd_data(hin_m_rd_data));
  logic signed [BFP_EXP_W -1:0] hin_e   [0:NT_D-1];

  // n1_m: explicit bfp_sdpram (same pattern as mlp_m).  Single-write
  // from S_NORM1_WAIT, single registered-read in S_QMV/KMV/VMV_DRIVE.
  // n1_e stays inferred (NT_D ≤ 60 × 4b — too small for BRAM tile).
  logic                       n1_m_we;
  logic [CW_D-1:0]            n1_m_wr_addr;
  logic [BFP_MANT_W-1:0]      n1_m_wr_data;
  logic [CW_D-1:0]            n1_m_rd_addr;
  wire  [BFP_MANT_W-1:0]      n1_m_rd_data;
  bfp_sdpram #(.DEPTH(D), .WIDTH(BFP_MANT_W))
    i_n1_m_bram (.clk(clk), .rst(rst),
                 .we(n1_m_we), .wr_addr(n1_m_wr_addr), .wr_data(n1_m_wr_data),
                 .rd_addr(n1_m_rd_addr), .rd_data(n1_m_rd_data));
  logic signed [BFP_EXP_W -1:0] n1_e    [0:NT_D-1];

  // q_m: packed bfp_sdpram.  TWO LANES-parallel write sites:
  //   S_QMV_REQ:    chunk-indexed, src = mv_out_*_drain_r (matvec drain)
  //   S_ROPEQ_RQ_B: cnt-indexed (tile), src = q_rot_m_rd_data_packed (rope)
  // TWO sequential reads:
  //   S_ROPEQ:      `rp_x_m <= q_m_rd_data` per cnt
  //   S_QK_DRIVE:   `mv_x_m <= q_m_rd_data` per cnt (consume_j = cnt-1)
  // Both reads use 1-cycle BRAM latency; rd_addr lookahead is state-conditional
  // (see the comb-mux below the BRAM decl).
  logic                            q_m_we;
  logic [$clog2((D+LANES-1)/LANES)-1:0] q_m_wr_addr_tile;
  logic [LANES*BFP_MANT_W-1:0]     q_m_wr_data_packed;
  logic [CW_D-1:0]                 q_m_rd_addr;
  wire  [BFP_MANT_W-1:0]           q_m_rd_data;
  bfp_sdpram_packed #(
    .LANES         (LANES),
    .LOGICAL_DEPTH (D),
    .WIDTH         (BFP_MANT_W)
  ) i_q_m_bram (
    .clk            (clk),
    .rst            (rst),
    .we             (q_m_we),
    .wr_addr_tile   (q_m_wr_addr_tile),
    .wr_data_packed (q_m_wr_data_packed),
    .rd_addr        (q_m_rd_addr),
    .rd_data        (q_m_rd_data)
  );
  logic signed [BFP_EXP_W -1:0] q_e     [0:NT_D-1];

  // k_m: packed bfp_sdpram, same pattern as q_m but H_KV*HD-deep.
  // Writes: S_KMV_REQ (chunk) + S_ROPEK_RQ_B (cnt-tile).
  // Reads:  S_ROPEK (rp_x_m), S_KVWR_M (kv_k_m_wr_data — comb-fed by
  //         always_comb at top of module).  S_KVWR_M consumes 1 entry
  //         per cycle starting at cnt=0, so rd_addr=cnt+1 lookahead.
  logic                                    k_m_we;
  logic [$clog2((H_KV*HD+LANES-1)/LANES)-1:0] k_m_wr_addr_tile;
  logic [LANES*BFP_MANT_W-1:0]             k_m_wr_data_packed;
  logic [CW_KV-1:0]                        k_m_rd_addr;
  wire  [BFP_MANT_W-1:0]                   k_m_rd_data;
  bfp_sdpram_packed #(
    .LANES         (LANES),
    .LOGICAL_DEPTH (H_KV*HD),
    .WIDTH         (BFP_MANT_W)
  ) i_k_m_bram (
    .clk            (clk),
    .rst            (rst),
    .we             (k_m_we),
    .wr_addr_tile   (k_m_wr_addr_tile),
    .wr_data_packed (k_m_wr_data_packed),
    .rd_addr        (k_m_rd_addr),
    .rd_data        (k_m_rd_data)
  );
  logic signed [BFP_EXP_W -1:0] k_e     [0:NT_KV-1];
  // v_m: packed bfp_sdpram_packed_pr (packed write + LANES-parallel read).
  // Both write (S_VMV_REQ, LANES-parallel at chunk*LANES) and reads
  //   - S_KVWR_M @ cnt[3:0]==15 (always_comb above), tile = cnt[CW_KV-1:4]
  //   - S_AV_DRIVE @ consume_t==kv_pos (in always_ff), tile = head_grp*(HD/LANES)+chunk
  // hit LANES contiguous entries at LANES-aligned base.  rd_addr_tile is
  // state-conditional, driven 1 cycle ahead of consumption.
  localparam int VMT_AW = $clog2(((H_KV*HD)+LANES-1)/LANES);
  logic                                    v_m_we;
  logic [VMT_AW-1:0]                       v_m_wr_addr_tile;
  logic [LANES*BFP_MANT_W-1:0]             v_m_wr_data_packed;
  logic [VMT_AW-1:0]                       v_m_rd_addr_tile;
  wire  [LANES*BFP_MANT_W-1:0]             v_m_rd_data_packed;
  bfp_sdpram_packed_pr #(
    .LANES         (LANES),
    .LOGICAL_DEPTH (H_KV*HD),
    .WIDTH         (BFP_MANT_W)
  ) i_v_m_bram (
    .clk            (clk),
    .rst            (rst),
    .we             (v_m_we),
    .wr_addr_tile   (v_m_wr_addr_tile),
    .wr_data_packed (v_m_wr_data_packed),
    .rd_addr_tile   (v_m_rd_addr_tile),
    .rd_data_packed (v_m_rd_data_packed)
  );
  logic signed [BFP_EXP_W -1:0] v_e     [0:NT_KV-1];

  // q_rot_m: explicit lane-striped bfp_sdpram (LANES BRAMs × NT_D-deep).
  // Writes: 1 entry/cycle from S_ROPEQ_WAIT at idx = head_idx*HD + cnt[CW_HD-1:0]
  //   → lane = idx[3:0], wr_addr_within_lane = idx[CW_D-1:4].
  // Reads:  LANES-parallel from S_ROPEQ_RQ_B at base = cnt*BFP_TILE
  //   → rd_addr = cnt[$clog2(NT_D+1)-1:0]; output is 16-element tile.
  //   rd_addr drive happens during S_ROPEQ_RQ_A (one cycle ahead) so
  //   the BRAM's 1-cycle latency lines up with B's consumption.
  logic                            q_rot_m_we;
  logic [CW_D-1:0]                 q_rot_m_wr_addr;
  logic [BFP_MANT_W-1:0]           q_rot_m_wr_data;
  logic [$clog2(NT_D+1)-1:0]       q_rot_m_rd_addr;
  wire  [LANES*BFP_MANT_W-1:0]     q_rot_m_rd_data_packed;
  bfp_sdpram_striped #(
    .LANES         (LANES),
    .LOGICAL_DEPTH (D),
    .WIDTH         (BFP_MANT_W)
  ) i_q_rot_m_bram (
    .clk            (clk),
    .rst            (rst),
    .we             (q_rot_m_we),
    .wr_addr        (q_rot_m_wr_addr),
    .wr_data        (q_rot_m_wr_data),
    .rd_addr        (q_rot_m_rd_addr),
    .rd_data_packed (q_rot_m_rd_data_packed)
  );
  logic signed [BFP_EXP_W -1:0] q_rot_e [0:D-1];
  // k_rot_m: same lane-striped bfp_sdpram pattern as q_rot_m.
  // Logical depth H_KV*HD (=320 at smollm360), striped across LANES BRAMs.
  logic                            k_rot_m_we;
  logic [$clog2(H_KV*HD)-1:0]      k_rot_m_wr_addr;
  logic [BFP_MANT_W-1:0]           k_rot_m_wr_data;
  logic [$clog2(NT_KV+1)-1:0]      k_rot_m_rd_addr;
  wire  [LANES*BFP_MANT_W-1:0]     k_rot_m_rd_data_packed;
  bfp_sdpram_striped #(
    .LANES         (LANES),
    .LOGICAL_DEPTH (H_KV*HD),
    .WIDTH         (BFP_MANT_W)
  ) i_k_rot_m_bram (
    .clk            (clk),
    .rst            (rst),
    .we             (k_rot_m_we),
    .wr_addr        (k_rot_m_wr_addr),
    .wr_data        (k_rot_m_wr_data),
    .rd_addr        (k_rot_m_rd_addr),
    .rd_data_packed (k_rot_m_rd_data_packed)
  );
  logic signed [BFP_EXP_W -1:0] k_rot_e [0:H_KV*HD-1];

  logic signed [BFP_MANT_W-1:0] scores_m  [0:MAX_CTX-1];
  logic signed [BFP_EXP_W -1:0] qk_score_e[0:MAX_CTX-1];
  logic signed [BFP_EXP_W -1:0] scores_e_shared;
  logic signed [BFP_MANT_W-1:0] probs_m   [0:MAX_CTX-1];
  logic signed [BFP_EXP_W -1:0] probs_e_shared;

  // attn_m: packed bfp_sdpram (LANES entries per BRAM word).
  // Write: S_AV_REQ, LANES-parallel for-loop → packed_wr_data combinational.
  // Read:  S_OMV_DRIVE, single-element `mv_x_m <= attn_m[cnt]` → wire from
  //        packed wrapper's rd_data (1-cycle latency, lane-mux is internal).
  logic                            attn_m_we;
  logic [$clog2((D+LANES-1)/LANES)-1:0] attn_m_wr_addr_tile;
  logic [LANES*BFP_MANT_W-1:0]     attn_m_wr_data_packed;
  logic [CW_D-1:0]                 attn_m_rd_addr;
  wire  [BFP_MANT_W-1:0]           attn_m_rd_data;
  bfp_sdpram_packed #(
    .LANES         (LANES),
    .LOGICAL_DEPTH (D),
    .WIDTH         (BFP_MANT_W)
  ) i_attn_m_bram (
    .clk            (clk),
    .rst            (rst),
    .we             (attn_m_we),
    .wr_addr_tile   (attn_m_wr_addr_tile),
    .wr_data_packed (attn_m_wr_data_packed),
    .rd_addr        (attn_m_rd_addr),
    .rd_data        (attn_m_rd_data)
  );
  logic signed [BFP_EXP_W -1:0] attn_e  [0:NT_D-1];

  // o_m: packed bfp_sdpram.  Write S_OMV_REQ (LANES-parallel for-loop),
  // read combinationally into rs_b_m_c during S_RES1 — rd_addr driven
  // as `cnt+1` for 1-cycle lookahead so the BRAM-registered output
  // matches what the OLD async LUTRAM read gave at the same cycle.
  logic                            o_m_we;
  logic [$clog2((D+LANES-1)/LANES)-1:0] o_m_wr_addr_tile;
  logic [LANES*BFP_MANT_W-1:0]     o_m_wr_data_packed;
  logic [CW_D-1:0]                 o_m_rd_addr;
  wire  [BFP_MANT_W-1:0]           o_m_rd_data;
  bfp_sdpram_packed #(
    .LANES         (LANES),
    .LOGICAL_DEPTH (D),
    .WIDTH         (BFP_MANT_W)
  ) i_o_m_bram (
    .clk            (clk),
    .rst            (rst),
    .we             (o_m_we),
    .wr_addr_tile   (o_m_wr_addr_tile),
    .wr_data_packed (o_m_wr_data_packed),
    .rd_addr        (o_m_rd_addr),
    .rd_data        (o_m_rd_data)
  );
  logic signed [BFP_EXP_W -1:0] o_e     [0:NT_D-1];

  // h1_m: bfp_sdpram, same shape as hin_m.  Writes 1 entry/cycle during
  // S_RES1 / S_RES1_WAIT (gated by rs_y_valid, addressed by out_cnt).
  // Reads combinationally into rs_a_m_c during S_RES2 (cnt+1 lookahead)
  // and registered into rn_x_m during S_NORM2 (cnt+1 lookahead).
  logic                       h1_m_we;
  logic [CW_D-1:0]            h1_m_wr_addr;
  logic [BFP_MANT_W-1:0]      h1_m_wr_data;
  logic [CW_D-1:0]            h1_m_rd_addr;
  wire  [BFP_MANT_W-1:0]      h1_m_rd_data;
  bfp_sdpram #(.DEPTH(D), .WIDTH(BFP_MANT_W))
    i_h1_m_bram (.clk(clk), .rst(rst),
                 .we(h1_m_we), .wr_addr(h1_m_wr_addr), .wr_data(h1_m_wr_data),
                 .rd_addr(h1_m_rd_addr), .rd_data(h1_m_rd_data));
  logic signed [BFP_EXP_W -1:0] h1_e    [0:NT_D-1];

  // n2_m: explicit bfp_sdpram (same pattern as n1_m / mlp_m).  Single-
  // write from S_NORM2_WAIT, single registered-read in S_GMV/UMV_DRIVE.
  logic                       n2_m_we;
  logic [CW_D-1:0]            n2_m_wr_addr;
  logic [BFP_MANT_W-1:0]      n2_m_wr_data;
  logic [CW_D-1:0]            n2_m_rd_addr;
  wire  [BFP_MANT_W-1:0]      n2_m_rd_data;
  bfp_sdpram #(.DEPTH(D), .WIDTH(BFP_MANT_W))
    i_n2_m_bram (.clk(clk), .rst(rst),
                 .we(n2_m_we), .wr_addr(n2_m_wr_addr), .wr_data(n2_m_wr_data),
                 .rd_addr(n2_m_rd_addr), .rd_data(n2_m_rd_data));
  logic signed [BFP_EXP_W -1:0] n2_e    [0:NT_D-1];

  // g_m / u_m: packed bfp_sdpram, mirrors o_m / d_m.  Single LANES-parallel
  // write (S_GMV_REQ / S_UMV_REQ) and single combinational read in the
  // always_comb that drives the SwiGLU engine.  Use cnt+1 lookahead during
  // S_SWG so g_m_rd_data / u_m_rd_data line up with the consumer.
  logic                            g_m_we;
  logic [$clog2((FFN+LANES-1)/LANES)-1:0] g_m_wr_addr_tile;
  logic [LANES*BFP_MANT_W-1:0]     g_m_wr_data_packed;
  logic [CW_FFN-1:0]               g_m_rd_addr;
  wire  [BFP_MANT_W-1:0]           g_m_rd_data;
  bfp_sdpram_packed #(
    .LANES         (LANES),
    .LOGICAL_DEPTH (FFN),
    .WIDTH         (BFP_MANT_W)
  ) i_g_m_bram (
    .clk            (clk),
    .rst            (rst),
    .we             (g_m_we),
    .wr_addr_tile   (g_m_wr_addr_tile),
    .wr_data_packed (g_m_wr_data_packed),
    .rd_addr        (g_m_rd_addr),
    .rd_data        (g_m_rd_data)
  );
  logic signed [BFP_EXP_W -1:0] g_e     [0:NT_FFN-1];
  logic                            u_m_we;
  logic [$clog2((FFN+LANES-1)/LANES)-1:0] u_m_wr_addr_tile;
  logic [LANES*BFP_MANT_W-1:0]     u_m_wr_data_packed;
  logic [CW_FFN-1:0]               u_m_rd_addr;
  wire  [BFP_MANT_W-1:0]           u_m_rd_data;
  bfp_sdpram_packed #(
    .LANES         (LANES),
    .LOGICAL_DEPTH (FFN),
    .WIDTH         (BFP_MANT_W)
  ) i_u_m_bram (
    .clk            (clk),
    .rst            (rst),
    .we             (u_m_we),
    .wr_addr_tile   (u_m_wr_addr_tile),
    .wr_data_packed (u_m_wr_data_packed),
    .rd_addr        (u_m_rd_addr),
    .rd_data        (u_m_rd_data)
  );
  logic signed [BFP_EXP_W -1:0] u_e     [0:NT_FFN-1];
  // mlp_m: was `logic signed [BFP_MANT_W-1:0] mlp_m [0:FFN-1];` (inferred
  // as ~3 kLUTRAM at FFN=2560).  Now explicit bfp_sdpram (RAMB36E1).
  // Write side: S_SWG / S_SWG_WAIT when sg_y_valid (single write/cycle).
  // Read side:  S_DMV_DRIVE consumes one entry per cycle.  The bfp_sdpram's
  // own internal output register provides the same 1-cycle latency the
  // OLD `mv_x_m <= mlp_m[cnt]` had — mv_x_m_eff mux at the matvec instance
  // selects mlp_m_rd_data when state==S_DMV_DRIVE.  rd_addr tracks cnt
  // directly (no lookahead).  mlp_e stays inferred (NT_FFN=160 × 4b).
  logic                       mlp_m_we;
  logic [CW_FFN-1:0]          mlp_m_wr_addr;
  logic [BFP_MANT_W-1:0]      mlp_m_wr_data;
  logic [CW_FFN-1:0]          mlp_m_rd_addr;
  wire  [BFP_MANT_W-1:0]      mlp_m_rd_data;
  bfp_sdpram #(.DEPTH(FFN), .WIDTH(BFP_MANT_W))
    i_mlp_m_bram (.clk(clk), .rst(rst),
                  .we(mlp_m_we), .wr_addr(mlp_m_wr_addr), .wr_data(mlp_m_wr_data),
                  .rd_addr(mlp_m_rd_addr), .rd_data(mlp_m_rd_data));
  logic signed [BFP_EXP_W -1:0] mlp_e   [0:NT_FFN-1];

  // d_m: same pattern as o_m.  Write S_DMV_REQ, read combinationally
  // into rs_b_m_c during S_RES2 with `cnt+1` lookahead.
  logic                            d_m_we;
  logic [$clog2((D+LANES-1)/LANES)-1:0] d_m_wr_addr_tile;
  logic [LANES*BFP_MANT_W-1:0]     d_m_wr_data_packed;
  logic [CW_D-1:0]                 d_m_rd_addr;
  wire  [BFP_MANT_W-1:0]           d_m_rd_data;
  bfp_sdpram_packed #(
    .LANES         (LANES),
    .LOGICAL_DEPTH (D),
    .WIDTH         (BFP_MANT_W)
  ) i_d_m_bram (
    .clk            (clk),
    .rst            (rst),
    .we             (d_m_we),
    .wr_addr_tile   (d_m_wr_addr_tile),
    .wr_data_packed (d_m_wr_data_packed),
    .rd_addr        (d_m_rd_addr),
    .rd_data        (d_m_rd_data)
  );
  logic signed [BFP_EXP_W -1:0] d_e     [0:NT_D-1];

  logic signed [BFP_MANT_W-1:0] hout_m  [0:D-1];
  logic signed [BFP_EXP_W -1:0] hout_e  [0:NT_D-1];
  // Per-layer hidden snapshot — written in parallel with hout (element-by-
  // element, so it infers a TDP BRAM, not FFs) only for the host-selected
  // (layer, token-step); read back via wr_kind 12/13.
  logic signed [BFP_MANT_W-1:0] snap_m  [0:D-1];
  logic signed [BFP_EXP_W -1:0] snap_e  [0:NT_D-1];
  wire snap_capture = (layer_idx == snap_layer_sel) && (pos == snap_step_sel);

  // ---------------------------------------------------------------------------
  // FSM
  // ---------------------------------------------------------------------------
  typedef enum logic [6:0] {
    S_IDLE,
    S_LATCH_IN,
    S_NORM1, S_NORM1_WAIT,
    S_QMV_PRIME, S_QMV_DRIVE, S_QMV_DRAIN, S_QMV_REQ, S_QMV_NEXT,
    S_KMV_PRIME, S_KMV_DRIVE, S_KMV_DRAIN, S_KMV_REQ, S_KMV_NEXT,
    S_VMV_PRIME, S_VMV_DRIVE, S_VMV_DRAIN, S_VMV_REQ, S_VMV_NEXT,
    S_ROPEQ, S_ROPEQ_WAIT, S_ROPEQ_RQ_A, S_ROPEQ_RQ_B,
    S_ROPEK, S_ROPEK_WAIT, S_ROPEK_RQ_A, S_ROPEK_RQ_B,
    S_KVWR_M, S_KVWR_E,
    S_QK_PRIME, S_QK_PREFETCH, S_QK_DRIVE, S_QK_DRAIN,
    S_SM_DRIVE, S_SM_WAIT,
    S_AV_PRIME, S_AV_EMAX_PREFETCH, S_AV_EMAX_SCAN, S_AV_PREFETCH, S_AV_DRIVE, S_AV_DRAIN, S_AV_REQ, S_AV_NEXT,
    S_OMV_PRIME, S_OMV_DRIVE, S_OMV_DRAIN, S_OMV_REQ, S_OMV_NEXT,
    S_RES1, S_RES1_WAIT,
    S_NORM2, S_NORM2_WAIT,
    S_GMV_PRIME, S_GMV_DRIVE, S_GMV_DRAIN, S_GMV_REQ, S_GMV_NEXT,
    S_UMV_PRIME, S_UMV_DRIVE, S_UMV_DRAIN, S_UMV_REQ, S_UMV_NEXT,
    S_SWG, S_SWG_WAIT,
    S_DMV_PRIME, S_DMV_DRIVE, S_DMV_DRAIN, S_DMV_REQ, S_DMV_NEXT,
    S_RES2, S_RES2_WAIT,
    S_DONE
  } state_t;
  state_t state;

  // Generic counters.
  // - chunk: iterates over CHUNKS_D/KV/FFN/HD inside a single matvec.
  //   FFN dominates (CHUNKS_FFN=160 at 360M, 96 at 135M, 512 at 1.7B);
  //   use $clog2(CHUNKS_FFN+1) so the width tracks FFN automatically.
  //   Previously hardcoded 7 bits → silent overflow + FSM hang at any
  //   FFN ≥ 2048 (CHUNKS_FFN ≥ 128).
  // - kv_t: iterates over kv_pos = 0..MAX_CTX-1 during attention.
  //   Previously hardcoded 5 bits → broken latent at any active prompt
  //   + N_GEN > 32 even on the current 135M build; tied to CW_CTX now
  //   so it scales with MAX_CTX.
  logic [11:0]                              cnt;
  logic [$clog2(CHUNKS_FFN+1)-1:0]          chunk;
  logic [4:0]                               head_idx;
  logic [CW_CTX-1:0]                        kv_t;
  logic signed [BFP_EXP_W-1:0] score_emax;

  // ---------------------------------------------------------------------------
  // Per-matvec requant pipeline (2-stage).  S_*_DRAIN now latches the engine
  // outputs and computes the max of the two 8-lane halves of mv_out_e; the
  // next-cycle S_*_REQ takes that pair, finalises emax, applies
  // requant_mant to all 16 lanes, and writes the chunk's mantissa+exp.
  // Splits the original 58-LUT-level chain (15-deep emax cascade + barrel
  // shift) into two ~7-LUT-level stages so each clock fits in the 20 ns
  // budget at 50 MHz core_clk.
  // ---------------------------------------------------------------------------
  logic signed [LANES*BFP_MANT_W-1:0] mv_out_m_drain_r;
  logic signed [LANES*BFP_EXP_W -1:0] mv_out_e_drain_r;
  logic signed [BFP_EXP_W-1:0]        emax_h0_r, emax_h1_r;

  // ---------------------------------------------------------------------------
  // Structural pipeline barrier — Vivado's retimer was pulling the
  // FSM-gated emax pipeline backward into the matvec engine's MAC,
  // producing a 27 ns cnt → q_m chain.  This module-scope always_ff
  // latches the engine output ONCE more (1 extra cycle) with no
  // FSM-state condition, so the synthesiser cannot collapse it as
  // "redundant".  The S_*_DRAIN stages then consume the latched copies
  // — chain becomes cnt → engine → mv_out_{m,e}_lat FF → emax FFs →
  // requant FFs → q_m, with hard register barriers at each step.
  // ---------------------------------------------------------------------------
  logic                                       mv_out_valid_lat;
  logic signed [LANES*BFP_MANT_W-1:0]         mv_out_m_lat;
  logic signed [LANES*BFP_EXP_W -1:0]         mv_out_e_lat;
  always_ff @(posedge clk) begin
    if (rst) begin
      mv_out_valid_lat <= 1'b0;
      mv_out_m_lat     <= '0;
      mv_out_e_lat     <= '0;
    end else begin
      mv_out_valid_lat <= mv_out_valid;
      if (mv_out_valid) begin
        mv_out_m_lat <= mv_out_m;
        mv_out_e_lat <= mv_out_e;
      end
    end
  end
  // Comb-derived signed view of the matvec engine's lane-0 exp output.
  // The part-select mv_out_e[0+:8] is unsigned by SV rules, so explicit
  // re-typing is needed for signed comparisons against score_emax.
  wire signed [BFP_EXP_W-1:0] mv_lane0_e = $signed(mv_out_e[0 +: BFP_EXP_W]);
  wire signed [BFP_MANT_W-1:0] mv_lane0_m = $signed(mv_out_m[0 +: BFP_MANT_W]);

  // Sub-engine reset (one-shot pulses between matvecs / etc.)
  logic eng_rst;

  // ---------------------------------------------------------------------------
  // Engine instances (all driven via registered signals below)
  // ---------------------------------------------------------------------------
  logic                                      mv_start;
  logic signed [BFP_MANT_W-1:0]              mv_x_m;
  logic signed [BFP_EXP_W -1:0]              mv_x_e;
  logic                                      mv_valid, mv_last;
  logic signed [LANES*BFP_MANT_W-1:0]        mv_w_m;
  logic signed [LANES*BFP_EXP_W -1:0]        mv_w_e;
  wire  signed [LANES*BFP_MANT_W-1:0]        mv_out_m;
  wire  signed [LANES*BFP_EXP_W -1:0]        mv_out_e;
  wire                                       mv_out_valid;

  // mv_w_m_eff / mv_w_e_eff: muxed weight bus presented to the matvec
  // engine.  For the 7 W?_* matvecs (Q/K/V/O/G/U/DN) the streamer's
  // bank_m / bank_e outputs drive the engine.  QK and AV matvecs read
  // kv_k_m / kv_v_m / k_m / v_m (BRAM-backed) and use mv_w_m instead.
  // `is_stream_matvec` is set in each W?_PRIME and cleared in QK_PRIME
  // / AV_PRIME, holding through DRIVE / DRAIN / REQ.
  logic is_stream_matvec;
  wire signed [LANES*BFP_MANT_W-1:0]         mv_w_m_eff;
  wire signed [LANES*BFP_EXP_W -1:0]         mv_w_e_eff;
  wire        [255:0]                        ws_weight_m_out;
  wire        [127:0]                        ws_weight_e_out;
  assign mv_w_m_eff = is_stream_matvec ? $signed(ws_weight_m_out) : mv_w_m;
  assign mv_w_e_eff = is_stream_matvec ? $signed(ws_weight_e_out) : mv_w_e;

  // is_stream_matvec: set when entering a W?_* matvec via its PRIME,
  // cleared when entering QK_PRIME or AV_PRIME.  Held through DRIVE /
  // DRAIN / REQ / NEXT so the engine consumes streamer data only when
  // appropriate.  Always_ff is intentionally separate from the main
  // FSM so the flag isn't tied to the existing case-table indentation.

  always_ff @(posedge clk) begin
    if (rst) is_stream_matvec <= 1'b0;
    else begin
      case (state)
        S_QMV_PRIME, S_KMV_PRIME, S_VMV_PRIME, S_OMV_PRIME,
        S_GMV_PRIME, S_UMV_PRIME, S_DMV_PRIME: is_stream_matvec <= 1'b1;
        S_QK_PRIME, S_AV_PRIME:                is_stream_matvec <= 1'b0;
        default: ;  // hold through DRIVE / DRAIN / REQ / NEXT
      endcase
    end
  end

  // Comparative hash on the streamed weight bus.  Folds 256 + 128 =
  // 384 bits (mv_w_m_eff || mv_w_e_eff) into a 32-bit accumulator
  // each cycle the engine is consuming streamer-sourced data.  Each
  // 32-bit chunk of the bus is XOR-tree'd into the accumulator after a
  // 1-bit rotate so order matters.  Resets when `start` is asserted
  // (one new run per restart).
  wire [31:0] mv_w_xor =
        mv_w_m_eff[ 31:  0] ^ mv_w_m_eff[ 63: 32]
      ^ mv_w_m_eff[ 95: 64] ^ mv_w_m_eff[127: 96]
      ^ mv_w_m_eff[159:128] ^ mv_w_m_eff[191:160]
      ^ mv_w_m_eff[223:192] ^ mv_w_m_eff[255:224]
      ^ mv_w_e_eff[ 31:  0] ^ mv_w_e_eff[ 63: 32]
      ^ mv_w_e_eff[ 95: 64] ^ mv_w_e_eff[127: 96];
  // NB: do NOT reset on `start` — the multilayer pulses start once per
  // (layer × step), 570 times in a full SmolLM2-135M run.  Resetting
  // here would leave weight_hash holding only the *last* layer's reads,
  // not the whole run.  rst is driven from ~core_resetn | lay_restart_core
  // at the top, so the host's restart pulse cleanly re-arms it.
  always_ff @(posedge clk) begin
    if (rst) weight_hash <= 32'hFFFFFFFF;
    else if (mv_valid && is_stream_matvec)
      weight_hash <= {weight_hash[30:0], weight_hash[31]} ^ mv_w_xor;
  end

  // mv_x_m_eff mux — for DRIVE states whose activation source has been
  // converted to a bfp_sdpram instance, route the BRAM's registered
  // output directly (the BRAM provides the same 1-cycle latency the
  // OLD `mv_x_m <= arr[cnt]` had).  Other DRIVE states still write
  // their source into the mv_x_m register, which is the fallback.
  // Extend this chain as further arrays migrate to bfp_sdpram.
  // DRIVE + matching DRAIN both route their array's BRAM rd_data:
  // matvec_bfp_engine consumes the LAST element on the cycle last_elem
  // samples high, which is the FIRST cycle of S_*_DRAIN (state has
  // already advanced from DRIVE).  Since the conversion removed
  // `mv_x_m <= arr[cnt]` from the DRIVE bodies, a mux that misses the
  // DRAIN state would deliver a stale mv_x_m as element D-1.
  // cnt is not reset on DRIVE→DRAIN so the BRAM rd_addr still points at
  // D-1 and rd_data carries arr[D-1] for the DRAIN first cycle.
  // Other states fall back to mv_x_m (the still-registered path used
  // by S_QK_DRIVE / S_AV_DRIVE etc).
  logic signed [BFP_MANT_W-1:0] mv_x_m_eff;
  always_comb begin
    unique case (state)
      S_QMV_DRIVE, S_KMV_DRIVE, S_VMV_DRIVE,
      S_QMV_DRAIN, S_KMV_DRAIN, S_VMV_DRAIN: mv_x_m_eff = n1_m_rd_data;
      S_GMV_DRIVE, S_UMV_DRIVE,
      S_GMV_DRAIN, S_UMV_DRAIN:              mv_x_m_eff = n2_m_rd_data;
      S_DMV_DRIVE, S_DMV_DRAIN:              mv_x_m_eff = mlp_m_rd_data;
      S_OMV_DRIVE, S_OMV_DRAIN:              mv_x_m_eff = attn_m_rd_data;
      default:                               mv_x_m_eff = mv_x_m;
    endcase
  end

  matvec_bfp_engine #(.LANES(LANES)) i_mv (
    .clk(clk), .rst(rst | eng_rst),
    .start_matvec(mv_start),
    .in_x_mant(mv_x_m_eff), .in_x_exp(mv_x_e),
    .in_valid(mv_valid), .last_elem(mv_last),
    .w_mant(mv_w_m_eff), .w_exp(mv_w_e_eff),
    .out_mant(mv_out_m), .out_exp(mv_out_e), .out_valid(mv_out_valid)
  );

  logic                                rn_start, rn_valid, rn_last;
  logic signed [BFP_MANT_W-1:0]        rn_x_m, rn_g_m;
  logic signed [BFP_EXP_W -1:0]        rn_x_e, rn_g_e;
  wire  signed [BFP_MANT_W-1:0]        rn_y_m;
  wire  signed [BFP_EXP_W -1:0]        rn_y_e;
  wire                                 rn_y_valid, rn_done;

  rmsnorm_bfp #(.D(D)) i_rn (
    .clk(clk), .rst(rst),
    .start(rn_start),
    .in_x_mant(rn_x_m), .in_x_exp(rn_x_e),
    .in_g_mant(rn_g_m), .in_g_exp(rn_g_e),
    .in_valid(rn_valid), .last_elem(rn_last),
    .out_y_mant(rn_y_m), .out_y_exp(rn_y_e),
    .out_valid(rn_y_valid), .done(rn_done)
  );

  logic                                rp_start, rp_valid;
  logic signed [BFP_MANT_W-1:0]        rp_x_m;
  logic signed [BFP_EXP_W -1:0]        rp_x_e;
  wire  signed [BFP_MANT_W-1:0]        rp_y_m;
  wire  signed [BFP_EXP_W -1:0]        rp_y_e;
  wire                                 rp_y_valid, rp_done;

  rope_bfp #(.HD(HD)) i_rp (
    .clk(clk), .rst(rst),
    .start(rp_start),
    .in_x_mant(rp_x_m), .in_x_exp(rp_x_e), .in_valid(rp_valid),
    .pos(pos),
    .out_y_mant(rp_y_m), .out_y_exp(rp_y_e),
    .out_valid(rp_y_valid), .done(rp_done)
  );

  // SwiGLU engine — same comb-drive pattern as residual (load+emit
  // alternation with in_ready back-pressure at tile boundaries).
  logic                                sg_start;
  logic                                sg_drive;
  logic signed [BFP_MANT_W-1:0]        sg_g_m_c, sg_u_m_c;
  logic signed [BFP_EXP_W -1:0]        sg_g_e_c, sg_u_e_c;
  logic                                sg_valid_c, sg_last_c;
  wire                                 sg_in_ready;
  wire  signed [BFP_MANT_W-1:0]        sg_y_m;
  wire  signed [BFP_EXP_W -1:0]        sg_y_e;
  wire                                 sg_y_valid, sg_done;

  always_comb sg_drive = (state == S_SWG);
  always_comb begin
    sg_g_m_c   = '0; sg_g_e_c = '0;
    sg_u_m_c   = '0; sg_u_e_c = '0;
    sg_valid_c = 1'b0;
    sg_last_c  = 1'b0;
    if (sg_drive) begin
      sg_g_m_c   = g_m_rd_data;
      sg_g_e_c   = g_e[cnt[CW_FFN-1:0] / BFP_TILE];
      sg_u_m_c   = u_m_rd_data;
      sg_u_e_c   = u_e[cnt[CW_FFN-1:0] / BFP_TILE];
      // NB: gate in_valid by in_ready (mirrors the standalone testbench).
      // This makes the 1-cycle valid_r pipeline inside swiglu_bfp line up
      // with the comb up_mant_r / silu_at_lut registration — otherwise the
      // first prod_buf write after S_EMIT→S_LOAD would store the stale
      // (last-S_EMIT-cycle) input.
      sg_valid_c = sg_in_ready && (cnt < FFN);
      sg_last_c  = sg_in_ready && (cnt == FFN-1);
    end
  end

  swiglu_bfp #(.D(FFN)) i_sg (
    .clk(clk), .rst(rst),
    .start(sg_start),
    .in_gate_mant(sg_g_m_c), .in_gate_exp(sg_g_e_c),
    .in_up_mant(sg_u_m_c),   .in_up_exp(sg_u_e_c),
    .in_valid(sg_valid_c), .last_elem(sg_last_c),
    .in_ready(sg_in_ready),
    .out_y_mant(sg_y_m), .out_y_exp(sg_y_e),
    .out_valid(sg_y_valid), .done(sg_done)
  );

  logic                                sm_start, sm_valid;
  logic [CW_CTX-1:0]                   sm_n;
  logic signed [BFP_MANT_W-1:0]        sm_x_m;
  logic signed [BFP_EXP_W -1:0]        sm_x_e;
  wire  signed [BFP_MANT_W-1:0]        sm_y_m;
  wire  signed [BFP_EXP_W -1:0]        sm_y_e;
  wire                                 sm_y_valid, sm_done;

  softmax_bfp #(.N_MAX(MAX_CTX)) i_sm (
    .clk(clk), .rst(rst),
    .start(sm_start),
    .n_elems(sm_n),
    .in_x_mant(sm_x_m), .in_x_exp(sm_x_e), .in_valid(sm_valid),
    .out_y_mant(sm_y_m), .out_y_exp(sm_y_e),
    .out_valid(sm_y_valid), .done(sm_done)
  );

  // Residual engine — combinational input drive so the engine's S_LOAD→
  // S_EMIT back-pressure (in_ready dropping at tile boundary) directly stops
  // cnt advancement WITHOUT losing the next-to-be-consumed sample.  See the
  // commented timing trace in S_RES1 / S_RES2 for why a registered drive
  // off-by-ones at tile boundaries.
  logic                                rs_start;
  logic                                rs_drive_res1;     // 1 in S_RES1
  logic                                rs_drive_res2;     // 1 in S_RES2
  logic signed [BFP_MANT_W-1:0]        rs_a_m_c, rs_b_m_c;
  logic signed [BFP_EXP_W -1:0]        rs_a_e_c, rs_b_e_c;
  logic                                rs_valid_c, rs_last_c;
  wire                                 rs_in_ready;
  wire  signed [BFP_MANT_W-1:0]        rs_y_m;
  wire  signed [BFP_EXP_W -1:0]        rs_y_e;
  wire                                 rs_y_valid, rs_done;

  always_comb rs_drive_res1 = (state == S_RES1);
  always_comb rs_drive_res2 = (state == S_RES2);
  always_comb begin
    rs_a_m_c   = '0; rs_a_e_c = '0;
    rs_b_m_c   = '0; rs_b_e_c = '0;
    rs_valid_c = 1'b0;
    rs_last_c  = 1'b0;
    if (rs_drive_res1) begin
      rs_a_m_c   = hin_m_rd_data;
      rs_a_e_c   = hin_e[cnt[CW_D-1:0] / BFP_TILE];
      rs_b_m_c   = o_m_rd_data;
      rs_b_e_c   = o_e  [cnt[CW_D-1:0] / BFP_TILE];
      rs_valid_c = rs_in_ready && (cnt < D);
      rs_last_c  = rs_in_ready && (cnt == D-1);
    end else if (rs_drive_res2) begin
      rs_a_m_c   = h1_m_rd_data;
      rs_a_e_c   = h1_e[cnt[CW_D-1:0] / BFP_TILE];
      rs_b_m_c   = d_m_rd_data;
      rs_b_e_c   = d_e [cnt[CW_D-1:0] / BFP_TILE];
      rs_valid_c = rs_in_ready && (cnt < D);
      rs_last_c  = rs_in_ready && (cnt == D-1);
    end
  end

  residual_bfp #(.D(D)) i_rs (
    .clk(clk), .rst(rst),
    .start(rs_start),
    .in_a_mant(rs_a_m_c), .in_a_exp(rs_a_e_c),
    .in_b_mant(rs_b_m_c), .in_b_exp(rs_b_e_c),
    .in_valid(rs_valid_c), .last_elem(rs_last_c), .in_ready(rs_in_ready),
    .out_y_mant(rs_y_m), .out_y_exp(rs_y_e),
    .out_valid(rs_y_valid), .done(rs_done)
  );

  // ---------------------------------------------------------------------------
  // Output bus packing
  // ---------------------------------------------------------------------------
  always_comb begin
    for (int i = 0; i < D; i++)
      hidden_out_m[i*BFP_MANT_W +: BFP_MANT_W] = hout_m[i];
    for (int t = 0; t < NT_D; t++)
      hidden_out_e[t*BFP_EXP_W  +: BFP_EXP_W ] = hout_e[t];
  end

  // ---------------------------------------------------------------------------
  // K-group mapping helper (for GQA)
  // ---------------------------------------------------------------------------
  function automatic int kv_grp_of;
    input int h;
    kv_grp_of = (H_Q == H_KV) ? h : (h * H_KV) / H_Q;
  endfunction

  // ---------------------------------------------------------------------------
  // matvec chunk re-tile-quantize: matvec_bfp_engine emits a per-lane
  // exponent in mv_out_e but BFP requires one shared exponent per TILE.  When
  // LANES == BFP_TILE (=16) each chunk's LANES outputs form exactly one tile.
  // This helper aligns each lane's mantissa to max(per-lane exp) and returns
  // the shared tile mantissas + tile exp.
  // ---------------------------------------------------------------------------
  // mv_out_m/mv_out_e are the engine's per-lane outputs.  After computing
  // emax = max(mv_out_e[i]), we shift each mantissa right by (emax - mv_out_e[i]).
  function automatic logic signed [BFP_MANT_W-1:0] requant_mant;
    input logic signed [BFP_MANT_W-1:0]  m;
    input logic signed [BFP_EXP_W -1:0]  e_lane;
    input logic signed [BFP_EXP_W -1:0]  e_max;
    logic signed [7:0] sh;
    sh = e_max - e_lane;
    if (sh >= 16)      requant_mant = 16'sd0;
    else if (sh >= 0)  requant_mant = m >>> sh[3:0];
    else               requant_mant = m;
  endfunction

  // ---------------------------------------------------------------------------
  // DDR3 weight streamer.
  //
  //   Active matvec is identified by ws_matvec_id:
  //     0=Q  1=K  2=V  3=O  4=G  5=U  6=DN
  //
  //   Per chunk, the layer FSM pulses ws_load_req with ws_chunk_idx=chunk
  //   and waits for ws_ready before transitioning into the matvec DRIVE
  //   phase.  Then ws_rd_col / ws_rd_tile (driven from `cnt`) deliver
  //   weight_m_out / weight_e_out one cycle later — same pipeline as the
  //   on-chip ROM read it replaces.
  // ---------------------------------------------------------------------------
  logic [2:0]                       ws_matvec_id;
  logic                             ws_load_req;
  wire                              ws_ready, ws_busy_unused;
  logic [AXI_ADDR_WIDTH-1:0]        ws_base_m_mux, ws_base_e_mux;
  logic [11:0]                      ws_in_dim_mux, ws_in_dim_tiles_mux;
  logic [11:0]                      ws_rd_col, ws_rd_tile;

  // 4-phase per-chunk load handshake.  Mirrors smollm_layer.sv's MV_LOAD_
  // {REQ,HOLD,WAIT} pattern: pulse load_req, wait for ws_ready to fall
  // (HOLD — avoids reading stale ready=1 from a previous chunk), then
  // wait for it to rise (WAIT — fresh tile is in bank).  Stays in WSP_READY
  // once data is in bank, until *_NEXT resets it for the next chunk.
  typedef enum logic [1:0] {
    WSP_KICK, WSP_HOLD, WSP_WAIT, WSP_READY
  } ws_phase_t;
  ws_phase_t ws_phase;

  // rd_col / rd_tile track cnt during DRIVE: the streamer has a 1-cycle
  // BRAM read latency, matching the existing `mv_w_m <= rom[cnt]` pipeline
  // alignment exactly.  Outside DRIVE the values don't matter.
  always_comb ws_rd_col  = cnt;
  always_comb ws_rd_tile = cnt >> 4;

  always_comb begin
    // Default — Q.
    ws_base_m_mux        = ws_base_WQ_m;
    ws_base_e_mux        = ws_base_WQ_e;
    ws_in_dim_mux        = 12'(D);
    ws_in_dim_tiles_mux  = 12'(NT_D);
    unique case (ws_matvec_id)
      3'd0: begin ws_base_m_mux = ws_base_WQ_m;  ws_base_e_mux = ws_base_WQ_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd1: begin ws_base_m_mux = ws_base_WK_m;  ws_base_e_mux = ws_base_WK_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd2: begin ws_base_m_mux = ws_base_WV_m;  ws_base_e_mux = ws_base_WV_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd3: begin ws_base_m_mux = ws_base_WO_m;  ws_base_e_mux = ws_base_WO_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd4: begin ws_base_m_mux = ws_base_WG_m;  ws_base_e_mux = ws_base_WG_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd5: begin ws_base_m_mux = ws_base_WU_m;  ws_base_e_mux = ws_base_WU_e;
                  ws_in_dim_mux = 12'(D);   ws_in_dim_tiles_mux = 12'(NT_D);   end
      3'd6: begin ws_base_m_mux = ws_base_WDN_m; ws_base_e_mux = ws_base_WDN_e;
                  ws_in_dim_mux = 12'(FFN); ws_in_dim_tiles_mux = 12'(NT_FFN); end
      default:;
    endcase
  end

  weight_streamer_bfp_mt #(
    .AXI_DATA_WIDTH (512),
    .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH),
    .AXI_ID_WIDTH   (AXI_ID_WIDTH),
    .IN_DIM_MAX     (FFN > D ? FFN : D),
    .IN_DIM_BITS    (12),
    // chunks_out = max(D, FFN) / LANES.  For smollm360 FFN=2560/16=160
    // → needs 8 bits.  7 bits silently wraps chunks 128..159 to 0..31
    // on G/U/DN matvecs, corrupting the upper third of the mlp output.
    .CHUNK_BITS     (8),
    // Sim-only selftest shadow size = max matvec line count.  Largest
    // is max(D*D, D*FFN, FFN*D)/16 = D * max(D, FFN) / 16.
    .SIM_M_DEPTH_P  ((D * (D > FFN ? D : FFN)) / 16 + 16)
  ) i_ws (
    .clk_core      (clk),
    .rst_core      (rst),
    .matrix_base_m (ws_base_m_mux),
    .matrix_base_e (ws_base_e_mux),
    .chunk_idx     (chunk),
    .in_dim        (ws_in_dim_mux),
    .in_dim_tiles  (ws_in_dim_tiles_mux),
    .load_req      (ws_load_req),
    .ready         (ws_ready),
    .busy          (ws_busy_unused),
    .rd_col        (ws_rd_col),
    .weight_m_out  (ws_weight_m_out),
    .rd_tile       (ws_rd_tile),
    .weight_e_out  (ws_weight_e_out),
    .clk_axi       (clk_axi),
    .rst_axi       (rst_axi),
    .m_axi_arvalid (m_axi_arvalid),
    .m_axi_arready (m_axi_arready),
    .m_axi_arid    (m_axi_arid),
    .m_axi_araddr  (m_axi_araddr),
    .m_axi_arlen   (m_axi_arlen),
    .m_axi_arsize  (m_axi_arsize),
    .m_axi_arburst (m_axi_arburst),
    .m_axi_arlock  (m_axi_arlock),
    .m_axi_arcache (m_axi_arcache),
    .m_axi_arprot  (m_axi_arprot),
    .m_axi_arqos   (m_axi_arqos),
    .m_axi_rvalid  (m_axi_rvalid),
    .m_axi_rready  (m_axi_rready),
    .m_axi_rid     (m_axi_rid),
    .m_axi_rdata   (m_axi_rdata),
    .m_axi_rresp   (m_axi_rresp),
    .m_axi_rlast   (m_axi_rlast)
`ifdef VERILATOR
    , .sim_matvec_id (ws_matvec_id)
`endif
  );

  // ---------------------------------------------------------------------------
  // Main FSM — all writes use non-blocking assignment; engine inputs are
  // driven from registers so they pipeline-align with mv_valid one cycle
  // later (matvec engine samples on the clock edge after we register).
  // ---------------------------------------------------------------------------
  integer ii;       // for-loop scratch in always_ff (synth-safe with int loop body)
  logic [4:0] head_grp;
  logic [11:0] av_row_base;      // current AV row offset = head*HD + chunk*LANES
  // Common w_e for AV — max of kv_v_e[t][chunk] over t in [0..kv_pos].
  // Computed per-chunk in S_AV_PRIME; used by S_AV_DRIVE to align each
  // timestep's V mantissa before feeding the matvec engine.
  logic signed [BFP_EXP_W-1:0] av_emax;
  // Sequential scan counter for AV_EMAX_SCAN — walks t=0..kv_pos-1
  // (one cycle each) to compute av_emax from kv_v_e without a 64-port
  // comb fan-in (which Vivado can't infer as BRAM and won't dissolve
  // at full SmolLM2 dims).
  logic [CW_CTX-1:0] av_scan_cnt;
  // Residual / SwiGLU output write-index — independent of cnt (which tracks
  // input drive) because the engines emit interleaved with loading.
  logic [11:0] out_cnt;

  always_ff @(posedge clk) begin
    if (rst) begin
      state           <= S_IDLE;
      done            <= 1'b0;
      eng_rst         <= 1'b1;
      cnt             <= '0;
      chunk           <= '0;
      head_idx        <= '0;
      kv_t            <= '0;
      score_emax      <= -8'sd128;
      mv_out_m_drain_r<= '0;
      mv_out_e_drain_r<= '0;
      emax_h0_r       <= -8'sd128;
      emax_h1_r       <= -8'sd128;
      mv_start        <= 1'b0;
      mv_valid        <= 1'b0;
      mv_last         <= 1'b0;
      mv_x_m          <= '0;
      mv_x_e          <= '0;
      mv_w_m          <= '0;
      mv_w_e          <= '0;
      rn_start        <= 1'b0;
      rn_valid        <= 1'b0;
      rn_last         <= 1'b0;
      rn_x_m          <= '0;
      rn_x_e          <= '0;
      rn_g_m          <= '0;
      rn_g_e          <= '0;
      rp_start        <= 1'b0;
      rp_valid        <= 1'b0;
      rp_x_m          <= '0;
      rp_x_e          <= '0;
      sg_start        <= 1'b0;
      sm_start        <= 1'b0;
      sm_valid        <= 1'b0;
      sm_n            <= '0;
      sm_x_m          <= '0;
      sm_x_e          <= '0;
      rs_start        <= 1'b0;
      scores_e_shared <= '0;
      probs_e_shared  <= '0;
      head_grp        <= '0;
      av_row_base     <= '0;
      av_scan_cnt     <= '0;
      out_cnt         <= '0;
      ws_matvec_id    <= 3'd0;
      ws_load_req     <= 1'b0;
      ws_phase        <= WSP_KICK;
    end else begin
      // Default pulses → 0
      mv_start <= 1'b0;
      mv_valid <= 1'b0;
      mv_last  <= 1'b0;
      rn_start <= 1'b0;
      rn_valid <= 1'b0;
      rn_last  <= 1'b0;
      rp_start <= 1'b0;
      rp_valid <= 1'b0;
      sg_start <= 1'b0;
      sm_start <= 1'b0;
      sm_valid <= 1'b0;
      rs_start <= 1'b0;
      eng_rst  <= 1'b0;
      done     <= 1'b0;
      ws_load_req <= 1'b0;

      case (state)
        // -------------------------------------------------------------------
        S_IDLE: if (start) begin
          state   <= S_LATCH_IN;
          cnt     <= '0;
          eng_rst <= 1'b1;
          ws_phase <= WSP_KICK;
        end

        // Copy hidden_in bus into BRAM-style arrays one element/cycle.
        // hin_m mantissa write handled by hin_m_we above; hin_e is still
        // an inferred small (NT_D × 8b) LUTRAM array.
        S_LATCH_IN: begin
          if (cnt < NT_D)
            hin_e[cnt[CW_D-1:0]] <= hidden_in_e[cnt[CW_D-1:0]*BFP_EXP_W +: BFP_EXP_W];
          if (cnt == D-1) begin
            state    <= S_NORM1;
            cnt      <= '0;
            rn_start <= 1'b1;
          end else cnt <= cnt + 1'b1;
        end

        // -------------------------------------------------------------------
        // RMSNorm1 — feed hin + G1, capture norm1 (n1).
        // -------------------------------------------------------------------
        S_NORM1: begin
          rn_valid <= 1'b1;
          rn_x_m   <= hin_m_rd_data;
          rn_x_e   <= hin_e[cnt[CW_D-1:0] / BFP_TILE];
          rn_g_m   <= rom_G1_m[layer_idx * D    + cnt[CW_D-1:0]];
          rn_g_e   <= rom_G1_e[layer_idx * NT_D + cnt[CW_D-1:0] / BFP_TILE];
          rn_last  <= (cnt == D-1);
          if (cnt == D-1) begin
            state <= S_NORM1_WAIT;
            cnt   <= '0;
          end else cnt <= cnt + 1'b1;
        end

        S_NORM1_WAIT: begin
          if (rn_y_valid) begin
            // n1_m write handled by always_comb-equivalent assigns above
            if (cnt[3:0] == 4'd0)
              n1_e[cnt[CW_D-1:0] / BFP_TILE] <= rn_y_e;
            if (cnt == D-1) begin
              cnt      <= '0;
              chunk    <= '0;
              eng_rst  <= 1'b1;
              state    <= S_QMV_PRIME;
            end else cnt <= cnt + 1'b1;
          end
        end

        // ===================================================================
        // Q matvec — 7 phases (PRIME/DRIVE/DRAIN/NEXT) — input dim = D, out_dim = D
        // ===================================================================
        S_QMV_PRIME: begin
          if (ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd0;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1;
            cnt      <= '0;
            state    <= S_QMV_DRIVE;
          end
        end
        S_QMV_DRIVE: begin
          mv_valid <= 1'b1;
          // mv_x_m comes from n1_m_rd_data via the mv_x_m_eff mux.
          mv_x_e   <= n1_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_QMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_QMV_DRAIN: if (mv_out_valid_lat) begin : qmv_pipeA
          // Stage 1: latch + compute max of two 8-lane halves.
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_QMV_REQ;
        end
        S_QMV_REQ: begin : qmv_pipeB
          // Stage 2: final max + write (q_m_we + g_q_m_pack drive the packed BRAM).
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          q_e[chunk] <= emax_f;
          state <= S_QMV_NEXT;
        end
        S_QMV_NEXT: begin
          ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_D - 1) begin
            chunk   <= '0;
            eng_rst <= 1'b1;
            state   <= S_KMV_PRIME;
          end else begin
            chunk   <= chunk + 1'b1;
            eng_rst <= 1'b1;
            state   <= S_QMV_PRIME;
          end
        end

        // ===================================================================
        // K matvec
        // ===================================================================
        S_KMV_PRIME: begin
          if (ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd1;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_KMV_DRIVE;
          end
        end
        S_KMV_DRIVE: begin
          mv_valid <= 1'b1;
          // mv_x_m comes from n1_m_rd_data via the mv_x_m_eff mux.
          mv_x_e   <= n1_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_KMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_KMV_DRAIN: if (mv_out_valid_lat) begin : kmv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_KMV_REQ;
        end
        S_KMV_REQ: begin : kmv_pipeB
          // k_m LANES-parallel write driven by k_m_we + g_k_m_pack (kmv_data).
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          k_e[chunk] <= emax_f;
          state <= S_KMV_NEXT;
        end
        S_KMV_NEXT: begin
          ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_KV - 1) begin
            chunk <= '0; eng_rst <= 1'b1; state <= S_VMV_PRIME;
          end else begin
            chunk <= chunk + 1'b1; eng_rst <= 1'b1; state <= S_KMV_PRIME;
          end
        end

        // ===================================================================
        // V matvec
        // ===================================================================
        S_VMV_PRIME: begin
          if (ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd2;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_VMV_DRIVE;
          end
        end
        S_VMV_DRIVE: begin
          mv_valid <= 1'b1;
          // mv_x_m comes from n1_m_rd_data via the mv_x_m_eff mux.
          mv_x_e   <= n1_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_VMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_VMV_DRAIN: if (mv_out_valid_lat) begin : vmv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_VMV_REQ;
        end
        S_VMV_REQ: begin : vmv_pipeB
          // v_m LANES-parallel write driven by v_m_we + g_v_m_pack above.
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          v_e[chunk] <= emax_f;
          state <= S_VMV_NEXT;
        end
        S_VMV_NEXT: begin
          ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_KV - 1) begin
            chunk <= '0; head_idx <= '0; cnt <= '0;
            eng_rst <= 1'b1; rp_start <= 1'b1; state <= S_ROPEQ;
          end else begin
            chunk <= chunk + 1'b1; eng_rst <= 1'b1; state <= S_VMV_PRIME;
          end
        end

        // ===================================================================
        // RoPE Q — H_Q heads
        // ===================================================================
        S_ROPEQ: begin
          rp_valid <= 1'b1;
          rp_x_m   <= q_m_rd_data;
          rp_x_e   <= q_e[(head_idx * HD + cnt[CW_HD-1:0]) / BFP_TILE];
          if (cnt == HD-1) begin
            state <= S_ROPEQ_WAIT; cnt <= '0;
          end else cnt <= cnt + 1'b1;
        end
        S_ROPEQ_WAIT: begin
          if (rp_y_valid) begin
            // q_rot_m write handled by always_comb-equivalent assigns above
            q_rot_e[head_idx * HD + cnt[CW_HD-1:0]] <= rp_y_e;
            cnt <= cnt + 1'b1;
          end
          if (rp_done) begin
            if (head_idx == H_Q - 1) begin
              cnt   <= '0;
              state <= S_ROPEQ_RQ_A;
            end else begin
              head_idx <= head_idx + 1'b1; cnt <= '0;
              rp_start <= 1'b1; state <= S_ROPEQ;
            end
          end
        end

        // Re-tile-quantize q_rot — 2-stage pipeline.
        // Stage A: scan the tile's q_rot_e[base..base+15], compute the
        //          max of each 8-element half; latch into emax_h{0,1}_r.
        // Stage B: final emax = max(halves); apply requant_mant to all
        //          16 elements; write q_m + q_e; advance cnt.
        S_ROPEQ_RQ_A: begin : rq_q_a
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          automatic int base;
          base = cnt[$clog2(NT_D+1)-1:0] * BFP_TILE;
          e_lo = -8'sd128;
          for (ii = 0; ii < 8; ii++)
            if (base + ii < D)
              if (q_rot_e[base + ii] > e_lo) e_lo = q_rot_e[base + ii];
          e_hi = -8'sd128;
          for (ii = 8; ii < BFP_TILE; ii++)
            if (base + ii < D)
              if (q_rot_e[base + ii] > e_hi) e_hi = q_rot_e[base + ii];
          emax_h0_r <= e_lo;
          emax_h1_r <= e_hi;
          state <= S_ROPEQ_RQ_B;
        end
        S_ROPEQ_RQ_B: begin : rq_q_b
          // q_m LANES-parallel write driven by q_m_we + g_q_m_pack
          // (qrq_data branch selected because state==S_ROPEQ_RQ_B).
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          q_e[cnt[$clog2(NT_D+1)-1:0]] <= emax_f;
          if (cnt == NT_D - 1) begin
            head_idx <= '0; cnt <= '0;
            rp_start <= 1'b1; state <= S_ROPEK;
          end else begin
            cnt   <= cnt + 1'b1;
            state <= S_ROPEQ_RQ_A;
          end
        end

        // RoPE K — H_KV heads
        S_ROPEK: begin
          rp_valid <= 1'b1;
          rp_x_m   <= k_m_rd_data;
          rp_x_e   <= k_e[(head_idx * HD + cnt[CW_HD-1:0]) / BFP_TILE];
          if (cnt == HD-1) begin
            state <= S_ROPEK_WAIT; cnt <= '0;
          end else cnt <= cnt + 1'b1;
        end
        S_ROPEK_WAIT: begin
          if (rp_y_valid) begin
            // k_rot_m write handled by always_comb-equivalent assigns above
            k_rot_e[head_idx * HD + cnt[CW_HD-1:0]] <= rp_y_e;
            cnt <= cnt + 1'b1;
          end
          if (rp_done) begin
            if (head_idx == H_KV - 1) begin
              cnt   <= '0;
              state <= S_ROPEK_RQ_A;
            end else begin
              head_idx <= head_idx + 1'b1; cnt <= '0;
              rp_start <= 1'b1; state <= S_ROPEK;
            end
          end
        end

        // Re-tile-quantize k_rot — 2-stage pipeline (mirror of S_ROPEQ_RQ).
        S_ROPEK_RQ_A: begin : rq_k_a
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          automatic int base;
          base = cnt[$clog2(NT_KV+1)-1:0] * BFP_TILE;
          e_lo = -8'sd128;
          for (ii = 0; ii < 8; ii++)
            if (base + ii < H_KV*HD)
              if (k_rot_e[base + ii] > e_lo) e_lo = k_rot_e[base + ii];
          e_hi = -8'sd128;
          for (ii = 8; ii < BFP_TILE; ii++)
            if (base + ii < H_KV*HD)
              if (k_rot_e[base + ii] > e_hi) e_hi = k_rot_e[base + ii];
          emax_h0_r <= e_lo;
          emax_h1_r <= e_hi;
          state <= S_ROPEK_RQ_B;
        end
        S_ROPEK_RQ_B: begin : rq_k_b
          // k_m LANES-parallel write driven by k_m_we + g_k_m_pack (krq_data).
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          k_e[cnt[$clog2(NT_KV+1)-1:0]] <= emax_f;
          if (cnt == NT_KV - 1) begin
            cnt <= '0;
            state <= S_KVWR_M;
          end else begin
            cnt   <= cnt + 1'b1;
            state <= S_ROPEK_RQ_A;
          end
        end

        // ===================================================================
        // KV-cache write: store k_rot and v (no rope for V) at slot kv_pos.
        // Phase M: stream HD*H_KV mantissas.  Phase E: collapse rope
        // per-element exps into one shared tile-exp via max.
        // ===================================================================
        // After re-tile-quant, k_m/k_e is the post-rope K in per-tile BFP.
        // Just copy mantissas + per-tile exponents into the KV cache slot.
        S_KVWR_M: begin : kvwr_m_blk
          // kv_k_m / kv_v_m_chk writes fired by comb drivers above.
          if (cnt == H_KV*HD - 1) begin
            cnt   <= '0; state <= S_KVWR_E;
          end else cnt <= cnt + 1'b1;
        end
        S_KVWR_E: begin
          // kv_k_e / kv_v_e writes fired by comb drivers above.
          if (cnt == NT_KV - 1) begin
            cnt <= '0; head_idx <= '0; kv_t <= '0;
            score_emax <= -8'sd128;
            eng_rst <= 1'b1;
            state <= S_QK_PRIME;
          end else cnt <= cnt + 1'b1;
        end

        // ===================================================================
        // Attention QK^T — one timestep at a time.  Engine reused: stream
        // Q[head] mantissas, with K[grp,t] mantissas on lane 0 (other lanes 0).
        // ===================================================================
        S_QK_PRIME: begin
          // Latch the new head_grp; PREFETCH the cycle after then sees
          // it and issues kv_k_m/e rd_addr for cnt=0.
          head_grp <= 5'(kv_grp_of(head_idx));
          mv_start <= 1'b1; cnt <= '0; state <= S_QK_PREFETCH;
        end
        S_QK_PREFETCH: begin
          // Comb driver issues rd_addr=addr(kv_t, head_grp, cnt=0); the
          // BRAM internal register latches it.  Advance cnt to 1 so the
          // first DRIVE cycle's rd_addr = addr(j=1) while we consume
          // rd_data for j=0.
          cnt <= 'd1;
          state <= S_QK_DRIVE;
        end
        S_QK_DRIVE: begin : qk_drive_blk
          // consume_j tracks the index whose rd_data the BRAM just
          // delivered (issued by the comb driver one cycle earlier).
          automatic logic [CW_HD-1:0] consume_j;
          consume_j = cnt[CW_HD-1:0] - 1'b1;
          mv_valid <= 1'b1;
          mv_x_m   <= q_m_rd_data;
          mv_x_e   <= q_e[(head_idx * HD + consume_j) / BFP_TILE];
          mv_last  <= (consume_j == HD-1);
          // Lane 0: K from kv_k_m / kv_k_e BRAMs.  Since S_KVWR_M wrote
          // kv_k_m[kv_pos] = k_m before reaching here, the kv_t==kv_pos
          // path no longer needs the FF fallback.
          mv_w_m[0 +: BFP_MANT_W] <= kv_k_m_rd_data;
          mv_w_e[0 +: BFP_EXP_W ] <= kv_k_e_rd_data;
          for (ii = 1; ii < LANES; ii++) begin
            mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= '0;
            mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <= '0;
          end
          if (consume_j == HD-1) state <= S_QK_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_QK_DRAIN: if (mv_out_valid) begin
`ifdef LBFP_STAGE_DUMP
          $display("[QK_DRAIN] kv_t=%0d mv_lane0_m=%0d mv_lane0_e=%0d score_emax=%0d this_>max=%0d",
                   kv_t, mv_lane0_m, mv_lane0_e, score_emax,
                   (mv_lane0_e > score_emax));
`endif
          scores_m  [kv_t] <= mv_lane0_m;
          qk_score_e[kv_t] <= mv_lane0_e;
          if (mv_lane0_e > score_emax) score_emax <= mv_lane0_e;
          if (kv_t == kv_pos) begin
            scores_e_shared <= (mv_lane0_e > score_emax) ? mv_lane0_e : score_emax;
            cnt      <= '0;
            eng_rst  <= 1'b1;
            sm_start <= 1'b1;          // fire start one cycle before first valid
            state    <= S_SM_DRIVE;
          end else begin
            kv_t    <= kv_t + 1'b1;
            eng_rst <= 1'b1;
            state   <= S_QK_PRIME;
          end
        end

        // -------------------------------------------------------------------
        // Softmax: pre-align each score mantissa to scores_e_shared, apply
        // 1/sqrt(HD) = 2^-3 (HD=64) by tweaking the shared exponent.  Stream
        // (kv_pos+1) values into softmax_bfp.
        // -------------------------------------------------------------------
        S_SM_DRIVE: begin
          sm_valid <= 1'b1;
          sm_n     <= CW_CTX'(kv_pos + 1);
          // Pre-shift mantissa
          begin : sm_drv
            automatic logic signed [BFP_EXP_W:0] sh;
            sh = scores_e_shared - qk_score_e[cnt[CW_CTX-1:0]];
            if (sh >= 16) sm_x_m <= '0;
            else          sm_x_m <= $signed(scores_m[cnt[CW_CTX-1:0]]) >>> sh[3:0];
          end
          // 1/sqrt(HD=64) = 2^-3
          sm_x_e <= scores_e_shared - 8'sd3;
          if (cnt == kv_pos) begin
            state <= S_SM_WAIT; cnt <= '0;
          end else cnt <= cnt + 1'b1;
        end
        S_SM_WAIT: begin
          if (sm_y_valid) begin
            probs_m[cnt[CW_CTX-1:0]] <= sm_y_m;
            probs_e_shared           <= sm_y_e;
            cnt <= cnt + 1'b1;
          end
          if (sm_done) begin
            // Begin AV: HD outputs in CHUNKS_HD chunks of LANES
            chunk        <= '0;
            kv_t         <= '0;
            av_row_base  <= head_idx * HD;
            eng_rst      <= 1'b1;
            state        <= S_AV_PRIME;
          end
        end

        // ===================================================================
        // AV — for each chunk of HD outputs, matvec input = probs (length
        // kv_pos+1), weight row = V[grp, t, j_base..j_base+LANES-1].
        // ===================================================================
        // S_AV_PRIME: init head_grp + seed av_emax with the current-step
        // V tile exponent.  Then S_AV_EMAX_SCAN walks past timesteps
        // 0..kv_pos-1 one per cycle to fold each kv_v_e[t] into the max.
        // Replaces the original MAX_CTX-wide combinational fan-in,
        // which made kv_v_e_reg unsynthesizable at NL=30 / MAX_CTX=64.
        S_AV_PRIME: begin : av_prime_blk
          automatic int tile_idx;
          tile_idx = (kv_grp_of(head_idx) * HD + chunk * LANES) / BFP_TILE;
          head_grp    <= 5'(kv_grp_of(head_idx));
          av_emax     <= $signed(v_e[tile_idx]);   // seed with current step
          av_scan_cnt <= '0;
          state       <= S_AV_EMAX_PREFETCH;
        end
        S_AV_EMAX_PREFETCH: begin
          // Comb driver issues kv_v_e rd_addr=addr(av_scan_cnt=0); BRAM
          // latches it.  Bump av_scan_cnt to 1 so the SCAN loop's first
          // cycle sees consume_av_t=0 with rd_data already valid.
          av_scan_cnt <= 'd1;
          state       <= S_AV_EMAX_SCAN;
        end
        S_AV_EMAX_SCAN: begin : av_emax_scan
          automatic logic signed [BFP_EXP_W-1:0] e_t;
          automatic logic [CW_CTX-1:0] consume_av_t;
          consume_av_t = av_scan_cnt[CW_CTX-1:0] - 1'b1;
          if (av_scan_cnt <= kv_pos) begin
            // rd_data on the kv_v_e BRAM is kv_v_e[t = consume_av_t]
            // (issued by comb driver the prior cycle when av_scan_cnt
            // was consume_av_t).  Fold it into the running max if that
            // timestep is < kv_pos (the current-step exponent was
            // already seeded into av_emax by S_AV_PRIME via v_e).
            e_t = $signed(kv_v_e_rd_data);
            if (consume_av_t < kv_pos && e_t > av_emax) av_emax <= e_t;
            av_scan_cnt <= av_scan_cnt + 1'b1;
          end else begin
            // Scan complete; pre-issue first kv_v_m_chk read in
            // S_AV_PREFETCH (rd_addr = addr(cnt=0)) so by the time
            // S_AV_DRIVE's first consumption cycle hits, kv_v_chk_rd_data
            // already holds data(t=0).  consume_t = cnt - 1 in DRIVE
            // picks the matching probs / v_e exponent.
            mv_start <= 1'b1;
            cnt      <= '0;
            state    <= S_AV_PREFETCH;
          end
        end

        S_AV_PREFETCH: begin
          // rd_addr_comb is already driving addr(cnt=0) → xpm/BRAM
          // latches it this cycle; data(0) is visible next cycle.
          // Advance cnt to 1 so during the first DRIVE cycle:
          //   - cnt = 1 → rd_addr_comb = addr(1) (prefetch for next)
          //   - consume_t = cnt - 1 = 0 (process data(0) just arrived)
          cnt   <= 'd1;
          state <= S_AV_DRIVE;
        end
        // matvec engine requires input dim to be a multiple of TILE so that
        // tile_done can fire AND all 16 elements of a tile share one w_e.
        // For AV the per-timestep V values have different per-tile exps, so
        // we pre-shift each timestep's V mantissa to av_emax and feed av_emax
        // as the shared w_e for every cycle (including the post-kv_pos
        // padding zeros).
        // cnt is the *prefetch* counter (drives kv_v_chk_rd_addr comb).
        // consume_t = cnt - 1 is the timestep whose data we process this
        // cycle (rd_data arrived from xpm/BRAM 1 cycle after the addr
        // we drove last cycle).  PREFETCH bumped cnt to 1 so DRIVE
        // starts at consume_t=0 with rd_data=data(0) already valid.
        S_AV_DRIVE: begin : av_drive_blk
          automatic int tile_idx;
          automatic logic signed [BFP_EXP_W-1:0] v_e_this;
          automatic logic signed [BFP_EXP_W-1:0] shamt;
          // consume_t must be wide enough to represent BFP_TILE-1=15 so the
          // S_AV_DRAIN transition fires; CW_CTX (= $clog2(MAX_CTX+1)) can be
          // smaller (3 bits at MAX_CTX=4) and would let consume_t wrap.
          automatic logic [$clog2(BFP_TILE+1)-1:0] consume_t;
          tile_idx  = (head_grp * HD + chunk * LANES) / BFP_TILE;
          consume_t = cnt[$clog2(BFP_TILE+1)-1:0] - 1'b1;
          mv_valid <= 1'b1;
          mv_x_e   <= probs_e_shared;
          if (consume_t <= kv_pos) begin
            mv_x_m <= probs_m[consume_t];
            if (consume_t == kv_pos) v_e_this = $signed(v_e[tile_idx]);
            else v_e_this = $signed(kv_v_e_rd_data);
            shamt = av_emax - v_e_this;
            for (ii = 0; ii < LANES; ii++) begin
              automatic logic signed [BFP_MANT_W-1:0] m_raw;
              if (consume_t == kv_pos)
                m_raw = $signed(v_m_rd_data_packed[ii*BFP_MANT_W +: BFP_MANT_W]);
              else
                m_raw = $signed(kv_v_chk_rd_data[ii*BFP_MANT_W +: BFP_MANT_W]);
              if (shamt >= 16)     mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= '0;
              else if (shamt >= 0) mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= m_raw >>> shamt[3:0];
              else                 mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= m_raw;
              mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <= av_emax;
            end
          end else begin
            // Padding cycles past kv_pos.  Mantissas zero; w_e == av_emax so
            // the matvec engine's tile_exp ends up at probs_e + av_emax (the
            // intended scale).
            mv_x_m <= '0;
            for (ii = 0; ii < LANES; ii++) begin
              mv_w_m[ii*BFP_MANT_W +: BFP_MANT_W] <= '0;
              mv_w_e[ii*BFP_EXP_W  +: BFP_EXP_W ] <= av_emax;
            end
          end
          mv_last <= (consume_t == BFP_TILE - 1);
          if (consume_t == BFP_TILE - 1) state <= S_AV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_AV_DRAIN: if (mv_out_valid_lat) begin : av_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_AV_REQ;
        end
        S_AV_REQ: begin : av_pipeB
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          // attn_m's LANES-parallel write handled by g_attn_m_pack +
          // attn_m_we above (packed bfp_sdpram).  Only the exponent
          // companion attn_e (LUTRAM) is still updated here.
          attn_e[(av_row_base + chunk * LANES) / BFP_TILE] <= emax_f;
          state <= S_AV_NEXT;
        end
        S_AV_NEXT: begin
          if (chunk == CHUNKS_HD - 1) begin
            // head done — next head, or move to O matvec
            if (head_idx == H_Q - 1) begin
              chunk    <= '0; cnt <= '0; eng_rst <= 1'b1; state <= S_OMV_PRIME;
            end else begin
              head_idx <= head_idx + 1'b1;
              chunk    <= '0; cnt <= '0; kv_t <= '0;
              score_emax <= -8'sd128;
              eng_rst <= 1'b1;
              state <= S_QK_PRIME;
            end
          end else begin
            chunk   <= chunk + 1'b1;
            eng_rst <= 1'b1;
            state   <= S_AV_PRIME;
          end
        end

        // ===================================================================
        // O matvec (D inputs, D outputs)
        // ===================================================================
        S_OMV_PRIME: begin
          if (ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd3;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_OMV_DRIVE;
          end
        end
        S_OMV_DRIVE: begin
          mv_valid <= 1'b1;
          // mv_x_m comes from attn_m_rd_data via the mv_x_m_eff mux.
          mv_x_e   <= attn_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_OMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_OMV_DRAIN: if (mv_out_valid_lat) begin : omv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_OMV_REQ;
        end
        S_OMV_REQ: begin : omv_pipeB
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          // o_m write handled by g_o_m_pack + o_m_we above (packed bfp_sdpram).
          o_e[chunk] <= emax_f;
          state <= S_OMV_NEXT;
        end
        S_OMV_NEXT: begin
          ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_D - 1) begin
            chunk   <= '0; cnt <= '0; out_cnt <= '0;
            eng_rst <= 1'b1; rs_start <= 1'b1; state <= S_RES1;
          end else begin
            chunk   <= chunk + 1'b1;
            eng_rst <= 1'b1; state <= S_OMV_PRIME;
          end
        end

        // ===================================================================
        // Residual 1: h1 = hin + o.
        //
        // rs_a_*/rs_b_*/rs_valid/rs_last are driven COMBINATIONALLY from cnt
        // (see always_comb above and the rs_drive_res1 flag).  The engine's
        // in_ready gate naturally pauses cnt during S_EMIT phases.  When
        // in_valid && in_ready both high THIS cycle, the engine WILL consume
        // the comb data into its registers this same cycle — so cnt can
        // advance without losing data at tile boundaries.
        // ===================================================================
        S_RES1: begin
          // (comb drive is via rs_drive_res1 / cnt above)
          // h1_m mantissa write handled by h1_m_we above.
          if (rs_in_ready && rs_valid_c) begin
            if (cnt == D-1) state <= S_RES1_WAIT;
            else            cnt <= cnt + 1'b1;
          end
          // Capture outputs concurrently — engine emits during S_LOAD gaps.
          if (rs_y_valid) begin
            if (out_cnt[3:0] == 4'd0)
              h1_e[out_cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            out_cnt <= out_cnt + 1'b1;
          end
        end
        S_RES1_WAIT: begin
          // h1_m mantissa write handled by h1_m_we above.
          if (rs_y_valid) begin
            if (out_cnt[3:0] == 4'd0)
              h1_e[out_cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            out_cnt <= out_cnt + 1'b1;
          end
          if (rs_done) begin
            state    <= S_NORM2;
            eng_rst  <= 1'b1;
            cnt      <= '0;
            out_cnt  <= '0;
            rn_start <= 1'b1;
          end
        end

        // ===================================================================
        // RMSNorm2
        // ===================================================================
        S_NORM2: begin
          rn_valid <= 1'b1;
          rn_x_m   <= h1_m_rd_data;
          rn_x_e   <= h1_e[cnt[CW_D-1:0] / BFP_TILE];
          rn_g_m   <= rom_G2_m[layer_idx * D    + cnt[CW_D-1:0]];
          rn_g_e   <= rom_G2_e[layer_idx * NT_D + cnt[CW_D-1:0] / BFP_TILE];
          rn_last  <= (cnt == D-1);
          if (cnt == D-1) begin
            state <= S_NORM2_WAIT; cnt <= '0;
          end else cnt <= cnt + 1'b1;
        end
        S_NORM2_WAIT: begin
          if (rn_y_valid) begin
            // n2_m write handled by always_comb-equivalent assigns above
            if (cnt[3:0] == 4'd0)
              n2_e[cnt[CW_D-1:0] / BFP_TILE] <= rn_y_e;
            if (cnt == D-1) begin
              cnt   <= '0; chunk <= '0; eng_rst <= 1'b1; state <= S_GMV_PRIME;
            end else cnt <= cnt + 1'b1;
          end
        end

        // ===================================================================
        // Gate matvec (D in, FFN out)
        // ===================================================================
        S_GMV_PRIME: begin
          if (ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd4;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_GMV_DRIVE;
          end
        end
        S_GMV_DRIVE: begin
          mv_valid <= 1'b1;
          // mv_x_m comes from n2_m_rd_data via the mv_x_m_eff mux.
          mv_x_e   <= n2_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_GMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_GMV_DRAIN: if (mv_out_valid_lat) begin : gmv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_GMV_REQ;
        end
        S_GMV_REQ: begin : gmv_pipeB
          // g_m LANES-parallel write driven by g_m_we + g_g_m_pack above.
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          g_e[chunk] <= emax_f;
          state <= S_GMV_NEXT;
        end
        S_GMV_NEXT: begin
          ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_FFN - 1) begin
            chunk <= '0; eng_rst <= 1'b1; state <= S_UMV_PRIME;
          end else begin
            chunk <= chunk + 1'b1; eng_rst <= 1'b1; state <= S_GMV_PRIME;
          end
        end

        // ===================================================================
        // Up matvec (D in, FFN out)
        // ===================================================================
        S_UMV_PRIME: begin
          if (ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd5;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_UMV_DRIVE;
          end
        end
        S_UMV_DRIVE: begin
          mv_valid <= 1'b1;
          // mv_x_m comes from n2_m_rd_data via the mv_x_m_eff mux.
          mv_x_e   <= n2_e[cnt[CW_D-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_UMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_UMV_DRAIN: if (mv_out_valid_lat) begin : umv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_UMV_REQ;
        end
        S_UMV_REQ: begin : umv_pipeB
          // u_m LANES-parallel write driven by u_m_we + g_u_m_pack above.
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          u_e[chunk] <= emax_f;
          state <= S_UMV_NEXT;
        end
        S_UMV_NEXT: begin
          ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_FFN - 1) begin
            chunk <= '0; cnt <= '0; out_cnt <= '0;
            eng_rst <= 1'b1; sg_start <= 1'b1; state <= S_SWG;
          end else begin
            chunk <= chunk + 1'b1; eng_rst <= 1'b1; state <= S_UMV_PRIME;
          end
        end

        // ===================================================================
        // SwiGLU (FFN -> FFN).  Comb drive via sg_drive flag.
        // ===================================================================
        S_SWG: begin
          if (sg_in_ready && sg_valid_c) begin
            if (cnt == FFN-1) state <= S_SWG_WAIT;
            else              cnt <= cnt + 1'b1;
          end
          if (sg_y_valid) begin
            // mlp_m write handled by always_comb (mlp_m_we/_wr_addr/_wr_data)
            if (out_cnt[3:0] == 4'd0)
              mlp_e[out_cnt[CW_FFN-1:0] / BFP_TILE] <= sg_y_e;
            out_cnt <= out_cnt + 1'b1;
          end
        end
        S_SWG_WAIT: begin
          if (sg_y_valid) begin
            // mlp_m write handled by always_comb (mlp_m_we/_wr_addr/_wr_data)
            if (out_cnt[3:0] == 4'd0)
              mlp_e[out_cnt[CW_FFN-1:0] / BFP_TILE] <= sg_y_e;
            out_cnt <= out_cnt + 1'b1;
          end
          if (sg_done) begin
            chunk <= '0; eng_rst <= 1'b1; state <= S_DMV_PRIME;
          end
        end

        // ===================================================================
        // Down matvec (FFN in, D out)
        // ===================================================================
        S_DMV_PRIME: begin
          if (ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_matvec_id <= 3'd6;
                ws_load_req  <= 1'b1;
                ws_phase     <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_DMV_DRIVE;
          end
        end
        S_DMV_DRIVE: begin
          mv_valid <= 1'b1;
          // mv_x_m no longer written here — mlp_m's bfp_sdpram output
          // register supplies it through the mv_x_m_eff mux at the
          // matvec instance.  Same edge-timing as the OLD inferred
          // `mv_x_m <= mlp_m[cnt]` (1-cycle latency, no lookahead).
          mv_x_e   <= mlp_e[cnt[CW_FFN-1:0] / BFP_TILE];
          mv_last  <= (cnt == FFN-1);
          if (cnt == FFN-1) state <= S_DMV_DRAIN;
          else cnt <= cnt + 1'b1;
        end
        S_DMV_DRAIN: if (mv_out_valid_lat) begin : dmv_pipeA
          automatic logic signed [BFP_EXP_W-1:0] e_lo, e_hi;
          e_lo = $signed(mv_out_e_lat[0 +: BFP_EXP_W]);
          for (ii = 1; ii < 8; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_lo)
              e_lo = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          e_hi = $signed(mv_out_e_lat[8*BFP_EXP_W +: BFP_EXP_W]);
          for (ii = 9; ii < LANES; ii++)
            if ($signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]) > e_hi)
              e_hi = $signed(mv_out_e_lat[ii*BFP_EXP_W +: BFP_EXP_W]);
          mv_out_m_drain_r <= mv_out_m_lat;
          mv_out_e_drain_r <= mv_out_e_lat;
          emax_h0_r        <= e_lo;
          emax_h1_r        <= e_hi;
          state <= S_DMV_REQ;
        end
        S_DMV_REQ: begin : dmv_pipeB
          automatic logic signed [BFP_EXP_W-1:0] emax_f;
          emax_f = (emax_h0_r > emax_h1_r) ? emax_h0_r : emax_h1_r;
          // d_m write handled by g_d_m_pack + d_m_we above (packed bfp_sdpram).
          d_e[chunk] <= emax_f;
          state <= S_DMV_NEXT;
        end
        S_DMV_NEXT: begin
          ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_D - 1) begin
            chunk    <= '0; cnt <= '0; out_cnt <= '0;
            eng_rst  <= 1'b1; rs_start <= 1'b1; state <= S_RES2;
          end else begin
            chunk    <= chunk + 1'b1;
            eng_rst  <= 1'b1; state <= S_DMV_PRIME;
          end
        end

        // ===================================================================
        // Residual 2: hout = h1 + d.  Comb drive via rs_drive_res2.
        // ===================================================================
        S_RES2: begin
          if (rs_in_ready && rs_valid_c) begin
            if (cnt == D-1) state <= S_RES2_WAIT;
            else            cnt <= cnt + 1'b1;
          end
          if (rs_y_valid) begin
            hout_m[out_cnt[CW_D-1:0]] <= rs_y_m;
            if (out_cnt[3:0] == 4'd0)
              hout_e[out_cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            if (snap_capture) begin
              snap_m[out_cnt[CW_D-1:0]] <= rs_y_m;
              if (out_cnt[3:0] == 4'd0)
                snap_e[out_cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            end
            out_cnt <= out_cnt + 1'b1;
          end
        end
        S_RES2_WAIT: begin
          if (rs_y_valid) begin
            hout_m[out_cnt[CW_D-1:0]] <= rs_y_m;
            if (out_cnt[3:0] == 4'd0)
              hout_e[out_cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            if (snap_capture) begin
              snap_m[out_cnt[CW_D-1:0]] <= rs_y_m;
              if (out_cnt[3:0] == 4'd0)
                snap_e[out_cnt[CW_D-1:0] / BFP_TILE] <= rs_y_e;
            end
            out_cnt <= out_cnt + 1'b1;
          end
          if (rs_done) state <= S_DONE;
        end

        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  always_comb dbg_state = state;
  always_comb dbg_cnt   = cnt;
  always_comb dbg_chunk = chunk;

`ifdef LBFP_STAGE_DUMP
  // Sim-only intermediate dumps.  $writememh runs on the cycle the FSM
  // transitions OUT of each post-stage state.  Triggers use the enum
  // names so they survive FSM reordering (the previous numeric literals
  // had drifted several places out of date after Phase 2 added more
  // states).  Vivado warn-skips $writememh; Verilator emits the file.
  state_t prev_state;
  // Unpack buffers for packed-BRAM arrays (q_m, k_m, v_m, attn_m, o_m,
  // d_m, g_m, u_m).  Filled lane-by-lane when each stage's dump fires.
  logic [BFP_MANT_W-1:0] sim_o_m_flat   [0:D-1];
  logic [BFP_MANT_W-1:0] sim_d_m_flat   [0:D-1];
  logic [BFP_MANT_W-1:0] sim_q_m_flat   [0:D-1];
  logic [BFP_MANT_W-1:0] sim_k_m_flat   [0:H_KV*HD-1];
  logic [BFP_MANT_W-1:0] sim_v_m_flat   [0:H_KV*HD-1];
  logic [BFP_MANT_W-1:0] sim_attn_m_flat[0:D-1];
  logic [BFP_MANT_W-1:0] sim_g_m_flat   [0:FFN-1];
  logic [BFP_MANT_W-1:0] sim_u_m_flat   [0:FFN-1];
  // Gate: only dump at the target layer (set via +define+LBFP_DUMP_LAYER,
  // default 0).  In the autoregress stream-sim 30 layers × 19 tokens
  // would otherwise overwrite each other; gating to one layer keeps the
  // last-token's view of that layer's state.  The single-layer selftest
  // (tb_smollm_layer_bfp) drives layer_idx=0 + kv_pos=3 and runs once,
  // so the default LBFP_DUMP_LAYER=0 matches.
`ifndef LBFP_DUMP_LAYER
  `define LBFP_DUMP_LAYER 0
`endif
  wire dump_ok = (layer_idx == `LBFP_DUMP_LAYER);
  always_ff @(posedge clk) begin
    if (rst) prev_state <= S_IDLE;
    else begin
      prev_state <= state;
      if (dump_ok && prev_state == S_NORM1_WAIT && state == S_QMV_PRIME) begin
        $writememh("rtl_n1_m.hex", i_n1_m_bram.mem);
        $writememh("rtl_n1_e.hex", n1_e);
      end
      if (dump_ok && prev_state == S_QMV_REQ && state == S_QMV_NEXT) begin
        // q_m now packed in i_q_m_bram.i_word_ram.mem — dump suppressed.
        $writememh("rtl_qpre_e.hex", q_e);
      end
      if (dump_ok && prev_state == S_KMV_REQ && state == S_KMV_NEXT) begin
        // k_m now packed in i_k_m_bram.i_word_ram.mem — dump suppressed.
        $writememh("rtl_kpre_e.hex", k_e);
      end
      if (dump_ok && prev_state == S_VMV_REQ && state == S_VMV_NEXT) begin
        // v_m packed in i_v_m_bram.i_word_ram.mem — dump suppressed.
        $writememh("rtl_v_e.hex", v_e);
      end
      // Post-RoPE requant — both Q and K finished, transitioning to KVWR.
      if (dump_ok && prev_state == S_ROPEK_RQ_B && state == S_KVWR_M) begin
        for (int t = 0; t < (D + LANES - 1) / LANES; t++)
          for (int l = 0; l < LANES; l++)
            sim_q_m_flat[t*LANES + l] =
                i_q_m_bram.i_word_ram.mem[t][l*BFP_MANT_W +: BFP_MANT_W];
        for (int t = 0; t < (H_KV*HD + LANES - 1) / LANES; t++)
          for (int l = 0; l < LANES; l++)
            sim_k_m_flat[t*LANES + l] =
                i_k_m_bram.i_word_ram.mem[t][l*BFP_MANT_W +: BFP_MANT_W];
        $writememh("rtl_q_m.hex", sim_q_m_flat);
        $writememh("rtl_k_m.hex", sim_k_m_flat);
        $writememh("rtl_q_e.hex", q_e);
        $writememh("rtl_k_e.hex", k_e);
      end
      // After attention (S_AV_NEXT → S_OMV_PRIME)
      if (dump_ok && prev_state == S_AV_NEXT && state == S_OMV_PRIME) begin
        // attn_m is now packed (LANES entries per BRAM word) inside
        // i_attn_m_bram.i_word_ram.mem — dump suppressed pending an
        // unpacking helper.  [[task-4]]
        $writememh("rtl_attn_e.hex", attn_e);
      end
      // After softmax wait (S_SM_WAIT → S_AV_PRIME): scores + probs ready.
      if (dump_ok && prev_state == S_SM_WAIT && state == S_AV_PRIME) begin
        $writememh("rtl_scores_m.hex", scores_m);
        $writememh("rtl_qkscore_e.hex", qk_score_e);
        $writememh("rtl_probs_m.hex", probs_m);
        $display("scores_e_shared=%0d probs_e_shared=%0d",
                 $signed(scores_e_shared), $signed(probs_e_shared));
      end
      // After O matvec (S_OMV_NEXT → S_RES1).
      if (dump_ok && prev_state == S_OMV_NEXT && state == S_RES1) begin
        // o_m is packed in i_o_m_bram.i_word_ram.mem (LANES entries per
        // 256-bit word).  Unpack lane-by-lane into a flat shadow for the
        // diff harness.
        for (int t = 0; t < (D + LANES - 1) / LANES; t++)
          for (int l = 0; l < LANES; l++)
            sim_o_m_flat[t*LANES + l] =
                i_o_m_bram.i_word_ram.mem[t][l*BFP_MANT_W +: BFP_MANT_W];
        $writememh("rtl_o_m.hex", sim_o_m_flat);
        $writememh("rtl_o_e.hex", o_e);
      end
      // After RES1 (S_RES1_WAIT → S_NORM2).
      if (dump_ok && prev_state == S_RES1_WAIT && state == S_NORM2) begin
        $writememh("rtl_h1_m.hex", i_h1_m_bram.mem);
        $writememh("rtl_h1_e.hex", h1_e);
      end
      // After NORM2 (S_NORM2_WAIT → S_GMV_PRIME).
      if (dump_ok && prev_state == S_NORM2_WAIT && state == S_GMV_PRIME) begin
        $writememh("rtl_n2_m.hex", i_n2_m_bram.mem);
        $writememh("rtl_n2_e.hex", n2_e);
      end
      // After GMV (S_GMV_NEXT → S_UMV_PRIME).
      if (dump_ok && prev_state == S_GMV_NEXT && state == S_UMV_PRIME) begin
        // Unpack g_m from packed BRAM (LANES per 256-bit word) so the
        // diff harness can see per-output values.  Needed for chasing
        // the smollm360 mlp-stage divergence — diff_stages.py was
        // blind to g_m / u_m until this dump.
        for (int t = 0; t < (FFN + LANES - 1) / LANES; t++)
          for (int l = 0; l < LANES; l++)
            sim_g_m_flat[t*LANES + l] =
                i_g_m_bram.i_word_ram.mem[t][l*BFP_MANT_W +: BFP_MANT_W];
        $writememh("rtl_g_m.hex", sim_g_m_flat);
        $writememh("rtl_g_e.hex", g_e);
      end
      // After UMV (S_UMV_NEXT → S_SWG).
      if (dump_ok && prev_state == S_UMV_NEXT && state == S_SWG) begin
        for (int t = 0; t < (FFN + LANES - 1) / LANES; t++)
          for (int l = 0; l < LANES; l++)
            sim_u_m_flat[t*LANES + l] =
                i_u_m_bram.i_word_ram.mem[t][l*BFP_MANT_W +: BFP_MANT_W];
        $writememh("rtl_u_m.hex", sim_u_m_flat);
        $writememh("rtl_u_e.hex", u_e);
      end
      // After SWG (S_SWG_WAIT → S_DMV_PRIME).
      if (dump_ok && prev_state == S_SWG_WAIT && state == S_DMV_PRIME) begin
        $writememh("rtl_mlp_m.hex", i_mlp_m_bram.mem);
        $writememh("rtl_mlp_e.hex", mlp_e);
      end
      // After DMV (S_DMV_NEXT → S_RES2).
      if (dump_ok && prev_state == S_DMV_NEXT && state == S_RES2) begin
        // d_m is now packed inside i_d_m_bram.i_word_ram.mem (see attn_m).
        $writememh("rtl_d_e.hex", d_e);
      end
    end
  end
`endif

endmodule

`default_nettype wire
