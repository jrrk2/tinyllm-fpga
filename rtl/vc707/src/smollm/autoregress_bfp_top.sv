// autoregress_bfp_top.sv — self-contained block-FP autoregressive token
// generator.  Single `start` pulse runs the prompt + autoregress loop on
// chip; emits `done` when finished plus a packed bus of all generated
// tokens.  Suitable as the FPGA top-level instantiation for the SmolLM2
// BFP path.
//
// The three sub-modules each own an AXI master that fetches their
// slice of the weight set from DDR3.  Because the model stages are
// strictly sequential —
// EMBED → NL × LAYER → DECODE — only one master is active at a time,
// so this module collapses the three onto a single AR/R channel via a
// priority OR-mux (the source whose arvalid is high wins, and rvalid
// is broadcast back to all three).

`include "bfp_format.svh"

`default_nettype none

module autoregress_bfp_top #(
  parameter int D        = 576,
  parameter int H_Q      = 9,
  parameter int H_KV     = 3,
  parameter int HD       = 64,
  parameter int FFN      = 1536,
  parameter int NL       = 30,
  parameter int MAX_CTX  = 64,
  parameter int VOCAB    = 49152,
  // NPROMPT_MAX sizes the host-writable prompt_rom and the result_tokens
  // output buffer at synthesis time.  N_PROMPT is the boot-time default
  // active prompt length (the value the host can override at runtime
  // via n_prompt_active).  Any prompt of length 1..NPROMPT_MAX can run
  // on the same bitstream — no Vivado rebuild needed when changing the
  // prompt length.
  parameter int NPROMPT_MAX = 64,
  parameter int N_PROMPT    = 4,
  parameter int N_GEN       = 15,
  parameter int N_STEPS     = NPROMPT_MAX + N_GEN,
  parameter     PREFIX   = "lbfp_full_",
  parameter bit STREAM_LOOKUP  = 1'b0,
  parameter int AXI_ADDR_WIDTH = 30,
  parameter int AXI_ID_WIDTH   = 5
)(
  input  wire                          clk,
  input  wire                          rst,
  input  wire                          start,
  // Active prompt length — number of prompt_rom entries the FSM
  // consumes before switching to autoregressive generation.  Must be
  // in [1, NPROMPT_MAX].  Already 2-FF synced into the core_clk
  // domain by the caller; stable for the duration of a run.
  input  wire [$clog2(NPROMPT_MAX+1)-1:0] n_prompt_active,
  output logic                         done,
  output logic [N_STEPS*16-1:0]        result_tokens,
  output wire  [31:0]                  weight_hash,
  // Host BRAM-write port (core_clk).  Decoded internally:
  //   0..3 → layer gammas, 4..5 → decode head norm_w, 6 → prompt_rom.
  input  wire [4:0]                    wr_kind,
  input  wire [17:0]                   wr_addr,
  input  wire [15:0]                   wr_data,
  input  wire                          wr_en,
  input  wire                          clk_wr,   // BRAM write clock (eth_clk at top)
  // Read-back of the BRAM at wr_addr — muxed across all sub-modules by
  // wr_kind (registered, 1-cycle latency in clk_wr).  Returns zero for
  // unmapped kinds.
  output logic [15:0]                  wr_rdata,
  // Per-layer hidden-state snapshot select — passed to the multilayer engine,
  // which latches the chosen (layer, token-step) hidden_out for host read-back
  // via wr_kind 12 (mantissa) / 13 (per-tile exponent).
  input  wire [4:0]                    snap_layer_sel,
  input  wire [10:0]                   snap_step_sel,
  // Freeze the engine at the programmed (snap_layer_sel, snap_step_sel) for
  // ILA / hout inspection — passed straight to the multilayer engine.
  input  wire                          freeze_en,
  // Absolute-cycle freeze trigger + frozen-state status (logic-analyser).
  input  wire                          trig_cyc_en,
  input  wire [31:0]                   trig_cyc,
  output wire [31:0]                   dbg_cyc,
  output wire [4:0]                    dbg_cur_layer,
  output wire                          dbg_frozen,
  // Layer-0 / region base offsets — caller patches in lbfp_ddr3.svh
  // constants.  Ignored when neither stream is enabled.
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WQ_m,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WQ_e,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WK_m,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WK_e,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WV_m,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WV_e,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WO_m,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WO_e,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WG_m,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WG_e,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WU_m,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WU_e,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WDN_m,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_WDN_e,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_EMBED_m,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_EMBED_e,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_EMBED_LU_m,
  input  wire [AXI_ADDR_WIDTH-1:0]     ws_base_EMBED_LU_e,
  // AXI master to MIG (clk_axi domain).
  input  wire                          clk_axi,
  input  wire                          rst_axi,
  output wire                          m_axi_arvalid,
  input  wire                          m_axi_arready,
  output wire [AXI_ID_WIDTH-1:0]       m_axi_arid,
  output wire [AXI_ADDR_WIDTH-1:0]     m_axi_araddr,
  output wire [7:0]                    m_axi_arlen,
  output wire [2:0]                    m_axi_arsize,
  output wire [1:0]                    m_axi_arburst,
  output wire                          m_axi_arlock,
  output wire [3:0]                    m_axi_arcache,
  output wire [2:0]                    m_axi_arprot,
  output wire [3:0]                    m_axi_arqos,
  input  wire                          m_axi_rvalid,
  output wire                          m_axi_rready,
  input  wire [AXI_ID_WIDTH-1:0]       m_axi_rid,
  input  wire [511:0]                  m_axi_rdata,
  input  wire [1:0]                    m_axi_rresp,
  input  wire                          m_axi_rlast
);

  // ---------------------------------------------------------------------------
  // Prompt ROM.
  // ---------------------------------------------------------------------------
  // prompt_rom is host-loaded at boot via wr_* (kind=6, addr=0..N_PROMPT-1).
  // Power-on contents are all-zero — autoregress would treat that as
  // <|endoftext|> tokens until host calls bfp_client load-roms.
  (* ram_style = "block" *) logic [15:0] prompt_rom [0:NPROMPT_MAX-1];
`ifdef VERILATOR
  // Sim has no host-write driver for prompt_rom, so the testbench would
  // otherwise see all-zero prompt tokens and the autoregress would loop
  // emitting <|endoftext|> forever (token 0 → embed[0] → argmax back to
  // 0 → repeat).  Load directly from the same .hex the FPGA load-roms
  // would upload.
  initial $readmemh("../generated/lbfp_full_PROMPT.hex", prompt_rom);
`endif
  logic [15:0] rd_prompt;
  logic [4:0]  wr_kind_q;
  always_ff @(posedge clk_wr) begin
    if (wr_en && wr_kind == 5'd6)
      prompt_rom[wr_addr[$clog2(NPROMPT_MAX)-1:0]] <= wr_data;
    rd_prompt <= prompt_rom[wr_addr[$clog2(NPROMPT_MAX)-1:0]];
    wr_kind_q <= wr_kind;
  end

  // Read-back MUX.  Layer / decode-head MUXes (kinds 0..3 / 4..5)
  // arrive on lay_wr_rdata / dec_wr_rdata (their own wr_kind_q select).
  wire [15:0] lay_wr_rdata;
  wire [15:0] dec_wr_rdata;
  always_comb begin
    case (wr_kind_q)
      5'd0, 5'd1, 5'd2, 5'd3: wr_rdata = lay_wr_rdata;
      5'd4, 5'd5:             wr_rdata = dec_wr_rdata;
      5'd6:                   wr_rdata = rd_prompt;
      // wr_kind 10/11 — debug peek of the LAST layer's hout_m / hout_e
      // (i.e. the decode-head input).  See smollm_layer_bfp's wr_rdata
      // mux for the actual read path.
      5'd10, 5'd11:           wr_rdata = lay_wr_rdata;
      // wr_kind 12/13 — per-layer hidden snapshot (snap_m/snap_e), captured
      // and read back inside smollm_multilayer_tm_bfp (→ lay_wr_rdata).
      5'd12, 5'd13:           wr_rdata = lay_wr_rdata;
      default:                wr_rdata = 16'h0000;
    endcase
  end

  localparam int NT_D = D / BFP_TILE;

  // ---------------------------------------------------------------------------
  // Inner sub-module wires (3 AXI masters that share m_axi_*).
  // ---------------------------------------------------------------------------
  wire                          emb_arvalid, lay_arvalid, dec_arvalid;
  wire [AXI_ID_WIDTH-1:0]       emb_arid,    lay_arid,    dec_arid;
  wire [AXI_ADDR_WIDTH-1:0]     emb_araddr,  lay_araddr,  dec_araddr;
  wire [7:0]                    emb_arlen,   lay_arlen,   dec_arlen;
  wire [2:0]                    emb_arsize,  lay_arsize,  dec_arsize;
  wire [1:0]                    emb_arburst, lay_arburst, dec_arburst;
  wire                          emb_arlock,  lay_arlock,  dec_arlock;
  wire [3:0]                    emb_arcache, lay_arcache, dec_arcache;
  wire [2:0]                    emb_arprot,  lay_arprot,  dec_arprot;
  wire [3:0]                    emb_arqos,   lay_arqos,   dec_arqos;
  wire                          emb_rready,  lay_rready,  dec_rready;

  // ---------------------------------------------------------------------------
  // Three-way AR-channel priority OR-mux.
  // Because EMB → LAY → DEC are strictly sequential, only one arvalid is
  // ever high; the priority ordering protects against transient overlap.
  // ---------------------------------------------------------------------------
  assign m_axi_arvalid = emb_arvalid | lay_arvalid | dec_arvalid;
  assign m_axi_arid    = emb_arvalid ? emb_arid    : lay_arvalid ? lay_arid    : dec_arid;
  assign m_axi_araddr  = emb_arvalid ? emb_araddr  : lay_arvalid ? lay_araddr  : dec_araddr;
  assign m_axi_arlen   = emb_arvalid ? emb_arlen   : lay_arvalid ? lay_arlen   : dec_arlen;
  assign m_axi_arsize  = emb_arvalid ? emb_arsize  : lay_arvalid ? lay_arsize  : dec_arsize;
  assign m_axi_arburst = emb_arvalid ? emb_arburst : lay_arvalid ? lay_arburst : dec_arburst;
  assign m_axi_arlock  = emb_arvalid ? emb_arlock  : lay_arvalid ? lay_arlock  : dec_arlock;
  assign m_axi_arcache = emb_arvalid ? emb_arcache : lay_arvalid ? lay_arcache : dec_arcache;
  assign m_axi_arprot  = emb_arvalid ? emb_arprot  : lay_arvalid ? lay_arprot  : dec_arprot;
  assign m_axi_arqos   = emb_arvalid ? emb_arqos   : lay_arvalid ? lay_arqos   : dec_arqos;
  // rready: broadcast `1`; each sub-module ignores rvalid when inactive.
  assign m_axi_rready  = emb_rready & lay_rready & dec_rready;

  // arready arrives back; route to whichever source's arvalid is winning.
  wire emb_arready = m_axi_arready & emb_arvalid;
  wire lay_arready = m_axi_arready & (~emb_arvalid) & lay_arvalid;
  wire dec_arready = m_axi_arready & (~emb_arvalid) & (~lay_arvalid) & dec_arvalid;
  // rvalid / rid / rdata / rresp / rlast broadcast — at most one
  // sub-module's load FSM is in its R-state at any moment.
  wire emb_rvalid_b = m_axi_rvalid;
  wire lay_rvalid_b = m_axi_rvalid;
  wire dec_rvalid_b = m_axi_rvalid;

  // ---------------------------------------------------------------------------
  // embed_lookup_bfp.
  // ---------------------------------------------------------------------------
  logic                                mdl_start;
  logic [15:0]                         mdl_token_in;
  logic [10:0]                         mdl_pos;
  logic [6:0]                          mdl_kv_pos;
  wire  signed [D*BFP_MANT_W-1:0]      emb_m;
  wire  signed [NT_D*BFP_EXP_W-1:0]    emb_e;
  wire                                 emb_done;

  embed_lookup_bfp #(
    .D(D), .VOCAB(VOCAB), .PREFIX(PREFIX),
    .STREAM_LOOKUP(STREAM_LOOKUP),
    .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
    .AXI_ID_WIDTH(AXI_ID_WIDTH)
  ) i_emb (
    .clk(clk), .rst(rst), .start(mdl_start),
    .token_id(mdl_token_in),
    .hidden_m(emb_m), .hidden_e(emb_e), .done(emb_done),
    .ws_base_EMBED_LU_m(ws_base_EMBED_LU_m),
    .ws_base_EMBED_LU_e(ws_base_EMBED_LU_e),
    .clk_axi(clk_axi), .rst_axi(rst_axi),
    .m_axi_arvalid(emb_arvalid),
    .m_axi_arready(emb_arready),
    .m_axi_arid   (emb_arid),
    .m_axi_araddr (emb_araddr),
    .m_axi_arlen  (emb_arlen),
    .m_axi_arsize (emb_arsize),
    .m_axi_arburst(emb_arburst),
    .m_axi_arlock (emb_arlock),
    .m_axi_arcache(emb_arcache),
    .m_axi_arprot (emb_arprot),
    .m_axi_arqos  (emb_arqos),
    .m_axi_rvalid (emb_rvalid_b),
    .m_axi_rready (emb_rready),
    .m_axi_rid    (m_axi_rid),
    .m_axi_rdata  (m_axi_rdata),
    .m_axi_rresp  (m_axi_rresp),
    .m_axi_rlast  (m_axi_rlast)
  );

  // ---------------------------------------------------------------------------
  // smollm_multilayer_tm_bfp.
  // ---------------------------------------------------------------------------
  logic signed [D*BFP_MANT_W-1:0]      lay_in_m;
  logic signed [NT_D*BFP_EXP_W-1:0]    lay_in_e;
  logic                                lay_start;
  wire  signed [D*BFP_MANT_W-1:0]      lay_out_m;
  wire  signed [NT_D*BFP_EXP_W-1:0]    lay_out_e;
  wire                                 lay_done;

  smollm_multilayer_tm_bfp #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD),
    .FFN(FFN), .MAX_CTX(MAX_CTX), .NL(NL), .PREFIX(PREFIX),
    .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
    .AXI_ID_WIDTH(AXI_ID_WIDTH)
  ) i_lay (
    .clk(clk), .rst(rst), .start(lay_start),
    .pos(mdl_pos), .kv_pos(mdl_kv_pos),
    .hidden_in_m(lay_in_m), .hidden_in_e(lay_in_e),
    .hidden_out_m(lay_out_m), .hidden_out_e(lay_out_e),
    .done(lay_done),
    .weight_hash(weight_hash),
    .ws_base_WQ_m(ws_base_WQ_m),   .ws_base_WQ_e(ws_base_WQ_e),
    .ws_base_WK_m(ws_base_WK_m),   .ws_base_WK_e(ws_base_WK_e),
    .ws_base_WV_m(ws_base_WV_m),   .ws_base_WV_e(ws_base_WV_e),
    .ws_base_WO_m(ws_base_WO_m),   .ws_base_WO_e(ws_base_WO_e),
    .ws_base_WG_m(ws_base_WG_m),   .ws_base_WG_e(ws_base_WG_e),
    .ws_base_WU_m(ws_base_WU_m),   .ws_base_WU_e(ws_base_WU_e),
    .ws_base_WDN_m(ws_base_WDN_m), .ws_base_WDN_e(ws_base_WDN_e),
    .clk_axi(clk_axi), .rst_axi(rst_axi),
    .m_axi_arvalid(lay_arvalid),
    .m_axi_arready(lay_arready),
    .m_axi_arid   (lay_arid),
    .m_axi_araddr (lay_araddr),
    .m_axi_arlen  (lay_arlen),
    .m_axi_arsize (lay_arsize),
    .m_axi_arburst(lay_arburst),
    .m_axi_arlock (lay_arlock),
    .m_axi_arcache(lay_arcache),
    .m_axi_arprot (lay_arprot),
    .m_axi_arqos  (lay_arqos),
    .m_axi_rvalid (lay_rvalid_b),
    .m_axi_rready (lay_rready),
    .m_axi_rid    (m_axi_rid),
    .m_axi_rdata  (m_axi_rdata),
    .m_axi_rresp  (m_axi_rresp),
    .m_axi_rlast  (m_axi_rlast),
    .wr_kind(wr_kind), .wr_addr(wr_addr), .wr_data(wr_data), .wr_en(wr_en),
    .clk_wr(clk_wr), .wr_rdata(lay_wr_rdata),
    .snap_layer_sel(snap_layer_sel), .snap_step_sel(snap_step_sel),
    .freeze_en(freeze_en),
    .trig_cyc_en(trig_cyc_en), .trig_cyc(trig_cyc),
    .dbg_cyc(dbg_cyc), .dbg_cur_layer(dbg_cur_layer), .dbg_frozen(dbg_frozen)
  );

  // ---------------------------------------------------------------------------
  // smollm_decode_head_bfp.
  // ---------------------------------------------------------------------------
  logic                                dec_start;
  logic signed [D*BFP_MANT_W-1:0]      dec_in_m;
  logic signed [NT_D*BFP_EXP_W-1:0]    dec_in_e;
  wire  [15:0]                         dec_token;
  wire                                 dec_done;

  smollm_decode_head_bfp #(
    .D(D), .VOCAB(VOCAB), .PREFIX(PREFIX),
    .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
    .AXI_ID_WIDTH(AXI_ID_WIDTH)
  ) i_dec (
    .clk(clk), .rst(rst), .start(dec_start),
    .hidden_in_m(dec_in_m), .hidden_in_e(dec_in_e),
    .token_out(dec_token), .done(dec_done),
    .ws_base_EMBED_m(ws_base_EMBED_m),
    .ws_base_EMBED_e(ws_base_EMBED_e),
    .clk_axi(clk_axi), .rst_axi(rst_axi),
    .m_axi_arvalid(dec_arvalid),
    .m_axi_arready(dec_arready),
    .m_axi_arid   (dec_arid),
    .m_axi_araddr (dec_araddr),
    .m_axi_arlen  (dec_arlen),
    .m_axi_arsize (dec_arsize),
    .m_axi_arburst(dec_arburst),
    .m_axi_arlock (dec_arlock),
    .m_axi_arcache(dec_arcache),
    .m_axi_arprot (dec_arprot),
    .m_axi_arqos  (dec_arqos),
    .m_axi_rvalid (dec_rvalid_b),
    .m_axi_rready (dec_rready),
    .m_axi_rid    (m_axi_rid),
    .m_axi_rdata  (m_axi_rdata),
    .m_axi_rresp  (m_axi_rresp),
    .m_axi_rlast  (m_axi_rlast),
    .wr_kind(wr_kind), .wr_addr(wr_addr), .wr_data(wr_data), .wr_en(wr_en),
    .clk_wr(clk_wr), .wr_rdata(dec_wr_rdata)
  );

  // ---------------------------------------------------------------------------
  // Outer FSM
  // ---------------------------------------------------------------------------
  typedef enum logic [3:0] {
    S_IDLE, S_DRIVE,
    S_EMB, S_EMB_WAIT,
    S_LAY, S_LAY_WAIT,
    S_DEC, S_DEC_WAIT,
    S_CAPTURE, S_NEXT,
    S_ALL_DONE
  } st_t;
  st_t state;
  logic [$clog2(N_STEPS+1)-1:0]        step;
  logic [15:0]                         last_token;
  logic [15:0] result_buf [0:N_STEPS-1];

  always_ff @(posedge clk) begin
    if (rst) begin
      state       <= S_IDLE;
      step        <= '0;
      last_token  <= '0;
      mdl_start   <= 1'b0;
      lay_start   <= 1'b0;
      dec_start   <= 1'b0;
      mdl_token_in<= '0;
      mdl_pos     <= '0;
      mdl_kv_pos  <= '0;
      lay_in_m    <= '0;
      lay_in_e    <= '0;
      dec_in_m    <= '0;
      dec_in_e    <= '0;
      done        <= 1'b0;
    end else begin
      mdl_start <= 1'b0;
      lay_start <= 1'b0;
      dec_start <= 1'b0;
      done      <= 1'b0;
      case (state)
        S_IDLE: if (start) begin
          step       <= '0;
          last_token <= '0;
          state      <= S_DRIVE;
        end
        S_DRIVE: begin
          if (step < n_prompt_active)
                               mdl_token_in <= prompt_rom[step[$clog2(NPROMPT_MAX)-1:0]];
          else                 mdl_token_in <= last_token;
          mdl_pos    <= 11'(step);
          mdl_kv_pos <= 7'(step);
          mdl_start  <= 1'b1;
          state      <= S_EMB;
        end
        S_EMB:       state <= S_EMB_WAIT;
        S_EMB_WAIT:  if (emb_done) begin
          lay_in_m  <= emb_m;
          lay_in_e  <= emb_e;
          lay_start <= 1'b1;
          state     <= S_LAY;
        end
        S_LAY:       state <= S_LAY_WAIT;
        S_LAY_WAIT:  if (lay_done) begin
          dec_in_m  <= lay_out_m;
          dec_in_e  <= lay_out_e;
          dec_start <= 1'b1;
          state     <= S_DEC;
        end
        S_DEC:       state <= S_DEC_WAIT;
        S_DEC_WAIT:  if (dec_done) begin
          last_token       <= dec_token;
          result_buf[step] <= dec_token;
          state            <= S_NEXT;
        end
        S_NEXT: begin
          // Terminate after n_prompt_active prompt tokens + N_GEN generated
          // tokens.  N_STEPS (= NPROMPT_MAX + N_GEN) sizes the buffer at
          // synth — but if the active prompt is shorter than NPROMPT_MAX
          // (the common case) the FSM would otherwise iterate over the
          // empty slack slots, blowing the host poll timeout for no gain.
          if (step == ({{($bits(step)-$clog2(NPROMPT_MAX+1)){1'b0}},
                        n_prompt_active} + N_GEN - 1))
            state <= S_ALL_DONE;
          else begin
            step  <= step + 1'b1;
            state <= S_DRIVE;
          end
        end
        S_ALL_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  always_comb begin
    for (int i = 0; i < N_STEPS; i++)
      result_tokens[i*16 +: 16] = result_buf[i];
  end

endmodule

`default_nettype wire
