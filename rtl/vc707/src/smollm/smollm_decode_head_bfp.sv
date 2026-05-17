// smollm_decode_head_bfp.sv — block-FP final norm + lm_head + argmax.
//
// Pipeline:
//   1. RMSNorm of hidden_in with norm_w (the model's final norm gamma).
//   2. lm_head matvec — logits = h_normed @ EMBED.T   (tied embeddings).
//      EMBED is (VOCAB, D) and the matvec engine treats it row-by-row,
//      LANES=16 rows per chunk × CHUNKS_VOCAB = VOCAB/16 chunks.
//   3. Running argmax across the VOCAB logits — compare each lane's
//      (mant, exp) to the running best, keep the lane index that wins.
//
// Output: 16-bit `token_out` (the argmax token id) and `done` pulse.

`include "bfp_format.svh"

`default_nettype none

module smollm_decode_head_bfp #(
  parameter int D     = 576,
  parameter int VOCAB = 49152,         // assumed LANES-aligned (baker pads)
  parameter     PREFIX = "lbfp_full_",
  parameter int AXI_ADDR_WIDTH = 30,
  parameter int AXI_ID_WIDTH   = 5
)(
  input  wire                                       clk,
  input  wire                                       rst,
  input  wire                                       start,
  input  wire signed [D*BFP_MANT_W-1:0]             hidden_in_m,
  input  wire signed [(D/BFP_TILE)*BFP_EXP_W-1:0]   hidden_in_e,
  output logic [15:0]                               token_out,
  output logic                                      done,
  // DDR3 streamer interface.  The EMBED matrix is wide-packed exactly
  // like a W?_* weight matrix, so it reuses weight_streamer_bfp_mt
  // with the same chunk_idx (=chunk) and in_dim=D semantics.  Norm
  // gamma stays on-chip BRAM.
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_EMBED_m,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_EMBED_e,
  input  wire                                       clk_axi,
  input  wire                                       rst_axi,
  output wire                                       m_axi_arvalid,
  input  wire                                       m_axi_arready,
  output wire [AXI_ID_WIDTH-1:0]                    m_axi_arid,
  output wire [AXI_ADDR_WIDTH-1:0]                  m_axi_araddr,
  output wire [7:0]                                 m_axi_arlen,
  output wire [2:0]                                 m_axi_arsize,
  output wire [1:0]                                 m_axi_arburst,
  output wire                                       m_axi_arlock,
  output wire [3:0]                                 m_axi_arcache,
  output wire [2:0]                                 m_axi_arprot,
  output wire [3:0]                                 m_axi_arqos,
  input  wire                                       m_axi_rvalid,
  output wire                                       m_axi_rready,
  input  wire [AXI_ID_WIDTH-1:0]                    m_axi_rid,
  input  wire [511:0]                               m_axi_rdata,
  input  wire [1:0]                                 m_axi_rresp,
  input  wire                                       m_axi_rlast,
  // Host write port for rom_NW_m (kind=4) and rom_NW_e (kind=5).
  // Plumbed in from vc707_microgpt_eth via the autoregress top.
  input  wire [4:0]                                 wr_kind,
  input  wire [17:0]                                wr_addr,
  input  wire [15:0]                                wr_data,
  input  wire                                       wr_en,
  input  wire                                       clk_wr,  // BRAM write clock
  // Read-back of the BRAM at wr_addr — port-A read on the same TDP
  // BRAM.  Muxed by wr_kind so the caller can verify what was loaded.
  output logic [15:0]                               wr_rdata
);

  localparam int NT_D          = D / BFP_TILE;
  localparam int LANES         = 16;
  localparam int CHUNKS_VOCAB  = VOCAB / LANES;

  // ---------------------------------------------------------------------------
  // Final norm gamma (norm_w) — narrow ROM.
  // ---------------------------------------------------------------------------
  (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] rom_NW_m [0:D-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] rom_NW_e [0:NT_D-1];

  // norm_w is host-loaded at boot via the wr_* port (kind=4 for NW_m,
  // kind=5 for NW_e).  Write port is on clk_wr (eth_clk); read port
  // stays on clk (core_clk) — true-dual-port BRAM, no CDC.
  logic [BFP_MANT_W-1:0] rd_NW_m;
  logic [BFP_EXP_W -1:0] rd_NW_e;
  logic [4:0]            wr_kind_q;

  always_ff @(posedge clk_wr) begin
    if (wr_en) begin
      case (wr_kind)
        5'd4: rom_NW_m[wr_addr[$clog2(D)-1:0]]    <= $signed(wr_data);
        5'd5: rom_NW_e[wr_addr[$clog2(NT_D)-1:0]] <= $signed(wr_data[BFP_EXP_W-1:0]);
        default: ;
      endcase
    end
    rd_NW_m   <= rom_NW_m[wr_addr[$clog2(D)-1:0]];
    rd_NW_e   <= rom_NW_e[wr_addr[$clog2(NT_D)-1:0]];
    wr_kind_q <= wr_kind;
  end

  always_comb begin
    case (wr_kind_q)
      5'd4: wr_rdata = {{(16-BFP_MANT_W){rd_NW_m[BFP_MANT_W-1]}}, rd_NW_m};
      5'd5: wr_rdata = {{(16-BFP_EXP_W ){rd_NW_e[BFP_EXP_W -1]}}, rd_NW_e};
      default: wr_rdata = 16'h0000;
    endcase
  end

  // ---------------------------------------------------------------------------
  // Latched hidden_in (BRAM-style)
  // ---------------------------------------------------------------------------
  (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] hin_m [0:D-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] hin_e [0:NT_D-1];
  // RMSNorm output buffer
  (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] hn_m [0:D-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] hn_e [0:NT_D-1];

  // ---------------------------------------------------------------------------
  // FSM
  // ---------------------------------------------------------------------------
  typedef enum logic [3:0] {
    S_IDLE, S_LATCH,
    S_NORM, S_NORM_WAIT,
    S_MV_PRIME, S_MV_DRIVE, S_MV_DRAIN,
    S_MV_LANE_CMP, S_MV_MERGE,
    S_MV_NEXT,
    S_DONE
  } st_t;
  st_t state;

  // ---------------------------------------------------------------------------
  // rmsnorm_bfp instance
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // matvec_bfp_engine instance — LANES outputs per chunk
  // ---------------------------------------------------------------------------
  logic                                       mv_start, mv_valid, mv_last;
  logic                                       mv_eng_rst;
  logic signed [BFP_MANT_W-1:0]               mv_x_m;
  logic signed [BFP_EXP_W -1:0]               mv_x_e;
  wire  signed [LANES*BFP_MANT_W-1:0]         mv_out_m;
  wire  signed [LANES*BFP_EXP_W -1:0]         mv_out_e;
  wire                                        mv_out_valid;

  wire signed [LANES*BFP_MANT_W-1:0]          mv_w_m_eff;
  wire signed [LANES*BFP_EXP_W -1:0]          mv_w_e_eff;
  wire        [255:0]                         ws_weight_m_out;
  wire        [127:0]                         ws_weight_e_out;
  assign mv_w_m_eff = $signed(ws_weight_m_out);
  assign mv_w_e_eff = $signed(ws_weight_e_out);

  matvec_bfp_engine #(.LANES(LANES)) i_mv (
    .clk(clk), .rst(rst | mv_eng_rst),
    .start_matvec(mv_start),
    .in_x_mant(mv_x_m), .in_x_exp(mv_x_e),
    .in_valid(mv_valid), .last_elem(mv_last),
    .w_mant(mv_w_m_eff), .w_exp(mv_w_e_eff),
    .out_mant(mv_out_m), .out_exp(mv_out_e), .out_valid(mv_out_valid)
  );

  // ---------------------------------------------------------------------------
  // DDR3 weight streamer (EMBED matrix).
  // chunk_idx = layer FSM `chunk`, in_dim = D.
  // ---------------------------------------------------------------------------
  logic                             ws_load_req;
  wire                              ws_ready, ws_busy_unused;
  logic [11:0]                      ws_rd_col, ws_rd_tile;
  typedef enum logic [1:0] {
    WSP_KICK, WSP_HOLD, WSP_WAIT, WSP_READY
  } ws_phase_t;
  ws_phase_t ws_phase;
  always_comb ws_rd_col  = cnt;
  always_comb ws_rd_tile = cnt >> 4;

  weight_streamer_bfp_mt #(
    .AXI_DATA_WIDTH (512),
    .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH),
    .AXI_ID_WIDTH   (AXI_ID_WIDTH),
    .IN_DIM_MAX     (D),
    .IN_DIM_BITS    (12),
    .CHUNK_BITS     ($clog2(CHUNKS_VOCAB > 0 ? CHUNKS_VOCAB : 1))
  ) i_ws (
    .clk_core      (clk),
    .rst_core      (rst),
    .matrix_base_m (ws_base_EMBED_m),
    .matrix_base_e (ws_base_EMBED_e),
    .chunk_idx     (chunk),
    .in_dim        (12'(D)),
    .in_dim_tiles  (12'(NT_D)),
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
  );

  // ---------------------------------------------------------------------------
  // Counters / argmax tracking
  // ---------------------------------------------------------------------------
  logic [11:0]                  cnt;
  logic [$clog2(CHUNKS_VOCAB)-1:0] chunk;
  logic signed [BFP_MANT_W-1:0] best_m;
  logic signed [BFP_EXP_W -1:0] best_e;
  logic [15:0]                  best_idx;

  // Pipelined argmax state — walking 1 lane per cycle so each clock
  // sees one bfp_gt instead of 16 cascaded.  Replaces the 147-LUT-level
  // combinational chain that blew the 50 MHz timing budget by 45 ns.
  logic signed [LANES*BFP_MANT_W-1:0] mv_out_m_r;
  logic signed [LANES*BFP_EXP_W -1:0] mv_out_e_r;
  logic [3:0]                         lane_cnt;
  logic signed [BFP_MANT_W-1:0]       local_best_m;
  logic signed [BFP_EXP_W -1:0]       local_best_e;
  logic [3:0]                         local_best_lane;

  // Compare (m_a, e_a) > (m_b, e_b) returning 1 if a is larger in real value.
  // Both signed.  Bring to common exp = max(e_a, e_b), align mantissas, compare.
  function automatic logic bfp_gt;
    input logic signed [BFP_MANT_W-1:0] m_a;
    input logic signed [BFP_EXP_W -1:0] e_a;
    input logic signed [BFP_MANT_W-1:0] m_b;
    input logic signed [BFP_EXP_W -1:0] e_b;
    logic signed [BFP_EXP_W:0] de;
    logic signed [BFP_EXP_W:0] sh;
    logic signed [BFP_MANT_W-1:0] a_al, b_al;
    de = $signed({e_a[BFP_EXP_W-1], e_a}) - $signed({e_b[BFP_EXP_W-1], e_b});
    if (de == 0) begin
      bfp_gt = (m_a > m_b);
    end else if (de > 0) begin
      // a has larger exp; shift b right by de (saturate to 0 if too far).
      b_al = (de >= 16) ? 16'sd0 : (m_b >>> de[3:0]);
      bfp_gt = (m_a > b_al);
    end else begin
      // b has larger exp; shift a right by (-de).
      sh = -de;
      a_al = (sh >= 16) ? 16'sd0 : (m_a >>> sh[3:0]);
      bfp_gt = (a_al > m_b);
    end
  endfunction


  always_ff @(posedge clk) begin
    if (rst) begin
      state      <= S_IDLE;
      cnt        <= '0;
      chunk      <= '0;
      // Sentinel value: -1.0 * 2^127 ≈ -1.7e38 (BFP equivalent of -infinity).
      // Any real logit, positive or negative, beats this — fixes the
      // step-12 <unk> failure where every logit was negative and the old
      // sentinel (-2^-128, a tiny negative near zero) was never beaten.
      best_m     <= 16'sh8000;
      best_e     <= 8'sd127;
      best_idx   <= '0;
      mv_start   <= 1'b0;
      mv_valid   <= 1'b0;
      mv_last    <= 1'b0;
      mv_eng_rst <= 1'b0;
      rn_start   <= 1'b0;
      rn_valid   <= 1'b0;
      rn_last    <= 1'b0;
      done       <= 1'b0;
      token_out  <= 16'd0;
      ws_load_req <= 1'b0;
      ws_phase    <= WSP_KICK;
      mv_out_m_r      <= '0;
      mv_out_e_r      <= '0;
      lane_cnt        <= 4'd0;
      local_best_m    <= 16'sh8000;
      local_best_e    <= -8'sd128;
      local_best_lane <= 4'd0;
    end else begin
      mv_start   <= 1'b0;
      mv_valid   <= 1'b0;
      mv_last    <= 1'b0;
      mv_eng_rst <= 1'b0;
      rn_start   <= 1'b0;
      rn_valid   <= 1'b0;
      rn_last    <= 1'b0;
      done       <= 1'b0;
      ws_load_req <= 1'b0;

      case (state)
        S_IDLE: if (start) begin
          state    <= S_LATCH;
          cnt      <= '0;
          chunk    <= '0;
          best_m   <= 16'sh8000;
          best_e   <= -8'sd128;
          best_idx <= '0;
          ws_phase <= WSP_KICK;
        end

        // Copy hidden_in bus into BRAM array (1 elem/cycle for BRAM inference)
        S_LATCH: begin
          hin_m[cnt[$clog2(D)-1:0]] <= hidden_in_m[cnt[$clog2(D)-1:0]*BFP_MANT_W +: BFP_MANT_W];
          if (cnt < NT_D)
            hin_e[cnt[$clog2(NT_D+1)-1:0]] <=
              hidden_in_e[cnt[$clog2(NT_D+1)-1:0]*BFP_EXP_W +: BFP_EXP_W];
          if (cnt == D-1) begin
            state    <= S_NORM;
            cnt      <= '0;
            rn_start <= 1'b1;
          end else cnt <= cnt + 1'b1;
        end

        // RMSNorm: stream D inputs (hin + norm_w as gamma)
        S_NORM: begin
          rn_valid <= 1'b1;
          rn_x_m   <= hin_m[cnt[$clog2(D)-1:0]];
          rn_x_e   <= hin_e[cnt[$clog2(D)-1:0] / BFP_TILE];
          rn_g_m   <= rom_NW_m[cnt[$clog2(D)-1:0]];
          rn_g_e   <= rom_NW_e[cnt[$clog2(D)-1:0] / BFP_TILE];
          rn_last  <= (cnt == D-1);
          if (cnt == D-1) begin
            state <= S_NORM_WAIT; cnt <= '0;
          end else cnt <= cnt + 1'b1;
        end

        S_NORM_WAIT: begin
          if (rn_y_valid) begin
            hn_m[cnt[$clog2(D)-1:0]] <= rn_y_m;
            if (cnt[3:0] == 4'd0)
              hn_e[cnt[$clog2(D)-1:0] / BFP_TILE] <= rn_y_e;
            if (cnt == D-1) begin
              cnt        <= '0;
              chunk      <= '0;
              mv_eng_rst <= 1'b1;
              state      <= S_MV_PRIME;
            end else cnt <= cnt + 1'b1;
          end
        end

        // ---- lm_head matvec: CHUNKS_VOCAB chunks of LANES outputs each ----
        S_MV_PRIME: begin
          if (ws_phase != WSP_READY) begin
            unique case (ws_phase)
              WSP_KICK: begin
                ws_load_req <= 1'b1;
                ws_phase    <= WSP_HOLD;
              end
              WSP_HOLD: if (!ws_ready) ws_phase <= WSP_WAIT;
              WSP_WAIT: if ( ws_ready) ws_phase <= WSP_READY;
              default: ;
            endcase
          end else begin
            mv_start <= 1'b1; cnt <= '0; state <= S_MV_DRIVE;
          end
        end
        S_MV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= hn_m[cnt[$clog2(D)-1:0]];
          mv_x_e   <= hn_e[cnt[$clog2(D)-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_MV_DRAIN;
          else cnt <= cnt + 1'b1;
        end

        // S_MV_DRAIN: capture the engine's 16-lane output and seed
        // local_best with lane 0.  Subsequent lanes 1..15 are compared
        // one per cycle in S_MV_LANE_CMP, and S_MV_MERGE folds the
        // chunk's best into the global best.  Pipelining is required
        // for 50 MHz timing — the original one-shot 16-cascade was
        // 147 LUT levels.
        S_MV_DRAIN: if (mv_out_valid) begin
          mv_out_m_r      <= mv_out_m;
          mv_out_e_r      <= mv_out_e;
          local_best_m    <= $signed(mv_out_m[0 +: BFP_MANT_W]);
          local_best_e    <= $signed(mv_out_e[0 +: BFP_EXP_W ]);
          local_best_lane <= 4'd0;
          lane_cnt        <= 4'd1;
          state           <= S_MV_LANE_CMP;
        end

        S_MV_LANE_CMP: begin : lane_cmp
          automatic logic signed [BFP_MANT_W-1:0] m_i;
          automatic logic signed [BFP_EXP_W -1:0] e_i;
          m_i = $signed(mv_out_m_r[lane_cnt*BFP_MANT_W +: BFP_MANT_W]);
          e_i = $signed(mv_out_e_r[lane_cnt*BFP_EXP_W  +: BFP_EXP_W ]);
          if (bfp_gt(m_i, e_i, local_best_m, local_best_e)) begin
            local_best_m    <= m_i;
            local_best_e    <= e_i;
            local_best_lane <= lane_cnt;
          end
          if (lane_cnt == 4'd15) state <= S_MV_MERGE;
          else lane_cnt <= lane_cnt + 4'd1;
        end

        S_MV_MERGE: begin
          if (bfp_gt(local_best_m, local_best_e, best_m, best_e)) begin
            best_m   <= local_best_m;
            best_e   <= local_best_e;
            best_idx <= 16'(chunk * LANES + local_best_lane);
          end
          state <= S_MV_NEXT;
        end

        S_MV_NEXT: begin
          ws_phase <= WSP_KICK;
          if (chunk == CHUNKS_VOCAB - 1) begin
            token_out <= best_idx;
            state     <= S_DONE;
          end else begin
            chunk      <= chunk + 1'b1;
            mv_eng_rst <= 1'b1;
            state      <= S_MV_PRIME;
          end
        end

        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
