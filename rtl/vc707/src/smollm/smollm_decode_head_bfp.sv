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
  parameter     PREFIX = "lbfp_full_"
)(
  input  wire                                       clk,
  input  wire                                       rst,
  input  wire                                       start,
  input  wire signed [D*BFP_MANT_W-1:0]             hidden_in_m,
  input  wire signed [(D/BFP_TILE)*BFP_EXP_W-1:0]   hidden_in_e,
  output logic [15:0]                               token_out,
  output logic                                      done
);

  localparam int NT_D          = D / BFP_TILE;
  localparam int LANES         = 16;
  localparam int CHUNKS_VOCAB  = VOCAB / LANES;

  // ---------------------------------------------------------------------------
  // Final norm gamma (norm_w) — narrow ROM.
  // ---------------------------------------------------------------------------
  (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] rom_NW_m [0:D-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] rom_NW_e [0:NT_D-1];

  // ---------------------------------------------------------------------------
  // EMBED (used as lm_head via tied embeddings).  Wide-packed BRAM:
  //   rom_EMBED_m: CHUNKS_VOCAB * D entries of 256 bits each.
  //   rom_EMBED_e: CHUNKS_VOCAB * NT_D entries of 128 bits each.
  // ---------------------------------------------------------------------------
  localparam int LANE_M_W = LANES * BFP_MANT_W;
  localparam int LANE_E_W = LANES * BFP_EXP_W;
  (* ram_style = "block" *) logic [LANE_M_W-1:0] rom_EMBED_m [0:CHUNKS_VOCAB*D-1];
  (* ram_style = "block" *) logic [LANE_E_W-1:0] rom_EMBED_e [0:CHUNKS_VOCAB*NT_D-1];

`ifdef MICROGPT_WEIGHT_DIR
  initial begin
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "NORM_W_m.hex"}, rom_NW_m);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "NORM_W_e.hex"}, rom_NW_e);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "EMBED_m.hex"},  rom_EMBED_m);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "EMBED_e.hex"},  rom_EMBED_e);
  end
`else
  initial begin
    $readmemh({PREFIX, "NORM_W_m.hex"}, rom_NW_m);
    $readmemh({PREFIX, "NORM_W_e.hex"}, rom_NW_e);
    $readmemh({PREFIX, "EMBED_m.hex"},  rom_EMBED_m);
    $readmemh({PREFIX, "EMBED_e.hex"},  rom_EMBED_e);
  end
`endif

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
    S_MV_PRIME, S_MV_DRIVE, S_MV_DRAIN, S_MV_NEXT,
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
  logic signed [LANES*BFP_MANT_W-1:0]         mv_w_m;
  logic signed [LANES*BFP_EXP_W -1:0]         mv_w_e;
  wire  signed [LANES*BFP_MANT_W-1:0]         mv_out_m;
  wire  signed [LANES*BFP_EXP_W -1:0]         mv_out_e;
  wire                                        mv_out_valid;

  matvec_bfp_engine #(.LANES(LANES)) i_mv (
    .clk(clk), .rst(rst | mv_eng_rst),
    .start_matvec(mv_start),
    .in_x_mant(mv_x_m), .in_x_exp(mv_x_e),
    .in_valid(mv_valid), .last_elem(mv_last),
    .w_mant(mv_w_m), .w_exp(mv_w_e),
    .out_mant(mv_out_m), .out_exp(mv_out_e), .out_valid(mv_out_valid)
  );

  // ---------------------------------------------------------------------------
  // Counters / argmax tracking
  // ---------------------------------------------------------------------------
  logic [11:0]                  cnt;
  logic [$clog2(CHUNKS_VOCAB)-1:0] chunk;
  logic signed [BFP_MANT_W-1:0] best_m;
  logic signed [BFP_EXP_W -1:0] best_e;
  logic [15:0]                  best_idx;

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

  integer ii;

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
    end else begin
      mv_start   <= 1'b0;
      mv_valid   <= 1'b0;
      mv_last    <= 1'b0;
      mv_eng_rst <= 1'b0;
      rn_start   <= 1'b0;
      rn_valid   <= 1'b0;
      rn_last    <= 1'b0;
      done       <= 1'b0;

      case (state)
        S_IDLE: if (start) begin
          state    <= S_LATCH;
          cnt      <= '0;
          chunk    <= '0;
          best_m   <= 16'sh8000;
          best_e   <= -8'sd128;
          best_idx <= '0;
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
          mv_start <= 1'b1; cnt <= '0; state <= S_MV_DRIVE;
        end
        S_MV_DRIVE: begin
          mv_valid <= 1'b1;
          mv_x_m   <= hn_m[cnt[$clog2(D)-1:0]];
          mv_x_e   <= hn_e[cnt[$clog2(D)-1:0] / BFP_TILE];
          mv_w_m   <= rom_EMBED_m[chunk * D    + cnt[$clog2(D)-1:0]];
          mv_w_e   <= rom_EMBED_e[chunk * NT_D + cnt[$clog2(D)-1:0] / BFP_TILE];
          mv_last  <= (cnt == D-1);
          if (cnt == D-1) state <= S_MV_DRAIN;
          else cnt <= cnt + 1'b1;
        end

        S_MV_DRAIN: if (mv_out_valid) begin : drain_blk
          // Two-pass argmax inside the chunk:
          //   1) Sequentially track local_best across all 16 lanes (blocking,
          //      so each iteration sees the running max).
          //   2) Compare that single local_best against the global running
          //      best_m / best_e via one non-blocking update.
          // The previous one-pass loop wrote best_* via non-blocking inside
          // every iteration that beat the OLD best — last-NBA-wins semantics
          // meant the highest-index passing lane won, not the actual max.
          automatic logic signed [BFP_MANT_W-1:0] local_m;
          automatic logic signed [BFP_EXP_W -1:0] local_e;
          automatic logic [3:0]                   local_lane;
          local_m    = $signed(mv_out_m[0 +: BFP_MANT_W]);
          local_e    = $signed(mv_out_e[0 +: BFP_EXP_W ]);
          local_lane = 4'd0;
          for (ii = 1; ii < LANES; ii++) begin
            automatic logic signed [BFP_MANT_W-1:0] m_i;
            automatic logic signed [BFP_EXP_W -1:0] e_i;
            m_i = $signed(mv_out_m[ii*BFP_MANT_W +: BFP_MANT_W]);
            e_i = $signed(mv_out_e[ii*BFP_EXP_W  +: BFP_EXP_W ]);
            if (bfp_gt(m_i, e_i, local_m, local_e)) begin
              local_m    = m_i;
              local_e    = e_i;
              local_lane = ii[3:0];
            end
          end
          if (bfp_gt(local_m, local_e, best_m, best_e)) begin
            best_m   <= local_m;
            best_e   <= local_e;
            best_idx <= 16'(chunk * LANES + local_lane);
          end
          state <= S_MV_NEXT;
        end

        S_MV_NEXT: begin
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
