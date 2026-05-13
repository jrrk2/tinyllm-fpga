// tb_full_bfp.sv — Verilator test top.  Token-step interface:
//   in  : token_in (16-bit), pos (11-bit), kv_pos (5-bit), start pulse
//   out : token_out (argmax over VOCAB logits), done pulse
//
//   sequence inside one start..done window:
//     embed_lookup_bfp  → BFP hidden vector for token_in
//     smollm_multilayer_tm_bfp  → 30-layer forward pass
//     smollm_decode_head_bfp    → final RMSNorm + lm_head + argmax
//
//   C++ harness handles the token loop: looks up prompt[k] for prefill
//   steps then feeds generated[k-1] for autoregress, compares each
//   token_out to the Python BFP golden (lbfp_full_GOLDEN_TOKENS.txt).

`include "lbfp_full_cfg.svh"
`include "bfp_format.svh"

`default_nettype none

module tb_full_bfp (
  input  wire        clk,
  input  wire        rst,
  input  wire        start,
  input  wire [15:0] token_in,
  input  wire [10:0] pos,
  input  wire [6:0]  kv_pos,
  output wire [15:0] token_out,
  output wire        done
);

  localparam int D       = `LBFP_FULL_D;
  localparam int H_Q     = `LBFP_FULL_HQ;
  localparam int H_KV    = `LBFP_FULL_HKV;
  localparam int HD      = `LBFP_FULL_HD;
  localparam int FFN     = `LBFP_FULL_FFN;
  localparam int NL      = `LBFP_FULL_NL;
  localparam int MAX_CTX = `LBFP_FULL_MAX_CTX;
  localparam int VOCAB   = `LBFP_FULL_VOCAB;
  localparam int NT_D    = D / BFP_TILE;

  // Three-stage FSM:  EMBED → LAYERS → DECODE.
  typedef enum logic [2:0] {
    S_IDLE, S_EMB, S_EMB_WAIT, S_LAY, S_LAY_WAIT, S_DEC, S_DEC_WAIT, S_DONE
  } st_t;
  st_t state;

  // -- embed lookup --
  logic                                 emb_start;
  wire  signed [D*BFP_MANT_W-1:0]       emb_m;
  wire  signed [NT_D*BFP_EXP_W-1:0]     emb_e;
  wire                                  emb_done;
  embed_lookup_bfp #(.D(D), .VOCAB(VOCAB), .PREFIX("../generated/lbfp_full_")) i_emb (
    .clk(clk), .rst(rst), .start(emb_start),
    .token_id(token_in),
    .hidden_m(emb_m), .hidden_e(emb_e), .done(emb_done)
  );

  // Latched embed output (so it stays stable for the multilayer to consume)
  logic signed [D*BFP_MANT_W-1:0]       lay_in_m;
  logic signed [NT_D*BFP_EXP_W-1:0]     lay_in_e;

  // -- multilayer --
  logic                                 lay_start;
  wire  signed [D*BFP_MANT_W-1:0]       lay_out_m;
  wire  signed [NT_D*BFP_EXP_W-1:0]     lay_out_e;
  wire                                  lay_done;
  smollm_multilayer_tm_bfp #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD),
    .FFN(FFN), .MAX_CTX(MAX_CTX), .NL(NL), .PREFIX("../generated/lbfp_full_")
  ) i_lay (
    .clk(clk), .rst(rst), .start(lay_start),
    .pos(pos), .kv_pos(kv_pos),
    .hidden_in_m(lay_in_m), .hidden_in_e(lay_in_e),
    .hidden_out_m(lay_out_m), .hidden_out_e(lay_out_e),
    .done(lay_done)
  );

  // -- decode head --
  logic                                 dec_start;
  logic signed [D*BFP_MANT_W-1:0]       dec_in_m;
  logic signed [NT_D*BFP_EXP_W-1:0]     dec_in_e;
  wire  [15:0]                          dec_token;
  wire                                  dec_done;
  smollm_decode_head_bfp #(.D(D), .VOCAB(VOCAB), .PREFIX("../generated/lbfp_full_")) i_dec (
    .clk(clk), .rst(rst), .start(dec_start),
    .hidden_in_m(dec_in_m), .hidden_in_e(dec_in_e),
    .token_out(dec_token), .done(dec_done)
  );

  // FSM driving the three stages
  logic [15:0] tok_r;
  logic        done_r;
  always_ff @(posedge clk) begin
    if (rst) begin
      state      <= S_IDLE;
      emb_start  <= 1'b0;
      lay_start  <= 1'b0;
      dec_start  <= 1'b0;
      lay_in_m   <= '0; lay_in_e <= '0;
      dec_in_m   <= '0; dec_in_e <= '0;
      tok_r      <= '0;
      done_r     <= 1'b0;
    end else begin
      emb_start <= 1'b0;
      lay_start <= 1'b0;
      dec_start <= 1'b0;
      done_r    <= 1'b0;
      case (state)
        S_IDLE: if (start) begin
          emb_start <= 1'b1;
          state     <= S_EMB;
        end
        S_EMB: state <= S_EMB_WAIT;
        S_EMB_WAIT: if (emb_done) begin
          lay_in_m  <= emb_m;
          lay_in_e  <= emb_e;
          lay_start <= 1'b1;
          state     <= S_LAY;
        end
        S_LAY: state <= S_LAY_WAIT;
        S_LAY_WAIT: if (lay_done) begin
          dec_in_m  <= lay_out_m;
          dec_in_e  <= lay_out_e;
          dec_start <= 1'b1;
          state     <= S_DEC;
        end
        S_DEC: state <= S_DEC_WAIT;
        S_DEC_WAIT: if (dec_done) begin
          tok_r <= dec_token;
          state <= S_DONE;
        end
        S_DONE: begin
          done_r <= 1'b1;
          state  <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  assign token_out = tok_r;
  assign done      = done_r;

endmodule

`default_nettype wire
