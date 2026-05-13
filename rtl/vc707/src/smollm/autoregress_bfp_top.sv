// autoregress_bfp_top.sv — self-contained block-FP autoregressive token
// generator.  Single `start` pulse runs the prompt + autoregress loop on
// chip; emits done when finished plus a packed bus of all generated
// tokens.  Suitable as the FPGA top-level instantiation for the SmolLM2
// BFP path.
//
// Architecture:
//   prompt ROM ($readmemh-loaded with N_PROMPT 16-bit token ids)
//      |
//      v
//   FSM (drives token_in / pos / kv_pos / model_start, captures token_out)
//      |
//      v
//   single-token model stack:
//      embed_lookup_bfp -> smollm_multilayer_tm_bfp -> smollm_decode_head_bfp
//      |
//      v
//   result RAM (N_STEPS 16-bit tokens, packed to result_tokens output)
//
// One full forward = N_STEPS = N_PROMPT + N_GEN token-step model calls.
// kv_pos advances each step (0..N_STEPS-1).  In prompt phase the FSM
// drives prompt[step]; in autoregress phase it drives the previous
// token_out.

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
  parameter int N_PROMPT = 4,
  parameter int N_GEN    = 15,
  parameter int N_STEPS  = N_PROMPT + N_GEN,
  parameter     PREFIX   = "lbfp_full_"
)(
  input  wire                          clk,
  input  wire                          rst,
  input  wire                          start,
  output logic                         done,
  // Packed bus of N_STEPS 16-bit token ids (step 0 in LSBs).
  output logic [N_STEPS*16-1:0]        result_tokens
);

  // ---------------------------------------------------------------------------
  // Prompt ROM — 4 token ids baked into a hex file by the host generator.
  // One 16-bit value per line (16'd0..49152, $readmemh format).
  // ---------------------------------------------------------------------------
  (* ram_style = "block" *) logic [15:0] prompt_rom [0:N_PROMPT-1];
`ifdef MICROGPT_WEIGHT_DIR
  initial $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "PROMPT.hex"}, prompt_rom);
`else
  initial $readmemh({PREFIX, "PROMPT.hex"}, prompt_rom);
`endif

  // ---------------------------------------------------------------------------
  // Inner single-token model: embed_lookup → multilayer × NL → decode_head
  // ---------------------------------------------------------------------------
  localparam int NT_D = D / BFP_TILE;
  logic                                mdl_start;
  logic [15:0]                         mdl_token_in;
  logic [10:0]                         mdl_pos;
  logic [6:0]                          mdl_kv_pos;
  wire  signed [D*BFP_MANT_W-1:0]      emb_m;
  wire  signed [NT_D*BFP_EXP_W-1:0]    emb_e;
  wire                                 emb_done;

  embed_lookup_bfp #(.D(D), .VOCAB(VOCAB), .PREFIX(PREFIX)) i_emb (
    .clk(clk), .rst(rst), .start(mdl_start),
    .token_id(mdl_token_in),
    .hidden_m(emb_m), .hidden_e(emb_e), .done(emb_done)
  );

  // Three-stage inner FSM: EMB → LAY → DEC.
  logic signed [D*BFP_MANT_W-1:0]      lay_in_m;
  logic signed [NT_D*BFP_EXP_W-1:0]    lay_in_e;
  logic                                lay_start;
  wire  signed [D*BFP_MANT_W-1:0]      lay_out_m;
  wire  signed [NT_D*BFP_EXP_W-1:0]    lay_out_e;
  wire                                 lay_done;

  smollm_multilayer_tm_bfp #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD),
    .FFN(FFN), .MAX_CTX(MAX_CTX), .NL(NL), .PREFIX(PREFIX)
  ) i_lay (
    .clk(clk), .rst(rst), .start(lay_start),
    .pos(mdl_pos), .kv_pos(mdl_kv_pos),
    .hidden_in_m(lay_in_m), .hidden_in_e(lay_in_e),
    .hidden_out_m(lay_out_m), .hidden_out_e(lay_out_e),
    .done(lay_done)
  );

  logic                                dec_start;
  logic signed [D*BFP_MANT_W-1:0]      dec_in_m;
  logic signed [NT_D*BFP_EXP_W-1:0]    dec_in_e;
  wire  [15:0]                         dec_token;
  wire                                 dec_done;
  smollm_decode_head_bfp #(.D(D), .VOCAB(VOCAB), .PREFIX(PREFIX)) i_dec (
    .clk(clk), .rst(rst), .start(dec_start),
    .hidden_in_m(dec_in_m), .hidden_in_e(dec_in_e),
    .token_out(dec_token), .done(dec_done)
  );

  typedef enum logic [3:0] {
    S_IDLE, S_DRIVE,                // outer loop: pick prompt or last gen
    S_EMB, S_EMB_WAIT,
    S_LAY, S_LAY_WAIT,
    S_DEC, S_DEC_WAIT,
    S_CAPTURE, S_NEXT,
    S_ALL_DONE
  } st_t;
  st_t state;
  logic [$clog2(N_STEPS+1)-1:0]        step;
  logic [15:0]                         last_token;
  // Per-step result storage.
  logic [15:0] result_buf [0:N_STEPS-1];

  // ---------------------------------------------------------------------------
  // Main FSM
  // ---------------------------------------------------------------------------
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
          // Choose token_in: prompt[step] for prompt phase, last_token for autoregress
          if (step < N_PROMPT) mdl_token_in <= prompt_rom[step[$clog2(N_PROMPT+1)-1:0]];
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
          last_token            <= dec_token;
          result_buf[step]      <= dec_token;
          state                 <= S_NEXT;
        end
        S_NEXT: begin
          if (step == N_STEPS - 1) state <= S_ALL_DONE;
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

  // Pack result_buf into the wide output bus
  always_comb begin
    for (int i = 0; i < N_STEPS; i++)
      result_tokens[i*16 +: 16] = result_buf[i];
  end

endmodule

`default_nettype wire
