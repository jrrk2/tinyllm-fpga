// sampler.sv — argmax / temperature sampler for the lm_head logits.
//
// SKELETON. See PLAN.md "Sampler".
//
// Inputs: 49,152 logits streamed one per cycle (or 16 per cycle from
//         the matvec engine).
// Output: chosen token ID (0..49151).
//
// Modes:
//   sample_mode = 0  →  argmax (greedy).
//   sample_mode = 1  →  temperature sampling. logit'_i = logit_i / T,
//                       softmax → cumulative dist → use rng to pick.
//
// argmax mode is trivial (running max + index).
// Temperature mode needs a softmax pass + cumulative-prob walk; can
// reuse the existing microgpt_categorical_sampler.sv pattern.

`default_nettype none

module sampler #(
  parameter int VOCAB = 49152
)(
  input  wire        clk,
  input  wire        rst,

  input  wire        start,
  input  wire        sample_mode,
  input  wire [15:0] temperature_q8_8,
  input  wire [31:0] rng_state_in,
  output logic       busy,
  output logic       done,
  output logic [15:0] chosen_token,
  output logic [31:0] rng_state_out,

  // Logit stream from matvec engine (one FP16 logit per cycle, or 16/cycle
  // if hooked directly to the matvec out_value bus).
  input  wire [15:0] logit_data,
  input  wire        logit_valid,
  input  wire        logit_last
);
  // TODO: argmax + temperature softmax + categorical pick.

  always_ff @(posedge clk) begin
    if (rst) begin busy<=0; done<=0; chosen_token<=0; rng_state_out<=0; end
  end
endmodule

`default_nettype wire
