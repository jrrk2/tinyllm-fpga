// tb_autoregress_bfp.sv — Verilator wrapper exercising the on-chip
// autoregressive BFP token generator.  Trivially small: a single `go`
// pulse kicks off the loop, then we wait for `done` and snapshot the
// 19-token result bus.  The C++ harness reads result_tokens and decodes.

`include "../generated/lbfp_full_cfg.svh"
`include "bfp_format.svh"

`default_nettype none

module tb_autoregress_bfp (
  input  wire                                                 clk,
  input  wire                                                 rst,
  input  wire                                                 go,
  output wire                                                 done,
  output wire [(`LBFP_FULL_NPROMPT + `LBFP_FULL_NGEN)*16-1:0] result_tokens
);

  autoregress_bfp_top #(
    .D       (`LBFP_FULL_D),
    .H_Q     (`LBFP_FULL_HQ),
    .H_KV    (`LBFP_FULL_HKV),
    .HD      (`LBFP_FULL_HD),
    .FFN     (`LBFP_FULL_FFN),
    .NL      (`LBFP_FULL_NL),
    .MAX_CTX (`LBFP_FULL_MAX_CTX),
    .VOCAB   (`LBFP_FULL_VOCAB),
    .N_PROMPT(`LBFP_FULL_NPROMPT),
    .N_GEN   (`LBFP_FULL_NGEN),
    .PREFIX  ("../generated/lbfp_full_")
  ) dut (
    .clk(clk), .rst(rst), .start(go),
    .done(done), .result_tokens(result_tokens)
  );

endmodule

`default_nettype wire
