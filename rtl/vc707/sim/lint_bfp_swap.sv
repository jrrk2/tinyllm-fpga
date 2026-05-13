// Stub module that exercises the same instantiation pattern as the
// MICROGPT_USE_BFP branch of vc707_microgpt_eth.sv.  Lets Verilator
// lint-check the port connections without pulling in Xilinx primitives.
`include "bfp_format.svh"
module lint_bfp_swap (
  input  wire        clk,
  input  wire        rst,
  input  wire        restart,
  input  wire [3:0]  scale_wr_kind_core,
  input  wire [15:0] scale_wr_addr_core,
  input  wire [15:0] scale_wr_data_core,
  input  wire        scale_wr_pulse_core,
  output wire [9215:0] lay_result,
  output wire          lay_done_core,
  output wire [31:0]   factor_rd_data_core
);
  wire [16*16-1:0] bfp_result_m;
  wire [4*8-1:0]   bfp_result_e;
  smollm_layer_bfp_selftest #(
    .D(64), .H_Q(1), .H_KV(1), .HD(64), .FFN(128), .MAX_CTX(4),
    .RESULT_LANES(16)
  ) i_lay_st (
    .clk      ( clk                         ),
    .rst      ( rst                         ),
    .restart  ( restart                     ),
    .wr_kind  ( {1'b0, scale_wr_kind_core}  ),
    .wr_addr  ( {2'b0, scale_wr_addr_core}  ),
    // NB: the selftest pins layer_idx=0 internally.

    .wr_data  ( scale_wr_data_core          ),
    .wr_en    ( scale_wr_pulse_core         ),
    .result_m ( bfp_result_m                ),
    .result_e ( bfp_result_e                ),
    .done     ( lay_done_core               )
  );
  assign lay_result = {{(9216-16*16-4*8){1'b0}}, bfp_result_e, bfp_result_m};
  assign factor_rd_data_core = '0;
endmodule
