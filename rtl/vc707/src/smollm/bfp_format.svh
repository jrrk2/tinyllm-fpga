// bfp_format.svh — block-floating-point (per-tile FP) data format for the
// SmolLM2 forward pass.
//
//   TILE elements share one int8 signed exponent.
//   Each element has an int16 signed mantissa.
//   The value of element k in tile t is:
//        v[t,k] = m[t,k] / 2^15 * 2^e[t]
//                = m[t,k] * 2^(e[t] - 15)
//
//   Python emulator (host/sim_rtl_fp_hw.py) verifies coherent autoregressive
//   generation across TILE=16/32/64.  RTL uses TILE=16 to match matvec LANES.

`ifndef BFP_FORMAT_SVH
`define BFP_FORMAT_SVH

// Concrete widths
localparam int BFP_TILE       = 16;            // elements per shared-exponent tile
localparam int BFP_MANT_W     = 16;            // signed mantissa width (fits DSP48 B input as 18-bit signed)
localparam int BFP_EXP_W      = 8;             // signed exponent width
localparam int BFP_TILE_BITS  = BFP_TILE * BFP_MANT_W + BFP_EXP_W;   // 264 bits per tile

// Mantissa MAC pipeline widths
localparam int BFP_MULT_W     = 2 * BFP_MANT_W;        // 32-bit mantissa product
localparam int BFP_TILE_ACC_W = BFP_MULT_W + $clog2(BFP_TILE);  // 32 + 4 = 36-bit within-tile sum
localparam int BFP_ACC_W      = 48;                    // DSP48E1 P register / cross-tile accumulator
localparam int BFP_ALIGN_MAX  = BFP_ACC_W - 1;         // max barrel-shift before contribution → 0
localparam int BFP_EXP_SUM_W  = BFP_EXP_W + 1;         // 9-bit signed exponent sum (x_e + w_e)

// Packed tile representation
//   bits [BFP_EXP_W-1 : 0]                    = exponent (signed)
//   bits [BFP_EXP_W + k*BFP_MANT_W +: BFP_MANT_W] = mantissa[k]  (signed)
`define BFP_TILE_PACK(M, E)  {M, E}
`define BFP_GET_EXP(T)       $signed((T)[BFP_EXP_W-1:0])
`define BFP_GET_MANT(T, K)   $signed((T)[BFP_EXP_W + (K)*BFP_MANT_W +: BFP_MANT_W])

// FP24 normalize output: a wider int (e.g., 48-bit acc) at a given exponent
// → 16-bit mantissa + 8-bit exponent (output FP24 value).
//   leading_bit_pos = position of leading 1 in |acc| (priority encoder)
//   shift_amt       = leading_bit_pos - 15   (shift right to land mantissa in [-2^15, 2^15))
//   mantissa_out    = (acc >>> shift_amt) [BFP_MANT_W-1 : 0]   (signed)
//   exponent_out    = acc_exp + shift_amt + 1
// (Implementation goes in fabric — see matvec_bfp_engine.sv for the
// integrated normalizer.)

`endif  // BFP_FORMAT_SVH
