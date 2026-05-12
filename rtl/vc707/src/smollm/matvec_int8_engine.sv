// matvec_int8_engine.sv — N-lane INT8 × Q1.15 matrix-vector engine.
//
//  Per lane i of N:
//      acc[i] += sext(in_value_q15) * sext(w_int8[i])      while in_valid
//      out[i]  = saturate( (acc[i] * scale_q15[i]) >> 15 ) when scale_valid
//
//  Quantization scheme (per-channel symmetric):
//     w_real    = w_int8 * scale_real     (scale_real is the per-row FP scale)
//     scale_q15 = round(scale_real * 2^15)   (host-side; assumes |scale|<1)
//     in_q15    = round(in_real    * 2^15)
//     out_q15   = round(out_real   * 2^15)
//
//  After in_dim cycles of in_valid the accumulator holds:
//     acc[i] = sum_k (in_q15[k] * w_int8[i,k])
//  which equals out_real[i] * 2^15 / scale_real[i].
//  Scaling by scale_q15[i] (which is scale_real[i] * 2^15) and right-
//  shifting by 15 recovers out_q15[i]:
//     (acc * scale_q15) >> 15
//   = (out_real * 2^15 / scale_real) * (scale_real * 2^15) / 2^15
//   = out_real * 2^15
//   = out_q15
//
//  Resource: per lane, one DSP48E1 for the MAC (16×8 + 40-bit accumulator
//  fits easily) and one DSP for the final scale (40 × 16 → 56-bit, but we
//  only keep 40 bits after the shift). 16 lanes ≈ 32 DSPs.

`default_nettype none

module matvec_int8_engine #(
  parameter int LANES = 16,
  parameter int ACC_W = 40                  // 24-bit product + ~16-bit headroom
)(
  input  wire                       clk,
  input  wire                       rst,

  // Streaming input vector (one Q1.15 element per cycle, in_dim total)
  input  wire signed [15:0]         in_value,
  input  wire                       in_valid,
  input  wire                       in_last,

  // Streaming weight tile (LANES INT8 weights per cycle, lock-step)
  input  wire signed [LANES*8-1:0]  w_int8,

  // Final scaling: assert scale_valid one cycle after in_last; output
  // appears on out_value/out_valid the cycle after that.
  input  wire        [LANES*16-1:0] scale_q15,
  input  wire                       scale_valid,

  // Result vector (LANES Q1.15 elements)
  output logic signed [LANES*16-1:0] out_value,
  output logic                       out_valid,

  // Async clear (e.g. between separate matvec ops)
  input  wire                        acc_clear
);

  genvar gi;
  generate for (gi = 0; gi < LANES; gi++) begin : g_lane
    logic signed [7:0]       w;
    logic signed [23:0]      mul_c;
    logic signed [ACC_W-1:0] acc_r;

    always_comb w     = w_int8[gi*8 +: 8];
    always_comb mul_c = in_value * w;          // signed 16 × signed 8 = 24

    // Single-cycle MAC.  Maps to DSP48E1 P=P+A*B mode at 56 MHz easily.
    // Doing mul and add in one clock avoids the "drop the last sample"
    // hazard a separate mul_r register would have when in_valid drops.
    always_ff @(posedge clk) begin
      if (rst || acc_clear)
        acc_r <= '0;
      else if (in_valid)
        acc_r <= acc_r + {{(ACC_W-24){mul_c[23]}}, mul_c};
    end

    // Final scaling: multiply 40-bit acc by 16-bit Q1.15 scale, then
    // arithmetic right-shift by 15 to recover Q1.15 output.  Saturate
    // to ±32767 on overflow so blow-ups become clamps instead of wraps.
    logic signed [ACC_W+15:0] scaled;
    logic signed [15:0]       scale;

    always_comb begin
      scale  = scale_q15[gi*16 +: 16];
      scaled = acc_r * scale;
    end

    logic signed [ACC_W:0] shifted;       // (acc*scale) >> 15
    always_comb shifted = scaled >>> 15;

    always_ff @(posedge clk) begin
      if (rst) begin
        out_value[gi*16 +: 16] <= '0;
      end else if (scale_valid) begin
        if      (shifted >  $signed({1'b0, 15'h7FFF}))
          out_value[gi*16 +: 16] <= 16'sh7FFF;
        else if (shifted < -$signed({1'b0, 16'h8000}))
          out_value[gi*16 +: 16] <= 16'sh8000;
        else
          out_value[gi*16 +: 16] <= shifted[15:0];
      end
    end
  end endgenerate

  // out_valid pulses one cycle after scale_valid (registered output).
  always_ff @(posedge clk) begin
    if (rst)              out_valid <= 1'b0;
    else                  out_valid <= scale_valid;
  end

endmodule

`default_nettype wire
