// swiglu.sv — SiLU(gate) * up element-wise activation for the MLP.
//
//   y_real = silu(gate_real) * up_real
//
// where gate, up and y carry per-tensor calibrated scales (lsc[gate],
// lsc[up], lsc[mlp]) — they are NOT all Q1.15 at unit scale.  The
// previous implementation silently assumed they were, which produced
// near-zero outputs and broke the entire MLP path on the FPGA.
//
// The LUT now stores SiLU at scale SILU_LUT_SCALE (= 32 — covers
// SmolLM2 calibration range up to ±21).  Per-layer factors rescale
// the matvec output (lsc[gate]) into LUT scale on the way in, and
// LUT scale into lsc[mlp] on the way out.
//
// All three factors come from gen_smollm_blockfp.py:
//   gate_in_factor  = round(lsc[gate] / SILU_LUT_SCALE * 32768)  // Q1.15 ≤ 1
//   up_in_factor    = round(lsc[up]   / SILU_LUT_SCALE * 32768)  // Q1.15 ≤ 1
//   mlp_out_factor  = round(SILU_LUT_SCALE^2 / lsc[mlp] * 256)   // Q16.8 unsigned
//
// LUT file: generated/silu_lut.hex (gen_swiglu_lut.py with SILU_LUT_SCALE=32).
// Pipeline latency = 2 cycles (extra register stage for the rescale path).

`default_nettype none

module swiglu (
  input  wire                clk,
  input  wire                rst,

  input  wire signed [15:0]  in_gate,
  input  wire signed [15:0]  in_up,
  input  wire                in_valid,

  // Per-layer scale factors — driven from per-tensor calibration brom.
  input  wire        [15:0]  gate_in_factor,    // Q1.15 ≤ 1
  input  wire        [15:0]  up_in_factor,      // Q1.15 ≤ 1
  input  wire        [23:0]  mlp_out_factor,    // Q16.8 unsigned

  output logic signed [15:0] out_y,
  output logic               out_valid
);

  // ------------------------------------------------------------------
  // SiLU LUT — 65536 entries at SILU_LUT_SCALE (= 32).
  // ------------------------------------------------------------------
  logic signed [15:0] silu_lut [0:65535];

`ifdef MICROGPT_WEIGHT_DIR
  initial $readmemh({`MICROGPT_WEIGHT_DIR, "/silu_lut.hex"}, silu_lut);
`else
  initial $readmemh("../generated/silu_lut.hex", silu_lut);
`endif

  // ------------------------------------------------------------------
  // Stage 1 (comb): rescale gate and up from per-tensor scales into
  // LUT scale.  Both factors are ≤ 1 (lsc[gate], lsc[up] ≤ SILU_LUT_SCALE),
  // so the >>>15 result fits in 16 bits without saturation in the common
  // case.  Saturate defensively for any outlier layer.
  // ------------------------------------------------------------------
  wire signed [31:0] gate_scaled_w = $signed(in_gate) * $signed({1'b0, gate_in_factor});
  wire signed [16:0] gate_shifted  = gate_scaled_w[31:15];
  wire signed [15:0] gate_lut_idx  = (gate_shifted >  17'sd32767) ? 16'sh7FFF :
                                     (gate_shifted < -17'sd32768) ? 16'sh8000 :
                                                                    gate_shifted[15:0];

  wire signed [31:0] up_scaled_w   = $signed(in_up) * $signed({1'b0, up_in_factor});
  wire signed [16:0] up_shifted    = up_scaled_w[31:15];
  wire signed [15:0] up_at_lut     = (up_shifted >  17'sd32767) ? 16'sh7FFF :
                                     (up_shifted < -17'sd32768) ? 16'sh8000 :
                                                                  up_shifted[15:0];

  // Stage 1 register — LUT read is comb, plus pipeline for timing.
  logic signed [15:0] silu_lut_r;
  logic signed [15:0] up_at_lut_r;
  logic               valid_s1;
  always_ff @(posedge clk) begin
    if (rst) begin
      silu_lut_r  <= '0;
      up_at_lut_r <= '0;
      valid_s1    <= 1'b0;
    end else begin
      silu_lut_r  <= silu_lut[gate_lut_idx[15:0]];
      up_at_lut_r <= up_at_lut;
      valid_s1    <= in_valid;
    end
  end

  // ------------------------------------------------------------------
  // Stage 2 (comb): silu * up at LUT scale, then rescale to lsc[mlp].
  // ------------------------------------------------------------------
  wire signed [31:0] prod_w        = silu_lut_r * up_at_lut_r;        // Q-fixed at LUT_SCALE^2
  wire signed [15:0] prod_at_lut2  = prod_w >>> 15;                   // signed Q1.15 of LUT_SCALE^2
  // prod_at_lut2 is signed; mlp_out_factor is UNSIGNED Q16.8.  Keep them
  // signed/unsigned correctly so negative prod values don't masquerade as
  // huge positives.  Pre-multiplier widths: signed 16 × signed 25 → 41,
  // saved into signed 40 (top bit always equals next-to-top → safe to drop).
  wire signed [39:0] result_w      = $signed(prod_at_lut2) * $signed({1'b0, mlp_out_factor});
  wire signed [31:0] result_shifted = result_w >>> 8;                 // Q16.8 factor → Q1.15 result

  always_ff @(posedge clk) begin
    if (rst) begin
      out_y     <= '0;
      out_valid <= 1'b0;
    end else begin
      out_valid <= valid_s1;
      if (valid_s1) begin
        if      (result_shifted >  32'sd32767)  out_y <= 16'sh7FFF;
        else if (result_shifted < -32'sd32768)  out_y <= 16'sh8000;
        else                                    out_y <= result_shifted[15:0];
      end
    end
  end

endmodule

`default_nettype wire
