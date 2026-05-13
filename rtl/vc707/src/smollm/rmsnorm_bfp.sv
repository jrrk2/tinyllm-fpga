// rmsnorm_bfp.sv — block-FP RMSNorm.
//
//   y[i] = x[i] * gamma[i] / sqrt( mean_k(x[k]^2) + eps )
//
// All I/O in the block-FP format (TILE elements share an 8-bit exponent,
// 16-bit signed mantissa per element).  Output exponents are computed
// per-tile from the actual mantissa magnitudes — i.e. the module
// re-tile-quantizes the output stream.
//
// Pipeline:
//   Stage A  (D cycles): Σ x[k]^2 — mantissa MAC reusing the matvec MAC
//                        pattern, then cross-tile align and accumulate.
//   Stage B  (constant cycles): scalar normalize, multiply by INV_D, add
//                        eps; produce v_norm (mantissa + exponent).
//   Stage C  (constant cycles): 1/sqrt(v_norm) via 64K-entry LUT keyed on
//                        the leading-normalized mantissa, plus 1-2 NR
//                        iterations.  Exponent: -⌈e_v/2⌉ (odd-exp handled
//                        by 1-bit pre-shift of the mantissa).
//   Stage D  (D cycles): stream output y[i] = x[i] * gamma[i] * inv_rms;
//                        accumulate per-tile max-|mantissa-product| to
//                        derive the output tile exponent on the fly,
//                        then ship out (mantissa, exponent) tiles.
//
// Current state: SKELETON ONLY.  Sum-of-squares stage compiles and is
// directly portable from matvec_bfp_engine; the rsqrt LUT and FP-output
// stage are pending.  Use this file as the interface contract for the
// surrounding wiring while the body is fleshed out.

`include "bfp_format.svh"

`default_nettype none

module rmsnorm_bfp #(
  parameter int D = 576,
  parameter logic [31:0] EPS_Q30 = 32'd10737   // round(1e-5 * 2^30)
)(
  input  wire                          clk,
  input  wire                          rst,

  input  wire                          start,      // pulse one cycle before first element
  input  wire signed [BFP_MANT_W-1:0]  in_x_mant,
  input  wire signed [BFP_EXP_W -1:0]  in_x_exp,   // tile-shared, held constant for TILE cycles
  input  wire signed [BFP_MANT_W-1:0]  in_g_mant,
  input  wire signed [BFP_EXP_W -1:0]  in_g_exp,
  input  wire                          in_valid,
  input  wire                          last_elem,

  output logic signed [BFP_MANT_W-1:0] out_y_mant,
  output logic signed [BFP_EXP_W -1:0] out_y_exp,
  output logic                         out_valid,
  output logic                         done
);

  // -----------------------------------------------------------------------
  // Stage A: per-tile sum-of-squares (one DSP48 doing m_x * m_x)
  // -----------------------------------------------------------------------
  localparam int CW = $clog2(BFP_TILE);

  // Counter & tile-boundary pulses (mirror matvec_bfp_engine)
  logic [11:0]      cnt;
  logic [CW-1:0]    tile_idx;
  logic             tile_done, tile_done_r, last_elem_r;
  always_ff @(posedge clk) begin
    if (rst) cnt <= '0;
    else if (start) cnt <= '0;
    else if (in_valid) cnt <= cnt + 1'b1;
  end
  always_comb tile_idx = cnt[CW-1:0];
  always_comb tile_done = in_valid && (tile_idx == BFP_TILE - 1);
  always_ff @(posedge clk) begin
    if (rst) begin tile_done_r <= 1'b0; last_elem_r <= 1'b0; end
    else begin
      tile_done_r <= tile_done;
      last_elem_r <= in_valid && last_elem;
    end
  end

  // x*x squared into 32-bit unsigned, accumulated into 48-bit P
  logic signed [BFP_MULT_W-1:0]   sq_w;
  always_comb sq_w = in_x_mant * in_x_mant;   // always non-negative

  logic [BFP_ACC_W-1:0]           p_reg;
  always_ff @(posedge clk) begin
    if (rst) p_reg <= '0;
    else if (in_valid) begin
      if (tile_idx == '0)
        p_reg <= {{(BFP_ACC_W-BFP_MULT_W){1'b0}}, sq_w};
      else
        p_reg <= p_reg + {{(BFP_ACC_W-BFP_MULT_W){1'b0}}, sq_w};
    end
  end

  // Tile sum + 2*e_x exponent captured on tile boundary
  logic signed [BFP_EXP_W-1:0]      x_exp_r;
  always_ff @(posedge clk) begin
    if (rst) x_exp_r <= '0;
    else if (tile_done) x_exp_r <= in_x_exp;
  end

  logic [BFP_ACC_W-1:0]              sq_tile_r;
  logic signed [BFP_EXP_SUM_W:0]    sq_tile_exp_r;   // 10-bit: 2*8-bit + sign + 1
  logic                              sq_tile_valid_r;
  logic                              sq_last_r;
  always_ff @(posedge clk) begin
    if (rst) begin
      sq_tile_r <= '0; sq_tile_exp_r <= '0; sq_tile_valid_r <= 1'b0; sq_last_r <= 1'b0;
    end else begin
      sq_tile_valid_r <= tile_done_r;
      sq_last_r       <= last_elem_r;
      if (tile_done_r) begin
        sq_tile_r     <= p_reg;
        // exponent for x*x is 2*(e_x - 15) = 2*e_x - 30
        sq_tile_exp_r <= ($signed({x_exp_r[BFP_EXP_W-1], x_exp_r}) <<< 1);
        // NOTE: subtraction of 30 happens once at the cross-tile sum below
      end
    end
  end

  // Cross-tile accumulator — same pattern as matvec_bfp_engine.  We track
  // (sq_acc, sq_acc_exp) and align each new tile sum to the max exponent.
  // TODO(claude): replicate the align-and-add block from matvec_bfp_engine,
  // then normalize the final sq_acc to obtain (m_v, e_v).  Both are needed
  // before Stage B/C can be filled in.

  // -----------------------------------------------------------------------
  // STUB: stage B, C, D
  //
  // Stage B (1/D scalar multiply, eps add, normalize):
  //   v_q30  = (sq_acc * INV_D_Q32) >> 32 + EPS_Q30
  //   v_norm = normalize(v_q30 + sq_acc_exp - 30)
  //
  // Stage C (1/sqrt via LUT + NR):
  //   inv_rms_mant, inv_rms_exp = rsqrt_lut(v_norm_mant_msb)
  //   refine via 1-2 NR iterations in mantissa space
  //
  // Stage D (per-element output, D cycles):
  //   prod = m_x * m_g  (32-bit)
  //   prod = prod * m_inv_rms  (48-bit)
  //   shift to int16 mantissa with leading bit at position 14
  //   out_exp = e_x + e_g + e_inv_rms - 30 + shift
  //   per-tile output: take max(|m|) within tile → choose tile exponent,
  //   re-align mantissas, emit (m, e) pair
  //
  // Until those are wired in, the module simply holds outputs at 0.
  // -----------------------------------------------------------------------

  always_ff @(posedge clk) begin
    if (rst) begin
      out_y_mant <= '0; out_y_exp <= '0; out_valid <= 1'b0; done <= 1'b0;
    end
  end

endmodule

`default_nettype wire
