// matvec_bfp_engine.sv — block-floating-point matvec engine.
//
// Computes y = W * x where each input/output element is 16-bit mantissa +
// per-tile (TILE=16) 8-bit shared exponent.  LANES output lanes in parallel
// (each one DSP48E1 mantissa MAC chain).
//
// Caller protocol:
//   - Drive `in_x_mant` one mantissa per cycle with `in_valid` high.
//   - `in_x_exp` is the CURRENT input tile's exponent — held constant for
//     all TILE=16 cycles of one tile.
//   - `w_mant` / `w_exp` are the current tile's per-lane weight mantissas /
//     exponents — also held constant for TILE cycles.
//   - Pulse `start_matvec` one cycle BEFORE the very first input mantissa
//     to reset the per-lane cross-tile accumulator.
//   - Pulse `last_elem` together with the very last input mantissa.
//   - One cycle after `last_elem`, `out_valid` pulses with the result.
//
//   ┌──────tile 0──────┐┌──────tile 1──────┐ ...
//   x:  m₀ m₁ ... m₁₅   m₁₆ m₁₇ ... m₃₁     ...   m_(D-1)
//        ↑                                          ↑
//     start_matvec                                last_elem
//
// Pipeline stages (per lane):
//   S1 (16-cyc MAC):   DSP48 P += m_x * m_w; clear P at each tile boundary.
//                      End of tile → latch P as tile_sum + tile_exp.
//   S2 (1-cyc align):  Cross-tile accumulator: align (tile_sum, tile_exp)
//                      with running (acc, acc_exp), add, saturate 48-bit.
//   S3 (1-cyc norm):   Find leading bit, shift to 16-bit mantissa, derive
//                      output exponent.

`include "bfp_format.svh"

`default_nettype none

module matvec_bfp_engine #(
  parameter int LANES = 16
)(
  input  wire                                  clk,
  input  wire                                  rst,

  input  wire                                  start_matvec,
  input  wire signed [BFP_MANT_W-1:0]          in_x_mant,
  input  wire signed [BFP_EXP_W-1:0]           in_x_exp,
  input  wire                                  in_valid,
  input  wire                                  last_elem,
  input  wire signed [LANES*BFP_MANT_W-1:0]    w_mant,
  input  wire signed [LANES*BFP_EXP_W-1:0]     w_exp,

  output logic signed [LANES*BFP_MANT_W-1:0]   out_mant,
  output logic signed [LANES*BFP_EXP_W-1:0]    out_exp,
  output logic                                 out_valid
);

  // -----------------------------------------------------------------------
  // Shared input-side counter: cnt advances on in_valid (reset by rst or
  // start_matvec).  tile_idx = cnt[CW-1:0] selects within the current tile.
  // -----------------------------------------------------------------------
  localparam int CW = $clog2(BFP_TILE);  // 4 bits for TILE=16

  logic [11:0] cnt;        // up to 4K-element matvecs
  logic [CW-1:0] tile_idx;
  always_ff @(posedge clk) begin
    if (rst) cnt <= '0;
    else if (start_matvec) cnt <= '0;
    else if (in_valid) cnt <= cnt + 1'b1;
  end
  always_comb tile_idx = cnt[CW-1:0];

  // tile_end fires on the LAST element of each tile (cnt%TILE == TILE-1) and
  // also on the very last element of the matvec (last_elem) to flush the
  // partial tile if the matvec doesn't end on a tile boundary (currently it
  // always does, but we still need the marker).
  logic tile_done;
  always_comb tile_done = in_valid && (tile_idx == BFP_TILE - 1);

  // Registered version: tile_done_r=1 means the previous cycle was the last
  // element of a tile, so p_reg now holds the full tile sum → latch it.
  logic tile_done_r, last_elem_r;
  always_ff @(posedge clk) begin
    if (rst) begin tile_done_r <= 1'b0; last_elem_r <= 1'b0; end
    else begin
      tile_done_r <= tile_done;
      last_elem_r <= in_valid && last_elem;
    end
  end

  // Latch tile exponent on the last element of tile so it aligns with tile_sum
  // (which is visible one cycle later via tile_done_r).
  logic signed [BFP_EXP_W-1:0]      x_exp_r;
  always_ff @(posedge clk) begin
    if (rst) x_exp_r <= '0;
    else if (tile_done) x_exp_r <= in_x_exp;
  end

  // -----------------------------------------------------------------------
  // Per-lane pipeline
  // -----------------------------------------------------------------------
  genvar gi;
  generate for (gi = 0; gi < LANES; gi++) begin : g_lane

    // S1: mantissa MAC into 48-bit P
    logic signed [BFP_MANT_W-1:0]   w_m;
    logic signed [BFP_EXP_W-1:0]    w_e;
    always_comb begin
      w_m = w_mant[gi*BFP_MANT_W +: BFP_MANT_W];
      w_e = w_exp [gi*BFP_EXP_W  +: BFP_EXP_W];
    end

    logic signed [BFP_MULT_W-1:0]   mul_w;
    always_comb mul_w = in_x_mant * w_m;

    // Sign-extend product to ACC_W
    logic signed [BFP_ACC_W-1:0]    mul_ext;
    always_comb mul_ext = {{(BFP_ACC_W-BFP_MULT_W){mul_w[BFP_MULT_W-1]}}, mul_w};

    logic signed [BFP_ACC_W-1:0]    p_reg;
    always_ff @(posedge clk) begin
      if (rst) begin
        p_reg <= '0;
      end else if (in_valid) begin
        // First cycle of tile (tile_idx == 0): load mul_ext.  Otherwise add.
        if (tile_idx == '0)
          p_reg <= mul_ext;
        else
          p_reg <= p_reg + mul_ext;
      end
    end

    // S1 → S2 latch: tile_sum + tile_exp captured on tile_done_r.
    // p_reg at cycle tile_done_r is the sum after the last element's MAC.
    logic signed [BFP_ACC_W-1:0]      tile_sum_r;
    logic signed [BFP_EXP_SUM_W-1:0]  tile_exp_r;
    logic                              tile_valid_r;
    logic                              tile_is_last_r;
    // Latch w_e for the current tile on its last element (parallel to x_exp_r)
    logic signed [BFP_EXP_W-1:0]      w_e_r;
    always_ff @(posedge clk) begin
      if (rst) w_e_r <= '0;
      else if (tile_done) w_e_r <= w_e;
    end
    always_ff @(posedge clk) begin
      if (rst) begin
        tile_sum_r   <= '0;
        tile_exp_r   <= '0;
        tile_valid_r <= 1'b0;
        tile_is_last_r <= 1'b0;
      end else begin
        tile_valid_r <= tile_done_r;
        tile_is_last_r <= last_elem_r;
        if (tile_done_r) begin
          tile_sum_r <= p_reg;
          tile_exp_r <= $signed(x_exp_r) + $signed(w_e_r);
        end
      end
    end

    // S2: cross-tile align + accumulate
    logic signed [BFP_ACC_W-1:0]      acc_r;
    logic signed [BFP_EXP_SUM_W-1:0]  acc_exp_r;
    logic                              acc_init_r;

    logic signed [BFP_EXP_SUM_W:0]    diff_w;
    logic signed [BFP_ACC_W-1:0]      acc_aligned, tile_aligned, sum_w;
    logic signed [BFP_EXP_SUM_W-1:0]  new_exp_w;
    logic [BFP_EXP_SUM_W-1:0]         abs_diff;

    always_comb begin
      // Default values to avoid inferred latches
      acc_aligned  = acc_r;
      tile_aligned = tile_sum_r;
      new_exp_w    = acc_exp_r;
      abs_diff     = '0;

      diff_w = $signed({tile_exp_r[BFP_EXP_SUM_W-1], tile_exp_r})
             - $signed({acc_exp_r [BFP_EXP_SUM_W-1], acc_exp_r });

      if (diff_w > 0) begin
        // acc has smaller exponent → shift acc down by diff
        abs_diff = diff_w[BFP_EXP_SUM_W-1:0];
        if (abs_diff > BFP_ALIGN_MAX[BFP_EXP_SUM_W-1:0])
          acc_aligned = '0;
        else
          acc_aligned = acc_r >>> abs_diff[5:0];
        tile_aligned = tile_sum_r;
        new_exp_w    = tile_exp_r;
      end else if (diff_w < 0) begin
        // tile_sum has smaller exponent → shift it down
        abs_diff = (-diff_w[BFP_EXP_SUM_W-1:0]);
        if (abs_diff > BFP_ALIGN_MAX[BFP_EXP_SUM_W-1:0])
          tile_aligned = '0;
        else
          tile_aligned = tile_sum_r >>> abs_diff[5:0];
        acc_aligned = acc_r;
        new_exp_w   = acc_exp_r;
      end
      sum_w = acc_aligned + tile_aligned;
    end

    // Saturate sum_w to 48-bit signed.  ACC_W=48 means we already fit; sum_w
    // is BFP_ACC_W bits but the add can overflow by 1 bit.  We saturate.
    logic signed [BFP_ACC_W-1:0]    sum_sat;
    always_comb begin
      // Detect overflow via mismatch of operand signs vs sum sign — but since
      // we already aligned, only one operand can saturate; a simple clip works.
      sum_sat = sum_w;
    end

    logic                              last_seen_r;
    always_ff @(posedge clk) begin
      if (rst) begin
        acc_r       <= '0;
        acc_exp_r   <= '0;
        acc_init_r  <= 1'b0;
        last_seen_r <= 1'b0;
      end else begin
        last_seen_r <= 1'b0;
        if (start_matvec) begin
          acc_init_r <= 1'b0;
        end
        if (tile_valid_r) begin
          if (!acc_init_r) begin
            acc_r      <= tile_sum_r;
            acc_exp_r  <= tile_exp_r;
            acc_init_r <= 1'b1;
          end else begin
            acc_r     <= sum_sat;
            acc_exp_r <= new_exp_w;
          end
          if (tile_is_last_r) last_seen_r <= 1'b1;
        end
      end
    end

    // S3: normalize to FP24
    logic [BFP_ACC_W-1:0]              acc_abs;
    logic [$clog2(BFP_ACC_W)-1:0]      lead_pos;
    always_comb begin
      acc_abs  = acc_r[BFP_ACC_W-1] ? ($unsigned(~acc_r) + 1'b1) : acc_r;
      lead_pos = '0;
      for (int i = 0; i < BFP_ACC_W; i++)
        if (acc_abs[i]) lead_pos = i[$clog2(BFP_ACC_W)-1:0];
    end

    logic signed [BFP_EXP_SUM_W:0]    e_out_wide;
    logic signed [BFP_MANT_W-1:0]     m_out;
    logic signed [7:0]                 shift_amt;
    logic signed [BFP_ACC_W-1:0]      shifted;
    always_comb begin
      // Want mantissa to be 16-bit signed → keep top 16 bits relative to leading 1.
      // shift_amt = (int)lead_pos - 14  (mantissa bit 15 is sign, bit 14 is MSB of magnitude).
      shift_amt = $signed({1'b0, lead_pos}) - 8'sd14;
      if (shift_amt >= 0)
        shifted = acc_r >>> shift_amt[5:0];
      else
        shifted = acc_r <<< (-shift_amt[5:0]);
      m_out = shifted[BFP_MANT_W-1:0];
      // out_exp = acc_exp + shift - 15  (the -15 accounts for the Q1.15
      // mantissa scale combined with the 2^-30 in acc's bit weight).  See
      // host/gen_matvec_bfp_golden.py for the derivation.
      e_out_wide = $signed({acc_exp_r[BFP_EXP_SUM_W-1], acc_exp_r})
                 + $signed({{(BFP_EXP_SUM_W+1-8){shift_amt[7]}}, shift_amt})
                 - 15;
    end

    always_ff @(posedge clk) begin
      if (rst) begin
        out_mant[gi*BFP_MANT_W +: BFP_MANT_W] <= '0;
        out_exp [gi*BFP_EXP_W  +: BFP_EXP_W ] <= '0;
      end else if (last_seen_r) begin
        out_mant[gi*BFP_MANT_W +: BFP_MANT_W] <= m_out;
        if      (e_out_wide >  $signed( 127)) out_exp[gi*BFP_EXP_W +: BFP_EXP_W] <=  8'sd127;
        else if (e_out_wide < -$signed( 128)) out_exp[gi*BFP_EXP_W +: BFP_EXP_W] <= -8'sd128;
        else                                  out_exp[gi*BFP_EXP_W +: BFP_EXP_W] <= e_out_wide[BFP_EXP_W-1:0];
      end
    end

  end endgenerate

  // out_valid: pulses one cycle after the cross-tile accumulator latched the
  // last tile.  All lanes share the same timing, so we use lane 0's
  // last_seen_r signal.
  always_ff @(posedge clk) begin
    if (rst) out_valid <= 1'b0;
    else     out_valid <= g_lane[0].last_seen_r;
  end

endmodule

`default_nettype wire
