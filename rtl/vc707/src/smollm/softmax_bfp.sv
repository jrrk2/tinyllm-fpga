// softmax_bfp.sv — block-FP softmax (variable-length input).
//
//   y[i] = exp(x[i] - max(x)) / Σ_j exp(x[j] - max(x))
//
// Input: a single block-FP "vector" of N mantissas sharing one exponent
// (caller responsible — for attention the QK^T scores naturally share an
// exponent across all timesteps).
//
// Output: N probabilities in block-FP, single shared output exponent
// (which is ≤ 0 since all probs are in [0, 1]).
//
// Algorithm (FSM-driven, similar shape to softmax_q15.sv):
//   S_LOAD     accept N mantissas (single tile of scores), track max-mant
//   S_EXP      compute diff[k] = m[k] - max_m, LUT-index = clamp((diff <<
//              (e - 8)) + 1023, 0, 1023), look up e_buf[k] = exp_lut[idx],
//              accumulate sum_e
//   S_RECIP    Newton-Raphson reciprocal of sum_e (same pattern as
//              softmax_q15.sv: y_{n+1} = y_n*(2^33 - sum_e*y_n) >> 32)
//   S_OUTPUT   stream y[k] = (e_buf[k] * inv_sum) >> SHIFT, with per-tile
//              re-quantize at the end (track max-|m| across the output
//              vector, normalize tile exp).  Output mantissas in Q1.15.

`include "bfp_format.svh"

`default_nettype none

module softmax_bfp #(
  parameter int N_MAX = 64
)(
  input  wire                          clk,
  input  wire                          rst,
  input  wire                          start,
  input  wire [$clog2(N_MAX+1)-1:0]    n_elems,    // 1..N_MAX
  input  wire signed [BFP_MANT_W-1:0]  in_x_mant,
  input  wire signed [BFP_EXP_W -1:0]  in_x_exp,   // shared across all N
  input  wire                          in_valid,
  output logic signed [BFP_MANT_W-1:0] out_y_mant,
  output logic signed [BFP_EXP_W -1:0] out_y_exp,  // single shared output exp
  output logic                         out_valid,
  output logic                         done
);

  // 1024-entry exp LUT (same one as softmax_q15.sv)
  logic [15:0] exp_lut [0:1023];
  initial $readmemh("../generated/exp_lut.hex", exp_lut);

  localparam int AW = $clog2(N_MAX + 1);

  // Per-element score buffer (mantissas only — exponent is shared)
  logic signed [BFP_MANT_W-1:0] x_buf  [0:N_MAX-1];
  logic [15:0]                  e_buf  [0:N_MAX-1];

  typedef enum logic [2:0] {S_IDLE, S_LOAD, S_EXP, S_RECIP, S_NORM, S_OUTPUT, S_DONE} state_t;
  state_t state;
  logic [AW-1:0] in_cnt, exp_cnt, out_cnt;

  logic signed [BFP_MANT_W-1:0] max_r;
  logic signed [BFP_EXP_W -1:0] x_exp_r;
  logic [AW-1:0]                n_r;

  // S_EXP datapath
  logic signed [BFP_MANT_W:0]   diff17;
  logic signed [BFP_MANT_W:0]   diff_shifted;
  logic signed [BFP_EXP_W:0]    sm_shamt;
  logic [9:0]                   lut_idx;
  always_comb begin
    diff17 = {x_buf[exp_cnt][BFP_MANT_W-1], x_buf[exp_cnt]}
           - {max_r[BFP_MANT_W-1], max_r};
    // Pre-shift to LUT index space.  exp_lut covers diff_real ∈ [-8, 0],
    // step = 1/128.  shamt = e_shared - 8.
    sm_shamt = $signed({x_exp_r[BFP_EXP_W-1], x_exp_r}) - 8;
    if (sm_shamt >= 0) diff_shifted = diff17 <<< sm_shamt[4:0];
    else               diff_shifted = diff17 >>> (-sm_shamt[4:0]);
    // idx = clamp(diff_shifted + 1023, 0, 1023).  diff_shifted ≤ 0 since
    // max - max = 0 is the largest value.
    if      (diff_shifted >=  17'sd0)         lut_idx = 10'd1023;
    else if (diff_shifted <= -17'sd1023)      lut_idx = 10'd0;
    else                                      lut_idx = 10'(diff_shifted + 17'sd1023);
  end

  logic [31:0] sum_e_r;

  // S_RECIP Newton-Raphson reciprocal (same pattern as softmax_q15.sv)
  logic [4:0]  rcp_msb;
  logic [31:0] rcp_y, rcp_y0_seed;
  always_comb begin
    rcp_msb = 5'd0;
    for (int i = 0; i < 32; i++) if (sum_e_r[i]) rcp_msb = i[4:0];
    rcp_y0_seed = 32'(1'b1) << (5'd31 - rcp_msb);
  end
  logic [63:0] rcp_prod, rcp_diff, rcp_next;
  always_comb begin
    rcp_prod = {32'd0, sum_e_r} * {32'd0, rcp_y};
    rcp_diff = 64'h200000000 - rcp_prod;
    rcp_next = {32'd0, rcp_y} * rcp_diff;
  end
  logic [31:0] inv_sum_r;
  logic [2:0]  nr_cnt;

  // S_NORM: scan e_buf[0..n-1] * inv_sum to find max → determine out_exp
  logic [BFP_ACC_W-1:0]         norm_prods [0:N_MAX-1];
  logic [4:0]                   max_norm_lead;
  logic [AW-1:0]                norm_cnt;

  // S_OUTPUT: emit aligned mantissas
  logic signed [7:0]            out_shift_r;

  // Compute leading-bit of one norm product
  function automatic logic [5:0] lead_bit_48(input logic [BFP_ACC_W-1:0] v);
    logic [5:0] p;
    p = 6'd0;
    for (int i = 0; i < BFP_ACC_W; i++) if (v[i]) p = i[5:0];
    return p;
  endfunction

  always_ff @(posedge clk) begin
    if (rst) begin
      state      <= S_IDLE;
      in_cnt     <= '0;
      exp_cnt    <= '0;
      out_cnt    <= '0;
      max_r      <= 16'sh8000;     // most negative
      sum_e_r    <= '0;
      rcp_y      <= '0;
      inv_sum_r  <= '0;
      nr_cnt     <= '0;
      max_norm_lead <= '0;
      norm_cnt   <= '0;
      out_y_mant <= '0;
      out_y_exp  <= '0;
      out_valid  <= 1'b0;
      done       <= 1'b0;
      out_shift_r <= '0;
    end else begin
      out_valid <= 1'b0;
      done      <= 1'b0;
      case (state)
        S_IDLE: if (start) begin
          state   <= S_LOAD;
          in_cnt  <= '0;
          max_r   <= 16'sh8000;
          sum_e_r <= '0;
        end
        S_LOAD: if (in_valid) begin
          x_buf[in_cnt] <= in_x_mant;
          if (in_x_mant > max_r) max_r <= in_x_mant;
          if (in_cnt == '0) begin
            x_exp_r <= in_x_exp;
            n_r     <= n_elems;
          end
          in_cnt <= in_cnt + 1'b1;
`ifdef LBFP_STAGE_DUMP
          $display("[sm] LOAD i=%0d x_m=%0d x_e=%0d", in_cnt, $signed(in_x_mant), $signed(in_x_exp));
`endif
          if (in_cnt == n_elems - 1) begin
            state   <= S_EXP;
            exp_cnt <= '0;
`ifdef LBFP_STAGE_DUMP
            $display("[sm] LOAD done max_r=%0d sum_will_compute_in_EXP", $signed(max_r));
`endif
          end
        end
        S_EXP: begin
          e_buf[exp_cnt] <= exp_lut[lut_idx];
          sum_e_r        <= sum_e_r + {16'd0, exp_lut[lut_idx]};
          exp_cnt        <= exp_cnt + 1'b1;
`ifdef LBFP_STAGE_DUMP
          $display("[sm] EXP i=%0d lut_idx=%0d e=%h sum=%h", exp_cnt, lut_idx, exp_lut[lut_idx], sum_e_r);
`endif
          if (exp_cnt == n_r - 1) begin
            state  <= S_RECIP;
            nr_cnt <= '0;
          end
        end
        S_RECIP: begin
          if (nr_cnt == 3'd0) begin
            rcp_y  <= rcp_y0_seed;
            nr_cnt <= 3'd1;
`ifdef LBFP_STAGE_DUMP
            $display("[sm] RECIP start sum_e_r=%h rcp_y0_seed=%h rcp_msb=%0d", sum_e_r, rcp_y0_seed, rcp_msb);
`endif
          end else begin
            rcp_y <= rcp_next[63:32];
            if (nr_cnt == 3'd4) begin
              inv_sum_r <= rcp_next[63:32];
              state     <= S_NORM;
              norm_cnt  <= '0;
              max_norm_lead <= '0;
`ifdef LBFP_STAGE_DUMP
              $display("[sm] RECIP done inv_sum=%h", rcp_next[63:32]);
`endif
            end
            nr_cnt <= nr_cnt + 1'b1;
          end
        end
        S_NORM: begin : norm_body
          // Compute norm_prod = e_buf[k] * inv_sum  (16 × 32 = 48-bit)
          automatic logic [BFP_ACC_W-1:0] prod;
          automatic logic [5:0]           lp;
          prod = {16'd0, e_buf[norm_cnt]} * {16'd0, inv_sum_r};
          norm_prods[norm_cnt] <= prod;
          lp = lead_bit_48(prod);
`ifdef LBFP_STAGE_DUMP
          $display("[sm] NORM i=%0d e_buf=%h inv_sum=%h prod=%h lp=%0d max_norm_lead_was=%0d",
                   norm_cnt, e_buf[norm_cnt], inv_sum_r, prod, lp, max_norm_lead);
`endif
          if (lp > max_norm_lead) max_norm_lead <= lp;
          if (norm_cnt == n_r - 1) begin
            // Current cycle's `lp` (for norm_cnt = n_r-1) hasn't been
            // folded into max_norm_lead yet (non-blocking <=), so
            // out_shift_r needs to compare against max(max_norm_lead, lp)
            // — otherwise when the LAST entry holds the leading bit
            // (e.g. current-step kv_t==kv_pos dominating attention) the
            // shift is computed from an under-counted max and the
            // output exponent ends up far too small.  Common case for
            // multi-head attention where the max prob is at kv_pos.
            automatic logic [4:0] eff_max =
                (lp > max_norm_lead) ? lp[4:0] : max_norm_lead;
            state    <= S_OUTPUT;
            out_cnt  <= '0;
            out_shift_r <= $signed({2'b0, eff_max}) - 8'sd14;
`ifdef LBFP_STAGE_DUMP
            $display("[sm] NORM done max_norm_lead=%0d eff_max=%0d out_shift_r=%0d",
                     max_norm_lead, eff_max, $signed({2'b0, eff_max}) - 8'sd14);
`endif
          end else begin
            norm_cnt <= norm_cnt + 1'b1;
          end
        end
        S_OUTPUT: begin
          begin : out_emit
            automatic logic signed [BFP_ACC_W-1:0] sh;
            if (out_shift_r >= 0)
              sh = $signed(norm_prods[out_cnt]) >>> out_shift_r[5:0];
            else
              sh = $signed(norm_prods[out_cnt]) <<< (-out_shift_r[5:0]);
            out_y_mant <= sh[BFP_MANT_W-1:0];
            // Output exponent shared (computed once)
            begin : exp_blk
              logic signed [BFP_EXP_SUM_W+1:0] e_wide;
              // out_e = shift - 17  (norm_prod = prob * 2^32; shift to Q1.15)
              e_wide = $signed({{(BFP_EXP_SUM_W-6){out_shift_r[7]}}, out_shift_r}) - 17;
              if      (e_wide >  127) out_y_exp <=  8'sd127;
              else if (e_wide < -128) out_y_exp <= -8'sd128;
              else                    out_y_exp <=  e_wide[BFP_EXP_W-1:0];
            end
            out_valid <= 1'b1;
            out_cnt   <= out_cnt + 1'b1;
            if (out_cnt == n_r - 1) state <= S_DONE;
          end
        end
        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
