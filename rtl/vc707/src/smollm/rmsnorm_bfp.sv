// rmsnorm_bfp.sv — block-FP RMSNorm.
//
//   y[i] = x[i] * gamma[i] / sqrt( mean_k(x[k]^2) + eps )
//
// All I/O in block-FP format (TILE elements per 8-bit-shared exponent,
// 16-bit signed mantissa per element).
//
// Pipeline (FSM-driven):
//   S_IDLE     wait for start
//   S_LOAD     D cycles — accept x and gamma into per-element buffers AND
//              accumulate Σ x[k]^2 in a 48-bit fabric accumulator via per-
//              tile MAC + cross-tile barrel-shift align.
//   S_COMPUTE  N cycles — normalize the sum-sq accumulator, multiply by
//              INV_D_Q32, add EPS_Q30, then derive inv_rms via:
//                seed   = LUT(msb of v_q30) in Q12.12
//                iter×3 of Newton-Raphson  y_{n+1} = y * (1.5 - 0.5*v*y²)
//                Final inv_rms = Q12.12 (24-bit unsigned).
//   S_OUTPUT   D cycles — stream y[i] = m_x[i] * m_g[i] * inv_rms (48-bit
//              product), find per-tile max-|mantissa| to set tile exponent,
//              align & emit.  out_valid pulses each cycle.
//   S_DONE     pulse `done` for 1 cycle.

`include "bfp_format.svh"

`default_nettype none

module rmsnorm_bfp #(
  parameter int D = 576,
  parameter logic [31:0] EPS_Q30 = 32'd10737   // round(1e-5 * 2^30)
)(
  input  wire                          clk,
  input  wire                          rst,

  input  wire                          start,
  input  wire signed [BFP_MANT_W-1:0]  in_x_mant,
  input  wire signed [BFP_EXP_W -1:0]  in_x_exp,
  input  wire signed [BFP_MANT_W-1:0]  in_g_mant,
  input  wire signed [BFP_EXP_W -1:0]  in_g_exp,
  input  wire                          in_valid,
  input  wire                          last_elem,

  output logic signed [BFP_MANT_W-1:0] out_y_mant,
  output logic signed [BFP_EXP_W -1:0] out_y_exp,
  output logic                         out_valid,
  output logic                         done
);

  localparam int CW   = $clog2(BFP_TILE);             // 4
  localparam int NT   = (D + BFP_TILE - 1) / BFP_TILE;
  localparam int DW   = $clog2(D + 1);

  localparam logic [31:0] INV_D_Q32 = 32'(((longint'(1)) << 32) / D);

  // -----------------------------------------------------------------------
  // Element buffers — D entries of (mantissa, tile-shared exponent)
  // -----------------------------------------------------------------------
  logic signed [BFP_MANT_W-1:0] x_mem [0:D-1];
  logic signed [BFP_MANT_W-1:0] g_mem [0:D-1];
  logic signed [BFP_EXP_W-1:0]  xe_mem [0:NT-1];
  logic signed [BFP_EXP_W-1:0]  ge_mem [0:NT-1];

  // -----------------------------------------------------------------------
  // FSM
  // -----------------------------------------------------------------------
  typedef enum logic [2:0] {S_IDLE, S_LOAD, S_NORM, S_NR, S_OUTPUT, S_DONE} state_t;
  state_t state;

  logic [DW-1:0]  in_cnt;
  logic [DW-1:0]  out_cnt;
  logic [CW-1:0]  in_tile_idx;
  logic           tile_done;
  logic           tile_done_r;
  logic           last_elem_r;

  always_comb tile_done   = (state == S_LOAD) && in_valid && (in_tile_idx == BFP_TILE-1);

  always_ff @(posedge clk) begin
    if (rst) begin
      tile_done_r <= 1'b0;
      last_elem_r <= 1'b0;
    end else begin
      tile_done_r <= tile_done;
      last_elem_r <= (state == S_LOAD) && in_valid && last_elem;
    end
  end

  // -----------------------------------------------------------------------
  // Stage A: per-tile sum-of-squares (one DSP doing m_x * m_x)
  // -----------------------------------------------------------------------
  logic signed [BFP_MULT_W-1:0]   sq_w;
  always_comb sq_w = in_x_mant * in_x_mant;            // signed×signed but always ≥0

  logic [BFP_ACC_W-1:0]           p_reg;
  always_ff @(posedge clk) begin
    if (rst || start) p_reg <= '0;
    else if (state == S_LOAD && in_valid) begin
      if (in_tile_idx == '0)
        p_reg <= {{(BFP_ACC_W-BFP_MULT_W){1'b0}}, sq_w};
      else
        p_reg <= p_reg + {{(BFP_ACC_W-BFP_MULT_W){1'b0}}, sq_w};
    end
  end

  // Latch tile exponent on the last element of each tile.
  logic signed [BFP_EXP_W-1:0]    x_exp_latch_r;
  always_ff @(posedge clk) begin
    if (rst) x_exp_latch_r <= '0;
    else if (tile_done) x_exp_latch_r <= in_x_exp;
  end

  // Tile-sum + tile_exp_sq pair, ready one cycle after tile_done
  logic [BFP_ACC_W-1:0]              tile_sum_r;
  logic signed [BFP_EXP_SUM_W:0]    tile_exp_sq_r;  // 2*(e_x), 10 bits
  logic                              tile_valid_r;
  always_ff @(posedge clk) begin
    if (rst) begin
      tile_sum_r <= '0; tile_exp_sq_r <= '0; tile_valid_r <= 1'b0;
    end else begin
      tile_valid_r <= tile_done_r;
      if (tile_done_r) begin
        tile_sum_r    <= p_reg;
        // 2*x_exp_latch_r → tile mantissa product scale's exponent
        tile_exp_sq_r <= $signed({x_exp_latch_r[BFP_EXP_W-1], x_exp_latch_r}) <<< 1;
      end
    end
  end

  // -----------------------------------------------------------------------
  // Cross-tile sum-sq accumulator with barrel-shift alignment (unsigned)
  // -----------------------------------------------------------------------
  logic [BFP_ACC_W-1:0]              sq_acc_r;
  logic signed [BFP_EXP_SUM_W:0]    sq_acc_exp_r;
  logic                              sq_acc_init_r;

  logic signed [BFP_EXP_SUM_W+1:0]  diff_w;
  logic [BFP_ACC_W-1:0]              acc_aligned, tile_aligned, sum_new;
  logic signed [BFP_EXP_SUM_W:0]    new_exp_w;
  logic [BFP_EXP_SUM_W+1:0]         abs_diff;

  always_comb begin
    acc_aligned  = sq_acc_r;
    tile_aligned = tile_sum_r;
    new_exp_w    = sq_acc_exp_r;
    abs_diff     = '0;
    diff_w = $signed({tile_exp_sq_r[BFP_EXP_SUM_W], tile_exp_sq_r})
           - $signed({sq_acc_exp_r [BFP_EXP_SUM_W], sq_acc_exp_r });
    if (diff_w > 0) begin
      abs_diff = diff_w[BFP_EXP_SUM_W+1:0];
      if (abs_diff > BFP_ALIGN_MAX[BFP_EXP_SUM_W+1:0]) acc_aligned = '0;
      else acc_aligned = sq_acc_r >> abs_diff[5:0];
      new_exp_w = tile_exp_sq_r;
    end else if (diff_w < 0) begin
      abs_diff = -diff_w[BFP_EXP_SUM_W+1:0];
      if (abs_diff > BFP_ALIGN_MAX[BFP_EXP_SUM_W+1:0]) tile_aligned = '0;
      else tile_aligned = tile_sum_r >> abs_diff[5:0];
      new_exp_w = sq_acc_exp_r;
    end
    sum_new = acc_aligned + tile_aligned;
  end

  always_ff @(posedge clk) begin
    if (rst || start) begin
      sq_acc_r       <= '0;
      sq_acc_exp_r   <= '0;
      sq_acc_init_r  <= 1'b0;
    end else if (tile_valid_r) begin
      if (!sq_acc_init_r) begin
        sq_acc_r      <= tile_sum_r;
        sq_acc_exp_r  <= tile_exp_sq_r;
        sq_acc_init_r <= 1'b1;
      end else begin
        sq_acc_r     <= sum_new;
        sq_acc_exp_r <= new_exp_w;
      end
    end
  end

  // -----------------------------------------------------------------------
  // S_LOAD: capture x, gamma and their per-tile exponents
  // -----------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (state == S_LOAD && in_valid) begin
      x_mem[in_cnt] <= in_x_mant;
      g_mem[in_cnt] <= in_g_mant;
      if (in_tile_idx == '0) begin
        xe_mem[in_cnt[CW +: $clog2(NT)]] <= in_x_exp;
        ge_mem[in_cnt[CW +: $clog2(NT)]] <= in_g_exp;
      end
    end
  end

  // -----------------------------------------------------------------------
  // S_NORM: produce a Q2.30 v_q30 value from (sq_acc_r, sq_acc_exp_r), then
  //         add EPS, then leading-bit extract for the rsqrt seed LUT.
  //
  // v_real = sq_acc_r * 2^(sq_acc_exp_r - 30 + 0) / D
  //        = sq_acc_r * INV_D_Q32 / 2^32 * 2^(sq_acc_exp_r - 30)
  //
  // To produce a single Q2.30 input to the seed/NR datapath, we normalize
  // sq_acc_r so its leading bit lands at bit 30 (Q2.30 = bit 30/31 set).
  // Net exponent absorbed into v_exp.
  // -----------------------------------------------------------------------
  logic [95:0]      prod_acc_invD;
  logic [BFP_ACC_W-1:0] mean_sq_int;
  logic [5:0]       lead_pos;
  logic [31:0]      v_q30;
  logic signed [BFP_EXP_SUM_W+1:0] v_exp;

  logic [5:0] lead_pos_adj;
  always_comb begin
    prod_acc_invD = sq_acc_r * INV_D_Q32;
    mean_sq_int   = prod_acc_invD[BFP_ACC_W+32-1:32];   // mean_sq * 2^(sq_acc_exp - 30)
    // Find leading bit of mean_sq_int
    lead_pos = '0;
    for (int i = 0; i < BFP_ACC_W; i++)
      if (mean_sq_int[i]) lead_pos = i[5:0];
    // Parity-adjust lead so (sq_acc_exp + lead) is even — keeps e_inv_excess
    // an integer.  Drop one bit of normalization if odd.
    if ((sq_acc_exp_r[0] ^ lead_pos[0]) == 1'b1)
      lead_pos_adj = (lead_pos == 0) ? 6'd0 : (lead_pos - 6'd1);
    else
      lead_pos_adj = lead_pos;
    // Shift so leading bit lands at 30 (or 31 in the odd-parity case)
    if (lead_pos_adj >= 30)
      v_q30 = mean_sq_int >> (lead_pos_adj - 30);
    else
      v_q30 = mean_sq_int << (30 - lead_pos_adj);
    v_q30 = v_q30 + EPS_Q30;
    // Track the "raw" v_exp = sq_acc_exp + lead_adj - 60 (used in
    // e_inv_excess derivation; the -60 absorbs both the 2^-30 from acc's
    // representation and the 2^-30 from Q2.30 interpretation of v_q30).
    v_exp = $signed({sq_acc_exp_r[BFP_EXP_SUM_W], sq_acc_exp_r}) + $signed({4'b0, lead_pos_adj}) - 60;
  end

  // -----------------------------------------------------------------------
  // Newton-Raphson 1/sqrt(v_q30) with 24-bit Q12.12 inv_rms (widened to fit
  // SmolLM2's outlier scales).
  // -----------------------------------------------------------------------
  logic [4:0]       v_msb;
  always_comb begin
    v_msb = 5'd0;
    for (int i = 0; i < 32; i++) if (v_q30[i]) v_msb = i[4:0];
  end

  logic [23:0] y_seed;
  always_comb begin
    case (v_msb)
      5'd31: y_seed = 24'd2365;     5'd30: y_seed = 24'd3344;
      5'd29: y_seed = 24'd4730;     5'd28: y_seed = 24'd6689;
      5'd27: y_seed = 24'd9459;     5'd26: y_seed = 24'd13377;
      5'd25: y_seed = 24'd18919;    5'd24: y_seed = 24'd26755;
      5'd23: y_seed = 24'd37837;    5'd22: y_seed = 24'd53510;
      5'd21: y_seed = 24'd75674;    5'd20: y_seed = 24'd107020;
      5'd19: y_seed = 24'd151349;   5'd18: y_seed = 24'd214040;
      5'd17: y_seed = 24'd302698;   5'd16: y_seed = 24'd428079;
      5'd15: y_seed = 24'd605396;   5'd14: y_seed = 24'd856159;
      5'd13: y_seed = 24'd1210791;  5'd12: y_seed = 24'd1712317;
      5'd11: y_seed = 24'd2421583;  5'd10: y_seed = 24'd3424635;
      5'd 9: y_seed = 24'd4843165;  5'd 8: y_seed = 24'd6849270;
      5'd 7: y_seed = 24'd9686330;  5'd 6: y_seed = 24'd13698540;
      default: y_seed = 24'hFFFFFF;
    endcase
  end

  localparam logic [31:0] C_1P5_Q30 = 32'h6000_0000;

  logic [23:0]  inv_rms;
  logic [2:0]   nr_cnt;
  logic [47:0]  nr_y_sq;
  logic [79:0]  nr_vy2;
  logic [55:0]  nr_vy2_q30;
  logic signed [32:0] nr_corr;
  logic signed [57:0] nr_y_new_full;
  logic [57:0]  nr_biased;
  logic [23:0]  nr_y_new;

  always_comb begin
    nr_y_sq    = inv_rms * inv_rms;
    nr_vy2     = {32'b0, v_q30} * {32'b0, nr_y_sq};
    nr_vy2_q30 = nr_vy2[79:24];
    if (nr_vy2_q30[55:32] != 24'h0)
      nr_corr = 33'sb0;
    else begin
      nr_corr = $signed({1'b0, C_1P5_Q30}) - $signed({2'b00, nr_vy2_q30[31:1]});
      if (nr_corr < 0) nr_corr = 33'sb0;
    end
    nr_y_new_full = $signed({1'b0, inv_rms}) * nr_corr;
    nr_biased     = $unsigned(nr_y_new_full) + 58'h0000_0000_2000_0000;
    if (nr_biased[57:54] != 4'b0000)
      nr_y_new = 24'hFFFFFF;
    else
      nr_y_new = nr_biased[53:30];
  end

  // -----------------------------------------------------------------------
  // S_OUTPUT: stream y[i] = x[i] * g[i] * inv_rms, find per-tile max-|m|
  //
  // To re-tile-quantize, we first compute the 48-bit product for each
  // element of the tile, then find max-|product| within the tile to set
  // its exponent, then emit aligned 16-bit mantissas.  Implementation: two
  // passes — first pass within a tile computes products and tracks max,
  // second pass emits.  Buffered between with a tile-sized register file.
  // -----------------------------------------------------------------------
  // Phase within S_OUTPUT: 0..TILE-1 = compute+track max ; TILE..2*TILE-1
  // = emit aligned mantissas.  out_cnt counts the EMITTED elements.
  logic [CW-1:0]                   out_tile_idx;   // 0..TILE-1
  logic                            out_phase_emit; // 0 = compute, 1 = emit

  // Per-element product buffer (within the current tile)
  logic signed [BFP_ACC_W-1:0]     prod_buf [0:BFP_TILE-1];
  logic [5:0]                      max_lead_pos_r;     // leading bit of max |prod|
  logic signed [BFP_EXP_SUM_W+1:0] tile_exp_out_r;     // exponent for current output tile

  // Element index for the compute pass (0..TILE-1 of current tile)
  logic [CW-1:0]                   cmp_idx;

  // Compute per-element product = m_x * m_g * inv_rms, exp = e_x + e_g + e_inv - 45
  // (the -45 comes from 3 mantissa scales: -15 -15 -12 -3 = -45; wait, let me redo)
  //   real_y = (m_x/2^15)*2^e_x * (m_g/2^15)*2^e_g * (inv_rms/2^12)*2^e_inv
  //          = m_x*m_g*inv_rms * 2^(e_x + e_g + e_inv - 42)
  //   For Q1.15 output stored at out_e: m_y/2^15 * 2^out_e = real_y
  //          → m_y = m_x*m_g*inv_rms * 2^(e_x + e_g + e_inv - 42 - out_e + 15)
  //                = m_x*m_g*inv_rms * 2^(e_x + e_g + e_inv - 27 - out_e)
  //   Output tile exp chosen so that the max |m_y| lands at int16 max (≈2^15).
  logic signed [BFP_MANT_W-1:0] cur_m_x;
  logic signed [BFP_MANT_W-1:0] cur_m_g;
  logic signed [31:0]           xg_prod_w;
  logic signed [55:0]           xgi_prod_w;
  always_comb begin
    cur_m_x   = x_mem[ {out_cnt[DW-1:CW], cmp_idx} ];
    cur_m_g   = g_mem[ {out_cnt[DW-1:CW], cmp_idx} ];
    xg_prod_w = cur_m_x * cur_m_g;
    xgi_prod_w = $signed(xg_prod_w) * $signed({1'b0, inv_rms});
  end

  // out_cnt counts EMITTED elements (advances only in emit phase)
  logic [CW-1:0] emit_idx;
  // Output exponent calculation: needs to absorb e_x + e_g + e_inv_rms
  // along with the leading-bit-based shift.
  logic signed [BFP_EXP_W-1:0] cur_xe, cur_ge;
  logic signed [BFP_EXP_SUM_W+1:0] e_inv_rms;

  // e_inv_excess = (30 - sq_acc_exp - lead) / 2.  Parity guaranteed even by
  // the lead_pos_adj logic above, so >>> 1 is exact.
  // Equivalently: -v_exp/2 - 15 (where v_exp = sq_acc_exp + lead - 60).
  always_comb e_inv_rms = -(v_exp >>> 1) - 15;

  // Output mantissa = (xgi_prod_w >> shift_amt) with shift_amt derived from
  // max_lead_pos_r so that max-|m| in the tile fits 16-bit signed.  shift
  // amount = max_lead_pos_r - 14 (when positive: right shift; negative: left).
  logic signed [7:0] shift_amt;
  logic signed [55:0] shifted_prod;
  always_comb begin
    shift_amt    = $signed({2'b0, max_lead_pos_r}) - 8'sd14;
    if (shift_amt >= 0) shifted_prod = xgi_prod_w >>> shift_amt[5:0];
    else                shifted_prod = xgi_prod_w <<< (-shift_amt[5:0]);
  end

  // Update tile_exp_out_r on the last cycle of the compute phase, based on
  // max_lead_pos and the per-tile e_x/e_g (constant within tile).
  always_comb begin
    cur_xe = xe_mem[ out_cnt[DW-1:CW] ];
    cur_ge = ge_mem[ out_cnt[DW-1:CW] ];
  end

  // out_y_exp formula: out_e = e_x + e_g + e_inv_rms + shift_amt - 27
  logic signed [BFP_EXP_SUM_W+1:0] out_e_wide;
  always_comb out_e_wide = $signed({cur_xe[BFP_EXP_W-1], cur_xe})
                         + $signed({cur_ge[BFP_EXP_W-1], cur_ge})
                         + e_inv_rms
                         + $signed({{(BFP_EXP_SUM_W-6){shift_amt[7]}}, shift_amt})
                         - 27;

  // S3 compute-pass leading-bit tracker
  logic [5:0]                       new_lead_pos;
  always_comb begin
    automatic logic [BFP_ACC_W-1:0] xgi_abs;
    xgi_abs = xgi_prod_w[55] ? ($unsigned(~xgi_prod_w[BFP_ACC_W-1:0]) + 1'b1)
                              :  $unsigned(xgi_prod_w[BFP_ACC_W-1:0]);
    new_lead_pos = 6'd0;
    for (int i = 0; i < BFP_ACC_W; i++) if (xgi_abs[i]) new_lead_pos = i[5:0];
  end

  // -----------------------------------------------------------------------
  // FSM proper
  // -----------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      state          <= S_IDLE;
      in_cnt         <= '0;
      in_tile_idx    <= '0;
      out_cnt        <= '0;
      out_tile_idx   <= '0;
      out_phase_emit <= 1'b0;
      cmp_idx        <= '0;
      emit_idx       <= '0;
      max_lead_pos_r <= '0;
      tile_exp_out_r <= '0;
      out_y_mant     <= '0;
      out_y_exp      <= '0;
      out_valid      <= 1'b0;
      done           <= 1'b0;
      inv_rms        <= 24'h001000;   // 1.0 in Q12.12
      nr_cnt         <= '0;
    end else begin
      out_valid <= 1'b0;
      done      <= 1'b0;
      case (state)
        S_IDLE: begin
          if (start) begin
            in_cnt      <= '0;
            in_tile_idx <= '0;
            state       <= S_LOAD;
          end
        end
        S_LOAD: begin
          if (in_valid) begin
            in_cnt <= in_cnt + 1'b1;
            if (in_tile_idx == BFP_TILE-1) in_tile_idx <= '0;
            else                            in_tile_idx <= in_tile_idx + 1'b1;
            if (last_elem) state <= S_NORM;
          end
        end
        S_NORM: begin
          // v_q30 is combinational from sq_acc_r/exp_r — settle one cycle
          state  <= S_NR;
          nr_cnt <= 3'd0;
          // Load seed
          inv_rms <= y_seed;
        end
        S_NR: begin
          nr_cnt <= nr_cnt + 1'b1;
          inv_rms <= nr_y_new;
          if (nr_cnt == 3'd2) begin
            // Final NR iter just latched (3 total iters: 0,1,2)
            state          <= S_OUTPUT;
            out_cnt        <= '0;
            out_tile_idx   <= '0;
            out_phase_emit <= 1'b0;
            cmp_idx        <= '0;
            emit_idx       <= '0;
            max_lead_pos_r <= '0;
          end
        end
        S_OUTPUT: begin
          if (!out_phase_emit) begin
            // Compute pass: stash product, track max leading bit
            prod_buf[cmp_idx] <= xgi_prod_w;
            if (new_lead_pos > max_lead_pos_r) max_lead_pos_r <= new_lead_pos;
            if (cmp_idx == BFP_TILE-1) begin
              cmp_idx        <= '0;
              out_phase_emit <= 1'b1;
              emit_idx       <= '0;
              // tile_exp_out_r will be captured this cycle for use next cycle
            end else begin
              cmp_idx <= cmp_idx + 1'b1;
            end
          end else begin
            // Emit pass: shift each stashed product, output (mant, exp)
            // Compute shift from max_lead_pos_r (just captured)
            // out_y_mant = shifted prod_buf[emit_idx][15:0]
            begin : emit_blk
              logic signed [55:0] sp;
              logic signed [7:0]  sh;
              sh = $signed({2'b0, max_lead_pos_r}) - 8'sd14;
              if (sh >= 0) sp = prod_buf[emit_idx] >>> sh[5:0];
              else         sp = prod_buf[emit_idx] <<< (-sh[5:0]);
              out_y_mant <= sp[BFP_MANT_W-1:0];
              // Exponent: combination of x_e, g_e, inv_rms e, shift, and the
              // -27 offset; truncate to 8-bit signed.
              if      (out_e_wide >  127) out_y_exp <=  8'sd127;
              else if (out_e_wide < -128) out_y_exp <= -8'sd128;
              else                        out_y_exp <= out_e_wide[BFP_EXP_W-1:0];
              out_valid <= 1'b1;
            end
            out_cnt <= out_cnt + 1'b1;
            if (emit_idx == BFP_TILE-1) begin
              emit_idx       <= '0;
              out_phase_emit <= 1'b0;
              max_lead_pos_r <= '0;
              if (out_cnt == D-1) begin
                state <= S_DONE;
              end
            end else begin
              emit_idx <= emit_idx + 1'b1;
            end
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
