// softmax_q15.sv — N-element softmax over a Q1.15 vector.
//
//   y_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
//
// Synthesizable implementation using a precomputed exp LUT and a
// Newton-Raphson reciprocal for 1/sum_e.  No `real`, no `$exp`.
//
// Pipeline stages:
//   S_LOAD   (N cycles): buffer x_buf[k], track running signed max.
//   S_EXP    (N cycles): LUT-exp each element, accumulate sum_e.
//   S_RECIP  (6 cycles): 4-iter Newton-Raphson reciprocal (1 seed + 4 iter cycles + 1 latch).
//   S_OUTPUT (N cycles): e_buf[k] * inv_sum >> 17, saturate to Q1.15.
//
// LUT: ../generated/exp_lut.hex  (1024 x 16-bit unsigned Q1.15).
//   index = clamp( signed_shift_right(diff17, 8) + 1023, 0, 1023 )
//   diff17 = {x_buf[k][15], x_buf[k]} - {max_r[15], max_r}  (17-bit signed)
//
// Newton-Raphson reciprocal (all fixed-point, 64-bit intermediates):
//   y0 = 2^(31 - msb(sum_e))   (lower-bound seed, guaranteed < true reciprocal)
//   y_{n+1} = y_n * (2^33 - sum_e * y_n) >> 32  (standard NR in Q0.32 notation)
//   After 4 iterations converges to within 1 LSB for sum_e in [2^9, 2^21].
//
// Output multiply:
//   y[k] = round( e_buf[k] * inv_sum >> 17 ), clamped to [0, 32767].
//   inv_sum represents (1/sum_e) * 2^32, so:
//     e[k] * inv_sum / 2^17 = (e[k]/32768) / (sum_e/32768) * 32768 = y_q15.

`default_nettype none

module softmax_q15 #(
  parameter int N = 64
)(
  input  wire                clk,
  input  wire                rst,

  input  wire                start,
  input  wire signed [15:0]  in_x,
  input  wire                in_valid,

  output logic signed [15:0] out_y,
  output logic               out_valid,
  output logic               done
);

  // N=64 needs 6-bit array index (0..63) and 7-bit counter (0..64).
  localparam int AW = $clog2(N);      // array index width (6 for N=64)
  localparam int CW = $clog2(N+1);   // counter width     (7 for N=64)

  // -----------------------------------------------------------------------
  // Exp LUT: 1024 x 16-bit unsigned Q1.15.
  // LUT[1023] = 32768 represents exp(0) = 1.0 (uint16 OK).
  // -----------------------------------------------------------------------
  logic [15:0] exp_lut [0:1023];
  initial $readmemh("../generated/exp_lut.hex", exp_lut);

  // -----------------------------------------------------------------------
  // Storage arrays (use AW-bit index to avoid WIDTHTRUNC warnings)
  // -----------------------------------------------------------------------
  logic signed [15:0] x_buf [0:N-1];   // buffered signed Q1.15 inputs
  logic        [15:0] e_buf [0:N-1];   // exp LUT outputs (unsigned Q1.15)

  // -----------------------------------------------------------------------
  // FSM states
  // -----------------------------------------------------------------------
  typedef enum logic [2:0] {
    S_IDLE, S_LOAD, S_EXP, S_RECIP, S_OUTPUT, S_DONE
  } state_t;
  state_t state;

  // -----------------------------------------------------------------------
  // Counters (CW-bit wide to count 0..N)
  // -----------------------------------------------------------------------
  logic [CW-1:0] in_cnt;
  logic [CW-1:0] exp_cnt;
  logic [CW-1:0] out_cnt;
  logic [2:0]    nr_cnt;     // 0..5

  // -----------------------------------------------------------------------
  // Running max and exp-sum accumulator
  // -----------------------------------------------------------------------
  logic signed [15:0] max_r;
  logic [31:0]        sum_e;   // up to N*32768 = 2^21, fits in 22 bits

  // -----------------------------------------------------------------------
  // Newton-Raphson state
  // -----------------------------------------------------------------------
  logic [31:0] y_nr;      // current estimate (represents inv/2^32)
  logic [31:0] inv_sum;   // final latched reciprocal

  // -----------------------------------------------------------------------
  // Combinational: MSB finder (highest set bit in sum_e) for NR seed.
  //
  // Loop ASCENDING from bit 0 to bit 31 so the last match is the highest bit.
  // -----------------------------------------------------------------------
  logic [4:0]  msb_pos;
  logic [31:0] y0_seed;

  always_comb begin : find_msb
    msb_pos = 5'd0;
    for (int i = 0; i <= 31; i++)     // ascending: last match = highest set bit
      if (sum_e[i]) msb_pos = 5'(unsigned'(i));
    // y0 = 2^(31 - msb_pos): guaranteed <= true reciprocal.
    // (sum_e is in [2^msb, 2^(msb+1)), so true_inv is in (2^(31-msb-1), 2^(31-msb)];
    // y0 = 2^(31-msb) is the upper bound and equals the true inv for exact powers of 2.)
    // Actually y0 = 2^(31-msb) is an upper bound, not lower bound. Hmm -- let's recalculate.
    // sum_e in [2^msb, 2^(msb+1)): true_inv = 2^32/sum_e in (2^(32-msb-1), 2^(32-msb)].
    // y0 = 2^(31-msb) = 2^(32-msb-1): this is the LOWER bound. Good - NR stays monotone.
    y0_seed = 32'(1'b1) << (5'd31 - msb_pos);
  end

  // -----------------------------------------------------------------------
  // Combinational: one NR iteration.
  //   y_{n+1} = y_n * (2^33 - sum_e * y_n) >> 32
  //
  // Widths (worst case for N=64):
  //   sum_e <= 2^21, y_nr <= 2^17
  //   nr_prod = sum_e * y_nr <= 2^38             -> 64 bits OK
  //   nr_diff = 2^33 - nr_prod; 2^33 = 64'h200000000, diff > 0 (y_nr is lower bound)
  //   nr_next = y_nr * nr_diff <= 2^17 * 2^33 = 2^50  -> 64 bits OK
  // -----------------------------------------------------------------------
  /* verilator lint_off UNUSEDSIGNAL */
  logic [63:0] nr_prod;
  logic [63:0] nr_diff;
  logic [63:0] nr_next;
  /* verilator lint_on UNUSEDSIGNAL */

  always_comb begin : nr_comb
    nr_prod = {32'd0, sum_e} * {32'd0, y_nr};
    nr_diff = 64'h200000000 - nr_prod;      // 2^33 - sum_e*y_nr (always positive)
    nr_next = {32'd0, y_nr} * nr_diff;      // y_nr * (2^33 - sum_e*y_nr)
  end

  // -----------------------------------------------------------------------
  // Combinational: LUT index for exp_cnt.
  //   diff17 = sign_extend(x_buf[k]) - sign_extend(max_r)  (17-bit signed)
  //   index  = (diff17 >>> 8) + 1023, clamped to [0, 1023]
  // -----------------------------------------------------------------------
  logic signed [16:0] diff17;
  logic signed [16:0] raw_idx17;
  logic [9:0]         lut_idx;

  always_comb begin : lut_index
    // Sign-extend 16-bit to 17-bit before subtraction.
    diff17     = {x_buf[exp_cnt[AW-1:0]][15], x_buf[exp_cnt[AW-1:0]]}
               - {max_r[15], max_r};
    raw_idx17  = (diff17 >>> 8) + 17'sd1023;
    if (raw_idx17 < 17'sd0)
      lut_idx = 10'd0;
    else if (raw_idx17 > 17'sd1023)
      lut_idx = 10'd1023;
    else
      lut_idx = raw_idx17[9:0];
  end

  // -----------------------------------------------------------------------
  // Combinational: output multiply for out_cnt.
  //   out_prod   = e_buf[k](16-bit unsigned) * inv_sum(32-bit) = 48-bit
  //   out_shifted = round(out_prod >> 17) = out_prod[47:17] + out_prod[16]
  //
  // e_buf max = 32768 = 2^15, inv_sum max = 2^17 (when sum_e = 2^15, i.e. 1 element at max)
  // out_prod max = 2^15 * 2^17 = 2^32 -> out_prod[47:17] max = 2^15 = 32768.
  // After rounding (+ bit 16), max = 32769.  Saturate to 32767 before output.
  // -----------------------------------------------------------------------
  /* verilator lint_off UNUSEDSIGNAL */
  logic [47:0] out_prod;      // lower 17 bits used for rounding only
  /* verilator lint_on UNUSEDSIGNAL */
  logic [31:0] out_shifted;   // 32-bit to safely hold rounded result before saturation

  always_comb begin : out_mul
    out_prod    = {32'd0, e_buf[out_cnt[AW-1:0]]} * {16'd0, inv_sum};
    out_shifted = {1'b0, out_prod[47:17]} + {31'd0, out_prod[16]};  // round
  end

  // -----------------------------------------------------------------------
  // Main FSM
  // -----------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      state     <= S_IDLE;
      in_cnt    <= '0;
      exp_cnt   <= '0;
      out_cnt   <= '0;
      nr_cnt    <= '0;
      out_valid <= 1'b0;
      done      <= 1'b0;
      out_y     <= '0;
      max_r     <= 16'sh8000;   // minimum signed Q1.15
      sum_e     <= '0;
      y_nr      <= '0;
      inv_sum   <= '0;
    end else begin
      out_valid <= 1'b0;
      done      <= 1'b0;

      case (state)

        // ------------------------------------------------------------------
        S_IDLE: begin
          if (start) begin
            in_cnt <= '0;
            max_r  <= 16'sh8000;
            state  <= S_LOAD;
          end
        end

        // ------------------------------------------------------------------
        // S_LOAD: buffer inputs and track running maximum (signed compare).
        // ------------------------------------------------------------------
        S_LOAD: begin
          if (in_valid) begin
            x_buf[in_cnt[AW-1:0]] <= in_x;
            if (in_x > max_r)
              max_r <= in_x;
            in_cnt <= in_cnt + 1'b1;
            if (in_cnt == CW'(N-1)) begin
              exp_cnt <= '0;
              sum_e   <= '0;
              state   <= S_EXP;
            end
          end
        end

        // ------------------------------------------------------------------
        // S_EXP: LUT-lookup and accumulate.
        //   diff17, lut_idx, exp_lut[lut_idx] are all combinational from exp_cnt.
        // ------------------------------------------------------------------
        S_EXP: begin
          e_buf[exp_cnt[AW-1:0]] <= exp_lut[lut_idx];
          sum_e                  <= sum_e + {16'd0, exp_lut[lut_idx]};
          exp_cnt                <= exp_cnt + 1'b1;
          if (exp_cnt == CW'(N-1)) begin
            nr_cnt <= 3'd0;
            state  <= S_RECIP;
          end
        end

        // ------------------------------------------------------------------
        // S_RECIP: Newton-Raphson reciprocal.
        //   Cycle 0: latch seed y0 = 2^(31 - msb(sum_e))
        //   Cycles 1-4: apply NR iteration from combinational nr_next.
        //   Cycle 5: (implicit) inv_sum was latched at end of cycle 4.
        // ------------------------------------------------------------------
        S_RECIP: begin
          if (nr_cnt == 3'd0) begin
            y_nr   <= y0_seed;
            nr_cnt <= 3'd1;
          end else begin
            // y_{n+1} = y_n * (2^33 - sum_e*y_n) >> 32
            y_nr   <= nr_next[63:32];
            nr_cnt <= nr_cnt + 3'd1;
            if (nr_cnt == 3'd4) begin
              inv_sum <= nr_next[63:32];
              out_cnt <= '0;
              state   <= S_OUTPUT;
            end
          end
        end

        // ------------------------------------------------------------------
        // S_OUTPUT: emit saturated Q1.15 for each element.
        //   out_shifted is combinational from (e_buf[out_cnt], inv_sum).
        // ------------------------------------------------------------------
        S_OUTPUT: begin
          // Saturate: both 0 (if e=0 or inv=0) and >32767 → clamp.
          if (out_shifted > 32'd32767)
            out_y <= 16'd32767;
          else
            out_y <= out_shifted[15:0];
          out_valid <= 1'b1;
          out_cnt   <= out_cnt + 1'b1;
          if (out_cnt == CW'(N-1))
            state <= S_DONE;
        end

        // ------------------------------------------------------------------
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
