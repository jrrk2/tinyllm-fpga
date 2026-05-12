// cordic_sincos.sv — 16-iteration CORDIC sine/cosine in fixed point.
//
// Replaces the 256 KB rope_cos/rope_sin LUT with ~256 bytes of constant
// arctangent / scale ROM and 16 cycles of shift-add iteration.
//
// Interface
//   angle_in  : signed Q3.27, range [-8.0, +8.0).  Caller is responsible
//               for reducing huge angles modulo 2π to this range.  We use
//               Q3.27 (not Q3.13) so the quadrant fold doesn't bottleneck
//               accuracy at the caller's input quantisation.
//   cos_out   : signed Q1.15 (+1.0 ≈ 32767, -1.0 = -32768)
//   sin_out   : signed Q1.15
//
// Latency: 1 cycle for quadrant fold + 16 cycles for the iteration =
//          17 cycles after `start`.  `valid` pulses once when result is ready.
//
// Internal state runs in Q3.27 (31-bit signed) throughout, then rounded
// to Q1.15 at the output.
//
// References
//   J. Volder, "The CORDIC trigonometric computing technique" (1959).
//   Constants verified against `numpy.arctan(2.0 ** -i)`.

`default_nettype none

module cordic_sincos (
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 start,
  input  wire signed [30:0]   angle_in,    // Q3.27, range [-8, +8)
  output logic signed [15:0]  cos_out,     // Q1.15
  output logic signed [15:0]  sin_out,     // Q1.15
  output logic                valid
);

  // Q3.27 constants (caller's full-precision angle space)
  localparam logic signed [30:0] PI_HALF_Q27 =  31'sd 210828714;  // round(π/2 * 2^27)
  localparam logic signed [30:0] PI_Q27      =  31'sd 421657428;  // round(π   * 2^27)

  // CORDIC scale factor K_inv = 1/∏ √(1+2^-2i) over 16 iters.
  // python: round(1/prod(sqrt(1+4**-i) for i in range(16)) * 2**27) = 81504109
  localparam logic signed [30:0] K_Q3_27 = 31'sd81504109;

  // arctan(2^-i) in Q3.27, for i=0..15.
  // python: round(atan(2.0**-i) * 2**27) — verified.
  localparam logic signed [30:0] ATAN_Q3_27 [0:15] = '{
    31'sd105414357,  // i= 0   atan(2^-0)
    31'sd 62229729,  // i= 1   atan(2^-1)
    31'sd 32880480,  // i= 2   atan(2^-2)
    31'sd 16690645,  // i= 3   atan(2^-3)
    31'sd  8377711,  // i= 4
    31'sd  4192939,  // i= 5
    31'sd  2096981,  // i= 6
    31'sd  1048555,  // i= 7
    31'sd   524285,  // i= 8
    31'sd   262144,  // i= 9
    31'sd   131072,  // i=10
    31'sd    65536,  // i=11
    31'sd    32768,  // i=12
    31'sd    16384,  // i=13
    31'sd     8192,  // i=14
    31'sd     4096   // i=15
  };

  // -----------------------------------------------------------------
  // FSM
  // -----------------------------------------------------------------
  typedef enum logic [1:0] {S_IDLE, S_FOLD, S_ITER, S_DONE} state_t;
  state_t state;
  logic [4:0] iter;             // 0..15

  // CORDIC working state in Q3.27 (signed 31-bit).
  logic signed [30:0] x_r, y_r, z_r;

  // Quadrant fold: if |angle| > π/2 we transform x_init / sign_cos so that
  // the CORDIC core only ever rotates a residual in [-π/2, π/2].
  //   angle' = angle              if |angle| ≤ π/2
  //   angle' = π   - angle        if  angle  >  π/2   (cos negate)
  //   angle' = -π  - angle        if  angle  < -π/2   (cos negate)
  logic                cos_neg;

  // -----------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      state    <= S_IDLE;
      iter     <= 5'd0;
      x_r      <= '0;
      y_r      <= '0;
      z_r      <= '0;
      cos_neg  <= 1'b0;
      cos_out  <= '0;
      sin_out  <= '0;
      valid    <= 1'b0;
    end else begin
      valid <= 1'b0;
      case (state)
        S_IDLE: begin
          if (start) state <= S_FOLD;
        end

        S_FOLD: begin
          // Compute z_init (residual angle in [-π/2, π/2]) and cos_neg.
          // Done in Q3.27 throughout — no precision loss vs caller input.
          if (angle_in > PI_HALF_Q27) begin
            z_r     <= PI_Q27 - angle_in;
            cos_neg <= 1'b1;
          end else if (angle_in < -PI_HALF_Q27) begin
            z_r     <= -PI_Q27 - angle_in;
            cos_neg <= 1'b1;
          end else begin
            z_r     <= angle_in;
            cos_neg <= 1'b0;
          end
          x_r   <= K_Q3_27;     // x_0 = K (post-iteration ‖xy‖ = 1)
          y_r   <= '0;
          iter  <= 5'd0;
          state <= S_ITER;
        end

        S_ITER: begin
          // d = sign(z_r): +1 if z_r ≥ 0, -1 otherwise.
          // x_{n+1} = x_n - d·(y_n >> i)
          // y_{n+1} = y_n + d·(x_n >> i)
          // z_{n+1} = z_n - d·atan(2^-i)
          begin
            logic signed [30:0] x_shift, y_shift;
            x_shift = x_r >>> iter;
            y_shift = y_r >>> iter;
            if (z_r >= 0) begin
              x_r <= x_r - y_shift;
              y_r <= y_r + x_shift;
              z_r <= z_r - ATAN_Q3_27[iter];
            end else begin
              x_r <= x_r + y_shift;
              y_r <= y_r - x_shift;
              z_r <= z_r + ATAN_Q3_27[iter];
            end
          end
          if (iter == 5'd15) state <= S_DONE;
          iter <= iter + 5'd1;
        end

        S_DONE: begin
          // Round Q3.27 → Q1.15 (>> 12, with rounding bias 2^11).
          // Saturate at ±32767 — the +1.0 case (angle≈0) needs it because
          // x_r can land at exactly K * (vector at 0) ≈ 1.0 in Q3.27.
          begin
            logic signed [30:0] x_round, y_round;
            logic signed [18:0] x_shift, y_shift;
            x_round = x_r + 31'sd2048;
            y_round = y_r + 31'sd2048;
            x_shift = x_round >>> 12;       // 19-bit signed
            y_shift = y_round >>> 12;
            // Saturate
            if      (x_shift >  19'sd32767)  cos_out <= cos_neg ? -16'sd32767 :  16'sd32767;
            else if (x_shift < -19'sd32768)  cos_out <= cos_neg ?  16'sd32767 : -16'sd32768;
            else                             cos_out <= cos_neg ? -x_shift[15:0] : x_shift[15:0];
            if      (y_shift >  19'sd32767)  sin_out <=  16'sd32767;
            else if (y_shift < -19'sd32768)  sin_out <= -16'sd32768;
            else                             sin_out <= y_shift[15:0];
          end
          valid <= 1'b1;
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
