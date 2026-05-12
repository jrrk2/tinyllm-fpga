// rmsnorm.sv — Root-Mean-Square layer norm for SmolLM2 / TinyStories.
//
//   y[i] = x[i] * gamma[i] / sqrt( mean_k(x[k]^2) + eps )
//
// Streaming protocol (one batch of D elements):
//   - pulse `start`
//   - drive `in_x`, `in_gamma`, `in_valid` for D cycles
//   - block buffers x/gamma, computes inv_rms via Newton-Raphson 1/sqrt,
//     then streams D output beats on out_y/out_valid
//   - `done` pulses one cycle after the last out beat
//
// Fixed-point arithmetic (synthesizable — no real / $sqrt):
//   x, gamma : Q1.15 signed 16-bit
//   x*x      : Q2.30 unsigned per term, summed into 64-bit sum_sq
//   mean_sq  : sum_sq >> LOG2D   (D must be a power of 2)
//   v        : mean_sq[31:0] + EPS_Q30  (Q2.30, fits 32 bits)
//   inv_rms  : 1/sqrt(v_real) in Q5.12 unsigned (16-bit)
//              32-entry LUT seed + 3 NR iterations → ≤ ±1 LSB vs numpy
//   y[i]     : sat_q15( x[i] * gamma[i] * inv_rms >>> 27 )

`default_nettype none

module rmsnorm #(
  parameter int          D       = 64,
  parameter logic [31:0] EPS_Q30 = 32'd10737   // round(1e-5 * 2^30)
)(
  input  wire                clk,
  input  wire                rst,

  input  wire                start,
  input  wire signed [15:0]  in_x,
  input  wire signed [15:0]  in_gamma,
  input  wire                in_valid,

  output logic signed [15:0] out_y,
  output logic               out_valid,
  output logic               done
);

  // mean_sq = sum_sq / D.  Implemented as a 64-bit × 32-bit multiply by
  // INV_D = round(2^32 / D), then take bits [63:32].  Works for any D
  // (replaces the previous power-of-two-only shift).
  localparam logic [31:0] INV_D = 32'(((longint'(1)) << 32) / D);

  localparam int CW = $clog2(D + 1);

  // Per-batch storage
  logic signed [15:0] x_mem [0:D-1];
  logic signed [15:0] g_mem [0:D-1];

  // Σ x[k]^2 accumulated in 64 bits (Q2.30 per term, i.e. x_int^2 counts)
  logic [63:0] sum_sq;

  // NR state
  logic [31:0] v_q30;    // mean_sq + eps, Q2.30
  logic [15:0] inv_rms;  // 1/sqrt(v_real) in Q5.12 unsigned
  logic  [2:0] nr_cnt;   // 0: compute v, 1: load seed, 2-4: 3 NR iters

  typedef enum logic [2:0] {S_IDLE, S_LOAD, S_COMPUTE, S_OUTPUT, S_DONE} state_t;
  state_t state;

  logic [CW-1:0] in_cnt;
  logic [CW-1:0] out_cnt;

  // x*x: signed 32-bit product, always non-negative (zero-extend into sum_sq)
  wire signed [31:0] xx_s32 = in_x * in_x;

  // -----------------------------------------------------------------------
  // Combinational: leading-1 (MSB) position of v_q30 [31:0].
  // v_q30 < 2^31, so bit 31 is never set in normal operation.
  // -----------------------------------------------------------------------
  logic [4:0] v_msb;
  always_comb begin : msb_finder
    casez (v_q30)
      32'b1???????????????????????????????: v_msb = 5'd31;
      32'b01??????????????????????????????: v_msb = 5'd30;
      32'b001?????????????????????????????: v_msb = 5'd29;
      32'b0001????????????????????????????: v_msb = 5'd28;
      32'b00001???????????????????????????: v_msb = 5'd27;
      32'b000001??????????????????????????: v_msb = 5'd26;
      32'b0000001?????????????????????????: v_msb = 5'd25;
      32'b00000001????????????????????????: v_msb = 5'd24;
      32'b000000001???????????????????????: v_msb = 5'd23;
      32'b0000000001??????????????????????: v_msb = 5'd22;
      32'b00000000001?????????????????????: v_msb = 5'd21;
      32'b000000000001????????????????????: v_msb = 5'd20;
      32'b0000000000001???????????????????: v_msb = 5'd19;
      32'b00000000000001??????????????????: v_msb = 5'd18;
      32'b000000000000001?????????????????: v_msb = 5'd17;
      32'b0000000000000001????????????????: v_msb = 5'd16;
      32'b00000000000000001???????????????: v_msb = 5'd15;
      32'b000000000000000001??????????????: v_msb = 5'd14;
      32'b0000000000000000001?????????????: v_msb = 5'd13;
      32'b00000000000000000001????????????: v_msb = 5'd12;
      32'b000000000000000000001???????????: v_msb = 5'd11;
      32'b0000000000000000000001??????????: v_msb = 5'd10;
      32'b00000000000000000000001?????????: v_msb = 5'd9;
      32'b000000000000000000000001????????: v_msb = 5'd8;
      32'b0000000000000000000000001???????: v_msb = 5'd7;
      32'b00000000000000000000000001??????: v_msb = 5'd6;
      32'b000000000000000000000000001?????: v_msb = 5'd5;
      32'b0000000000000000000000000001????: v_msb = 5'd4;
      32'b00000000000000000000000000001???: v_msb = 5'd3;
      32'b000000000000000000000000000001??: v_msb = 5'd2;
      32'b0000000000000000000000000000001?: v_msb = 5'd1;
      default:                              v_msb = 5'd0;
    endcase
  end

  // -----------------------------------------------------------------------
  // Combinational: 32-entry LUT → Q5.12 seed for 1/sqrt(v_q30).
  // Entry msb: seed = round( 1/sqrt(1.5 * 2^(msb-30)) * 2^12 ), clip 65535.
  // For msb ≤ 21 inv_rms > 32 overflows Q5.12 → clamp; output saturates.
  // -----------------------------------------------------------------------
  logic [15:0] y_seed;
  always_comb begin : lut_seed
    case (v_msb)
      5'd31: y_seed = 16'd2365;
      5'd30: y_seed = 16'd3344;
      5'd29: y_seed = 16'd4730;
      5'd28: y_seed = 16'd6689;
      5'd27: y_seed = 16'd9459;
      5'd26: y_seed = 16'd13377;
      5'd25: y_seed = 16'd18919;
      5'd24: y_seed = 16'd26755;
      5'd23: y_seed = 16'd37837;
      5'd22: y_seed = 16'd53510;
      default: y_seed = 16'd65535;   // msb ≤ 21: inv_rms > 32, saturate
    endcase
  end

  // -----------------------------------------------------------------------
  // Combinational: Newton-Raphson iteration (one per clock in S_COMPUTE).
  //   y_{n+1} = y_n * (1.5 - 0.5 * v_real * y_n^2)
  //
  // y in Q5.12 (unsigned 16-bit), v in Q2.30 (unsigned 32-bit):
  //   y_sq      = y * y                     Q10.24  32-bit unsigned
  //   vy2       = v * y_sq                  Q12.54  64-bit unsigned
  //   vy2_q30   = vy2 >> 24                 40-bit (Q2.30 of v*y^2)
  //   corr      = C_1P5_Q30 - vy2_q30>>1   signed 33-bit, clamped ≥ 0
  //   y_new     = (y * corr + 2^29) >> 30   16-bit unsigned, clipped
  // -----------------------------------------------------------------------
  localparam logic [31:0] C_1P5_Q30 = 32'h6000_0000; // 1.5 * 2^30

  logic [31:0]        nr_y_sq;
  logic [63:0]        nr_vy2;
  logic [39:0]        nr_vy2_q30;
  logic signed [32:0] nr_corr;
  logic signed [48:0] nr_y_new_full;
  logic [48:0]        nr_biased;
  logic [15:0]        nr_y_new;

  always_comb begin : nr_datapath
    nr_y_sq      = inv_rms * inv_rms;                              // 32-bit
    nr_vy2       = {32'b0, v_q30} * {32'b0, nr_y_sq};             // 64-bit
    nr_vy2_q30   = nr_vy2[63:24];                                  // 40-bit

    // correction = 1.5 - 0.5*v*y^2 in Q2.30, clamped to [0, 1.5_q30]
    if (nr_vy2_q30[39:32] != 8'h00) begin
      // vy2_q30 overflows 32 bits → v*y^2 >> 4 → correction → 0
      nr_corr = 33'sb0;
    end else begin
      nr_corr = $signed({1'b0, C_1P5_Q30})
              - $signed({2'b00, nr_vy2_q30[31:1]});
      if (nr_corr < 0) nr_corr = 33'sb0;
    end

    // y_new = round(y * correction / 2^30); clip to 16-bit unsigned
    nr_y_new_full = $signed({1'b0, inv_rms}) * nr_corr;    // 16 × 33 = 49 bits
    nr_biased     = $unsigned(nr_y_new_full) + 49'h0000_2000_0000; // + 2^29
    // Extract bits [45:30] as the 16-bit Q5.12 result; saturate on overflow
    if (nr_biased[48:46] != 3'b000)
      nr_y_new = 16'hFFFF;
    else
      nr_y_new = nr_biased[45:30];
  end

  // -----------------------------------------------------------------------
  // Output datapath: y[i] = sat_q15( x * gamma * inv_rms >>> 27 )
  //   xg     = x_int * g_int          Q2.30, signed 32-bit
  //   xg_inv = xg * inv_rms_q12       Q7.42, signed 48-bit
  //   y_sh   = xg_inv >>> 27          Q7.15, signed 21-bit
  //   sat    clamp to [-32768, 32767]
  // -----------------------------------------------------------------------
  logic signed [31:0] o_xg;
  logic signed [47:0] o_xg_inv;
  logic signed [20:0] o_y_shift;
  logic signed [15:0] o_y_sat;

  always_comb begin : out_datapath
    // Use full out_cnt — Sonnet's [CW-2:0] slice was a power-of-2-D
    // optimisation that wraps for non-power-of-2 D (e.g. D=576 only
    // addresses indices 0..511).
    o_xg     = $signed(x_mem[out_cnt]) * $signed(g_mem[out_cnt]);
    o_xg_inv = o_xg * $signed({1'b0, inv_rms});   // signed × unsigned → signed 48-bit
    o_y_shift = $signed(o_xg_inv[47:27]);          // arithmetic right shift by 27

    if      (o_y_shift > $signed(21'sh007FFF)) o_y_sat = 16'sh7FFF;
    else if (o_y_shift < $signed(21'shFF8000)) o_y_sat = 16'sh8000;
    else                                        o_y_sat = o_y_shift[15:0];
  end

  // -----------------------------------------------------------------------
  // Main FSM (synchronous)
  //
  // S_COMPUTE phase (uses nr_cnt):
  //   nr_cnt==0: latch v_q30 from sum_sq (all D elements now included);
  //              inv_rms stays 0; v_msb/y_seed become valid next cycle
  //   nr_cnt==1: load y_seed into inv_rms
  //   nr_cnt==2: NR iteration 1
  //   nr_cnt==3: NR iteration 2
  //   nr_cnt==4: NR iteration 3 → inv_rms is final; transition to S_OUTPUT
  // -----------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      state     <= S_IDLE;
      sum_sq    <= '0;
      in_cnt    <= '0;
      out_cnt   <= '0;
      out_valid <= 1'b0;
      done      <= 1'b0;
      out_y     <= '0;
      v_q30     <= '0;
      inv_rms   <= '0;
      nr_cnt    <= '0;
    end else begin
      out_valid <= 1'b0;
      done      <= 1'b0;

      case (state)

        S_IDLE: begin
          if (start) begin
            sum_sq <= '0;
            in_cnt <= '0;
            state  <= S_LOAD;
          end
        end

        S_LOAD: begin
          if (in_valid) begin
            x_mem[in_cnt] <= in_x;
            g_mem[in_cnt] <= in_gamma;
            sum_sq        <= sum_sq + {32'b0, xx_s32};
            in_cnt        <= in_cnt + 1'b1;
            if (in_cnt == CW'(D-1)) begin
              // Transition; sum_sq gains last element at this same edge.
              // v_q30 will be computed in S_COMPUTE from the updated sum_sq.
              inv_rms <= '0;
              nr_cnt  <= '0;
              state   <= S_COMPUTE;
            end
          end
        end

        // nr_cnt 0: capture v from fully-updated sum_sq; v_msb/y_seed valid next cycle
        // nr_cnt 1: load LUT seed into inv_rms
        // nr_cnt 2-4: three NR iterations
        S_COMPUTE: begin
          nr_cnt <= nr_cnt + 1'b1;

          case (nr_cnt)
            3'd0: begin
              // sum_sq now contains all D elements (updated at last S_LOAD edge).
              // mean_sq_q30 = sum_sq * INV_D / 2^32, where INV_D = 2^32/D.
              // Result stays in Q.30.  Watch for overflow: sum_sq is up to
              // D × (2^15)^2 = D × 2^30 ≈ 2^40 for D=1024, well within 64-bit.
              begin
                logic [95:0] prod;
                prod = sum_sq * INV_D;     // 64×32 = 96-bit unsigned
                if (prod[95:64] != '0)
                  v_q30 <= 32'hFFFF_FFFF;  // shouldn't happen — guard anyway
                else
                  v_q30 <= prod[63:32] + EPS_Q30;
              end
            end

            3'd1: begin
              // v_msb and y_seed are combinational from v_q30 latched at nr_cnt==0
              inv_rms <= y_seed;
            end

            3'd2, 3'd3: begin
              inv_rms <= nr_y_new;    // NR iterations 1 and 2
            end

            3'd4: begin
              inv_rms <= nr_y_new;    // NR iteration 3 — final value
              out_cnt <= '0;
              state   <= S_OUTPUT;
`ifdef MICROGPT_RMS_PROBE
              // Each rmsnorm call dumps: v_q30, v_msb, final inv_rms.
              // Use to bisect Python emulator vs RTL when inv_rms saturates.
              $display("RMS_PROBE  v_q30=%0d  v_msb=%0d  inv_rms=%0d",
                       v_q30, v_msb, nr_y_new);
`endif
            end

            default: state <= S_IDLE;
          endcase
        end

        S_OUTPUT: begin
          out_y     <= o_y_sat;
          out_valid <= 1'b1;
          out_cnt   <= out_cnt + 1'b1;
          if (out_cnt == CW'(D-1)) state <= S_DONE;
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
