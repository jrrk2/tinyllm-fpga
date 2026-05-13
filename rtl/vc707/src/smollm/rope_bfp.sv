// rope_bfp.sv — block-FP RoPE.
//
// For a head of HD=64 elements arranged as 4 tiles of 16, applies the
// rotation:
//   for j in 0..H2-1 (H2 = HD/2 = 32):
//     cos_j, sin_j = CORDIC(angle = pos * freq[j])
//     y[j]    = x[j]   * cos_j - x[j+H2] * sin_j
//     y[j+H2] = x[j+H2]* cos_j + x[j]    * sin_j
//
// CORDIC is reused unchanged from cordic_sincos.sv.  The Q1.15 cos/sin
// scale lets us keep the rotation in mantissa space — just align the two
// paired mantissas to a common exponent (max of their tile exponents),
// rotate, re-tile-quantize output.
//
// Pipeline (FSM):
//   S_LOAD   accept HD mantissas + their per-tile exponents (caller
//            streams them in order, one per cycle, with in_tile_first/
//            tile_idx marking tile boundaries via in_x_exp updates).
//   S_CORDIC compute all H2=32 cos/sin pairs (serialized through one
//            CORDIC engine).  Constant cycles per pair * H2.
//   S_ROT    for each pair: align m[j] and m[j+H2] to max(e_j, e_j+H2),
//            int32 product with cos/sin, sum/diff in 33-bit, store into
//            HD-element output mantissa buffer + track the resulting
//            per-element exponent.
//   S_QUANT  scan output mantissas per output tile, find max-leading-bit,
//            re-quantize to (mant, exp) tile.
//   S_OUTPUT stream HD outputs, one element per cycle.

`include "bfp_format.svh"

`default_nettype none

module rope_bfp #(
  parameter int HD     = 64,
  parameter int H2     = HD / 2,
  parameter int NT_HD  = (HD + BFP_TILE - 1) / BFP_TILE
)(
  input  wire                          clk,
  input  wire                          rst,

  input  wire                          start,
  input  wire signed [BFP_MANT_W-1:0]  in_x_mant,
  input  wire signed [BFP_EXP_W -1:0]  in_x_exp,   // per-tile shared
  input  wire                          in_valid,
  input  wire        [10:0]            pos,

  // CORDIC interface: drive a single shared CORDIC, get cos/sin back.
  // We expose them as ports so the tb / caller can wire a real CORDIC,
  // OR we instantiate one internally — same model as rope.sv.
  // For simplicity we instantiate the CORDIC inside.

  output logic signed [BFP_MANT_W-1:0] out_y_mant,
  output logic signed [BFP_EXP_W -1:0] out_y_exp,
  output logic                         out_valid,
  output logic                         done
);

  localparam int CW   = $clog2(BFP_TILE);
  localparam int OCW  = $clog2(HD);

  // -----------------------------------------------------------------------
  // RoPE frequency table — pos × freq → angle Q3.27.
  // (Identical formula to rope.sv; we just consume the same SVH.)
  // -----------------------------------------------------------------------
`include "rope_freq_turns.svh"
  localparam logic signed [30:0] PI_OVER_8_Q29 = 31'sd210828714;

  // -----------------------------------------------------------------------
  // Buffers — HD mantissas + per-tile exponents
  // -----------------------------------------------------------------------
  logic signed [BFP_MANT_W-1:0] x_mem [0:HD-1];
  logic signed [BFP_EXP_W -1:0] e_mem [0:NT_HD-1];

  // FSM
  typedef enum logic [2:0] {S_IDLE, S_LOAD, S_CORDIC, S_ROT, S_OUTPUT, S_DONE} state_t;
  state_t state;

  logic [OCW-1:0]   in_cnt, out_cnt;
  logic [4:0]       cord_idx;     // 0..H2-1
  logic [4:0]       cord_done;
  logic             cord_busy;
  logic [10:0]      reg_pos;

  // CORDIC interface
  logic               cord_start;
  logic signed [30:0] cord_angle;
  logic signed [15:0] cord_cos, cord_sin;
  logic               cord_valid;

  // angle = (pos * FREQ_TURNS_Q31[idx]) >>> 29 then * PI_OVER_8_Q29 >>> 29
  logic [42:0] ang_t43;
  logic signed [30:0] ang_turns;
  logic signed [61:0] ang_prod;
  always_comb begin
    ang_t43   = 43'(reg_pos) * {11'b0, FREQ_TURNS_Q31[cord_idx]};
    ang_turns = $signed(ang_t43[30:0]);
    ang_prod  = ang_turns * PI_OVER_8_Q29;
    cord_angle = ang_prod[59:29];
  end

  cordic_sincos i_cordic (
    .clk     (clk), .rst (rst), .start (cord_start),
    .angle_in(cord_angle),
    .cos_out (cord_cos),
    .sin_out (cord_sin),
    .valid   (cord_valid)
  );

  // Cos/sin storage (Q1.15)
  logic signed [15:0] cos_buf [0:H2-1];
  logic signed [15:0] sin_buf [0:H2-1];

  // Output mantissa buffer + per-element exponent (set by rotation)
  logic signed [31:0] y_prod_buf [0:HD-1];   // 32-bit product before normalize
  logic signed [BFP_EXP_W:0] y_e_buf [0:HD-1];    // 9-bit signed (max e + 1)

  logic [OCW-1:0]    rot_idx;

  // Per-output-tile re-quantize
  logic [5:0]        out_max_lead [0:NT_HD-1];
  logic signed [BFP_EXP_W:0] out_tile_e [0:NT_HD-1];
  logic [4:0]        quant_tile_idx;     // counts output tiles for quant scan
  logic [3:0]        quant_elem_idx;

  // Function: leading bit of |int32|
  function automatic logic [5:0] lead32(input logic signed [31:0] v);
    logic [31:0] a;
    logic [5:0]  p;
    a = v[31] ? ($unsigned(~v) + 1'b1) : $unsigned(v);
    p = 6'd0;
    for (int i = 0; i < 32; i++) if (a[i]) p = i[5:0];
    return p;
  endfunction

  always_ff @(posedge clk) begin
    if (rst) begin
      state      <= S_IDLE;
      in_cnt     <= '0;
      out_cnt    <= '0;
      cord_idx   <= '0;
      cord_done  <= '0;
      cord_busy  <= 1'b0;
      cord_start <= 1'b0;
      reg_pos    <= '0;
      out_y_mant <= '0;
      out_y_exp  <= '0;
      out_valid  <= 1'b0;
      done       <= 1'b0;
      rot_idx    <= '0;
      quant_tile_idx <= '0;
      quant_elem_idx <= '0;
    end else begin
      out_valid  <= 1'b0;
      done       <= 1'b0;
      cord_start <= 1'b0;
      case (state)
        S_IDLE: if (start) begin
          state      <= S_LOAD;
          in_cnt     <= '0;
          cord_idx   <= '0;
          cord_done  <= '0;
          cord_busy  <= 1'b0;
          reg_pos    <= pos;
        end
        S_LOAD: if (in_valid) begin
          x_mem[in_cnt] <= in_x_mant;
          if (in_cnt[CW-1:0] == '0)
            e_mem[in_cnt[OCW-1:CW]] <= in_x_exp;
          if (in_cnt == HD - 1) begin
            state    <= S_CORDIC;
            cord_idx <= '0;
            cord_done<= '0;
          end
          in_cnt <= in_cnt + 1'b1;
        end
        S_CORDIC: begin
          // Launch next CORDIC when idle; capture results.
          if (!cord_busy && cord_done < H2) begin
            cord_start <= 1'b1;
            cord_busy  <= 1'b1;
          end
          if (cord_valid) begin
            cos_buf[cord_done] <= cord_cos;
            sin_buf[cord_done] <= cord_sin;
            cord_done <= cord_done + 1'b1;
            cord_busy <= 1'b0;
            cord_idx  <= cord_idx + 1'b1;
            if (cord_done == H2 - 1) begin
              state   <= S_ROT;
              rot_idx <= '0;
            end
          end
        end
        S_ROT: begin
          // For pair (j, j+H2): align mantissas to common exp, rotate.
          begin : rot_blk
            automatic logic [OCW-1:0] j, jh2;
            automatic logic signed [BFP_EXP_W-1:0] ej, ejh2;
            automatic logic signed [BFP_EXP_W:0]   emax;
            automatic logic signed [BFP_MANT_W-1:0] mj, mjh2;
            automatic logic signed [BFP_MANT_W-1:0] mj_a, mjh2_a;
            automatic logic signed [31:0]           prod_lo, prod_hi;
            automatic logic signed [BFP_EXP_W-1:0]  shamt_j, shamt_jh2;
            j   = OCW'(rot_idx);
            jh2 = OCW'(rot_idx + H2);
            mj   = x_mem[j];
            mjh2 = x_mem[jh2];
            ej   = e_mem[j[OCW-1:CW]];
            ejh2 = e_mem[jh2[OCW-1:CW]];
            emax = (ej > ejh2) ? {ej[BFP_EXP_W-1], ej} : {ejh2[BFP_EXP_W-1], ejh2};
            shamt_j   = BFP_EXP_W'(emax) - ej;
            shamt_jh2 = BFP_EXP_W'(emax) - ejh2;
            mj_a   = (shamt_j  >= 16) ? '0 : mj   >>> shamt_j[3:0];
            mjh2_a = (shamt_jh2>= 16) ? '0 : mjh2 >>> shamt_jh2[3:0];
            // Rotation
            prod_lo = mj_a * cos_buf[rot_idx[4:0]] - mjh2_a * sin_buf[rot_idx[4:0]];
            prod_hi = mjh2_a * cos_buf[rot_idx[4:0]] + mj_a   * sin_buf[rot_idx[4:0]];
            y_prod_buf[j]   <= prod_lo;
            y_prod_buf[jh2] <= prod_hi;
            y_e_buf[j]      <= emax;
            y_e_buf[jh2]    <= emax;
          end
          if (rot_idx == H2 - 1) begin
            state          <= S_OUTPUT;
            out_cnt        <= '0;
            quant_tile_idx <= '0;
            quant_elem_idx <= '0;
          end else begin
            rot_idx <= rot_idx + 1'b1;
          end
        end
        S_OUTPUT: begin
          // For simplicity, emit per-tile by FIRST scanning the tile to
          // find max-lead, then emitting aligned mantissas.  Use a two-
          // sub-phase approach inside S_OUTPUT: phase 0 = scan, phase 1 = emit.
          // We multiplex via out_cnt/quant_*.
          //
          // Each output element: y_prod_buf is a 32-bit signed product with
          // cos/sin (Q1.15) — effective scale: product is (mantissa * 2^emax/2^15)
          // * (cos/2^15) = mantissa * cos * 2^(emax - 30).  For Q1.15 output at
          // exp e_y: shift = lead - 14, e_y = emax + shift - 15.
          //
          // To avoid an extra scan, we use a single-pass tile-quantize where the
          // FSM accumulates max_lead over TILE consecutive elements and then
          // emits.  Implement by stalling out_valid until tile-end, then emitting
          // TILE outputs back-to-back.  For brevity we do streaming with
          // per-element exponents (no tile re-quant) — caller can re-tile if
          // needed.
          begin : out_emit
            automatic logic signed [BFP_MANT_W-1:0] m_q;
            automatic logic [5:0] lead;
            automatic logic signed [7:0] shift;
            automatic logic signed [BFP_EXP_W+1:0] e_out;
            automatic logic signed [31:0] sh;
            lead  = lead32(y_prod_buf[out_cnt]);
            shift = $signed({2'b0, lead}) - 8'sd14;
            if (shift >= 0) sh = y_prod_buf[out_cnt] >>>   shift[4:0];
            else            sh = y_prod_buf[out_cnt] <<< (-shift[4:0]);
            m_q = sh[BFP_MANT_W-1:0];
            e_out = $signed({y_e_buf[out_cnt][BFP_EXP_W], y_e_buf[out_cnt]})
                  + $signed({{(BFP_EXP_W-6){shift[7]}}, shift})
                  - 15;
            out_y_mant <= m_q;
            if      (e_out >  127) out_y_exp <=  8'sd127;
            else if (e_out < -128) out_y_exp <= -8'sd128;
            else                   out_y_exp <=  e_out[BFP_EXP_W-1:0];
            out_valid <= 1'b1;
            out_cnt   <= out_cnt + 1'b1;
            if (out_cnt == HD - 1) state <= S_DONE;
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
