// swiglu_bfp.sv — block-FP SwiGLU activation.
//
//   y[i] = silu(gate[i]) * up[i]
//        = (gate[i] / (1 + exp(-gate[i]))) * up[i]
//
// SiLU uses the existing 65 536-entry Q1.15 LUT at scale SILU_LUT_SCALE=32
// (same one in generated/silu_lut.hex).  Each block-FP gate element is
// converted to a Q1.15-at-scale-32 LUT index by aligning its mantissa to
// the LUT's fixed scale; if the magnitude exceeds the LUT range, we
// bypass the table (silu(x)=x above, 0 below — see gate_ovf_* below;
// the old ±1-LSB "saturate" comment was wrong for smollm360, whose gate
// negligible vs the calibrated outlier behavior).
//
// Pipeline (streaming, one element per cycle):
//   S_LOAD     accept gate + up tiles, look up silu(gate), multiply by up,
//              stash 32-bit product into a tile-sized buffer + track max-|.|
//   S_EMIT     after each TILE elements, shift each product by (max_lead-14)
//              and emit (m, e) tile.  Output is tile-quantized block-FP.

`include "bfp_format.svh"

`default_nettype none

module swiglu_bfp #(
  parameter int D = 1536,
  // SILU_LUT_SCALE = 32 (matches sim/gen_swiglu_lut.py + existing swiglu.sv)
  parameter int LUT_SCALE_LOG2 = 5
)(
  input  wire                          clk,
  input  wire                          rst,
  input  wire                          start,
  input  wire signed [BFP_MANT_W-1:0]  in_gate_mant,
  input  wire signed [BFP_EXP_W -1:0]  in_gate_exp,
  input  wire signed [BFP_MANT_W-1:0]  in_up_mant,
  input  wire signed [BFP_EXP_W -1:0]  in_up_exp,
  input  wire                          in_valid,
  input  wire                          last_elem,
  output logic                         in_ready,    // 0 while emitting → driver stalls
  output logic signed [BFP_MANT_W-1:0] out_y_mant,
  output logic signed [BFP_EXP_W -1:0] out_y_exp,
  output logic                         out_valid,
  output logic                         done
);

  // in_ready = 1 only in S_LOAD (and after start, before S_DONE)
  // We compute it below using FSM state.

  localparam int CW = $clog2(BFP_TILE);   // 4

  // 65 536-entry Q1.15 LUT loaded from generated/silu_lut.hex
  logic signed [15:0] silu_lut [0:65535];
  initial $readmemh("../generated/silu_lut.hex", silu_lut);

  // -----------------------------------------------------------------------
  // Stage 1: LUT lookup + multiply.
  //
  //   gate_real      = m_gate / 2^15 * 2^e_gate
  //   gate_at_lut15  = gate_real / SILU_LUT_SCALE * 2^15
  //                  = m_gate * 2^(e_gate - LUT_SCALE_LOG2 - 15) * 2^15
  //                  = m_gate * 2^(e_gate - 5)
  //   LUT-index      = sat_int16(gate_at_lut15)
  //   silu_at_lut15  = silu_lut[LUT-index]   (Q1.15 at SILU_LUT_SCALE)
  // -----------------------------------------------------------------------
  // gate_shifted_w = gate at the LUT's fixed scale (Q1.15 @ SILU_LUT_SCALE),
  // kept at full width (48 b) so an out-of-range gate is NOT lost: the LUT
  // only covers [-32, +32), but SmolLM2-360M's layer-3 "massive activation"
  // neuron has gate ≈ 46 (u ≈ 64, mlp ≈ 2930).  The old 16-bit saturation
  // clipped it to silu(32) = 32, scaling that neuron — and through the
  // down-proj every lane of the layer-3 hidden — by 32/46 ≈ 0.70, which
  // skewed all later layers.  Outside the table silu(x) = x (x ≥ 32,
  // sigmoid == 1 to 16-bit precision) and silu(x) = 0 (x ≤ -32), so we
  // bypass the LUT with the unsaturated gate (or 0) in the same Q-format.
  logic signed [47:0]  gate_shifted_w;
  logic signed [BFP_MANT_W-1:0] gate_lut_idx;
  logic                gate_ovf_pos, gate_ovf_neg;
  always_comb begin
    logic signed [BFP_EXP_W:0] shamt;
    shamt = $signed({in_gate_exp[BFP_EXP_W-1], in_gate_exp}) - LUT_SCALE_LOG2;
    // shamt > 0 → left-shift m_gate (real value bigger); negative → right shift
    if (shamt >= 0) gate_shifted_w = 48'($signed(in_gate_mant)) <<<   shamt[4:0];
    else            gate_shifted_w = 48'($signed(in_gate_mant)) >>> (-shamt[4:0]);
    gate_ovf_pos = (gate_shifted_w >  48'sd32767);
    gate_ovf_neg = (gate_shifted_w < -48'sd32768);
    // Saturate the LUT index to int16 (only used when in range)
    if      (gate_ovf_pos)  gate_lut_idx =  16'sd32767;
    else if (gate_ovf_neg)  gate_lut_idx = -16'sd32768;
    else                    gate_lut_idx =  gate_shifted_w[15:0];
  end

  logic signed [BFP_MANT_W-1:0] silu_at_lut;
  logic signed [47:0]           gate_shifted_r;
  logic                         ovf_pos_r, ovf_neg_r;
  always_ff @(posedge clk) begin
    silu_at_lut    <= silu_lut[$unsigned(gate_lut_idx)];
    gate_shifted_r <= gate_shifted_w;
    ovf_pos_r      <= gate_ovf_pos;
    ovf_neg_r      <= gate_ovf_neg;
  end
  // silu operand at LUT scale: table value in range, else x (pos) / 0 (neg).
  logic signed [47:0] silu_w;
  always_comb silu_w = ovf_pos_r ? gate_shifted_r
                     : ovf_neg_r ? 48'sd0
                     :             48'($signed(silu_at_lut));

  // Pipeline up alongside the LUT registered output, so silu_at_lut and
  // up are aligned (1 cycle of LUT latency).
  logic signed [BFP_MANT_W-1:0] up_mant_r;
  logic signed [BFP_EXP_W -1:0] up_exp_r, gate_exp_r;
  logic                         valid_r, last_r;
  always_ff @(posedge clk) begin
    up_mant_r  <= in_up_mant;
    up_exp_r   <= in_up_exp;
    gate_exp_r <= in_gate_exp;
    valid_r    <= in_valid;
    last_r     <= in_valid && last_elem;
  end

  // Compute prod = silu_w * up_mant_r — 64-bit signed (48 × 16).
  // Real value: silu_at_lut15 * up_real = silu_w/2^15 * SILU_LUT_SCALE
  //                                     * m_up/2^15 * 2^e_up
  //           = silu_w * m_up * 2^(e_up + LUT_SCALE_LOG2 - 30)
  // So prod's effective exponent = e_up + LUT_SCALE_LOG2 - 30 (unchanged by
  // the out-of-range bypass, which stays in the same Q-format).
  logic signed [63:0] prod_w;
  always_comb prod_w = silu_w * up_mant_r;

  // -----------------------------------------------------------------------
  // Stage 2: tile-quantize the per-element products.
  //   - Buffer TILE products, track max-leading-bit.
  //   - At end of tile (or on last element), compute shift = max_lead-14,
  //     emit aligned mantissas with one shared exponent.
  // -----------------------------------------------------------------------
  logic signed [63:0]            prod_buf [0:BFP_TILE-1];
  logic [5:0]                    max_lead_r;
  logic [CW-1:0]                 load_idx;
  logic [CW-1:0]                 emit_idx;
  logic                          emit_phase;
  logic [11:0]                   out_cnt;
  logic                          last_seen;
  // Effective tile up_exp (shared) latched when load_idx==0
  logic signed [BFP_EXP_W-1:0]   up_exp_tile_r;

  // Compute new_lead for this product (6 bits since 64-bit signed)
  logic [5:0]  new_lead;
  always_comb begin
    automatic logic [63:0] abs_v;
    abs_v = prod_w[63] ? ($unsigned(~prod_w) + 1'b1) : $unsigned(prod_w);
    new_lead = 6'd0;
    for (int i = 0; i < 64; i++) if (abs_v[i]) new_lead = i[5:0];
  end

  // FSM
  typedef enum logic [1:0] {S_IDLE, S_LOAD, S_EMIT, S_DONE} state_t;
  state_t state;
  // Track in-flight inputs since tile start.  Once TILE inputs have been
  // accepted, deassert ready until emit completes — the LUT pipeline needs
  // to drain into the prod_buf without new inputs overlapping.
  logic [CW:0] in_flight;
  always_comb in_ready = (state == S_LOAD) && (in_flight < BFP_TILE);

  always_ff @(posedge clk) begin
    if (rst) begin
      state        <= S_IDLE;
      load_idx     <= '0;
      emit_idx     <= '0;
      emit_phase   <= 1'b0;
      max_lead_r   <= '0;
      out_cnt      <= '0;
      last_seen    <= 1'b0;
      out_y_mant   <= '0;
      out_y_exp    <= '0;
      out_valid    <= 1'b0;
      done         <= 1'b0;
    end else begin
      out_valid <= 1'b0;
      done      <= 1'b0;
      case (state)
        S_IDLE: begin
          if (start) begin
            state      <= S_LOAD;
            load_idx   <= '0;
            in_flight  <= '0;
            max_lead_r <= '0;
            out_cnt    <= '0;
            last_seen  <= 1'b0;
          end
        end
        S_LOAD: begin
          // Track newly accepted inputs (driver side: in_valid&&in_ready)
          if (in_valid && in_ready) in_flight <= in_flight + 1'b1;
          if (valid_r) begin
            prod_buf[load_idx] <= prod_w;
            if (new_lead > max_lead_r) max_lead_r <= new_lead;
            if (load_idx == '0) up_exp_tile_r <= up_exp_r;
            if (last_r) last_seen <= 1'b1;
            if (load_idx == BFP_TILE - 1 || last_r) begin
              state    <= S_EMIT;
              emit_idx <= '0;
            end else begin
              load_idx <= load_idx + 1'b1;
            end
          end
        end
        S_EMIT: begin
          begin : emit_blk
            automatic logic signed [7:0]  shamt_s;
            automatic logic signed [63:0] sh_w;
            shamt_s = $signed({2'b0, max_lead_r}) - 8'sd14;
            if (shamt_s >= 0) sh_w = prod_buf[emit_idx] >>>   shamt_s[5:0];
            else              sh_w = prod_buf[emit_idx] <<< (-shamt_s[5:0]);
            out_y_mant <= sh_w[BFP_MANT_W-1:0];
            // out_e = e_up + LUT_SCALE_LOG2 - 30 + shamt + 15 = e_up + shamt -10
            begin : exp_blk
              logic signed [BFP_EXP_SUM_W+1:0] e_wide;
              e_wide = $signed({up_exp_tile_r[BFP_EXP_W-1], up_exp_tile_r})
                     + $signed({{(BFP_EXP_SUM_W-6){shamt_s[7]}}, shamt_s})
                     - 10;
              if      (e_wide >  127) out_y_exp <=  8'sd127;
              else if (e_wide < -128) out_y_exp <= -8'sd128;
              else                    out_y_exp <=  e_wide[BFP_EXP_W-1:0];
            end
            out_valid <= 1'b1;
            out_cnt   <= out_cnt + 1'b1;
            if (emit_idx == BFP_TILE - 1) begin
              emit_idx   <= '0;
              max_lead_r <= '0;
              load_idx   <= '0;
              in_flight  <= '0;
              if (last_seen) state <= S_DONE;
              else           state <= S_LOAD;
            end else if (out_cnt + 1 == D) begin
              // Partial final tile (D % TILE != 0): emit only what we loaded.
              state <= S_DONE;
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
