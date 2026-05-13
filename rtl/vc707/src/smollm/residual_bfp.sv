// residual_bfp.sv — element-wise block-FP add.
//
//   y[k] = a[k] + b[k]
//
// Per-tile pipeline:
//   1. Accept TILE elements of a + b (with per-tile exponents).
//   2. Align mantissas of a and b to max(e_a, e_b); compute sums (17-bit
//      signed); buffer in tile-sized RAM along with max-|sum| tracking.
//   3. Once tile is loaded, shift each mantissa by (max_lead - 14) and
//      emit (mant, exp) tile.  Output exponent = max(e_a, e_b) + shift.

`include "bfp_format.svh"

`default_nettype none

module residual_bfp #(
  parameter int D = 576
)(
  input  wire                          clk,
  input  wire                          rst,
  input  wire                          start,
  input  wire signed [BFP_MANT_W-1:0]  in_a_mant,
  input  wire signed [BFP_EXP_W -1:0]  in_a_exp,
  input  wire signed [BFP_MANT_W-1:0]  in_b_mant,
  input  wire signed [BFP_EXP_W -1:0]  in_b_exp,
  input  wire                          in_valid,
  input  wire                          last_elem,
  output logic                         in_ready,
  output logic signed [BFP_MANT_W-1:0] out_y_mant,
  output logic signed [BFP_EXP_W -1:0] out_y_exp,
  output logic                         out_valid,
  output logic                         done
);

  localparam int CW = $clog2(BFP_TILE);

  // Per-tile sum buffer (17-bit signed each, TILE entries) + max-lead bit
  logic signed [BFP_MANT_W:0]  sum_buf [0:BFP_TILE-1];
  logic [4:0]                   max_lead_r;
  logic signed [BFP_EXP_W:0]    tile_e_out_r;
  logic [CW-1:0]                load_idx, emit_idx;
  logic                          emit_phase;
  logic [11:0]                   total_cnt;
  logic                          last_seen;
  logic [CW:0]                   in_flight;

  typedef enum logic [1:0] {S_IDLE, S_LOAD, S_EMIT, S_DONE} state_t;
  state_t state;
  always_comb in_ready = (state == S_LOAD) && (in_flight < BFP_TILE);

  // Compute aligned sum and its leading bit (within tile)
  logic signed [BFP_EXP_W:0]    e_max;
  logic signed [BFP_EXP_W:0]    shamt_a, shamt_b;
  logic signed [BFP_MANT_W-1:0] a_a, b_a;
  logic signed [BFP_MANT_W:0]   sum_w;
  logic [4:0]                   sum_lead;
  always_comb begin
    e_max   = (in_a_exp > in_b_exp) ? {in_a_exp[BFP_EXP_W-1], in_a_exp}
                                     : {in_b_exp[BFP_EXP_W-1], in_b_exp};
    shamt_a = e_max - {in_a_exp[BFP_EXP_W-1], in_a_exp};
    shamt_b = e_max - {in_b_exp[BFP_EXP_W-1], in_b_exp};
    a_a     = (shamt_a >= 16) ? 16'sd0 : ($signed(in_a_mant) >>> shamt_a[3:0]);
    b_a     = (shamt_b >= 16) ? 16'sd0 : ($signed(in_b_mant) >>> shamt_b[3:0]);
    sum_w   = $signed({a_a[BFP_MANT_W-1], a_a}) + $signed({b_a[BFP_MANT_W-1], b_a});
    begin : lead_calc
      automatic logic [BFP_MANT_W:0] abs_v;
      abs_v    = sum_w[BFP_MANT_W] ? ($unsigned(~sum_w) + 1'b1) : $unsigned(sum_w);
      sum_lead = 5'd0;
      for (int i = 0; i <= BFP_MANT_W; i++) if (abs_v[i]) sum_lead = i[4:0];
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      state        <= S_IDLE;
      load_idx     <= '0;
      emit_idx     <= '0;
      emit_phase   <= 1'b0;
      max_lead_r   <= '0;
      total_cnt    <= '0;
      last_seen    <= 1'b0;
      tile_e_out_r <= '0;
      out_y_mant   <= '0;
      out_y_exp    <= '0;
      out_valid    <= 1'b0;
      done         <= 1'b0;
      in_flight    <= '0;
    end else begin
      out_valid <= 1'b0;
      done      <= 1'b0;
      case (state)
        S_IDLE: if (start) begin
          state      <= S_LOAD;
          load_idx   <= '0;
          max_lead_r <= '0;
          total_cnt  <= '0;
          last_seen  <= 1'b0;
          in_flight  <= '0;
        end
        S_LOAD: begin
          if (in_valid && in_ready) in_flight <= in_flight + 1'b1;
          if (in_valid && in_ready) begin
            sum_buf[load_idx] <= sum_w;
            if (sum_lead > max_lead_r) max_lead_r <= sum_lead;
            if (load_idx == '0) tile_e_out_r <= e_max;
            if (last_elem) last_seen <= 1'b1;
            if (load_idx == BFP_TILE - 1 || last_elem) begin
              state    <= S_EMIT;
              emit_idx <= '0;
            end else begin
              load_idx <= load_idx + 1'b1;
            end
          end
        end
        S_EMIT: begin
          begin : emit_blk
            automatic logic signed [7:0]  shamt;
            automatic logic signed [BFP_MANT_W:0] sh;
            shamt = $signed({3'b0, max_lead_r}) - 8'sd14;
            if (shamt >= 0) sh = sum_buf[emit_idx] >>>   shamt[3:0];
            else            sh = sum_buf[emit_idx] <<< (-shamt[3:0]);
            out_y_mant <= sh[BFP_MANT_W-1:0];
            begin : exp_blk
              logic signed [BFP_EXP_W+1:0] e_wide;
              e_wide = tile_e_out_r + $signed({{(BFP_EXP_W-6){shamt[7]}}, shamt});
              if      (e_wide >  127) out_y_exp <=  8'sd127;
              else if (e_wide < -128) out_y_exp <= -8'sd128;
              else                    out_y_exp <=  e_wide[BFP_EXP_W-1:0];
            end
            out_valid <= 1'b1;
            total_cnt <= total_cnt + 1'b1;
            if (emit_idx == BFP_TILE - 1) begin
              emit_idx   <= '0;
              max_lead_r <= '0;
              load_idx   <= '0;
              in_flight  <= '0;
              if (last_seen) state <= S_DONE;
              else           state <= S_LOAD;
            end else if (last_seen && (total_cnt + 1 == D)) begin
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
