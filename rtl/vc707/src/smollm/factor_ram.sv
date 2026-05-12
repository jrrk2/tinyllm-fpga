// factor_ram.sv — runtime-tunable per-layer SwiGLU + Attn-AV scale factors.
//
// Defaults loaded at reset from tm_layer_swiglu_attn.svh's TM_SWIGLU_SCALES
// and TM_ATTN_FACTOR localparams.  Host can override any (kind, layer) entry
// via the write port; the same RAM is observable through the readback port
// AND through the compute-side cur_*_factor outputs.  Keeping the readback
// path live forces Vivado to maintain the RAM as one storage element rather
// than splitting it into a read-only ROM + a separate writable copy.
//
// Tested standalone under Verilator (sim/tb_factor_ram*) to confirm the
// read/write semantics before another FPGA bitstream rebuild.

`default_nettype none

(* keep_hierarchy = "yes" *)
module factor_ram #(
  parameter int NL = 30
)(
  input  wire                  clk,
  input  wire                  rst,

  // Compute-side: layer_idx selects which row drives cur_*_factor (registered).
  input  wire [4:0]            layer_idx,
  output logic [15:0]          cur_sg_gate_in_factor,
  output logic [15:0]          cur_sg_up_in_factor,
  output logic [23:0]          cur_sg_mlp_out_factor,
  output logic [23:0]          cur_attn_factor,

  // Host write port: one-cycle pulse on factor_wr_en_* selects which field
  // to write at factor_wr_layer with factor_wr_data.  Packed format on
  // *_swiglu_lo: [31:16]=up, [15:0]=gate.
  input  wire [4:0]            factor_wr_layer,
  input  wire [31:0]           factor_wr_data,
  input  wire                  factor_wr_en_swiglu_lo,
  input  wire                  factor_wr_en_swiglu_mlp,
  input  wire                  factor_wr_en_attn,

  // Host readback: select {kind[1:0], layer[4:0]}; data is combinational.
  input  wire [6:0]            factor_rd_sel,
  output wire [31:0]           factor_rd_data
);

`include "tm_layer_swiglu_attn.svh"

  // dont_touch forbids Vivado from constant-propagating through these
  // FFs even when it can prove each layer's reset value.  Without it,
  // Vivado folds the layer_idx-indexed compute reads back to the
  // TM_SWIGLU_SCALES constants while the readback path stays live —
  // so writes appear to "work" on readback but never reach swiglu.
  (* dont_touch = "true" *) logic [15:0] gate_in_factor_ram [0:NL-1];
  (* dont_touch = "true" *) logic [15:0] up_in_factor_ram   [0:NL-1];
  (* dont_touch = "true" *) logic [23:0] mlp_out_factor_ram [0:NL-1];
  (* dont_touch = "true" *) logic [23:0] attn_factor_ram    [0:NL-1];

  // STORAGE TRICK: hold the bitwise INVERSE of the real factor.  This
  // guarantees every (layer, bit) FF has a non-zero reset value for at
  // least *some* layer — Vivado can no longer constant-fold any FF to
  // a fixed 0 based on the default TM_SWIGLU_SCALES bit pattern.  The
  // inversion is undone at the read side (cur_*_factor_r and the
  // readback mux), so all external semantics stay identical.
  always_ff @(posedge clk) begin
    if (rst) begin
      for (int i = 0; i < NL; i++) begin
        gate_in_factor_ram[i] <= ~TM_SWIGLU_SCALES[i][15:0];
        up_in_factor_ram[i]   <= ~TM_SWIGLU_SCALES[i][31:16];
        mlp_out_factor_ram[i] <= ~TM_SWIGLU_SCALES[i][55:32];
        attn_factor_ram[i]    <= ~TM_ATTN_FACTOR[i];
      end
    end else begin
      if (factor_wr_en_swiglu_lo) begin
        gate_in_factor_ram[factor_wr_layer] <= ~factor_wr_data[15:0];
        up_in_factor_ram  [factor_wr_layer] <= ~factor_wr_data[31:16];
      end
      if (factor_wr_en_swiglu_mlp)
        mlp_out_factor_ram[factor_wr_layer] <= ~factor_wr_data[23:0];
      if (factor_wr_en_attn)
        attn_factor_ram   [factor_wr_layer] <= ~factor_wr_data[23:0];
    end
  end

  // Compute-side reads — registered so swiglu / attn-AV see a clean
  // FF output rather than a mux cone Vivado might constant-fold.
  // dont_touch on the output FFs prevents Vivado from short-circuiting
  // the layer-indexed read back to the dont_touch'd RAM.
  (* dont_touch = "true" *) logic [15:0] cur_sg_gate_in_factor_r;
  (* dont_touch = "true" *) logic [15:0] cur_sg_up_in_factor_r;
  (* dont_touch = "true" *) logic [23:0] cur_sg_mlp_out_factor_r;
  (* dont_touch = "true" *) logic [23:0] cur_attn_factor_r;
  // Un-invert on read.  XOR with all-1s recovers the real factor value.
  always_ff @(posedge clk) begin
    cur_sg_gate_in_factor_r <= ~gate_in_factor_ram[layer_idx];
    cur_sg_up_in_factor_r   <= ~up_in_factor_ram  [layer_idx];
    cur_sg_mlp_out_factor_r <= ~mlp_out_factor_ram[layer_idx];
    cur_attn_factor_r       <= ~attn_factor_ram   [layer_idx];
  end
  assign cur_sg_gate_in_factor = cur_sg_gate_in_factor_r;
  assign cur_sg_up_in_factor   = cur_sg_up_in_factor_r;
  assign cur_sg_mlp_out_factor = cur_sg_mlp_out_factor_r;
  assign cur_attn_factor       = cur_attn_factor_r;

  // Readback mux for host visibility.
  wire [4:0] rd_layer = factor_rd_sel[4:0];
  wire [1:0] rd_kind  = factor_rd_sel[6:5];
  // Un-invert on readback so host sees the real factor value.
  assign factor_rd_data =
       (rd_kind == 2'd0) ? {~up_in_factor_ram[rd_layer], ~gate_in_factor_ram[rd_layer]}
     : (rd_kind == 2'd1) ? {8'b0, ~mlp_out_factor_ram[rd_layer]}
     : (rd_kind == 2'd2) ? {8'b0, ~attn_factor_ram[rd_layer]}
     :                     32'h0;

endmodule

`default_nettype wire
