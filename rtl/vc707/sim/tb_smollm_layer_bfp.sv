// tb_smollm_layer_bfp.sv — Verilator testbench for smollm_layer_bfp.
//
// Loads hidden_in + weights baked by gen_smollm_blockfp_bfp.py, runs one
// layer forward pass, compares hidden_out mantissa+exp against the
// host-side golden HOUT_*.hex.  Reports pass/fail.

`include "../generated/lbfp_cfg.svh"
`include "bfp_format.svh"

`default_nettype none

module tb_smollm_layer_bfp (
  input  wire clk,
  input  wire rst,
  input  wire go,
  output wire done,
  output wire fail
);

  localparam int D       = `LBFP_D;
  localparam int H_Q     = `LBFP_HQ;
  localparam int H_KV    = `LBFP_HKV;
  localparam int HD      = `LBFP_HD;
  localparam int FFN     = `LBFP_FFN;
  localparam int MAX_CTX = `LBFP_MAX_CTX;
  localparam int NT_D    = D / BFP_TILE;

  // Hidden_in + golden hidden_out loaded from hex
  logic [BFP_MANT_W-1:0] hin_arr  [0:D-1];
  logic [BFP_EXP_W -1:0] hin_e    [0:NT_D-1];
  logic [BFP_MANT_W-1:0] hg_m     [0:D-1];
  logic [BFP_EXP_W -1:0] hg_e     [0:NT_D-1];

  initial begin
    $readmemh("../generated/lbfp_HIN_m.hex",  hin_arr);
    $readmemh("../generated/lbfp_HIN_e.hex",  hin_e);
    $readmemh("../generated/lbfp_HOUT_m.hex", hg_m);
    $readmemh("../generated/lbfp_HOUT_e.hex", hg_e);
  end

  // Pack hidden_in into the DUT bus form
  wire signed [D*BFP_MANT_W-1:0]      hin_m_bus;
  wire signed [NT_D*BFP_EXP_W-1:0]    hin_e_bus;
  genvar gi;
  generate
    for (gi = 0; gi < D; gi++) begin : g_hin
      assign hin_m_bus[gi*BFP_MANT_W +: BFP_MANT_W] = hin_arr[gi];
    end
    for (gi = 0; gi < NT_D; gi++) begin : g_hine
      assign hin_e_bus[gi*BFP_EXP_W  +: BFP_EXP_W ] = hin_e[gi];
    end
  endgenerate

  // One-cycle start pulse after go is asserted and the boot delay has elapsed.
  // Stays low for the rest of the run so the DUT FSM does not retrigger after
  // S_DONE → S_IDLE.
  logic start_r;
  logic started_r;
  logic [3:0] start_cnt;
  always_ff @(posedge clk) begin
    if (rst) begin
      start_r   <= 1'b0;
      started_r <= 1'b0;
      start_cnt <= '0;
    end else begin
      start_r <= 1'b0;
      if (!started_r) begin
        if (go) start_cnt <= start_cnt + 1'b1;
        if (start_cnt == 4'd3) begin
          start_r   <= 1'b1;
          started_r <= 1'b1;
        end
      end
    end
  end

  wire signed [D*BFP_MANT_W-1:0]   hout_m_bus;
  wire signed [NT_D*BFP_EXP_W-1:0] hout_e_bus;
  wire dut_done;
  wire [5:0]  dbg_state;
  wire [11:0] dbg_cnt;
  wire [6:0]  dbg_chunk;

  smollm_layer_bfp #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD),
    .FFN(FFN), .MAX_CTX(MAX_CTX), .PREFIX("../generated/lbfp_")
  ) dut (
    .clk(clk), .rst(rst), .start(start_r),
    .pos(11'd`LBFP_POS), .kv_pos(5'd`LBFP_KV_POS),
    .hidden_in_m(hin_m_bus), .hidden_in_e(hin_e_bus),
    .hidden_out_m(hout_m_bus), .hidden_out_e(hout_e_bus),
    .done(dut_done),
    .wr_kind(5'd0), .wr_addr(18'd0), .wr_data(16'd0), .wr_en(1'b0),
    .dbg_state(dbg_state), .dbg_cnt(dbg_cnt), .dbg_chunk(dbg_chunk)
  );

  // Trace state changes
  logic [5:0] last_state;
  always_ff @(posedge clk) begin
    if (rst) last_state <= 6'h3f;
    else if (dbg_state != last_state) begin
      $display("[T+%0t] state -> %0d  cnt=%0d  chunk=%0d",
               $time, dbg_state, dbg_cnt, dbg_chunk);
      last_state <= dbg_state;
    end
  end

  // Check
  logic fail_r, done_r;
  // Tolerance — BFP arithmetic compounds through 14 op-stages (rmsnorm,
  // 7 matvecs, rope, softmax, AV, swiglu, 2 residuals).  Each tile-quantize
  // rounds within ±0.5 LSB; rsqrt / softmax / swiglu LUTs add ~1 LSB; matvec
  // exp normalization adds ~0.5 LSB.  Total expected drift is ~ ±200 LSB
  // across the full chain (relative error <1% on Q1.15 outputs).
  localparam int MANT_TOL = 256;

  integer i;
  always_ff @(posedge clk) begin
    if (rst) begin fail_r <= 1'b0; done_r <= 1'b0; end
    else if (dut_done && !done_r) begin
      automatic int mism;
      automatic int max_diff;
      mism = 0; max_diff = 0;
      for (i = 0; i < D; i++) begin
        automatic int diff;
        diff = $signed(hout_m_bus[i*BFP_MANT_W +: BFP_MANT_W]) - $signed(hg_m[i]);
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        if (diff > MANT_TOL) begin
          if (mism < 8) begin
            $display("MISMATCH[%0d]: got m=%0d e=%0d  golden m=%0d e=%0d  diff=%0d",
              i, $signed(hout_m_bus[i*BFP_MANT_W +: BFP_MANT_W]),
                 $signed(hout_e_bus[(i/BFP_TILE)*BFP_EXP_W +: BFP_EXP_W]),
                 $signed(hg_m[i]),
                 $signed(hg_e[i/BFP_TILE]), diff);
          end
          mism = mism + 1;
        end
      end
      // Check tile exponents
      for (i = 0; i < NT_D; i++) begin
        automatic int ediff;
        ediff = $signed(hout_e_bus[i*BFP_EXP_W +: BFP_EXP_W]) - $signed(hg_e[i]);
        if (ediff < 0) ediff = -ediff;
        if (ediff > 1) begin
          $display("EXP MISMATCH tile[%0d]: got=%0d golden=%0d",
            i, $signed(hout_e_bus[i*BFP_EXP_W +: BFP_EXP_W]),
               $signed(hg_e[i]));
          mism = mism + 1;
        end
      end
      $display("layer_bfp: %0d mismatches (tol=%0d LSB)  max_mant_diff=%0d",
               mism, MANT_TOL, max_diff);
      if (mism > 0) fail_r <= 1'b1;
      done_r <= 1'b1;
    end
  end

  assign done = done_r;
  assign fail = fail_r;
endmodule

`default_nettype wire
