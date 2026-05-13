`include "rope_bfp_cfg.svh"
`include "bfp_format.svh"
`default_nettype none
module tb_rope_bfp (input wire clk, rst, go, output wire done, fail);
  localparam int HD = `ROPEBFP_HD;
  localparam int NT_HD = HD / BFP_TILE;
  logic [BFP_MANT_W-1:0] xm [0:HD-1];
  logic [BFP_EXP_W -1:0] xe [0:NT_HD-1];
  logic [BFP_MANT_W-1:0] ym_g [0:HD-1];
  logic [BFP_EXP_W -1:0] ye_g [0:HD-1];
  initial begin
    $readmemh("../generated/rope_bfp_xm.hex", xm);
    $readmemh("../generated/rope_bfp_xe.hex", xe);
    $readmemh("../generated/rope_bfp_ym.hex", ym_g);
    $readmemh("../generated/rope_bfp_ye.hex", ye_g);
  end

  typedef enum logic [1:0] {S_IDLE, S_STARTING, S_RUNNING, S_WAIT} st_t;
  st_t st;
  logic [$clog2(HD+1)-1:0] cnt;
  logic start_pulse, in_valid;
  always_ff @(posedge clk) begin
    if (rst) begin st<=S_IDLE; cnt<='0; start_pulse<=1'b0; end
    else begin
      start_pulse <= 1'b0;
      case (st)
        S_IDLE: if (go) begin start_pulse<=1'b1; st<=S_STARTING; end
        S_STARTING: begin cnt<='0; st<=S_RUNNING; end
        S_RUNNING: if (cnt < HD-1) cnt<=cnt+1'b1; else st<=S_WAIT;
        S_WAIT: ;
      endcase
    end
  end
  always_comb in_valid = (st == S_RUNNING);

  logic signed [BFP_MANT_W-1:0] m_in;
  logic signed [BFP_EXP_W -1:0] e_in;
  always_comb begin
    m_in = '0; e_in = '0;
    if (in_valid) begin
      m_in = xm[cnt];
      e_in = xe[cnt / BFP_TILE];
    end
  end

  wire signed [BFP_MANT_W-1:0] out_m;
  wire signed [BFP_EXP_W -1:0] out_e;
  wire out_valid_w, done_w;
  rope_bfp #(.HD(HD)) i_dut (
    .clk(clk), .rst(rst), .start(start_pulse),
    .in_x_mant(m_in), .in_x_exp(e_in), .in_valid(in_valid),
    .pos(11'd`ROPEBFP_POS),
    .out_y_mant(out_m), .out_y_exp(out_e),
    .out_valid(out_valid_w), .done(done_w)
  );

  logic [$clog2(HD+1)-1:0] chk;
  logic fail_r, done_r;
  // ±4 LSB tolerance for residual cordic rounding noise (the alignment +
  // rotation arithmetic is now bit-exact; remaining drift is sub-LSB
  // accumulation in the 16-bit silu mantissa, plus CORDIC's 1-LSB final
  // rounding).  Exponent must match.
  localparam int MANT_TOL = 4;
  always_ff @(posedge clk) begin
    if (rst) begin chk<='0; fail_r<=1'b0; done_r<=1'b0; end
    else begin
      if (out_valid_w) begin
        automatic int diff;
        diff = $signed(out_m) - $signed(ym_g[chk]);
        if ((diff > MANT_TOL || diff < -MANT_TOL) || out_e !== ye_g[chk]) begin
          $display("FAIL i=%0d got m=%0d e=%0d, exp m=%0d e=%0d  diff=%0d",
                   chk, $signed(out_m), $signed(out_e),
                   $signed(ym_g[chk]), $signed(ye_g[chk]), diff);
          fail_r <= 1'b1;
        end
        chk <= chk + 1'b1;
      end
      if (done_w) done_r <= 1'b1;
    end
  end
  assign done = done_r;
  assign fail = fail_r;
endmodule
`default_nettype wire
