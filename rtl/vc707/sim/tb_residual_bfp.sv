`include "res_bfp_cfg.svh"
`include "bfp_format.svh"
`default_nettype none
module tb_residual_bfp (input wire clk, rst, go, output wire done, fail);
  localparam int D = `RESBFP_D, NT = `RESBFP_NT;
  logic [BFP_MANT_W-1:0] am [0:D-1], bm [0:D-1], yg_m [0:D-1];
  logic [BFP_EXP_W -1:0] ae [0:NT-1], be [0:NT-1], yg_e [0:NT-1];
  initial begin
    $readmemh("../generated/res_bfp_am.hex", am);
    $readmemh("../generated/res_bfp_bm.hex", bm);
    $readmemh("../generated/res_bfp_ae.hex", ae);
    $readmemh("../generated/res_bfp_be.hex", be);
    $readmemh("../generated/res_bfp_ym.hex", yg_m);
    $readmemh("../generated/res_bfp_ye.hex", yg_e);
  end
  typedef enum logic [1:0] {S_IDLE, S_STARTING, S_RUNNING, S_WAIT} st_t;
  st_t st;
  logic [$clog2(D+1)-1:0] cnt;
  logic start_pulse, in_valid, last_elem;
  wire dut_ready;
  always_ff @(posedge clk) begin
    if (rst) begin st<=S_IDLE; cnt<='0; start_pulse<=1'b0; end
    else begin
      start_pulse <= 1'b0;
      case (st)
        S_IDLE: if (go) begin start_pulse<=1'b1; st<=S_STARTING; end
        S_STARTING: begin cnt<='0; st<=S_RUNNING; end
        S_RUNNING: if (dut_ready) begin
          if (cnt < D-1) cnt<=cnt+1'b1; else st<=S_WAIT;
        end
        S_WAIT: ;
      endcase
    end
  end
  always_comb in_valid  = (st == S_RUNNING) && dut_ready;
  always_comb last_elem = in_valid && (cnt == D-1);

  logic signed [BFP_MANT_W-1:0] am_in, bm_in;
  logic signed [BFP_EXP_W -1:0] ae_in, be_in;
  always_comb begin
    am_in='0; bm_in='0; ae_in='0; be_in='0;
    if (in_valid) begin
      am_in=am[cnt]; bm_in=bm[cnt];
      ae_in=ae[cnt/BFP_TILE]; be_in=be[cnt/BFP_TILE];
    end
  end

  wire signed [BFP_MANT_W-1:0] out_m;
  wire signed [BFP_EXP_W -1:0] out_e;
  wire out_valid_w, done_w;
  residual_bfp #(.D(D)) i_dut (
    .clk(clk), .rst(rst), .start(start_pulse),
    .in_a_mant(am_in), .in_a_exp(ae_in),
    .in_b_mant(bm_in), .in_b_exp(be_in),
    .in_valid(in_valid), .last_elem(last_elem), .in_ready(dut_ready),
    .out_y_mant(out_m), .out_y_exp(out_e),
    .out_valid(out_valid_w), .done(done_w)
  );

  logic [$clog2(D+1)-1:0] chk;
  logic fail_r, done_r;
  always_ff @(posedge clk) begin
    if (rst) begin chk<='0; fail_r<=1'b0; done_r<=1'b0; end
    else begin
      if (out_valid_w) begin
        if (out_m !== yg_m[chk] || out_e !== yg_e[chk/BFP_TILE]) begin
          $display("FAIL i=%0d got m=%0d e=%0d, exp m=%0d e=%0d",
                   chk, $signed(out_m), $signed(out_e),
                   $signed(yg_m[chk]), $signed(yg_e[chk/BFP_TILE]));
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
