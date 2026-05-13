`include "softmax_bfp_cfg.svh"
`include "bfp_format.svh"
`default_nettype none
module tb_softmax_bfp (
  input wire clk, rst, go,
  output wire done, fail
);
  localparam int N = `SMBFP_N;
  localparam int N_MAX = 64;
  logic [BFP_MANT_W-1:0] xm [0:N-1];
  logic [BFP_MANT_W-1:0] yg_m [0:N-1];
  logic [BFP_EXP_W -1:0] yg_e_arr [0:0];
  initial begin
    $readmemh("../generated/softmax_bfp_xm.hex", xm);
    $readmemh("../generated/softmax_bfp_ym.hex", yg_m);
    $readmemh("../generated/softmax_bfp_ye.hex", yg_e_arr);
  end

  typedef enum logic [1:0] {S_IDLE, S_STARTING, S_RUNNING, S_WAIT} st_t;
  st_t st;
  logic [$clog2(N+1)-1:0] cnt;
  logic start_pulse, in_valid;
  always_ff @(posedge clk) begin
    if (rst) begin st<=S_IDLE; cnt<='0; start_pulse<=1'b0; end
    else begin
      start_pulse <= 1'b0;
      case (st)
        S_IDLE: if (go) begin start_pulse<=1'b1; st<=S_STARTING; end
        S_STARTING: begin cnt<='0; st<=S_RUNNING; end
        S_RUNNING: if (cnt < N-1) cnt<=cnt+1'b1; else st<=S_WAIT;
        S_WAIT: ;
      endcase
    end
  end
  always_comb in_valid = (st == S_RUNNING);

  logic signed [BFP_MANT_W-1:0] xm_in;
  always_comb xm_in = in_valid ? xm[cnt] : '0;

  wire signed [BFP_MANT_W-1:0] out_y_mant;
  wire signed [BFP_EXP_W -1:0] out_y_exp;
  wire out_valid_w, done_w;
  softmax_bfp #(.N_MAX(N_MAX)) i_dut (
    .clk(clk), .rst(rst), .start(start_pulse),
    .n_elems($clog2(N_MAX+1)'(N)),
    .in_x_mant(xm_in), .in_x_exp(8'sd`SMBFP_XE),
    .in_valid(in_valid),
    .out_y_mant(out_y_mant), .out_y_exp(out_y_exp),
    .out_valid(out_valid_w), .done(done_w)
  );

  logic [$clog2(N+1)-1:0] chk;
  logic fail_r, done_r;
  always_ff @(posedge clk) begin
    if (rst) begin chk<='0; fail_r<=1'b0; done_r<=1'b0; end
    else begin
      if (out_valid_w) begin
        if (out_y_mant !== yg_m[chk] || out_y_exp !== yg_e_arr[0]) begin
          $display("FAIL i=%0d got m=%0d e=%0d, exp m=%0d e=%0d",
                   chk, $signed(out_y_mant), $signed(out_y_exp),
                   $signed(yg_m[chk]), $signed(yg_e_arr[0]));
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
