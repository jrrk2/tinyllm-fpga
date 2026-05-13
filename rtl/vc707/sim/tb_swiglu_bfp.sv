// tb_swiglu_bfp.sv — Verilator driver for swiglu_bfp.
`include "swiglu_bfp_cfg.svh"
`include "bfp_format.svh"

`default_nettype none
module tb_swiglu_bfp (
  input wire clk, input wire rst, input wire go,
  output wire done, output wire fail
);
  localparam int D = `SWBFP_D, NT = `SWBFP_NT;

  logic [BFP_MANT_W-1:0] gm_mem [0:D-1];
  logic [BFP_MANT_W-1:0] um_mem [0:D-1];
  logic [BFP_EXP_W -1:0] ge_mem [0:NT-1];
  logic [BFP_EXP_W -1:0] ue_mem [0:NT-1];
  logic [BFP_MANT_W-1:0] yg_m   [0:D-1];
  logic [BFP_EXP_W -1:0] yg_e   [0:NT-1];
  initial begin
    $readmemh("../generated/swiglu_bfp_gatem.hex", gm_mem);
    $readmemh("../generated/swiglu_bfp_upm.hex",   um_mem);
    $readmemh("../generated/swiglu_bfp_gatee.hex", ge_mem);
    $readmemh("../generated/swiglu_bfp_upe.hex",   ue_mem);
    $readmemh("../generated/swiglu_bfp_ym.hex",    yg_m);
    $readmemh("../generated/swiglu_bfp_ye.hex",    yg_e);
  end

  typedef enum logic [1:0] {S_IDLE, S_STARTING, S_RUNNING, S_WAIT} state_t;
  state_t state;
  logic [$clog2(D+1)-1:0] cnt;
  logic start_pulse, in_valid, last_elem;
  wire dut_ready;
  always_ff @(posedge clk) begin
    if (rst) begin state<=S_IDLE; cnt<='0; start_pulse<=1'b0; end
    else begin
      start_pulse <= 1'b0;
      case (state)
        S_IDLE: if (go) begin start_pulse<=1'b1; state<=S_STARTING; end
        S_STARTING: begin cnt<='0; state<=S_RUNNING; end
        S_RUNNING:
          if (dut_ready) begin
            if (cnt < D-1) cnt<=cnt+1'b1;
            else           state<=S_WAIT;
          end
        S_WAIT: ;
      endcase
    end
  end
  always_comb in_valid  = (state == S_RUNNING) && dut_ready;
  always_comb last_elem = in_valid && (cnt == D-1);

  logic signed [BFP_MANT_W-1:0] gm, um;
  logic signed [BFP_EXP_W -1:0] ge, ue;
  always_comb begin
    gm = '0; um = '0; ge = '0; ue = '0;
    if (in_valid) begin
      gm = gm_mem[cnt]; um = um_mem[cnt];
      ge = ge_mem[cnt/BFP_TILE]; ue = ue_mem[cnt/BFP_TILE];
    end
  end

  wire signed [BFP_MANT_W-1:0] out_y_mant;
  wire signed [BFP_EXP_W -1:0] out_y_exp;
  wire out_valid_w, done_w;
  swiglu_bfp #(.D(D)) i_dut (
    .clk(clk), .rst(rst), .start(start_pulse),
    .in_gate_mant(gm), .in_gate_exp(ge),
    .in_up_mant(um),   .in_up_exp(ue),
    .in_valid(in_valid), .last_elem(last_elem),
    .in_ready(dut_ready),
    .out_y_mant(out_y_mant), .out_y_exp(out_y_exp),
    .out_valid(out_valid_w), .done(done_w)
  );

  logic [$clog2(D+1)-1:0] chk;
  logic fail_r, done_r;
  always_ff @(posedge clk) begin
    if (rst) begin chk<='0; fail_r<=1'b0; done_r<=1'b0; end
    else begin
      if (out_valid_w) begin
        if (out_y_mant !== yg_m[chk] || out_y_exp !== yg_e[chk/BFP_TILE]) begin
          $display("FAIL i=%0d got m=%0d e=%0d, exp m=%0d e=%0d",
                   chk, $signed(out_y_mant), $signed(out_y_exp),
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
