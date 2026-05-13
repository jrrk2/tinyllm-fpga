// tb_rmsnorm_bfp.sv — Verilator driver for rmsnorm_bfp unit test.
// Loads golden hex files, drives the DUT cycle-by-cycle, compares output
// stream against the golden y stream.

`include "rmsnorm_bfp_cfg.svh"
`include "bfp_format.svh"

`default_nettype none

module tb_rmsnorm_bfp (
  input  wire                            clk,
  input  wire                            rst,
  input  wire                            go,
  output wire                            done,
  output wire                            fail
);

  localparam int D  = `RMSBFP_D;
  localparam int NT = `RMSBFP_NT;

  // Memories
  logic [BFP_MANT_W-1:0] x_m_mem [0:D-1];
  logic [BFP_MANT_W-1:0] g_m_mem [0:D-1];
  logic [BFP_EXP_W -1:0] x_e_mem [0:NT-1];
  logic [BFP_EXP_W -1:0] g_e_mem [0:NT-1];
  logic [BFP_MANT_W-1:0] y_m_gold [0:D-1];
  logic [BFP_EXP_W -1:0] y_e_gold [0:NT-1];
  initial begin
    $readmemh("../generated/rmsnorm_bfp_xm.hex", x_m_mem);
    $readmemh("../generated/rmsnorm_bfp_gm.hex", g_m_mem);
    $readmemh("../generated/rmsnorm_bfp_xe.hex", x_e_mem);
    $readmemh("../generated/rmsnorm_bfp_ge.hex", g_e_mem);
    $readmemh("../generated/rmsnorm_bfp_ym.hex", y_m_gold);
    $readmemh("../generated/rmsnorm_bfp_ye.hex", y_e_gold);
  end

  // Driver FSM
  typedef enum logic [1:0] {S_IDLE, S_STARTING, S_RUNNING, S_WAITDONE} state_t;
  state_t state;
  logic [$clog2(D+1)-1:0] cnt;
  logic                  start_pulse;
  logic                  in_valid;
  logic                  last_elem;

  always_ff @(posedge clk) begin
    if (rst) begin
      cnt <= '0; state <= S_IDLE; start_pulse <= 1'b0;
    end else begin
      start_pulse <= 1'b0;
      case (state)
        S_IDLE: if (go) begin start_pulse <= 1'b1; state <= S_STARTING; end
        S_STARTING: begin cnt <= '0; state <= S_RUNNING; end
        S_RUNNING: begin
          if (cnt < D-1) cnt <= cnt + 1'b1;
          else           state <= S_WAITDONE;
        end
        S_WAITDONE: ;  // wait for dut done
      endcase
    end
  end
  always_comb in_valid  = (state == S_RUNNING);
  always_comb last_elem = in_valid && (cnt == D-1);

  // DUT inputs
  logic signed [BFP_MANT_W-1:0] in_x_mant, in_g_mant;
  logic signed [BFP_EXP_W -1:0] in_x_exp,  in_g_exp;
  always_comb begin
    in_x_mant = '0; in_g_mant = '0; in_x_exp = '0; in_g_exp = '0;
    if (in_valid) begin
      in_x_mant = x_m_mem[cnt];
      in_g_mant = g_m_mem[cnt];
      in_x_exp  = x_e_mem[cnt / BFP_TILE];
      in_g_exp  = g_e_mem[cnt / BFP_TILE];
    end
  end

  wire signed [BFP_MANT_W-1:0] out_y_mant;
  wire signed [BFP_EXP_W -1:0] out_y_exp;
  wire                          out_valid_w;
  wire                          done_w;

  rmsnorm_bfp #(.D(D)) i_dut (
    .clk        (clk),
    .rst        (rst),
    .start      (start_pulse),
    .in_x_mant  (in_x_mant),
    .in_x_exp   (in_x_exp),
    .in_g_mant  (in_g_mant),
    .in_g_exp   (in_g_exp),
    .in_valid   (in_valid),
    .last_elem  (last_elem),
    .out_y_mant (out_y_mant),
    .out_y_exp  (out_y_exp),
    .out_valid  (out_valid_w),
    .done       (done_w)
  );

  // Compare emitted (mant, exp) stream against golden.  Tile exponent
  // applies to all 16 mantissas of each tile.
  logic [$clog2(D+1)-1:0] check_cnt;
  logic                   fail_r, done_r;
  always_ff @(posedge clk) begin
    if (rst) begin
      check_cnt <= '0; fail_r <= 1'b0; done_r <= 1'b0;
    end else begin
      if (out_valid_w) begin
        logic signed [BFP_MANT_W-1:0] expect_m;
        logic signed [BFP_EXP_W-1:0]  expect_e;
        expect_m = y_m_gold[check_cnt];
        expect_e = y_e_gold[check_cnt / BFP_TILE];
        if (out_y_mant !== expect_m || out_y_exp !== expect_e) begin
          $display("FAIL i=%0d: got m=%0d e=%0d, expected m=%0d e=%0d",
                   check_cnt, $signed(out_y_mant), $signed(out_y_exp),
                   $signed(expect_m), $signed(expect_e));
          fail_r <= 1'b1;
        end
        check_cnt <= check_cnt + 1'b1;
      end
      if (done_w) done_r <= 1'b1;
    end
  end

  assign done = done_r;
  assign fail = fail_r;

endmodule

`default_nettype wire
