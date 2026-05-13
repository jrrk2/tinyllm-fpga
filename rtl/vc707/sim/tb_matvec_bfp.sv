// tb_matvec_bfp.sv — testbench driver for matvec_bfp_engine.
// Loads golden vectors from generated/matvec_bfp_{x,w,y}.hex, drives the
// engine cycle-by-cycle, and exposes the result via C++-readable signals.

`include "matvec_bfp_cfg.svh"
`include "bfp_format.svh"

`default_nettype none

module tb_matvec_bfp (
  input  wire                              clk,
  input  wire                              rst,
  input  wire                              go,
  output wire                              done,
  output wire signed [`MVBFP_LANES*BFP_MANT_W-1:0]  y_mant,
  output wire signed [`MVBFP_LANES*BFP_EXP_W -1:0]  y_exp,
  output wire                              fail
);

  localparam int D     = `MVBFP_D;
  localparam int LANES = `MVBFP_LANES;
  localparam int NT    = `MVBFP_NT;

  // Memories — loaded via $readmemh
  logic [BFP_MANT_W-1:0] x_m_mem [0:D-1];
  logic [BFP_EXP_W -1:0] x_e_mem [0:NT-1];
  logic [BFP_MANT_W-1:0] w_m_mem [0:D*LANES-1];
  logic [BFP_EXP_W -1:0] w_e_mem [0:NT*LANES-1];

  initial begin
    $readmemh("../generated/matvec_bfp_xm.hex", x_m_mem);
    $readmemh("../generated/matvec_bfp_xe.hex", x_e_mem);
    $readmemh("../generated/matvec_bfp_wm.hex", w_m_mem);
    $readmemh("../generated/matvec_bfp_we.hex", w_e_mem);
  end

  // Driver FSM
  logic [$clog2(D+2)-1:0] cnt;
  logic                   running;
  logic                   start_pulse;
  logic                   in_valid;
  logic                   last_elem;
  logic signed [BFP_MANT_W-1:0]            in_x_mant;
  logic signed [BFP_EXP_W -1:0]            in_x_exp;
  logic signed [LANES*BFP_MANT_W-1:0]      w_mant_bus;
  logic signed [LANES*BFP_EXP_W -1:0]      w_exp_bus;

  // FSM: IDLE → STARTING (1-cycle start_pulse) → RUNNING (in_valid)
  typedef enum logic [1:0] {S_IDLE, S_STARTING, S_RUNNING} state_t;
  state_t state;
  always_ff @(posedge clk) begin
    if (rst) begin
      cnt         <= '0;
      state       <= S_IDLE;
      start_pulse <= 1'b0;
    end else begin
      start_pulse <= 1'b0;
      case (state)
        S_IDLE: if (go) begin
          start_pulse <= 1'b1;
          state       <= S_STARTING;
        end
        S_STARTING: begin
          cnt   <= '0;
          state <= S_RUNNING;
        end
        S_RUNNING: begin
          if (cnt < D-1) cnt <= cnt + 1'b1;
          else           state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end
  always_comb in_valid  = (state == S_RUNNING);
  always_comb last_elem = in_valid && (cnt == D-1);

  // Drive inputs combinationally from memories
  always_comb begin
    in_x_mant = '0;
    in_x_exp  = '0;
    w_mant_bus = '0;
    w_exp_bus  = '0;
    if (in_valid) begin
      in_x_mant = x_m_mem[cnt];
      in_x_exp  = x_e_mem[cnt / BFP_TILE];   // tile exponent
      for (int lane = 0; lane < LANES; lane = lane + 1) begin
        w_mant_bus[lane*BFP_MANT_W +: BFP_MANT_W] = w_m_mem[cnt*LANES + lane];
        w_exp_bus [lane*BFP_EXP_W  +: BFP_EXP_W ] = w_e_mem[(cnt/BFP_TILE)*LANES + lane];
      end
    end
  end

  wire                          out_valid;
  wire signed [LANES*BFP_MANT_W-1:0] out_mant;
  wire signed [LANES*BFP_EXP_W -1:0] out_exp;
  matvec_bfp_engine #(.LANES(LANES)) i_dut (
    .clk          (clk),
    .rst          (rst),
    .start_matvec (start_pulse),
    .in_x_mant    (in_x_mant),
    .in_x_exp     (in_x_exp),
    .in_valid     (in_valid),
    .last_elem    (last_elem),
    .w_mant       (w_mant_bus),
    .w_exp        (w_exp_bus),
    .out_mant     (out_mant),
    .out_exp      (out_exp),
    .out_valid    (out_valid)
  );

  // Golden output
  logic [BFP_MANT_W-1:0] gy_m_mem [0:LANES-1];
  logic [BFP_EXP_W -1:0] gy_e_mem [0:LANES-1];
  initial begin
    $readmemh("../generated/matvec_bfp_ym.hex", gy_m_mem);
    $readmemh("../generated/matvec_bfp_ye.hex", gy_e_mem);
  end

  logic signed [LANES*BFP_MANT_W-1:0] y_mant_r;
  logic signed [LANES*BFP_EXP_W -1:0] y_exp_r;
  logic                                done_r;
  logic                                fail_r;

  always_ff @(posedge clk) begin
    if (rst) begin
      y_mant_r <= '0;
      y_exp_r  <= '0;
      done_r   <= 1'b0;
      fail_r   <= 1'b0;
    end else if (out_valid) begin
      y_mant_r <= out_mant;
      y_exp_r  <= out_exp;
      done_r   <= 1'b1;
      // Compare lane-by-lane
      for (int lane = 0; lane < LANES; lane = lane + 1) begin
        if (out_mant[lane*BFP_MANT_W +: BFP_MANT_W] !== gy_m_mem[lane] ||
            out_exp [lane*BFP_EXP_W  +: BFP_EXP_W ] !== gy_e_mem[lane]) begin
          $display("FAIL lane %0d: got m=%0d e=%0d, expected m=%0d e=%0d",
                   lane,
                   $signed(out_mant[lane*BFP_MANT_W +: BFP_MANT_W]),
                   $signed(out_exp [lane*BFP_EXP_W  +: BFP_EXP_W ]),
                   $signed(gy_m_mem[lane]),
                   $signed(gy_e_mem[lane]));
          fail_r <= 1'b1;
        end
      end
    end
  end

  assign y_mant = y_mant_r;
  assign y_exp  = y_exp_r;
  assign done   = done_r;
  assign fail   = fail_r;

endmodule

`default_nettype wire
