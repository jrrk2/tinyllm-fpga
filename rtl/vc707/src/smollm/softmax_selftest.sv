// softmax_selftest.sv — drives `softmax_q15` once at boot with hardcoded test
// data, latches the N-element Q1.15 result for the host to read via the
// register map.  Re-triggers on a `restart` pulse.
//
// Test data comes from a packed localparam in ../generated/softmax_selftest_data.svh.
// `result` is a packed N × 16-bit bus, lane 0 in [15:0].

`default_nettype none

module softmax_selftest #(
  parameter int N = 64
)(
  input  wire              clk,
  input  wire              rst,
  input  wire              restart,
  output logic [N*16-1:0]  result,
  output logic             done
);

  localparam int CW = $clog2(N + 1);

  // ----------------------- ROM (packed localparam) -----------------------
  // Auto-generated SVH — uses packed localparam literal rather than
  // $readmemh, sidestepping Vivado synth quirks.
`include "softmax_selftest_data.svh"

  // ----------------------- softmax_q15 DUT -----------------------
  logic                smx_start;
  logic signed [15:0]  smx_in_x;
  logic                smx_in_valid;
  logic signed [15:0]  smx_out_y;
  logic                smx_out_valid;
  /* verilator lint_off UNUSEDSIGNAL */
  logic                smx_done;
  /* verilator lint_on UNUSEDSIGNAL */

  softmax_q15 #(.N(N)) i_smx (
    .clk      ( clk           ),
    .rst      ( rst           ),
    .start    ( smx_start     ),
    .in_x     ( smx_in_x      ),
    .in_valid ( smx_in_valid  ),
    .out_y    ( smx_out_y     ),
    .out_valid( smx_out_valid ),
    .done     ( smx_done      )
  );

  // ----------------------- FSM -----------------------
  typedef enum logic [2:0] {
    S_INIT,    // pulse softmax_q15.start
    S_DRIVE,   // stream N inputs
    S_COLLECT, // capture N outputs
    S_DONE
  } state_t;
  state_t state;

  logic [CW-1:0] in_cnt;
  logic [CW-1:0] out_cnt;

  always_ff @(posedge clk) begin
    if (rst) begin
      state        <= S_INIT;
      in_cnt       <= '0;
      out_cnt      <= '0;
      smx_start    <= 1'b0;
      smx_in_valid <= 1'b0;
      smx_in_x     <= '0;
      result       <= '0;
      done         <= 1'b0;
    end else if (restart) begin
      state        <= S_INIT;
      in_cnt       <= '0;
      out_cnt      <= '0;
      smx_start    <= 1'b0;
      smx_in_valid <= 1'b0;
      done         <= 1'b0;
    end else begin
      // defaults
      smx_start    <= 1'b0;
      smx_in_valid <= 1'b0;

      case (state)
        S_INIT: begin
          smx_start <= 1'b1;
          in_cnt    <= '0;
          out_cnt   <= '0;
          state     <= S_DRIVE;
        end

        S_DRIVE: begin
          // softmax_q15 enters S_LOAD on the cycle after start; we drive N
          // valid inputs starting THIS cycle.  Indexed slice from packed
          // localparam — Vivado synthesizes as a small ROM/LUT.
          smx_in_x     <= SOFTMAX_X_PACKED[in_cnt*16 +: 16];
          smx_in_valid <= 1'b1;
          in_cnt       <= in_cnt + 1'b1;
          if (in_cnt == CW'(N-1)) state <= S_COLLECT;
        end

        S_COLLECT: begin
          if (smx_out_valid) begin
            result[out_cnt*16 +: 16] <= smx_out_y;
            out_cnt                  <= out_cnt + 1'b1;
            if (out_cnt == CW'(N-1)) begin
              done  <= 1'b1;
              state <= S_DONE;
            end
          end
        end

        S_DONE: begin
          // park forever (until restart)
        end

        default: state <= S_INIT;
      endcase
    end
  end

endmodule

`default_nettype wire
