// swiglu_selftest.sv — drives `swiglu` once at boot with hardcoded test
// data, latches the N-element Q1.15 result for the host to read via the
// register map.  Re-triggers on a `restart` pulse.
//
// Test data comes from a packed localparam SVH in ../generated/:
//   swiglu_selftest_data.svh   N × 16-bit gate + up vectors
//
// `result` is a packed N × 16-bit bus, lane 0 in [15:0].
//
// Latency: swiglu is 1-cycle registered.  We drive gate/up/valid in the
// same state as we collect outputs (offset by 1 cycle via out_cnt starting
// after the first valid is driven).

`default_nettype none

module swiglu_selftest #(
  parameter int N = 64
)(
  input  wire              clk,
  input  wire              rst,
  input  wire              restart,
  output logic [N*16-1:0]  result,
  output logic             done
);

  localparam int CW = $clog2(N + 1);

  // ----------------------- ROM (packed localparams) -----------------------
`include "swiglu_selftest_data.svh"

  // ----------------------- swiglu DUT -----------------------
  logic signed [15:0]  sg_in_gate;
  logic signed [15:0]  sg_in_up;
  logic                sg_in_valid;
  logic signed [15:0]  sg_out_y;
  logic                sg_out_valid;

  swiglu i_swiglu (
    .clk      ( clk          ),
    .rst      ( rst          ),
    .in_gate  ( sg_in_gate   ),
    .in_up    ( sg_in_up     ),
    .in_valid ( sg_in_valid  ),
    .out_y    ( sg_out_y     ),
    .out_valid( sg_out_valid )
  );

  // ----------------------- FSM -----------------------
  typedef enum logic [1:0] {
    S_INIT,    // clear counters, go straight to S_RUN
    S_RUN,     // drive N inputs and collect N outputs (1-cycle offset)
    S_DONE
  } state_t;
  state_t state;

  logic [CW-1:0] in_cnt;   // counts inputs driven (0..N)
  logic [CW-1:0] out_cnt;  // counts outputs collected (0..N)

  always_ff @(posedge clk) begin
    if (rst) begin
      state        <= S_INIT;
      in_cnt       <= '0;
      out_cnt      <= '0;
      sg_in_valid  <= 1'b0;
      sg_in_gate   <= '0;
      sg_in_up     <= '0;
      result       <= '0;
      done         <= 1'b0;
    end else if (restart) begin
      state        <= S_INIT;
      in_cnt       <= '0;
      out_cnt      <= '0;
      sg_in_valid  <= 1'b0;
      done         <= 1'b0;
    end else begin
      // defaults
      sg_in_valid <= 1'b0;

      case (state)
        S_INIT: begin
          in_cnt  <= '0;
          out_cnt <= '0;
          state   <= S_RUN;
        end

        S_RUN: begin
          // --- drive side: feed inputs while in_cnt < N ---
          if (in_cnt < CW'(N)) begin
            sg_in_gate  <= SWIGLU_GATE_PACKED[in_cnt*16 +: 16];
            sg_in_up    <= SWIGLU_UP_PACKED  [in_cnt*16 +: 16];
            sg_in_valid <= 1'b1;
            in_cnt      <= in_cnt + 1'b1;
          end

          // --- collect side: capture valid outputs ---
          if (sg_out_valid) begin
            result[out_cnt*16 +: 16] <= sg_out_y;
            out_cnt                  <= out_cnt + 1'b1;
            if (out_cnt == CW'(N-1)) begin
              done  <= 1'b1;
              state <= S_DONE;
            end
          end
        end

        S_DONE: begin
          // park until restart
        end

        default: state <= S_INIT;
      endcase
    end
  end

endmodule

`default_nettype wire
