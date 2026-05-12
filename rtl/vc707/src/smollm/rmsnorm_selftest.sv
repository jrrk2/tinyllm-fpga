// rmsnorm_selftest.sv — drives `rmsnorm` once at boot with hardcoded test
// data, latches the D-element Q1.15 result for the host to read via the
// register map.  Re-triggers on a `restart` pulse.
//
// Test data comes from $readmemh files in ../generated/:
//   rmsnorm_selftest_x.hex      D × 16-bit Q1.15 input vector
//   rmsnorm_selftest_gamma.hex  D × 16-bit Q1.15 scale vector
//
// `result` is a packed D × 16-bit bus, lane 0 in [15:0].

`default_nettype none

module rmsnorm_selftest #(
  parameter int D = 64
)(
  input  wire              clk,
  input  wire              rst,
  input  wire              restart,
  output logic [D*16-1:0]  result,
  output logic             done
);

  localparam int CW = $clog2(D + 1);

  // ----------------------- ROM (packed localparams) -----------------------
  // Auto-generated SVH — uses packed localparam literals rather than
  // $readmemh, sidestepping the Vivado quirks we hit on matvec_selftest.
`include "rmsnorm_selftest_data.svh"

  // ----------------------- rmsnorm DUT -----------------------
  logic                rms_start;
  logic signed [15:0]  rms_in_x;
  logic signed [15:0]  rms_in_gamma;
  logic                rms_in_valid;
  logic signed [15:0]  rms_out_y;
  logic                rms_out_valid;
  logic                rms_done;

  rmsnorm #(.D(D)) i_rms (
    .clk      ( clk           ),
    .rst      ( rst           ),
    .start    ( rms_start     ),
    .in_x     ( rms_in_x      ),
    .in_gamma ( rms_in_gamma  ),
    .in_valid ( rms_in_valid  ),
    .out_y    ( rms_out_y     ),
    .out_valid( rms_out_valid ),
    .done     ( rms_done      )
  );

  // ----------------------- FSM -----------------------
  typedef enum logic [2:0] {
    S_INIT,    // pulse rmsnorm.start
    S_DRIVE,   // stream D inputs
    S_COLLECT, // capture D outputs
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
      rms_start    <= 1'b0;
      rms_in_valid <= 1'b0;
      rms_in_x     <= '0;
      rms_in_gamma <= '0;
      result       <= '0;
      done         <= 1'b0;
    end else if (restart) begin
      state        <= S_INIT;
      in_cnt       <= '0;
      out_cnt      <= '0;
      rms_start    <= 1'b0;
      rms_in_valid <= 1'b0;
      done         <= 1'b0;
    end else begin
      // defaults
      rms_start    <= 1'b0;
      rms_in_valid <= 1'b0;

      case (state)
        S_INIT: begin
          rms_start <= 1'b1;
          in_cnt    <= '0;
          out_cnt   <= '0;
          state     <= S_DRIVE;
        end

        S_DRIVE: begin
          // rmsnorm enters S_LOAD on the cycle after start; we drive D
          // valid inputs starting THIS cycle.  Indexed slice from packed
          // localparams — Vivado synthesizes as a small ROM/LUT.
          rms_in_x     <= RMSNORM_X_PACKED    [in_cnt*16 +: 16];
          rms_in_gamma <= RMSNORM_GAMMA_PACKED[in_cnt*16 +: 16];
          rms_in_valid <= 1'b1;
          in_cnt       <= in_cnt + 1'b1;
          if (in_cnt == CW'(D-1)) state <= S_COLLECT;
        end

        S_COLLECT: begin
          if (rms_out_valid) begin
            result[out_cnt*16 +: 16] <= rms_out_y;
            out_cnt                  <= out_cnt + 1'b1;
            if (out_cnt == CW'(D-1)) begin
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
