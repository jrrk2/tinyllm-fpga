// rope_selftest.sv — drives `rope` once at boot with hardcoded test data,
// latches the HEAD_DIM × Q1.15 result for the host to read via the register
// map.  Re-triggers on a `restart` pulse.
//
// Test data comes from ../generated/rope_selftest_data.svh (packed localparam,
// no $readmemh).
//
// `result` is a packed HEAD_DIM × 16-bit bus, lane 0 in [15:0].

`default_nettype none

module rope_selftest #(
  parameter int HEAD_DIM = 64,
  parameter int MAX_CTX  = 2048
)(
  input  wire                    clk,
  input  wire                    rst,
  input  wire                    restart,
  output logic [HEAD_DIM*16-1:0] result,
  output logic                   done
);

  localparam int CW = $clog2(HEAD_DIM + 1);

  // ----------------------- ROM (packed localparams) -----------------------
  // Auto-generated SVH — uses packed localparam literals rather than
  // $readmemh.
`include "rope_selftest_data.svh"

  // ----------------------- rope DUT -----------------------
  logic                                   rope_start;
  logic [$clog2(MAX_CTX)-1:0]             rope_pos;
  logic signed [15:0]                     rope_in_x;
  logic                                   rope_in_valid;
  logic signed [15:0]                     rope_out_y;
  logic                                   rope_out_valid;
  logic                                   rope_done;

  rope #(
    .HEAD_DIM ( HEAD_DIM ),
    .MAX_CTX  ( MAX_CTX  )
  ) i_rope (
    .clk       ( clk            ),
    .rst       ( rst            ),
    .start     ( rope_start     ),
    .pos       ( rope_pos       ),
    .in_x      ( rope_in_x      ),
    .in_valid  ( rope_in_valid  ),
    .out_y     ( rope_out_y     ),
    .out_valid ( rope_out_valid ),
    .done      ( rope_done      )
  );

  // ----------------------- FSM -----------------------
  typedef enum logic [2:0] {
    S_INIT,    // pulse rope.start with pos=ROPE_POS
    S_DRIVE,   // stream HEAD_DIM inputs
    S_COLLECT, // capture HEAD_DIM out_y/out_valid beats into result
    S_DONE
  } state_t;
  state_t state;

  logic [CW-1:0] in_cnt;
  logic [CW-1:0] out_cnt;

  always_ff @(posedge clk) begin
    if (rst) begin
      state          <= S_INIT;
      in_cnt         <= '0;
      out_cnt        <= '0;
      rope_start     <= 1'b0;
      rope_in_valid  <= 1'b0;
      rope_in_x      <= '0;
      rope_pos       <= '0;
      result         <= '0;
      done           <= 1'b0;
    end else if (restart) begin
      state          <= S_INIT;
      in_cnt         <= '0;
      out_cnt        <= '0;
      rope_start     <= 1'b0;
      rope_in_valid  <= 1'b0;
      done           <= 1'b0;
    end else begin
      // defaults
      rope_start    <= 1'b0;
      rope_in_valid <= 1'b0;

      case (state)
        S_INIT: begin
          rope_start <= 1'b1;
          rope_pos   <= $clog2(MAX_CTX)'(ROPE_POS);
          in_cnt     <= '0;
          out_cnt    <= '0;
          state      <= S_DRIVE;
        end

        S_DRIVE: begin
          // rope enters S_LOAD on the cycle after start; we drive HEAD_DIM
          // valid inputs starting THIS cycle.  Indexed slice from packed
          // localparam — synthesizes as a small ROM/LUT.
          rope_in_x     <= ROPE_X_PACKED[in_cnt*16 +: 16];
          rope_in_valid <= 1'b1;
          in_cnt        <= in_cnt + 1'b1;
          if (in_cnt == CW'(HEAD_DIM - 1)) state <= S_COLLECT;
        end

        S_COLLECT: begin
          if (rope_out_valid) begin
            result[out_cnt*16 +: 16] <= rope_out_y;
            out_cnt                  <= out_cnt + 1'b1;
            if (out_cnt == CW'(HEAD_DIM - 1)) begin
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
