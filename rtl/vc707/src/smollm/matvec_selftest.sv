// matvec_selftest.sv — runs matvec_int8_engine once at boot with hardcoded
// test data, latches the 16-lane Q1.15 output for the host to read via the
// Avalon-MM register map.
//
// Test data comes from $readmemh files in ../generated/:
//   matvec_selftest_act.hex      IN_DIM × 16-bit activation
//   matvec_selftest_weights.hex  IN_DIM × 128-bit packed weights (lane 0 in low byte)
//   matvec_selftest_scale.hex    16    × 16-bit Q1.15 scale (lane 0 first)
//
// `result` is a 256-bit packed Q1.15 vector (lane 0 in result[15:0]).
// `done` goes high one cycle after the engine produces its first output and
// stays high — there is no re-trigger.

`default_nettype none

module matvec_selftest #(
  parameter int IN_DIM           = 64,
  parameter     ACT_HEX_PATH     = "matvec_selftest_act.hex",
  parameter     WEIGHTS_HEX_PATH = "matvec_selftest_weights.hex",
  parameter     SCALE_HEX_PATH   = "matvec_selftest_scale.hex"
)(
  input  wire          clk,
  input  wire          rst,
  input  wire          restart,    // pulse high for 1 cycle to re-run
  output logic [255:0] result,
  output logic         done
);

  localparam int LANES = 16;
  localparam int CW    = $clog2(IN_DIM + 1);

  // ----------------------- ROM init -----------------------
  // act_rom and weight_rom are read with a registered counter so Vivado
  // infers them as proper memories and honours $readmemh.  scale_rom was
  // originally also $readmemh-loaded but Vivado silently rejected it
  // (combinational read pattern → "ignoring malformed $readmem task:
  // invalid memory name").  Replaced with a case-statement function
  // generated into matvec_selftest_scale.svh.
  logic signed [15:0]  act_rom    [0:IN_DIM-1];
  logic        [127:0] weight_rom [0:IN_DIM-1];

`ifdef MICROGPT_WEIGHT_DIR
  initial begin
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", ACT_HEX_PATH    }, act_rom);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", WEIGHTS_HEX_PATH}, weight_rom);
  end
`else
  initial begin
    $readmemh(ACT_HEX_PATH,     act_rom);
    $readmemh(WEIGHTS_HEX_PATH, weight_rom);
  end
`endif

  // Per-lane scale ROM, packed into a single 256-bit constant via an
  // auto-generated SVH file (defines `SCALE_PACKED_INIT`).
`include "matvec_selftest_scale.svh"
  wire [255:0] scale_packed = SCALE_PACKED_INIT;

  // ----------------------- engine instance -----------------------
  logic signed [15:0]  eng_in_value;
  logic                eng_in_valid;
  logic                eng_in_last;
  logic [LANES*8-1:0]  eng_w;
  logic [255:0]        eng_scale;
  logic                eng_scale_valid;
  logic                eng_acc_clear;
  logic [255:0]        eng_out;
  logic                eng_out_valid;

  matvec_int8_engine #(
    .LANES (LANES),
    .ACC_W (40)
  ) i_eng (
    .clk         ( clk             ),
    .rst         ( rst             ),
    .in_value    ( eng_in_value    ),
    .in_valid    ( eng_in_valid    ),
    .in_last     ( eng_in_last     ),
    .w_int8      ( eng_w           ),
    .scale_q15   ( eng_scale       ),
    .scale_valid ( eng_scale_valid ),
    .out_value   ( eng_out         ),
    .out_valid   ( eng_out_valid   ),
    .acc_clear   ( eng_acc_clear   )
  );

  // ----------------------- FSM -----------------------
  typedef enum logic [2:0] {
    S_INIT,    // 1 cycle to assert acc_clear
    S_DRIVE,   // IN_DIM cycles streaming activation+weights
    S_DRAIN,   // 1 cycle for MAC to settle
    S_SCALE,   // 1 cycle of scale_valid
    S_WAIT,    // wait for out_valid
    S_DONE
  } state_t;
  state_t state;

  logic [CW-1:0] cnt;

  always_ff @(posedge clk) begin
    if (rst) begin
      state           <= S_INIT;
      cnt             <= '0;
      eng_in_value    <= '0;
      eng_in_valid    <= 1'b0;
      eng_in_last     <= 1'b0;
      eng_w           <= '0;
      eng_scale       <= '0;
      eng_scale_valid <= 1'b0;
      eng_acc_clear   <= 1'b0;
      result          <= '0;
      done            <= 1'b0;
    end else if (restart) begin
      // Re-arm: jump back to S_INIT and clear done.  result is left as it
      // was so the host can still tell us we're mid-rerun via done=0.
      state           <= S_INIT;
      cnt             <= '0;
      eng_in_valid    <= 1'b0;
      eng_in_last     <= 1'b0;
      eng_scale_valid <= 1'b0;
      eng_acc_clear   <= 1'b0;
      done            <= 1'b0;
    end else begin
      eng_in_valid    <= 1'b0;
      eng_in_last     <= 1'b0;
      eng_scale_valid <= 1'b0;
      eng_acc_clear   <= 1'b0;

      case (state)
        S_INIT: begin
          eng_acc_clear <= 1'b1;
          cnt           <= '0;
          state         <= S_DRIVE;
        end

        S_DRIVE: begin
          eng_in_value <= act_rom[cnt];
          eng_w        <= weight_rom[cnt];
          eng_in_valid <= 1'b1;
          eng_in_last  <= (cnt == CW'(IN_DIM-1));
          cnt          <= cnt + 1'b1;
          if (cnt == CW'(IN_DIM-1)) state <= S_DRAIN;
        end

        S_DRAIN: begin
          // single-cycle MAC settles
          state <= S_SCALE;
        end

        S_SCALE: begin
          eng_scale       <= scale_packed;
          eng_scale_valid <= 1'b1;
          state           <= S_WAIT;
        end

        S_WAIT: begin
          if (eng_out_valid) begin
            result <= eng_out;
            done   <= 1'b1;
            state  <= S_DONE;
          end
        end

        S_DONE: begin
          // park forever
        end

        default: state <= S_INIT;
      endcase
    end
  end

endmodule

`default_nettype wire
