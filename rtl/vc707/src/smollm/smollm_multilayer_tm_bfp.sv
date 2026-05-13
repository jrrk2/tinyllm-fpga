// smollm_multilayer_tm_bfp.sv — block-FP time-mux wrapper.
//
// Wraps a single smollm_layer_bfp instance (NL layers in its ROM banks),
// calls it NL times per token-step, bumping layer_idx 0..NL-1 and feeding
// each layer's hidden_out back as the next layer's hidden_in.  KV cache
// inside the layer persists per-layer across token steps because the
// layer owns the per-layer cache slice indexed by {layer_idx, kv_pos, …}.
//
// The interface mirrors smollm_layer_bfp's single-layer interface — a
// token-step harness drives `start` once per token with the embedded
// hidden_in vector, current `pos`/`kv_pos`, and reads `hidden_out` when
// `done` pulses NL layer-calls later.
//
// No DDR3 streamer here.  Weights live in BRAM-loaded ROMs inside the
// inner smollm_layer_bfp (loaded via $readmemh from lbfp_full_*.hex).

`include "bfp_format.svh"

`default_nettype none

module smollm_multilayer_tm_bfp #(
  parameter int    D       = 576,
  parameter int    H_Q     = 9,
  parameter int    H_KV    = 3,
  parameter int    HD      = 64,
  parameter int    FFN     = 1536,
  parameter int    MAX_CTX = 64,
  parameter int    NL      = 30,
  parameter        PREFIX  = "lbfp_full_"
)(
  input  wire                                       clk,
  input  wire                                       rst,
  input  wire                                       start,
  input  wire [10:0]                                pos,
  input  wire [6:0]                                 kv_pos,
  input  wire signed [D*BFP_MANT_W-1:0]             hidden_in_m,
  input  wire signed [(D/BFP_TILE)*BFP_EXP_W-1:0]   hidden_in_e,
  output logic signed [D*BFP_MANT_W-1:0]            hidden_out_m,
  output logic signed [(D/BFP_TILE)*BFP_EXP_W-1:0]  hidden_out_e,
  output logic                                      done
);

  localparam int NT_D = D / BFP_TILE;

  // ---------------------------------------------------------------------------
  // Hidden state shuttle between layers
  // ---------------------------------------------------------------------------
  logic signed [D*BFP_MANT_W-1:0]      h_state_m;
  logic signed [NT_D*BFP_EXP_W-1:0]    h_state_e;

  // ---------------------------------------------------------------------------
  // Inner layer instance — NL ROMs preloaded from lbfp_full_*.hex
  // ---------------------------------------------------------------------------
  logic                                lay_start;
  logic [4:0]                          lay_idx;
  wire  signed [D*BFP_MANT_W-1:0]      lay_hidden_out_m;
  wire  signed [NT_D*BFP_EXP_W-1:0]    lay_hidden_out_e;
  wire                                 lay_done;
  wire  [5:0]  ignore_state;
  wire  [11:0] ignore_cnt;
  wire  [6:0]  ignore_chunk;

  smollm_layer_bfp #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD),
    .FFN(FFN), .MAX_CTX(MAX_CTX), .NL(NL), .PREFIX(PREFIX)
  ) i_lay (
    .clk(clk), .rst(rst),
    .start(lay_start),
    .pos(pos), .kv_pos(kv_pos),
    .layer_idx(lay_idx),
    .hidden_in_m(h_state_m), .hidden_in_e(h_state_e),
    .hidden_out_m(lay_hidden_out_m), .hidden_out_e(lay_hidden_out_e),
    .done(lay_done),
    .wr_kind(5'd0), .wr_addr(18'd0), .wr_data(16'd0), .wr_en(1'b0),
    .dbg_state(ignore_state), .dbg_cnt(ignore_cnt), .dbg_chunk(ignore_chunk)
  );

  // ---------------------------------------------------------------------------
  // FSM
  // ---------------------------------------------------------------------------
  typedef enum logic [2:0] { S_IDLE, S_LATCH, S_PULSE, S_WAIT, S_CAPTURE, S_NEXT, S_DONE } st_t;
  st_t state;

  always_ff @(posedge clk) begin
    if (rst) begin
      state        <= S_IDLE;
      lay_idx      <= '0;
      lay_start    <= 1'b0;
      h_state_m    <= '0;
      h_state_e    <= '0;
      hidden_out_m <= '0;
      hidden_out_e <= '0;
      done         <= 1'b0;
    end else begin
      lay_start <= 1'b0;
      done      <= 1'b0;
      case (state)
        S_IDLE: if (start) begin
          h_state_m <= hidden_in_m;
          h_state_e <= hidden_in_e;
          lay_idx   <= '0;
          state     <= S_LATCH;
        end
        S_LATCH: begin
          // 1 cycle for the layer's internal LATCH_IN to start cleanly.
          state <= S_PULSE;
        end
        S_PULSE: begin
          lay_start <= 1'b1;
          state     <= S_WAIT;
        end
        S_WAIT: if (lay_done) state <= S_CAPTURE;
        S_CAPTURE: begin
          h_state_m <= lay_hidden_out_m;
          h_state_e <= lay_hidden_out_e;
          state     <= S_NEXT;
        end
        S_NEXT: begin
          if (lay_idx == NL - 1) begin
            hidden_out_m <= h_state_m;
            hidden_out_e <= h_state_e;
            state        <= S_DONE;
          end else begin
            lay_idx <= lay_idx + 1'b1;
            state   <= S_PULSE;
          end
        end
        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
