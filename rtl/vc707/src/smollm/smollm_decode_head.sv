// smollm_decode_head.sv — final-norm + LM-head + argmax sampler.
//
// Takes a post-layers hidden_state (D-dim Q1.15) and produces a single
// next-token index.  Pipeline:
//
//   hidden_state ─→ final-RMSNorm(γ) ─→ matvec(W_lm) ─→ argmax ─→ next_token
//
// Reuses the same matvec_int8_engine and rmsnorm modules as the
// transformer layers.  argmax is computed inline with the LM-head matvec
// output stream — each chunk's 16 lane logits are compared against a
// running max and the best lane index latched.

`default_nettype none

module smollm_decode_head #(
  parameter int D     = 128,
  parameter int VOCAB = 128,
  parameter     PREFIX = "decodehead_"
)(
  input  wire                          clk,
  input  wire                          rst,
  input  wire                          start,
  input  wire signed [D*16-1:0]        hidden_state,   // post-layers
  output logic [$clog2(VOCAB)-1:0]     next_token,
  output logic signed [15:0]           top_logit,      // for diag
  output logic                         done
);

  localparam int CHUNKS_LM = VOCAB / 16;     // 8 for VOCAB=128

  // ----- ROMs (loaded via $readmemh) -----
  logic signed [15:0]  rom_GAMMA  [0:D-1];
  logic [127:0]        rom_W_LM   [0:CHUNKS_LM * D - 1];
  logic signed [15:0]  rom_S_LM   [0:VOCAB-1];

`ifdef MICROGPT_WEIGHT_DIR
  initial begin
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "GAMMA.hex"},   rom_GAMMA);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "W_LM.hex"},    rom_W_LM);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "SCALE_LM.hex"},rom_S_LM);
  end
`else
  initial begin
    $readmemh({PREFIX, "GAMMA.hex"},    rom_GAMMA);
    $readmemh({PREFIX, "W_LM.hex"},     rom_W_LM);
    $readmemh({PREFIX, "SCALE_LM.hex"}, rom_S_LM);
  end
`endif

  // ----- rmsnorm instance -----
  logic                rms_start, rms_in_valid, rms_out_valid, rms_done;
  logic signed [15:0]  rms_in_x, rms_in_gamma, rms_out_y;
  rmsnorm #(.D(D)) i_rms (
    .clk, .rst, .start(rms_start),
    .in_x(rms_in_x), .in_gamma(rms_in_gamma), .in_valid(rms_in_valid),
    .out_y(rms_out_y), .out_valid(rms_out_valid), .done(rms_done)
  );

  // ----- matvec engine for LM head -----
  logic signed [15:0]   eng_in_value;
  logic                 eng_in_valid, eng_in_last;
  logic [127:0]         eng_w;
  logic [255:0]         eng_scale;
  logic                 eng_scale_valid, eng_acc_clear;
  logic [255:0]         eng_out;
  logic                 eng_out_valid;
  matvec_int8_engine #(.LANES(16), .ACC_W(40)) i_eng (
    .clk, .rst,
    .in_value(eng_in_value),    .in_valid(eng_in_valid),
    .in_last(eng_in_last),      .w_int8(eng_w),
    .scale_q15(eng_scale),      .scale_valid(eng_scale_valid),
    .out_value(eng_out),        .out_valid(eng_out_valid),
    .acc_clear(eng_acc_clear)
  );

  // ----- Buffer for the normed hidden -----
  logic signed [15:0] normed_buf [0:D-1];

  // ----- Argmax tracking state -----
  logic signed [15:0]              best_logit;
  logic [$clog2(VOCAB)-1:0]        best_idx;

  // Combinational chunk-level argmax: find the max over the 16 lanes of
  // the current `eng_out` and report which lane it came from.
  logic signed [15:0] chunk_max;
  logic [3:0]         chunk_max_lane;
  always_comb begin : chunk_argmax
    chunk_max      = $signed(eng_out[15:0]);  // lane 0 seed
    chunk_max_lane = 4'd0;
    for (int l = 1; l < 16; l++) begin
      if ($signed(eng_out[l*16 +: 16]) > $signed(chunk_max)) begin
        chunk_max      = $signed(eng_out[l*16 +: 16]);
        chunk_max_lane = l[3:0];
      end
    end
  end

  // ----- FSM -----
  typedef enum logic [3:0] {
    S_IDLE,
    S_NORM_PULSE,        // 1 cycle: pulse rms.start
    S_NORM_LOAD,         // D cycles: drive rms.in_x / in_gamma / in_valid
    S_NORM_OUTPUT,       // D cycles: capture rms.out_y → normed_buf
    S_LM_CLEAR,
    S_LM_DRIVE,
    S_LM_DRAIN,
    S_LM_SCALE,
    S_LM_WAIT,           // capture eng_out, run argmax inline
    S_DONE
  } state_t;
  state_t state;

  logic [10:0] cnt;
  logic [6:0]  chunk;       // 0..CHUNKS_LM-1

  always_ff @(posedge clk) begin
    if (rst) begin
      state           <= S_IDLE;
      cnt             <= '0;
      chunk           <= '0;
      best_logit      <= -16'sd32768;
      best_idx        <= '0;
      next_token      <= '0;
      top_logit       <= '0;
      done            <= 1'b0;
      rms_start       <= 1'b0;
      rms_in_valid    <= 1'b0;
      eng_in_valid    <= 1'b0;
      eng_in_last     <= 1'b0;
      eng_scale_valid <= 1'b0;
      eng_acc_clear   <= 1'b0;
    end else begin
      rms_start       <= 1'b0;
      rms_in_valid    <= 1'b0;
      eng_in_valid    <= 1'b0;
      eng_in_last     <= 1'b0;
      eng_scale_valid <= 1'b0;
      eng_acc_clear   <= 1'b0;
      done            <= 1'b0;

      case (state)
        S_IDLE: begin
          if (start) begin
            state      <= S_NORM_PULSE;
            cnt        <= '0;
            chunk      <= '0;
            best_logit <= -16'sd32768;
            best_idx   <= '0;
          end
        end

        S_NORM_PULSE: begin
          rms_start <= 1'b1;
          state     <= S_NORM_LOAD;
        end

        S_NORM_LOAD: begin
          rms_in_x     <= hidden_state[cnt*16 +: 16];
          rms_in_gamma <= rom_GAMMA[cnt];
          rms_in_valid <= 1'b1;
          cnt          <= cnt + 1'b1;
          if (cnt == D-1) begin
            state <= S_NORM_OUTPUT;
            cnt   <= '0;
          end
        end

        S_NORM_OUTPUT: begin
          if (rms_out_valid) begin
            normed_buf[cnt] <= rms_out_y;
            cnt             <= cnt + 1'b1;
            if (cnt == D-1) begin
              state    <= S_LM_CLEAR;
              cnt      <= '0;
              chunk    <= '0;
            end
          end
        end

        // ----- LM-head matvec, 8 chunks of 16 lanes each -----
        S_LM_CLEAR: begin
          eng_acc_clear <= 1'b1;
          cnt           <= '0;
          state         <= S_LM_DRIVE;
        end

        S_LM_DRIVE: begin
          eng_in_value <= normed_buf[cnt];
          eng_w        <= rom_W_LM[chunk * D + cnt];
          eng_in_valid <= 1'b1;
          eng_in_last  <= (cnt == D-1);
          cnt          <= cnt + 1'b1;
          if (cnt == D-1) state <= S_LM_DRAIN;
        end

        S_LM_DRAIN: state <= S_LM_SCALE;

        S_LM_SCALE: begin
          for (int l = 0; l < 16; l++)
            eng_scale[l*16 +: 16] <= rom_S_LM[chunk*16 + l];
          eng_scale_valid <= 1'b1;
          state           <= S_LM_WAIT;
        end

        S_LM_WAIT: begin
          if (eng_out_valid) begin
            // Combinational chunk-argmax already computed (chunk_max,
            // chunk_max_lane); compare against running best.
            if ($signed(chunk_max) > $signed(best_logit)) begin
              best_logit <= chunk_max;
              best_idx   <= ($clog2(VOCAB))'(chunk * 16 + chunk_max_lane);
            end
            chunk <= chunk + 1'b1;
            if (chunk == CHUNKS_LM - 1) begin
              state <= S_DONE;
            end else begin
              state <= S_LM_CLEAR;
            end
          end
        end

        S_DONE: begin
          next_token <= best_idx;
          top_logit  <= best_logit;
          done       <= 1'b1;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
