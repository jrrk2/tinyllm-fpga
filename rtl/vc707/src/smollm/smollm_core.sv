// smollm_core.sv — top FSM for SmolLM2-135M-Instruct inference.
//
// SKELETON — port list and state diagram only. See PLAN.md.
//
// Replaces microgpt_exact_core for the SmolLM2 build. Driven by
// vc707_microgpt_eth.sv via the same Avalon-MM-style register interface
// that microgpt_exact_core used (start, busy, done, token_in, pos_in,
// next_token), so the eth_ctrl + register-map slave logic in the top
// is unchanged.
//
// FSM states (one per token; loop until BOS or out_len cap):
//
//   ST_LOAD_EMBED    : DDR3 → tile cache: token's embedding row.
//   ST_LAYER_ATTN_QKV: per layer, stream Wq/Wk/Wv through matvec.
//   ST_LAYER_ROPE    : apply RoPE to Q,K (head_dim=64).
//   ST_LAYER_KV_WR   : append K,V to DDR3 KV cache at (layer, t).
//   ST_LAYER_ATTN    : softmax(Q·K_cache / √64) · V_cache, per Q head.
//   ST_LAYER_O       : Wo projection.
//   ST_LAYER_RES_NRM : residual + RMSNorm.
//   ST_LAYER_MLP_GU  : Wg, Wu projections (interleaved).
//   ST_LAYER_SWIGLU  : SiLU(g) ⊙ u.
//   ST_LAYER_DOWN    : Wd projection.
//   ST_LAYER_RES_NRM2: residual + RMSNorm.
//                  (loop ST_LAYER_* for layer_idx = 0..29)
//   ST_FINAL_NORM    : final RMSNorm on hidden state.
//   ST_LM_HEAD       : tied lm_head projection → 49152 logits.
//   ST_SAMPLE        : argmax / temperature → next token ID.
//   ST_DONE          : assert done; idle until next start.
//
// All weight access goes through `weight_address_gen` → `weight_tile_cache`
// → matvec engine.  The hidden state vector (576 FP16) and per-head
// accumulators live in BRAM.

`default_nettype none

module smollm_core (
  input  wire           clk,         // core_clk (target 100 MHz)
  input  wire           resetn,

  // Token I/O (matches microgpt_exact_core port style)
  input  wire           start,
  input  wire           clear_cache,    // resets KV cache + position
  input  wire           sample_mode,    // 1 = sample, 0 = argmax
  input  wire  [15:0]   temperature_q8_8,
  input  wire  [31:0]   rng_state_in,
  input  wire  [15:0]   token_in,       // 16-bit ID (vocab=49152)
  input  wire  [9:0]    pos_in,         // 0..1023 supported context
  output logic          busy,
  output logic          done,
  output logic [15:0]   next_token,
  output logic [31:0]   rng_state_out,
  // (No argmax_token / top_logit / logits_flat — vocab 49152 is too big
  //  to emit each forward pass.  Add a debug readback if needed.)

  // AXI master to MIG (shared with the rest of the system; only one
  // active master at a time in this build)
  output logic          m_axi_arvalid,
  input  wire           m_axi_arready,
  output logic [4:0]    m_axi_arid,
  output logic [29:0]   m_axi_araddr,
  output logic [7:0]    m_axi_arlen,
  output logic [2:0]    m_axi_arsize,
  output logic [1:0]    m_axi_arburst,
  output logic          m_axi_arlock,
  output logic [3:0]    m_axi_arcache,
  output logic [2:0]    m_axi_arprot,
  output logic [3:0]    m_axi_arqos,
  input  wire           m_axi_rvalid,
  output wire           m_axi_rready,
  input  wire  [4:0]    m_axi_rid,
  input  wire  [511:0]  m_axi_rdata,
  input  wire  [1:0]    m_axi_rresp,
  input  wire           m_axi_rlast,
  // Write channel (for KV cache writes)
  output logic          m_axi_awvalid,
  input  wire           m_axi_awready,
  output logic [4:0]    m_axi_awid,
  output logic [29:0]   m_axi_awaddr,
  output logic [7:0]    m_axi_awlen,
  output logic [2:0]    m_axi_awsize,
  output logic [1:0]    m_axi_awburst,
  output logic          m_axi_awlock,
  output logic [3:0]    m_axi_awcache,
  output logic [2:0]    m_axi_awprot,
  output logic [3:0]    m_axi_awqos,
  output logic          m_axi_wvalid,
  input  wire           m_axi_wready,
  output logic [511:0]  m_axi_wdata,
  output logic [63:0]   m_axi_wstrb,
  output logic          m_axi_wlast,
  input  wire           m_axi_bvalid,
  output wire           m_axi_bready,
  input  wire  [4:0]    m_axi_bid,
  input  wire  [1:0]    m_axi_bresp
);

  // ================================================================
  //  SmolLM2-135M architectural constants
  // ================================================================
  localparam int HIDDEN          = 576;
  localparam int FFN             = 1536;
  localparam int N_LAYERS        = 30;
  localparam int N_HEADS         = 9;
  localparam int N_KV_HEADS      = 3;        // GQA ratio 3:1
  localparam int HEAD_DIM        = 64;
  localparam int VOCAB           = 49152;
  localparam int MAX_CTX         = 1024;     // we'll cap KV cache here
  localparam logic [15:0] RMSNORM_EPS_Q1_15 = 16'h0000; // ~1e-5 in Q1.15

  // ================================================================
  //  Sub-block instances will be added here:
  //
  //  weight_address_gen  i_addr_gen (...);   // (layer, op) -> DDR3 addr
  //  weight_tile_cache   i_wtc      (...);   // ping-pong tile streamer
  //  matvec_int8_engine  i_matvec   (...);   // 64-lane INT8 × FP16 MAC
  //  rmsnorm             i_rmsnorm  (...);
  //  rope                i_rope     (...);
  //  swiglu              i_swiglu   (...);
  //  gqa_attention       i_attn     (...);
  //  sampler             i_sampler  (...);
  //
  //  Hidden state BRAM (1 KiB):     hidden[0..575] FP16
  //  KV-cache write FIFO (small):   stages writes to DDR3
  // ================================================================

  // Skeleton — full FSM logic to be implemented per PLAN.md.

  initial begin
    // synthesizer-friendly placeholder
  end

  assign m_axi_rready = 1'b1;
  assign m_axi_bready = 1'b1;

  always_ff @(posedge clk or negedge resetn) begin
    if (!resetn) begin
      busy          <= 1'b0;
      done          <= 1'b0;
      next_token    <= '0;
      rng_state_out <= '0;
      m_axi_arvalid <= 1'b0;
      m_axi_awvalid <= 1'b0;
      m_axi_wvalid  <= 1'b0;
    end else begin
      // TODO: full per-token FSM
    end
  end

endmodule

`default_nettype wire
