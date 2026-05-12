// gqa_attention.sv — Grouped-query attention with KV cache.
//
// SKELETON. See PLAN.md "GQA attention".
//
// SmolLM2 uses 9 query heads but only 3 KV heads (3:1 ratio). Each KV
// head serves 3 Q heads.
//
// For each Q head h ∈ [0..8] at the current position pos:
//   kv_h = h / 3
//   For t = 0..pos:
//     score[t] = (Q_h · K_cache[layer][kv_h][t]) / sqrt(64)
//   softmax(score) → attn
//   out_h = sum_t( attn[t] * V_cache[layer][kv_h][t] )
//
// KV cache is in DDR3 with addressing (layer, kv_head, t, dim).  The
// new K,V at position pos must first be written to the cache before
// reading the (pos+1)-length sequence back.
//
// Per-head compute: (pos+1) × 64 muls for Q·K + softmax + (pos+1) × 64
// muls for weighted sum.  At pos=512 average and 9 heads: ~590K muls
// per layer attention block.  Memory-bound on KV cache reads.

`default_nettype none

module gqa_attention #(
  parameter int HIDDEN     = 576,
  parameter int N_HEADS    = 9,
  parameter int N_KV_HEADS = 3,
  parameter int HEAD_DIM   = 64,
  parameter int MAX_CTX    = 1024
)(
  input  wire        clk,
  input  wire        rst,

  input  wire        start,
  input  wire [4:0]  layer_idx,    // for KV cache addressing
  input  wire [9:0]  pos,          // current token position
  output logic       busy,
  output logic       done,

  // Q vector input (576 FP16, written by Wq+RoPE stage)
  output logic [$clog2(HIDDEN)-1:0] q_addr,
  input  wire  [15:0]               q_data,
  // K,V at current position (already in BRAM from Wk/Wv+RoPE stages)
  output logic [$clog2(N_KV_HEADS*HEAD_DIM)-1:0] kv_now_addr,
  input  wire  [31:0]                            kv_now_data,  // K hi, V lo

  // Output: attention result (HIDDEN FP16) written to a result BRAM
  output logic [$clog2(HIDDEN)-1:0] out_addr,
  output logic [15:0]               out_data,
  output logic                      out_en,

  // KV cache AXI interface (writes new K,V; reads past K,V)
  // (Wired into the smollm_core's outer AXI multiplexer)
  output logic                      kv_wr_req,
  output logic [29:0]               kv_wr_addr,
  output logic [511:0]              kv_wr_data,
  output logic                      kv_rd_req,
  output logic [29:0]               kv_rd_addr,
  input  wire  [511:0]              kv_rd_data,
  input  wire                       kv_rd_valid
);
  // TODO: full GQA implementation.

  always_ff @(posedge clk) begin
    if (rst) begin busy<=0; done<=0; out_en<=0; kv_wr_req<=0; kv_rd_req<=0; end
  end
endmodule

`default_nettype wire
