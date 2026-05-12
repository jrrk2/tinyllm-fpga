// weight_address_gen.sv — (layer, op) → (DDR3 byte addr, byte count).
//
// SKELETON. See PLAN.md "DDR3 weight layout" for the address map.
//
// Computes the start address and tile count for each tensor in the
// SmolLM2-135M model.  Driven by the smollm_core FSM at a layer/op
// boundary; output drives weight_tile_cache.load_addr.
//
// Layout (matches PLAN.md):
//
//   0x0000_0000 :  embed_tokens / lm_head      (28,311,552 bytes)
//   0x01B0_0000 :  layer_0 weights             (3,538,944 bytes)
//   0x01B3_5FC0 :  layer_1 weights
//   ...                                          (30 × 3.38 MB)
//   0x07F0_0000 :  per-channel FP16 scales
//   0x0800_0000 :  KV cache region
//
// Within each layer, tensors are packed in this order (matches the FSM):
//   Wq, Wk, Wv, Wo, Wg, Wu, Wd, gamma_attn, gamma_mlp

`default_nettype none

module weight_address_gen (
  input  wire        clk,
  input  wire        rst,

  input  wire  [4:0] op_code,        // see localparams below
  input  wire  [4:0] layer_idx,      // 0..29

  output logic [29:0] ddr_addr,
  output logic [19:0] byte_count
);

  // Op codes
  localparam [4:0] OP_EMBED   = 5'd0;
  localparam [4:0] OP_WQ      = 5'd1;
  localparam [4:0] OP_WK      = 5'd2;
  localparam [4:0] OP_WV      = 5'd3;
  localparam [4:0] OP_WO      = 5'd4;
  localparam [4:0] OP_WG      = 5'd5;
  localparam [4:0] OP_WU      = 5'd6;
  localparam [4:0] OP_WD      = 5'd7;
  localparam [4:0] OP_GAM_A   = 5'd8;
  localparam [4:0] OP_GAM_M   = 5'd9;
  localparam [4:0] OP_FINAL_N = 5'd10;
  localparam [4:0] OP_LM_HEAD = 5'd11;

  localparam int HIDDEN = 576;
  localparam int FFN    = 1536;

  // Tensor sizes in bytes (INT8 weights, 1 byte each)
  localparam int SZ_WQ    = HIDDEN * HIDDEN;        //   331,776
  localparam int SZ_WK    = HIDDEN *   192;         //   110,592
  localparam int SZ_WV    = HIDDEN *   192;         //   110,592
  localparam int SZ_WO    = HIDDEN * HIDDEN;        //   331,776
  localparam int SZ_WG    = HIDDEN *   FFN;         //   884,736
  localparam int SZ_WU    = HIDDEN *   FFN;         //   884,736
  localparam int SZ_WD    =    FFN * HIDDEN;        //   884,736
  localparam int SZ_GAM   = HIDDEN * 2;             // FP16 gamma
  localparam int SZ_LAYER = SZ_WQ + SZ_WK + SZ_WV + SZ_WO
                          + SZ_WG + SZ_WU + SZ_WD + 2 * SZ_GAM;

  localparam logic [29:0] BASE_EMBED   = 30'h00_000_000;
  localparam logic [29:0] BASE_LAYERS  = 30'h01_B00_000;
  localparam logic [29:0] BASE_SCALES  = 30'h07_F00_000;

  // TODO: implement combinational lookup of (op_code, layer_idx) →
  // (ddr_addr, byte_count).  Trivially:
  //
  //   case (op_code)
  //     OP_EMBED:    ddr_addr = BASE_EMBED;       byte_count = HIDDEN*VOCAB;
  //     OP_LM_HEAD:  ddr_addr = BASE_EMBED;       byte_count = ...;
  //     OP_WQ..OP_WD,OP_GAM_A,OP_GAM_M:
  //       ddr_addr = BASE_LAYERS + layer_idx*SZ_LAYER + offset_within_layer(op_code);
  //   endcase

  always_comb begin
    ddr_addr   = '0;
    byte_count = '0;
    // skeleton
  end

endmodule

`default_nettype wire
