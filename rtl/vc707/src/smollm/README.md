# SmolLM2-135M-Instruct port — skeleton

These files are the skeleton for replacing `microgpt_exact_core` with a
SmolLM2-135M-Instruct inference engine. Read **`PLAN.md`** first — it has
the architecture spec, weight layout, throughput analysis, and bring-up
phases.

## Skeleton modules (compile but no real logic yet)

| file | role |
|---|---|
| `smollm_core.sv` | top FSM that orchestrates per-token compute |
| `weight_address_gen.sv` | (layer, op) → DDR3 byte address + size |
| `matvec_int8_engine.sv` | 64-lane INT8 × FP16 matrix-vector engine |
| `rmsnorm.sv` | RMSNorm with rsqrt LUT |
| `rope.sv` | Rotary positional encoding for Q,K |
| `swiglu.sv` | SiLU(gate)⊙up activation |
| `gqa_attention.sv` | GQA attention with DDR3-resident KV cache |
| `sampler.sv` | argmax / temperature sampler |

The cache infrastructure (`weight_tile_cache.sv`) lives one level up, in
`rtl/vc707/src/`, since it's shared between SmolLM and any future model
ports.

## Phase order (matches PLAN.md)

1. `host/smollm/convert.py` — INT8-quantize the FP32 model and pack it
   into the DDR3 layout (`weights.bin`). Already written — needs the
   model checkpoint to actually run.
2. UDP DDR3 *write* path — extend `microgpt_eth_ctrl.sv` with an AXI
   write master, controlled via REG_WRITE so the host can stream
   `weights.bin` into DDR3. **Not yet built.**
3. `matvec_int8_engine.sv` — first real implementation. Bench against
   PyTorch reference for a single matvec.
4. Plug RMSNorm + matvec + RoPE + SwiGLU + GQA together for one layer.
5. Loop layers + lm_head + sampler.
6. `host/smollm/tokenizer.py` — BPE encode/decode on the host side.

## Throughput target (from PLAN.md)

| metric | value |
|---|---|
| weights/token (INT8) | 129 MB |
| memory ceiling (12 GB/s) | ~91 tok/s |
| compute @ 64-lane × 100 MHz | ~85 tok/s |
| **expected end-to-end** | **~85 tok/s** (memory-bound) |
