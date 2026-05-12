# SmolLM2-135M-Instruct port plan for VC707

This document is the bring-up plan for replacing `microgpt_exact_core` with
a SmolLM2-135M-Instruct inference core that streams INT8-quantized weights
from DDR3 through the `weight_tile_cache`.

## Why SmolLM2-135M

- **2024 release** by HuggingFaceTB, Apache-2.0, trained on ~2 T tokens of
  curated FineWeb-Edu data. Roughly matches GPT-2-large quality at 135 M
  parameters.
- Architecture is mainstream LLaMA-2-style, well-documented, no exotic
  ops. Same shapes as the public Llama-3.2-1B but smaller.
- Coherent English at small enough size that it fits VC707's 1 GB DDR3
  with INT8 quantization (~135 MB) and runs at ~87 tok/s on this board's
  measured bandwidth.

## Model spec (from `config.json`)

```
hidden_size:          576    (embed dim)
num_hidden_layers:    30
num_attention_heads:   9     (head_dim = 64)
num_key_value_heads:   3     (GQA, 3:1 ratio)
intermediate_size:   1536    (FFN inner dim)
vocab_size:        49152
max_position_embeddings: 8192
rope_theta:        100000.0
rms_norm_eps:      1e-5
hidden_act:         silu     (SwiGLU)
tie_word_embeddings: true    (lm_head shares with embed_tokens)
```

## Per-token compute breakdown

Each layer does:

1. **RMSNorm** on the hidden state.
2. **Attention** (grouped-query):
   - Q proj: `[H=576] @ [Wq:576×576]` → `[576]` → split into 9 heads of 64.
   - K proj: `[576] @ [Wk:576×192]` → `[192]` → 3 KV heads of 64.
   - V proj: same as K → `[192]`.
   - Apply **RoPE** to Q and K.
   - Update KV cache (append the new K,V at position `t`).
   - Compute attention scores: `Q @ K^T / sqrt(64)` → softmax → `@ V`.
     For batch=1 autoregressive, this means each Q head dot-products with
     the *full* sequence's K, with the GQA reuse factor of 3.
   - O proj: `[576] @ [Wo:576×576]` → `[576]`.
3. **Residual** add.
4. **RMSNorm**.
5. **MLP (SwiGLU)**:
   - Gate: `[576] @ [Wg:576×1536]` → `[1536]`.
   - Up:   `[576] @ [Wu:576×1536]` → `[1536]`.
   - SiLU(gate) ⊙ up → `[1536]`.
   - Down: `[1536] @ [Wd:1536×576]` → `[576]`.
6. **Residual** add.

Then after all 30 layers:
7. Final RMSNorm.
8. `lm_head` (= tied embed_tokens.T): `[576] @ [49152×576]^T` → `[49152]`
   logit vector.
9. Sample (argmax / temperature).

## Weight count and memory budget (INT8 + FP16 scales)

Per layer:
- Wq: 576 × 576 = 331,776
- Wk: 576 × 192 =  110,592
- Wv: 576 × 192 =  110,592
- Wo: 576 × 576 = 331,776
- Wg: 576 × 1536 = 884,736
- Wu: 576 × 1536 = 884,736
- Wd: 1536 × 576 = 884,736
- 2 × RMSNorm gamma: 1152 (negligible)

Per-layer subtotal = 3,538,944 weights × 1 byte (INT8) = **3.38 MB**.

30 layers = **101 MB**.

Plus:
- `embed_tokens` / `lm_head` (tied): 49,152 × 576 = 28,311,552 = **27 MB**.

**Total INT8 weights = ~128 MB.** Per-channel FP16 scales add ~1 MB.

Per-token bytes streamed (autoregressive, batch=1) = ~129 MB.

At measured 11.71 GB/s sustained: 129 MB / 11.71 GB/s = **~11 ms/token =
~91 tok/s** end-to-end (ignoring compute, which has plenty of headroom).

KV cache (FP16) for max context 1024 tokens:
- K + V each: 2 × 1024 × 192 × 2 bytes = 786 KB per layer.
- 30 layers × 1.57 MB = ~**47 MB total KV cache**.

This fits comfortably in DDR3 alongside the weights. We could cache it
in BRAM if convenient (47 MB doesn't fit in 4.6 MB BRAM, so DDR3 it is).

## DDR3 weight layout

Linear address space, INT8 weights ordered by access (the matvec engine
streams a layer's worth of output rows × all input columns sequentially):

```
0x0000_0000 .. 0x0000_FFFF   embed_tokens / lm_head (27 MB; aligned to 8K)
0x01B0_0000 .. 0x01B0_???    layer_0
0x01B3_???  ..                 .
...                            (30 × 3.38 MB = 101 MB)
0x07F0_0000 .. 0x07FF_FFFF   FP16 per-channel scales table (~1 MB)
0x0800_0000 ..                  KV cache region (47 MB, written as we go)
```

Within each layer the per-op weights are packed back-to-back in this
order (matches the compute order in the FSM):

```
Wq, Wk, Wv, Wo, Wg, Wu, Wd, gamma_attn, gamma_mlp
```

The `weight_address_gen.sv` block computes `(layer_idx, op_idx) →
(ddr3_byte_addr, byte_count)`.

## Quantization

**Per-channel symmetric INT8** for all matvec weights:
- Each output row of a weight matrix has its own FP16 scale.
- Quantize: `q = round(w / scale)`, where `scale = max(|w|) / 127`.
- Dequantize at compute time: `(int8 × scale)` accumulated into FP16.
- Embeddings + final layer norm: keep at FP16 (small fraction of memory).

This is the standard INT8 weight-only quantization that GPTQ-style
algorithms produce. SmolLM2 has published INT8 GGUF on HuggingFace; we
can also call AWQ/GPTQ ourselves on the FP16 checkpoint.

## Module hierarchy

```
smollm_core.sv                 -- top FSM, orchestrates per-token compute
├─ weight_address_gen.sv       -- (layer, op) -> DDR3 addr + size
├─ weight_tile_cache.sv        -- ping-pong DDR3 streamer (already built)
├─ matvec_int8_engine.sv       -- 16-lane INT8 × FP16 -> FP16 accumulator
├─ rmsnorm.sv                  -- variance + scale + gamma
├─ rope.sv                     -- rotary positional encoding for Q,K
├─ swiglu.sv                   -- SiLU(gate) * up + down projection control
├─ gqa_attention.sv            -- KV cache mgmt + scaled-dot-product
└─ sampler.sv                  -- argmax / temperature / top-p
```

`microgpt_exact_core.sv` is left in place; `vc707_microgpt_eth.sv` will
choose between cores via a build-time parameter.

## Per-cycle throughput plan

The `matvec_int8_engine` is the workhorse:
- 16 lanes × 8-bit INT8 × 16-bit FP16 → 32-bit accumulator per lane.
- 1 cycle = 16 outputs × 1 input dim contribution.
- For W:[Out=576, In=576], producing 16 rows takes 576 cycles (one input
  column per cycle); the next 16-row tile starts after 576 cycles.
- Total cycles per linear op = `(Out / 16) * In`.

| op | Out × In | cycles per op |
|----|---------|---------------|
| Wq | 576×576 | 36 × 576 = 20,736 |
| Wk | 192×576 | 12 × 576 = 6,912 |
| Wv | 192×576 | 12 × 576 = 6,912 |
| Wo | 576×576 | 20,736 |
| Wg | 1536×576 | 96 × 576 = 55,296 |
| Wu | 1536×576 | 55,296 |
| Wd | 576×1536 | 36 × 1536 = 55,296 |
| layer total | | **221,184 cycles** |
| 30 layers | | 6,635,520 cycles |
| lm_head | 49152×576 | 3072 × 576 = 1,769,472 |
| **per token** | | **~8.4 M cycles** |

At 100 MHz core_clk (post the speed-bump task we already noted): 84 ms
compute per token = **12 tok/s** compute-bound.

That's worse than the memory-bound 91 tok/s. So:
- Memory: 11 ms/token at 11.71 GB/s.
- Compute: 84 ms/token at 16 lanes × 100 MHz.

**Compute is the bottleneck** for SmolLM2-135M on this board with 16
lanes. To match the memory ceiling we'd need ~128 lanes (= 8× more DSPs
than the matvec currently uses). VC707 has 2,800 DSP slices and we're
using 38 — easy headroom.

Practical recommendation: build with 64 lanes to start (2× memory ceiling
on compute, leaving margin for non-matvec ops). 64 lanes × 100 MHz =
6.4 G FMAs/s; per-token compute = 6.5M FMAs / 6.4 GFMA = 1.0 ms which is
< memory budget, so we're cleanly memory-bound at ~85 tok/s.

## Bring-up phases

1. **Phase 1 — host conversion script** (`host/smollm/convert.py`):
   - Download `HuggingFaceTB/SmolLM2-135M-Instruct`.
   - INT8-quantize weights with per-channel scales.
   - Lay out as binary `weights.bin` matching DDR3 layout above.
   - Print expected DDR3 byte addresses for each tensor.

2. **Phase 2 — UDP DDR3 write path**:
   - Add an AXI write master to the eth_ctrl side, controlled by REG_WRITE.
   - Host streams `weights.bin` in 8 KiB chunks to fill DDR3.
   - Verify by reading back through the tile cache.

3. **Phase 3 — single-layer matvec** (`matvec_int8_engine.sv`):
   - INT8 × FP16 lanes, 16-row tile, FP16 accumulator.
   - Bench against a known input/output pair from PyTorch.

4. **Phase 4 — full layer** (RMSNorm + attention + MLP + residual):
   - Each block is its own SV file (skeletons in this directory).
   - KV cache: hold in DDR3 with simple (layer, head, t) addressing.

5. **Phase 5 — full model** (loop layers, then lm_head, then sampler).

6. **Phase 6 — host BPE tokenizer** (`host/smollm/tokenizer.py`):
   - Use `tokenizers` library with the bundled vocab/merges.
   - Encode prompt → token IDs over UDP.
   - Receive token IDs → decode for display.

## Per-block algorithm notes (for module skeletons)

### RMSNorm (`rmsnorm.sv`)

```
y = x * gamma / sqrt(mean(x^2) + eps)
```
- Need a sum-of-squares accumulator (576 muls per token), then a
  reciprocal-square-root (use a small Q4.12 LUT + 1-2 Newton steps).
- Output scaled by gamma (per-channel weight, 576 FP16 values).

### RoPE (`rope.sv`)

For position `t`, head dim 64, base θ = 100,000:
```
freq_i = 1 / θ^(2i/64), i = 0..31
angle = t * freq_i
(x_i, x_{i+32}) = (x_i*cos(a) - x_{i+32}*sin(a),
                   x_i*sin(a) + x_{i+32}*cos(a))
```
Cos/sin tables: 1024 positions × 32 freqs × 2 (cos+sin) × FP16 =
4 MB total. Too big for BRAM if pre-computed; use CORDIC instead, or
compute on-the-fly with an angle accumulator.

### SwiGLU (`swiglu.sv`)

```
y = silu(gate(x)) * up(x)
silu(x) = x * sigmoid(x) = x / (1 + e^-x)
```
- 1024-entry FP16 LUT for sigmoid is plenty (LUT input range ±8).
- Element-wise multiply with `up` projection output.

### GQA attention (`gqa_attention.sv`)

For each Q head h ∈ [0..8]:
```
kv_head = h / 3            // 3:1 GQA ratio
score_t = Q_h · K_kv_head[t] / sqrt(64)  for t in [0..pos]
attn = softmax(score)
out_h = sum_t(attn_t * V_kv_head[t])
```
- KV cache layout: `kv[layer][head][t][dim]` in DDR3.
- Each token append: write Q, K to (layer, kv_head, t, :).
- Per-token attention reads back the entire context at the current
  `kv_head` — that's what dominates KV-cache bandwidth.

### Sampler (`sampler.sv`)

- Take the 49,152-element logit vector.
- Apply temperature (divide each logit by T as Q-format).
- Compute argmax (or top-k, top-p — basic argmax is enough for the
  first cut).
- Output the chosen token ID.

## What's NOT in scope here

- **Sliding window** / streaming of long context: SmolLM2-135M doesn't
  use it, so we just maintain the full KV cache up to whatever max
  context we configure (probably 1024 to keep KV cache size ~50 MB).
- **Beam search / top-p sampling**: argmax + temperature is sufficient.
- **Speculative decoding**: requires a smaller draft model — out of
  scope until base model works.

## Open questions to revisit before Phase 3

- INT4 vs INT8: INT4 doubles tok/s ceiling (~175 vs ~87) but quality
  degradation on a 135 M model is real. Stick with INT8 first; explore
  INT4 if everything else lands.
- Whether `lm_head` gets its own pass (FP16) for higher quality — the
  vocab projection benefits more from precision than internal layers.
- Sampler placement: could be in core_clk domain or in eth_clk domain
  for easier readback. Probably core_clk for tighter control.
