# DSP48E1 review and a custom FP format for SmolLM2 inference

## DSP48E1 features relevant to FP normalisation

The Virtex-7 DSP48E1 is a 25×18 signed multiplier with a 48-bit
post-adder/accumulator and a small ALU.  Several features are
genuinely useful for the FP-normalise data path that I'd undersold
on first pass:

| Feature              | DSP48E1 capability                                           | Use for FP normalise                       |
|----------------------|--------------------------------------------------------------|--------------------------------------------|
| **17-bit C shift**   | `OPMODE` can route `C` shifted left by 17 into the post-adder | Build 96-bit acc; align pre-shifted operands |
| **Pre-adder shift**  | A-input can be sign-extended/concatenated for ±1-bit shifts   | Final ±1 LSB renormalise after mantissa add |
| **PATTERNDETECT**    | Flags when post-adder output matches a programmable pattern   | **Leading-zero detect via masked compare** |
| **PATTERNBDETECT**   | Flags inverse pattern (≠)                                     | Companion: detect when the bit *isn't* zero |
| **Mask register**    | Per-bit ignore on the pattern compare                         | Stride checks: "are bits [47:24] all zero?" |
| **AUTORESET on PD**  | Self-clears the accumulator on pattern match                  | Per-element acc clear without explicit ctrl |
| **PCIN/PCOUT**       | Cascades 48-bit P between adjacent DSPs                       | Wide acc chains (96-, 144-bit)              |
| **OVERFLOW/UNDERFLOW** registered overflow flags into the next DSP | Detect when normalise must shift left vs right |

The two features the user flagged that matter most:

1. **Limited shift** — the C-input 17-bit left-shift lets one DSP
   feed an aligned partial sum into the next (PCIN), and the A:B
   concatenation gives a single-LSB shift via `0&A` or `A&0`.  Larger
   shifts still need a barrel shifter in fabric.
2. **Leading-zero count via PATTERNDETECT + mask** — by setting the
   PATTERN to all-zeros and stepping the mask, a couple of
   DSP-cycles per element can produce the LZ count without a fabric
   priority encoder.  Most efficient for **chunked** LZ (one
   max-LZ per chunk of 16 elements), since a single DSP-pair can
   serialise the per-element checks at one comparison per cycle.

A sensible pattern for chunk-wise FP normalise on this fabric:

* DSPs do the heavy multiplies, accumulate in 48-bit
* DSP `PATTERNDETECT(0, mask)` with mask stepping `0xFFFF0000`,
  `0xFF000000`, `0xF0000000` … gives a coarse-then-fine LZ count
* DSP C-input 17-bit shift handles the fast cascade-align
* LUT-fabric barrel shifter handles the residual fine shift (≤ 17 bits)
* Exponent bookkeeping (8-bit per chunk) rides alongside in 4-8 LUTs

The matvec engine in `matvec_int8_engine.sv` already uses one DSP per
lane in MAC mode + one DSP per lane in scale mode — both stay in
fixed point and need no FP normalise.  The block-FP normaliser would
sit at the *bus output* (per chunk of 16 lanes) rather than per-lane.

## Numerical needs of SmolLM2-135M

Calibrated max-abs of the residual stream across NL=30 layers
(`host/calibrate_smollm.py` output, prompt = "Once upon a time…"):

| Bus              | Min      | Max      | log2 dynamic range |
|------------------|----------|----------|--------------------|
| residual stream  | ~0.001   | ~30 000  | 25 bits            |
| norm1/norm2      | 1 – 45   | 1.5×     | ~5–6 bits          |
| q / k / v        | 14 – 30  | 1.5×     | ~5 bits            |
| attn (post-O)    | 3 – 260  | 1.5×     | ~7–8 bits          |
| gate / up        | 5 – 110  | 1.5×     | ~5–7 bits          |
| down (post-MLP)  | 7 – 30 000 | 1.5×    | 12–15 bits         |

The narrow buses fit comfortably in 16-bit Q1.15 with a per-tensor
calibrated full-scale.  The **residual stream** does not — its 25-bit
range overruns any single fixed-point format that has < 25 mantissa
bits with useful sub-LSB precision for early-layer values
(layer 0 hidden values are O(1), but layer 28 hidden values reach
30 000 in the same lane).

## Format A — what we have today (block-FP, per-layer)

* 16-bit Q1.15 mantissa per element
* 4-bit power-of-2 scale per LAYER (TM_RESCALE.h_in_p2 / h_out_p2 / h1_p2)
* Wrapper rescales hidden_state at inter-layer cascade
* Within a layer, two residual sums apply signed shifts to align
* Storage: 576 × 16 = 9216 bits per buffer + ~12 bits per layer scale

Pros: simple, fits the existing 16-bit data path, no DSP changes,
sim_blockfp.py confirmed coherent text generation in Python.

Cons: per-layer precision tracking is too coarse — early-layer
elements with magnitude << layer-mean lose all variance.  Verilator
DUT decodes to weird tokens (`'itization'` family) even though the
overall magnitude is right.

## Format B — proposed: chunk-wise block-FP (MX-style)

Per **chunk of 16 elements**:

```
struct chunk_t {
    int16_t mant[16];   // Q1.15 signed mantissas
    int8_t  exp;        // shared signed exponent (range -127 .. +127)
};
```

* 16 elements share one exponent
* D = 576 → 36 chunks → 36 exponents per buffer
* Storage: 16 × 16 + 8 = 264 bits per chunk → 9504 bits total
  (1.03× current 9216 bits)

Operations (per chunk):

| Op             | Mantissa work               | Exponent work                            |
|----------------|-----------------------------|------------------------------------------|
| **add (resid)**| align via right-shift the chunk with the smaller exponent; sum mantissas; renormalise | exp_out = max(exp_a, exp_b) + carry |
| **mul (matvec)**| element-wise int16 × int8, sum with 32-bit acc | exp_out = exp_a + exp_w + (renormalise shift) |
| **norm (RMSNorm)**| sum-of-squares within chunk; combine across chunks via exponent alignment | tracks 2× exp + carry |

The matvec engine already produces a 16-lane chunk in one shot — its
output natively becomes one chunk_t.  Residual stages today walk
elements one-by-one; under chunk-FP they could process chunks of 16
in 1 + 1 cycles (compare + add).

Implementation cost on VC707:

* +8 bits storage per 16 elements ≈ 0.5 RAMB36 extra per buffer
* 1 priority encoder per chunk for leading-1 (clog2(16)=4 bits)
* 1 barrel shifter per chunk (5 bits)
* 1 max() per chunk for exponent compare
* Per-element multiply path unchanged

The matvec / RMSNorm / SwiGLU engines need their interfaces widened
from "16-bit element" to "(16-bit element + 8-bit chunk exp)" but
internal arithmetic stays the same width.

## Format C — for reference: per-element FP (microsoft msfp / NV TF32)

* 16-bit element = 1 sign + 8 exp + 7 mantissa  (msfp16)
* Or 19-bit TF32: 1 sign + 8 exp + 10 mantissa

Pros: standard, no chunking complexity.
Cons: needs full normaliser per operation, ~2× DSP cost on multiplies
(can't use simple int16×int8 anymore), wider buses everywhere.
Heavyweight for our matvec engine.

## Recommendation

**Format B (chunk-FP)** is the right next step.  It re-uses the
existing 16-element matvec lane-width, costs ~3% extra storage,
adds one normaliser per chunk per stage (cheap in LUTs), and
gives 256-fold per-chunk dynamic range — way more than the 25-bit
residual stream needs.

Implementation plan:

1. **Hidden state**: declare as `(int16 mant [D], int8 exp [D/16])`.
   Selftest wrapper bakes both arrays in `layer_hidden_in_packed.svh`.
2. **Matvec output → hidden**: when a chunk's int16 results are written
   to a buffer that participates in a residual, also emit a chunk
   exponent (= per-row scale_q15's exponent + the bus's calibrated
   exponent).
3. **Residuals**: per chunk, compare exponents, right-shift the
   smaller, sum mantissas, renormalise with priority encoder + shifter.
4. **RMSNorm input**: take chunked input; internally re-anchor to a
   single max exponent for sum-of-squares.
5. **Inter-layer cascade**: just pass the chunk_t through; no rescale
   needed because each layer reads its own chunk exponent.

No DSP48 mode changes — the existing fixed-point multiplies remain
correct, just wrapped in a chunk format.

## This rebuild

Independent of Format B (which is a substantial RTL effort, deferred),
**this rebuild widens the result-readback regmap from 64 to 576 lanes**
so we can decode the FPGA's *actual* hidden_out through SmolLM2's
lm_head and confirm coherent-or-not directly from hardware.  The block-FP
layer architecture stays as-is.
