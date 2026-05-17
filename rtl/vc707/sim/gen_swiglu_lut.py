#!/usr/bin/env python3
"""Generate silu_lut.hex for swiglu.sv.

The LUT has 65536 entries indexed by the 16-bit two's-complement gate
value (Q-fixed at scale SILU_LUT_SCALE).  Entry i stores SiLU(gate_float)
in the same Q-format, where gate_float = int16(i) / 32768.0 * SILU_LUT_SCALE.

For SILU_LUT_SCALE = 32 the LUT covers SiLU on input range [-32, +32) —
enough for SmolLM2 calibration scales (lsc[gate] observed up to ~21).

Output: ../generated/silu_lut.hex  (65536 lines of 4 hex digits)
"""

# Scale of the SiLU LUT input/output.  Must match SILU_LUT_SCALE in swiglu.sv
# and host/gen_smollm_blockfp.py.
SILU_LUT_SCALE = 32.0
import numpy as np
import pathlib, sys

OUT = pathlib.Path(__file__).parent.parent / "generated" / "silu_lut.hex"

def main():
    # All 65536 16-bit two's-complement values, in order 0,1,...,32767,-32768,...,-1
    idx = np.arange(65536, dtype=np.uint16)
    gate_q15 = idx.view(np.int16)          # reinterpret as signed
    # Q-fixed at scale SILU_LUT_SCALE:  i ∈ [-32768, 32767)  →  real ∈ [-SCALE, SCALE)
    gate_f = gate_q15.astype(np.float64) / 32768.0 * SILU_LUT_SCALE

    silu_f = gate_f * (1.0 / (1.0 + np.exp(-gate_f)))

    # Quantise back to Q-fixed at SILU_LUT_SCALE; clamp to representable range.
    # silu(x) ≤ |x| for all x, so silu(SCALE) ≈ SCALE and silu(-SCALE) ≈ 0 —
    # both representable in the same Q-fixed format.
    silu_q15 = np.round(silu_f * 32768.0 / SILU_LUT_SCALE).astype(np.int64)
    silu_q15 = np.clip(silu_q15, -32768, 32767).astype(np.int16)

    # Write as unsigned 16-bit hex, 4 digits per line
    silu_u16 = silu_q15.view(np.uint16)
    lines = [f"{v:04x}" for v in silu_u16]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} entries to {OUT}", file=sys.stderr)

if __name__ == "__main__":
    main()
