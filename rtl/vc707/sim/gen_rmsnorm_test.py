#!/usr/bin/env python3
"""Generate test vectors + reference output for rmsnorm.sv.

Output file test_rmsnorm.bin layout:
  uint32  D
  float64 eps
  int16   x[D]      (Q1.15)
  int16   gamma[D]  (Q1.15)
  int16   y_ref[D]  (Q1.15)
"""
import numpy as np
import struct
import sys

def saturate_q15(x):
    return np.clip(x, -32768, 32767).astype(np.int16)

def main():
    rng = np.random.default_rng(0xBEEF)
    D = 64
    EPS = 1e-5

    # Keep magnitudes modest so the post-norm output stays representable
    # in Q1.15 without saturating most lanes.
    x_real     = rng.normal(0, 0.25, size=D).astype(np.float64)
    gamma_real = rng.normal(0, 0.5,  size=D).astype(np.float64)

    # Quantize host-side to Q1.15 to mirror what the engine actually sees.
    x_q15     = saturate_q15(np.round(x_real     * (1 << 15)))
    gamma_q15 = saturate_q15(np.round(gamma_real * (1 << 15)))

    # Reference RMSNorm using the *quantized* x and gamma (so we measure
    # only the engine's arithmetic, not the host's quantization step).
    x_dq    = x_q15.astype(np.float64)     / (1 << 15)
    g_dq    = gamma_q15.astype(np.float64) / (1 << 15)
    mean_sq = np.mean(x_dq * x_dq)
    inv_rms = 1.0 / np.sqrt(mean_sq + EPS)
    y_real  = x_dq * g_dq * inv_rms
    y_q15   = saturate_q15(np.round(y_real * (1 << 15)))

    print(f"D={D}  mean_sq={mean_sq:.6e}  inv_rms={inv_rms:.4f}", file=sys.stderr)
    print(f"x_q15  range [{x_q15.min()},{x_q15.max()}]", file=sys.stderr)
    print(f"y_real range [{y_real.min():.4f},{y_real.max():.4f}]", file=sys.stderr)
    print(f"y_q15  range [{y_q15.min()},{y_q15.max()}]", file=sys.stderr)
    n_sat = int(np.sum((y_q15 == 32767) | (y_q15 == -32768)))
    print(f"saturated lanes: {n_sat}/{D}", file=sys.stderr)

    with open("test_rmsnorm.bin", "wb") as f:
        f.write(struct.pack("<I", D))
        f.write(struct.pack("<d", EPS))
        f.write(x_q15.tobytes())
        f.write(gamma_q15.tobytes())
        f.write(y_q15.tobytes())
    print("wrote test_rmsnorm.bin", file=sys.stderr)

if __name__ == "__main__":
    main()
