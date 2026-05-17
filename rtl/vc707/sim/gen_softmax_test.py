#!/usr/bin/env python3
"""Generate test vectors + reference for softmax_q15.sv.

  uint32  N
  int16   x[N]      (Q1.15)
  int16   y_ref[N]  (Q1.15, sums to ~1.0 across the row)
"""
import numpy as np
import struct
import sys

def saturate_q15_unsigned(x):
    return np.clip(x, 0, 32767).astype(np.int16)

def main():
    rng = np.random.default_rng(0xCAFE)
    N = 64

    # Inputs in [-1, 1) scaled — typical post-1/sqrt(d_k) attention scores.
    x_real = rng.normal(0, 0.5, size=N).astype(np.float64)
    x_real = np.clip(x_real, -0.99, 0.99)
    x_q15  = np.clip(np.round(x_real * (1 << 15)), -32768, 32767).astype(np.int16)

    # Reference using the *quantized* x (so we measure only engine error).
    x_dq = x_q15.astype(np.float64) / (1 << 15)
    m    = x_dq.max()
    e    = np.exp(x_dq - m)
    y    = e / e.sum()
    y_q15 = saturate_q15_unsigned(np.round(y * (1 << 15)))

    print(f"N={N}", file=sys.stderr)
    print(f"x range   [{x_dq.min():.4f},{x_dq.max():.4f}]", file=sys.stderr)
    print(f"y_ref sum = {y.sum():.6f}", file=sys.stderr)
    print(f"y_q15 sum = {y_q15.sum()}  (Q1.15 unit = 32768)", file=sys.stderr)

    with open("test_softmax.bin", "wb") as f:
        f.write(struct.pack("<I", N))
        f.write(x_q15.tobytes())
        f.write(y_q15.tobytes())
    print("wrote test_softmax.bin", file=sys.stderr)

if __name__ == "__main__":
    main()
