#!/usr/bin/env python3
"""Generate test vectors + reference for swiglu.sv.

  uint32  N
  int16   gate[N]   (Q1.15)
  int16   up[N]     (Q1.15)
  int16   y_ref[N]  (Q1.15)
"""
import numpy as np
import struct
import sys

def saturate_q15(x):
    return np.clip(x, -32768, 32767).astype(np.int16)

def main():
    rng = np.random.default_rng(0xACAC)
    N = 128

    gate_real = rng.normal(0, 0.5, size=N).astype(np.float64)
    up_real   = rng.normal(0, 0.5, size=N).astype(np.float64)
    gate_q15  = saturate_q15(np.round(gate_real * (1 << 15)))
    up_q15    = saturate_q15(np.round(up_real   * (1 << 15)))

    g = gate_q15.astype(np.float64) / (1 << 15)
    u = up_q15.astype(np.float64)   / (1 << 15)
    silu = g * (1.0 / (1.0 + np.exp(-g)))
    y_real = silu * u
    y_q15  = saturate_q15(np.round(y_real * (1 << 15)))

    print(f"N={N}", file=sys.stderr)
    print(f"y_real range [{y_real.min():.4f},{y_real.max():.4f}]", file=sys.stderr)

    with open("test_swiglu.bin", "wb") as f:
        f.write(struct.pack("<I", N))
        f.write(gate_q15.tobytes())
        f.write(up_q15.tobytes())
        f.write(y_q15.tobytes())
    print("wrote test_swiglu.bin", file=sys.stderr)

if __name__ == "__main__":
    main()
