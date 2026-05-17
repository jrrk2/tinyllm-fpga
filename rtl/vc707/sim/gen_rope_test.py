#!/usr/bin/env python3
"""Generate test vectors + reference output for rope.sv (LLaMA convention).

Output file test_rope.bin layout:
  uint32  HEAD_DIM
  uint32  pos
  float64 base
  int16   x[HEAD_DIM]    (Q1.15)
  int16   y_ref[HEAD_DIM] (Q1.15)
"""
import numpy as np
import struct
import sys

def saturate_q15(x):
    return np.clip(x, -32768, 32767).astype(np.int16)

def main():
    rng = np.random.default_rng(0xDADA)
    HEAD_DIM = 64
    POS  = 137
    BASE = 10000.0

    # Q vector — modest magnitudes so post-rotation stays in Q1.15.
    x_real = rng.normal(0, 0.4, size=HEAD_DIM).astype(np.float64)
    x_q15  = saturate_q15(np.round(x_real * (1 << 15)))

    # Re-dequantize for the reference so we measure only the engine,
    # not host quantization rounding.
    x_dq = x_q15.astype(np.float64) / (1 << 15)

    H2 = HEAD_DIM // 2
    j  = np.arange(H2)
    freq  = 1.0 / (BASE ** (2.0 * j / HEAD_DIM))
    angle = POS * freq                   # length H2
    c     = np.cos(angle)
    s     = np.sin(angle)

    x_lo = x_dq[:H2]
    x_hi = x_dq[H2:]
    y_lo =  x_lo * c - x_hi * s
    y_hi =  x_hi * c + x_lo * s
    y_real = np.concatenate([y_lo, y_hi])
    y_q15  = saturate_q15(np.round(y_real * (1 << 15)))

    print(f"HEAD_DIM={HEAD_DIM}  pos={POS}", file=sys.stderr)
    print(f"x_q15  range [{x_q15.min()},{x_q15.max()}]", file=sys.stderr)
    print(f"y_real range [{y_real.min():.4f},{y_real.max():.4f}]", file=sys.stderr)
    n_sat = int(np.sum((y_q15 == 32767) | (y_q15 == -32768)))
    print(f"saturated lanes: {n_sat}/{HEAD_DIM}", file=sys.stderr)

    with open("test_rope.bin", "wb") as f:
        f.write(struct.pack("<II", HEAD_DIM, POS))
        f.write(struct.pack("<d", BASE))
        f.write(x_q15.tobytes())
        f.write(y_q15.tobytes())
    print("wrote test_rope.bin", file=sys.stderr)

if __name__ == "__main__":
    main()
