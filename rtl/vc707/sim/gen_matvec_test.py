#!/usr/bin/env python3
"""Generate test vectors + reference output for matvec_int8_engine.

Layout written to test_matvec.bin:
  uint32 lanes
  uint32 in_dim
  int16  in_q15[in_dim]
  int8   w_int8[lanes][in_dim]   (row-major; lane-fastest changing along rows)
  int16  scale_q15[lanes]
  int16  out_ref_q15[lanes]
"""
import numpy as np
import struct
import sys

def saturate_q15(x):
    return np.clip(x, -32768, 32767).astype(np.int16)

def main():
    rng = np.random.default_rng(0xC0FFEE)
    LANES = 16
    IN_DIM = 64

    in_q15 = rng.integers(-32768, 32767, size=IN_DIM, dtype=np.int16)
    w_i8   = rng.integers(-127, 127,    size=(LANES, IN_DIM), dtype=np.int8)
    # Keep scales modest so post-shift values stay representable
    scale_q15 = rng.integers(-8192, 8192, size=LANES, dtype=np.int16)

    # Reference: acc[l] = sum_k in_q15[k] * w_i8[l,k]   (int64 to avoid overflow)
    acc = np.zeros(LANES, dtype=np.int64)
    for l in range(LANES):
        acc[l] = np.sum(in_q15.astype(np.int64) * w_i8[l].astype(np.int64))

    scaled  = acc * scale_q15.astype(np.int64)
    shifted = scaled >> 15  # arithmetic shift on signed numpy ints
    out_ref = saturate_q15(shifted)

    print(f"lanes={LANES} in_dim={IN_DIM}", file=sys.stderr)
    print(f"in_q15 range [{in_q15.min()},{in_q15.max()}]", file=sys.stderr)
    print(f"acc      range [{acc.min()},{acc.max()}]", file=sys.stderr)
    print(f"scaled   range [{scaled.min()},{scaled.max()}]", file=sys.stderr)
    print(f"shifted  range [{shifted.min()},{shifted.max()}]", file=sys.stderr)
    print(f"out_ref  = {out_ref.tolist()}", file=sys.stderr)

    with open("test_matvec.bin", "wb") as f:
        f.write(struct.pack("<II", LANES, IN_DIM))
        f.write(in_q15.tobytes())
        # weight layout: per-cycle the SV expects a packed LANES*8 bus = w_int8[lane],
        # one cycle per k.  Write as w[k][lane] (k-major) so the C++ tb can read
        # 16 bytes per cycle directly.
        for k in range(IN_DIM):
            for l in range(LANES):
                f.write(struct.pack("b", int(w_i8[l, k])))
        f.write(scale_q15.tobytes())
        f.write(out_ref.tobytes())
    print("wrote test_matvec.bin", file=sys.stderr)

if __name__ == "__main__":
    main()
