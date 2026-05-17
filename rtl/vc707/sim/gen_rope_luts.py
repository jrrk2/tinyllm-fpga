#!/usr/bin/env python3
"""Generate RoPE sin/cos LUT hex files for rope.sv.

Precomputes angle[pos, j] = pos / (theta^(2j/D)) for all (pos, j) pairs,
then stores cos and sin as Q1.15 signed 16-bit values.

Output files (written to ../generated/ relative to this script):
  rope_cos.hex  — MAX_CTX * (HEAD_DIM/2) lines, each a 4-hex-digit Q1.15 value
  rope_sin.hex  — same shape, sin values

Layout: row-major over pos (outer) then j (inner), so entry index = pos*(D/2) + j.
"""

import numpy as np
import os
import sys

def main(HEAD_DIM=64, MAX_CTX=2048, BASE=10000.0, out_dir=None):
    H2 = HEAD_DIM // 2
    total = MAX_CTX * H2

    j   = np.arange(H2, dtype=np.float64)
    pos = np.arange(MAX_CTX, dtype=np.float64)

    # freq[j] = 1 / BASE^(2j/D)
    freq = 1.0 / (BASE ** (2.0 * j / HEAD_DIM))   # shape (H2,)

    # angle[pos, j] = pos * freq[j]
    # broadcast: (MAX_CTX,1) * (H2,) -> (MAX_CTX, H2)
    angle = pos[:, None] * freq[None, :]            # shape (MAX_CTX, H2)

    c = np.cos(angle)   # (MAX_CTX, H2)
    s = np.sin(angle)   # (MAX_CTX, H2)

    # Quantise to Q1.15: multiply by 2^15, round, saturate
    def to_q15_hex(arr):
        q = np.round(arr * (1 << 15)).astype(np.int64)
        q = np.clip(q, -32768, 32767).astype(np.int16)
        # Format as 4 hex digits (two's complement, unsigned view)
        u = q.view(np.uint16).flatten()
        return [f"{v:04x}" for v in u]

    cos_lines = to_q15_hex(c)
    sin_lines = to_q15_hex(s)

    assert len(cos_lines) == total
    assert len(sin_lines) == total

    if out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "..", "generated")
    os.makedirs(out_dir, exist_ok=True)

    cos_path = os.path.join(out_dir, "rope_cos.hex")
    sin_path = os.path.join(out_dir, "rope_sin.hex")

    with open(cos_path, "w") as f:
        f.write("\n".join(cos_lines) + "\n")
    with open(sin_path, "w") as f:
        f.write("\n".join(sin_lines) + "\n")

    print(f"wrote {cos_path}  ({total} entries)", file=sys.stderr)
    print(f"wrote {sin_path}  ({total} entries)", file=sys.stderr)

if __name__ == "__main__":
    main()
