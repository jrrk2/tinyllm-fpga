#!/usr/bin/env python3
"""Diff per-layer FPGA dumps against Python sim references.

Usage:
    python3 host/sim_blockfp.py --dump-layers --prompt "Once upon a time"
    python3 host/fpga_per_layer_dump.py
    python3 host/diff_layers.py

Reports per-layer max absolute lane error and the first layer where
mismatch exceeds a threshold (default 16 LSBs).
"""
import sys, numpy as np

NL = 30
THRESH = 16   # signed-int LSB tolerance per lane

def load(path):
    lanes = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            lanes.append(int(ln))
    return np.array(lanes, dtype=np.int32)

def main():
    first_div = None
    print(f"{'layer':>5} {'max|err|':>10} {'mean|err|':>10} {'fpga rng':>16} {'py rng':>16}")
    for li in range(NL):
        try:
            f = load(f"fpga_layer_{li:02d}.txt")
            p = load(f"py_layer_{li:02d}.txt")
        except FileNotFoundError as e:
            print(f"  missing: {e}", file=sys.stderr); return
        if len(f) != len(p):
            print(f"  layer {li}: lane count mismatch ({len(f)} vs {len(p)})"); return
        err = np.abs(f - p)
        max_e = int(err.max()); mean_e = float(err.mean())
        flag = " <-- first" if first_div is None and max_e > THRESH else ""
        if first_div is None and max_e > THRESH: first_div = li
        print(f"{li:>5} {max_e:>10} {mean_e:>10.2f} "
              f"[{int(f.min()):>6}..{int(f.max()):>6}]  "
              f"[{int(p.min()):>6}..{int(p.max()):>6}]{flag}")
    if first_div is None:
        print("\nAll layers within tolerance (±{} LSBs).".format(THRESH))
    else:
        print(f"\nFirst divergent layer: {first_div}")

if __name__ == "__main__":
    main()
