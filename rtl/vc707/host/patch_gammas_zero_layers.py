#!/usr/bin/env python3
"""Zero rmsnorm gammas for a contiguous layer range, emit a patched ROM dir.

Usage: patch_gammas_zero_layers.py --src DIR --dst DIR --zero-layers K1:K2
                                   [--prefix lbfp_full_] [--D 960] [--NL 32]

Zeroing gammas for layer L means rmsnorm output ≈ 0 there → matvec input
is 0 → block output is 0 → residual passes the input through unchanged.
Lets us "disable" a layer range at runtime without re-synth or DDR3
reupload — only the gamma BRAMs need re-uploading via load-roms (~20 s).

--zero-layers K1:K2 zeroes layers in the half-open interval [K1, K2).
  --zero-layers 0:0   = no-op (all layers active)
  --zero-layers 16:32 = zero upper half (smollm360)
  --zero-layers 0:32  = zero everything

Files copied:
  G1_m.hex  (NL × D mantissas)         <- patched
  G1_e.hex  (NL × NT_D exponents)      <- patched
  G2_m.hex  (NL × D mantissas)         <- patched
  G2_e.hex  (NL × NT_D exponents)      <- patched
  NORM_W_m.hex, NORM_W_e.hex, PROMPT.hex  <- symlinked unchanged
  lbfp_full_DDR3.bin (if present)         <- symlinked unchanged
"""

import argparse
import os
import sys
from pathlib import Path


def read_hex(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def write_hex(path, lines):
    with open(path, "w") as f:
        for ln in lines:
            f.write(ln + "\n")


def patch_array(lines, entries_per_layer, k1, k2, name):
    """Zero entries for layers in [k1, k2). entries_per_layer = D or NT_D."""
    n_entries = len(lines)
    nl = n_entries // entries_per_layer
    if n_entries % entries_per_layer != 0:
        sys.exit(f"{name}: {n_entries} not divisible by {entries_per_layer}")
    out = list(lines)
    width = len(lines[0])  # mantissa hex is 4 chars (16-bit), exp is 2 chars (8-bit)
    zero = "0" * width
    for L in range(k1, k2):
        if L >= nl:
            break
        for i in range(entries_per_layer):
            out[L * entries_per_layer + i] = zero
    return out, nl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source ROM dir")
    ap.add_argument("--dst", required=True, help="dest dir to emit patched ROMs")
    ap.add_argument("--zero-layers", required=True,
                    help="K1:K2 — zero layers in [K1, K2). E.g. 16:32")
    ap.add_argument("--prefix", default="lbfp_full_")
    ap.add_argument("--D", type=int, default=960, help="hidden dim")
    ap.add_argument("--NL", type=int, default=32, help="number of layers")
    ap.add_argument("--BFP-TILE", type=int, default=16)
    args = ap.parse_args()

    k1, k2 = (int(x) for x in args.zero_layers.split(":"))
    nt_d = args.D // args.BFP_TILE

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for name, per_layer in [("G1_m", args.D), ("G1_e", nt_d),
                             ("G2_m", args.D), ("G2_e", nt_d)]:
        src_path = src / f"{args.prefix}{name}.hex"
        dst_path = dst / f"{args.prefix}{name}.hex"
        lines = read_hex(src_path)
        patched, nl_found = patch_array(lines, per_layer, k1, k2, name)
        write_hex(dst_path, patched)
        zeroed = max(0, min(k2, nl_found) - k1)
        print(f"  {name}: {len(lines)} lines, zeroed {zeroed} layer(s) "
              f"[{k1}:{min(k2, nl_found)}) → {dst_path}")

    # Symlink the unchanged files so load-roms finds them in dst/.
    for name in ["NORM_W_m.hex", "NORM_W_e.hex", "PROMPT.hex"]:
        src_path = (src / f"{args.prefix}{name}").resolve()
        dst_path = dst / f"{args.prefix}{name}"
        if dst_path.is_symlink() or dst_path.exists():
            dst_path.unlink()
        dst_path.symlink_to(src_path)
    ddr3 = (src / f"{args.prefix}DDR3.bin").resolve()
    if ddr3.exists():
        dst_ddr3 = dst / f"{args.prefix}DDR3.bin"
        if dst_ddr3.is_symlink() or dst_ddr3.exists():
            dst_ddr3.unlink()
        dst_ddr3.symlink_to(ddr3)

    print(f"[patch] zeroed gammas for layers [{k1}:{k2}); dst = {dst}")


if __name__ == "__main__":
    main()
