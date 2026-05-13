#!/usr/bin/env python3
"""Diff RTL stage dumps (rtl_*.hex) against Python golden (../generated/lbfp_STAGE_*.hex).

Prints first stage where RTL diverges from golden by more than MANT_TOL (~ 50 LSB)
or exp differs by more than 1.  Exit status: 1 if any stage off, 0 if all close.
"""
import os, sys

STAGES = [
    ('n1',   'N1'),
    ('qpre', 'QPRE'),
    ('kpre', 'KPRE'),
    ('v',    'V'),
    ('q',    'Q'),       # post-rope
    ('k',    'K'),       # post-rope
    ('attn', 'ATTN'),
    ('o',    'O'),
    ('h1',   'H1'),
    ('n2',   'N2'),
    ('g',    'G'),
    ('u',    'U'),
    ('mlp',  'MLP'),
    ('d',    'D'),
]

MANT_TOL = 200    # generous; we just want to spot the first divergence
EXP_TOL  = 1


def read_hex(path, signed=True, width=16):
    if not os.path.exists(path):
        return None
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            v = int(line, 16)
            if signed and (v & (1 << (width-1))):
                v -= (1 << width)
            out.append(v)
    return out


for short, long in STAGES:
    rtl_m = read_hex(f"rtl_{short}_m.hex", signed=True, width=16)
    rtl_e = read_hex(f"rtl_{short}_e.hex", signed=True, width=8)
    gld_m = read_hex(f"../generated/lbfp_STAGE_{long}_m.hex", signed=True, width=16)
    gld_e = read_hex(f"../generated/lbfp_STAGE_{long}_e.hex", signed=True, width=8)
    if rtl_m is None:
        print(f"{short:6s}  RTL DUMP MISSING")
        continue
    if gld_m is None:
        print(f"{short:6s}  GOLDEN DUMP MISSING")
        continue
    n = min(len(rtl_m), len(gld_m))
    ne = min(len(rtl_e), len(gld_e))
    # First-element comparison
    max_m_diff = 0
    bad_m = 0
    first_bad = -1
    # Real-value comparison: each element's real = mant/32768 * 2^(tile_exp).
    TILE = 16
    max_real_rel_diff = 0.0
    sum_rel = 0.0
    cnt = 0
    for i in range(n):
        diff = abs(rtl_m[i] - gld_m[i])
        if diff > max_m_diff: max_m_diff = diff
        if diff > MANT_TOL:
            bad_m += 1
            if first_bad < 0: first_bad = i
        # Real-value comparison
        rtl_real = rtl_m[i] / 32768.0 * (2.0 ** rtl_e[i // TILE])
        gld_real = gld_m[i] / 32768.0 * (2.0 ** gld_e[i // TILE])
        rel = abs(rtl_real - gld_real) / (abs(gld_real) + 1e-9)
        if rel > max_real_rel_diff: max_real_rel_diff = rel
        sum_rel += rel; cnt += 1
    max_e_diff = max(abs(rtl_e[i] - gld_e[i]) for i in range(ne))
    flag = "OK " if (max_m_diff <= MANT_TOL and max_e_diff <= EXP_TOL) else "BAD"
    mean_rel = sum_rel / cnt if cnt else 0.0
    real_flag = "OK " if max_real_rel_diff < 0.05 else "BAD"
    print(f"{short:6s}  N={n:4d}  rel_max={max_real_rel_diff:.3e}  rel_mean={mean_rel:.3e}  "
          f"mant_max={max_m_diff:6d}  bad={bad_m:3d}@{first_bad:3d}  exp_max={max_e_diff:2d}  "
          f"{real_flag}")
