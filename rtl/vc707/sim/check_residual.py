#!/usr/bin/env python3
"""Verify residual_bfp arithmetic by re-computing BFP-aligned add from
RTL hin and RTL o, then comparing to RTL h1 dump."""
import os, sys

TILE = 16

def read_hex(path, width=16, signed=True):
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


hin_m = read_hex("../generated/lbfp_HIN_m.hex", 16, True)
hin_e = read_hex("../generated/lbfp_HIN_e.hex", 8,  True)
o_m   = read_hex("rtl_o_m.hex", 16, True)
o_e   = read_hex("rtl_o_e.hex", 8,  True)
h1_m  = read_hex("rtl_h1_m.hex", 16, True)
h1_e  = read_hex("rtl_h1_e.hex", 8,  True)

D = len(hin_m)
NT = D // TILE
assert len(o_m) == D
assert len(h1_m) == D
assert len(hin_e) >= NT, f"hin_e has {len(hin_e)} entries, need ≥{NT}"

print(f"D={D} NT={NT}")
print(f"hin_e = {hin_e[:NT]}")
print(f"  o_e = {o_e[:NT]}")
print(f" h1_e = {h1_e[:NT]} (RTL)")

# BFP-aligned residual: for each tile, emax = max(hin_e, o_e), shift each
# mantissa to align, sum at 17 bits, find max-leading bit, re-quantize.
golden_h1_m = [0] * D
golden_h1_e = [0] * NT
for t in range(NT):
    ea = hin_e[t]; eb = o_e[t]
    emax = max(ea, eb)
    sums = []
    max_lead = 0
    for k in range(TILE):
        sh_a = emax - ea; sh_b = emax - eb
        # Arithmetic shift right; sh<16 to keep value.
        a_a = (hin_m[t*TILE+k] >> sh_a) if sh_a < 16 else 0
        b_a = (o_m  [t*TILE+k] >> sh_b) if sh_b < 16 else 0
        s = a_a + b_a
        sums.append(s)
        if s != 0:
            lp = abs(s).bit_length() - 1
            if lp > max_lead: max_lead = lp
    shift = max_lead - 14
    for k in range(TILE):
        if shift >= 0:
            sh = sums[k] >> shift
        else:
            sh = sums[k] << (-shift)
        m = sh & 0xFFFF
        if m & 0x8000: m -= 0x10000
        golden_h1_m[t*TILE+k] = m
    golden_h1_e[t] = max(-128, min(127, emax + shift))

mismatches = 0
max_diff = 0
for i in range(D):
    diff = abs(h1_m[i] - golden_h1_m[i])
    if diff > 2:
        mismatches += 1
        if mismatches <= 5:
            print(f"  h1[{i}] RTL={h1_m[i]} BFP_gold={golden_h1_m[i]} (e={h1_e[i//TILE]}/{golden_h1_e[i//TILE]})")
    if diff > max_diff: max_diff = diff
print(f"BFP-golden vs RTL h1: {mismatches} mismatches >2 LSB  max_diff={max_diff}")
print(f"Golden tile exps: {golden_h1_e}")
