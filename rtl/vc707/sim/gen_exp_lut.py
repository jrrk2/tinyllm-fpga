#!/usr/bin/env python3
"""Generate exp_lut.hex — 1024-entry Q1.15 LUT for softmax_q15.sv.

LUT maps a Q1.15 difference (x[k] - max_x) to exp(diff).

Indexing:
  diff is a 17-bit signed integer representing (x_buf[k] - max_r) in Q1.15 units.
  Range of diff: (-65536, 0] for inputs in (-1, 1).  The LUT is designed
  for diff_real in [-8, 0], giving 1024 entries spaced by 1/128 real units.

  index = clamp( (diff >> 8) + 1023, 0, 1023 )

  index 1023 -> diff_real = 0     -> exp(0) = 1.0  -> 0x8000 (32768 unsigned)
  index 767  -> diff_real = -2.0  -> exp(-2)        -> 4435
  index 0    -> diff_real = -8.0  -> exp(-8)        -> 11

Values are stored as 16-bit unsigned Q1.15 (0..32768).
32768 (0x8000) represents 1.0 and is valid for index 1023.

Output: ../generated/exp_lut.hex  (1024 lines, each a 4-hex-digit value)
"""
import math
import os

N_LUT = 1024
STEP  = 128.0  # 1/step real per index unit = 1/128

# Default = rtl/generated/ (canonical, matches gen_smollm_blockfp.py and
# Vivado's MICROGPT_WEIGHT_DIR).  Override with OUT=... e.g. sim/Makefile
# sets OUT=../generated so Verilator from rtl/vc707/sim/ finds it.
_default_out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'generated'))
out_dir = os.environ.get('OUT', _default_out)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'exp_lut.hex')

with open(out_path, 'w') as f:
    for i in range(N_LUT):
        diff_real = (i - 1023) / STEP   # in [-7.992, 0]
        val = round(math.exp(diff_real) * 32768)
        val = min(val, 65535)           # clamp to uint16
        val = max(val, 0)
        f.write(f'{val:04x}\n')

print(f'Wrote {N_LUT} entries to {out_path}')
# Spot-check
import subprocess
lines = open(out_path).readlines()
print(f'  LUT[0]    = {int(lines[0],16):5d}  (exp(-8)  = {math.exp(-8):.6f})')
print(f'  LUT[767]  = {int(lines[767],16):5d}  (exp(-2)  = {math.exp(-2):.6f})')
print(f'  LUT[895]  = {int(lines[895],16):5d}  (exp(-1)  = {math.exp(-1):.6f})')
print(f'  LUT[1023] = {int(lines[1023],16):5d}  (exp(0)   = {math.exp(0):.6f})')
