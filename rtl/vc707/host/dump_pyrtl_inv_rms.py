#!/usr/bin/env python3
"""Dump the same RMS_PROBE log as Verilator: v_q30, v_msb, inv_rms for every
rmsnorm call across the 4-token forward pass.  Output to stderr in the same
format as the Verilator $display so we can diff line-by-line.
"""
import sys, numpy as np, torch, math
sys.path.insert(0, 'host')
import sim_blockfp as S
import sim_rtl_exact as R
from transformers import AutoModelForCausalLM, AutoTokenizer

# Patch rmsnorm_rtl to also print v_q30, v_msb, inv_rms.
_orig = R.rmsnorm_rtl
def rmsnorm_rtl_probe(x_int, gamma_scaled):
    EPS_Q30 = 10737
    D = len(x_int)
    INV_D_Q32 = (1 << 32) // D
    sum_sq = int(np.sum(x_int.astype(np.int64) ** 2))
    v_q30 = ((sum_sq * INV_D_Q32) >> 32) + EPS_Q30
    v_msb = (int.bit_length(int(v_q30)) - 1) if v_q30 > 0 else 0
    res = _orig(x_int, gamma_scaled)
    # Re-derive final inv_rms (cheap: rerun the NR sequence, same as RTL).
    if R.RMS_INV_W == 16:
        inv_rms = R.RTL_SEED.get(v_msb, 65535)
    else:
        inv_rms = R.SEED.get(v_msb, R.RMS_INV_MAX)
    C = (3 * (1 << 30)) // 2
    for _ in range(3):
        y_sq = inv_rms * inv_rms
        vy2 = v_q30 * y_sq
        vy2_q30 = vy2 >> 24
        corr = 0 if vy2_q30 >= (1 << 32) else max(0, C - (vy2_q30 >> 1))
        inv_rms = min((inv_rms * corr + (1 << 29)) >> 30, R.RMS_INV_MAX)
    print(f"RMS_PROBE  v_q30={v_q30}  v_msb={v_msb}  inv_rms={inv_rms}",
          file=sys.stderr)
    return res

R.rmsnorm_rtl = rmsnorm_rtl_probe
R.OPS = ['all']
R.has = lambda op: True
R.main()
