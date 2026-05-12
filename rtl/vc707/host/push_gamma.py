#!/usr/bin/env python3
"""Recompute GAMMA1/GAMMA2 from CURRENT calibration (margin=1.5) and push
to the FPGA brom_GAMMA1/2 via Ethernet at runtime.  No bitstream rebuild.

Use after a bitstream that wires GAMMA1/GAMMA2 as scale_wr_kind 7/8.

The baked-in hex values were calibrated with a slightly different margin
(or cal_prompt), giving lsc[norm1] up to 18% larger for layers 11-22
than the current code produces.  That under-magnitudes RMSNorm output,
which over 30 layers degrades the next-token logit distribution into
the ' a'/' Highlands' attractor.  Pushing freshly-calibrated GAMMA
values fixes it without recalibrating per-row scales or rebuilding.
"""
import sys, time, numpy as np, torch
sys.path.insert(0, 'host')
import sim_blockfp as S
from transformers import AutoModelForCausalLM, AutoTokenizer
from microgpt_eth import open_socket, reg_write

PEER = ("192.168.1.42", 19783)
D = 576
NL = 30
KIND_GAMMA1 = 7
KIND_GAMMA2 = 8


def push_gamma_table(s, kind, table_int16):
    """Write NL*D int16 entries via scale_wr protocol (0x01C stage, 0x01D fire)."""
    n = len(table_int16)
    print(f"  pushing {n} entries with kind={kind} ...", file=sys.stderr)
    for addr in range(n):
        v = int(table_int16[addr]) & 0xFFFF
        reg_write(s, PEER, [(0x01C, (kind << 16) | (addr & 0xFFFF))])
        time.sleep(0.001)
        reg_write(s, PEER, [(0x01D, v)])
        time.sleep(0.001)
        if addr % 1024 == 0 and addr > 0:
            print(f"    {addr}/{n}", file=sys.stderr)


def main():
    print("loading model + calibrating ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(S.MODEL)
    model = AutoModelForCausalLM.from_pretrained(S.MODEL,
                                                  torch_dtype=torch.float32).eval()
    sc = S.calibrate(model, tok,
                     "Once upon a time there was a princess.", margin=1.5)

    gammas_1 = np.zeros(NL * D, dtype=np.int16)
    gammas_2 = np.zeros(NL * D, dtype=np.int16)
    with torch.no_grad():
        for L in range(NL):
            g1 = model.model.layers[L].input_layernorm.weight \
                  .detach().cpu().numpy().astype(np.float32)
            g2 = model.model.layers[L].post_attention_layernorm.weight \
                  .detach().cpu().numpy().astype(np.float32)
            # fold_gamma in gen_smollm_blockfp.py: round(g * 32768 / lsc), clip int16
            v1 = np.round(g1 * 32768.0 / sc[L]["norm1"])
            v2 = np.round(g2 * 32768.0 / sc[L]["norm2"])
            gammas_1[L * D:(L + 1) * D] = np.clip(v1, -32768, 32767).astype(np.int16)
            gammas_2[L * D:(L + 1) * D] = np.clip(v2, -32768, 32767).astype(np.int16)

    s = open_socket(2.0)
    push_gamma_table(s, KIND_GAMMA1, gammas_1)
    push_gamma_table(s, KIND_GAMMA2, gammas_2)
    print("done.  Trigger inference (host/fpga_predict.py) to validate.", file=sys.stderr)


if __name__ == "__main__":
    main()
