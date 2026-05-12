#!/usr/bin/env python3
"""Run the bit-accurate RTL emulator for positions 0..POS-1, then dump the
accumulated KV cache to K_CACHE_INIT.hex / V_CACHE_INIT.hex in the same
format gen_smollm_blockfp.py emits.  The bitstream's $readmemh picks these
up at boot and seeds the KV cache, so attention at pos=POS reads RTL-self-
consistent K,V values for the prior tokens (instead of FP-rmsnorm values
that don't match what the RTL would have computed).

This replaces the FP-based KV init with one that matches the RTL pipeline's
own behavior — closing the loop on the calibration drift that produced the
' a'/' Highlands' attention misroute.

usage: python3 host/gen_rtl_kv_init.py
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, 'host')
os.environ['RTL_KV_PRELOAD'] = '0'   # compute KV from scratch
os.environ['RMS_INV_W']      = '16'  # match deployed bitstream (Q5.12)
import sim_blockfp as S
import sim_rtl_exact as R
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    tok = AutoTokenizer.from_pretrained(S.MODEL)
    model = AutoModelForCausalLM.from_pretrained(S.MODEL,
                                                  torch_dtype=torch.float32).eval()
    cfg = dict(D=576, H_Q=9, H_KV=3, HD=64, FFN=1536, NL=30, MAX_CTX=4)
    sc = S.calibrate(model, tok, "Once upon a time there was a princess.",
                     margin=1.5)
    # Match sim_rtl_exact.py's eff_sc derivation (identical to sc for SmolLM2)
    eff_sc = sc

    h_p2 = [(S.pow2(sc[li].get('hidden_in', 1.0)),
             S.pow2(max(sc[li].get('hidden_in', 1.0) + sc[li]['attn'], 1.0)),
             S.pow2(sc[li].get('hidden_out', 1.0))) for li in range(30)]

    layers = []
    with torch.no_grad():
        for li in range(30):
            L = model.model.layers[li]; d = {}
            for nm, sub in [('Wq', L.self_attn.q_proj), ('Wk', L.self_attn.k_proj),
                            ('Wv', L.self_attn.v_proj), ('Wo', L.self_attn.o_proj),
                            ('Wg', L.mlp.gate_proj),    ('Wu', L.mlp.up_proj),
                            ('Wd', L.mlp.down_proj)]:
                W = sub.weight.detach().cpu().numpy().astype(np.float32)
                Wq, rsc = S.quantize_int8_row(W); d[nm] = Wq; d[nm + '_r'] = rsc
            d['g1'] = L.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            d['g2'] = L.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)
            layers.append(d)
        embed = model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float32)

    ids = tok('Once upon a time', return_tensors='pt').input_ids[0].tolist()
    POS = len(ids) - 1   # 3 — we want KV[0..2] then forward at pos=3 reads them

    kv = [{'k_int': np.zeros((cfg['MAX_CTX'], cfg['H_KV'], cfg['HD']), dtype=np.int16),
           'v_int': np.zeros((cfg['MAX_CTX'], cfg['H_KV'], cfg['HD']), dtype=np.int16)}
          for _ in range(30)]

    # Force bit-accurate ops everywhere
    R.OPS = ['all']
    R.has = lambda op: True

    print(f"computing RTL-self-consistent KV for tokens 0..{POS-1} ...",
          file=sys.stderr)
    for step in range(POS):
        tid = ids[step]
        e_real = embed[tid].astype(np.float32)
        h = R.qfloor(e_real, float(1 << h_p2[0][0]))
        for li in range(30):
            h_in_p2, h1_p2, h_out_p2 = h_p2[li]
            h = R.fwd(h, layers[li], sc[li], h_in_p2, h1_p2, h_out_p2,
                      pos=step, kv=kv[li], kv_pos=step, cfg=cfg)
            if li < 29:
                h_real = h.astype(np.float64) * float(1 << h_out_p2) / 32768.0
                h = R.qfloor(h_real, float(1 << h_p2[li + 1][0]))

    # Now kv[li]['k_int'] / 'v_int' are MAX_CTX × H_KV × HD int16 tables.
    # gen_smollm_blockfp.py emit format: for each layer, flatten the
    # MAX_CTX × H_KV × HD array in C-order (the same as numpy's .flatten()).
    def write_hex(name, key):
        out_path = os.path.join('generated', f'tm_layer_{name}.hex')
        n = 0
        with open(out_path, 'w') as f:
            for li in range(30):
                for v in kv[li][key].flatten():
                    f.write(f"{int(v) & 0xFFFF:04x}\n")
                    n += 1
        print(f"  wrote {out_path}  ({n} entries)", file=sys.stderr)

    os.makedirs('generated', exist_ok=True)
    write_hex('K_CACHE_INIT', 'k_int')
    write_hex('V_CACHE_INIT', 'v_int')


if __name__ == '__main__':
    main()
