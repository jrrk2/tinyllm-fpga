#!/usr/bin/env python3
"""Emit only the small cfg artifacts that Vivado synthesis needs:

    <PREFIX>cfg.svh           D / HQ / HKV / HD / FFN / NL / MAX_CTX /
                              VOCAB / NPROMPT / NPROMPT_MAX / NGEN
    <PREFIX>PROMPT_TOKENS.txt prompt token IDs (one per line — metadata
                              for the host C++ client, no synth use)

The bulky per-model `.hex` set (~163 files, ~hundreds of MB) is NOT
emitted here — see gen_smollm_blockfp_full.py for that.  The split
matters because the dual-port BRAM rework moved gammas / norm_w /
prompt to host-loaded BRAMs, and the streaming path moved all weight
matrices to DDR3 — so the Vivado build no longer `$readmemh`s any
per-model data and the heavy hex bake is purely runtime.

Uses AutoConfig + AutoTokenizer only (no AutoModelForCausalLM) so this
runs in ~2 seconds even from a cold HuggingFace cache, vs ~30 s for
the full bake.
"""
import os
import sys

from transformers import AutoConfig, AutoTokenizer


MODEL   = os.environ.get('MODEL',   'HuggingFaceTB/SmolLM2-135M')
D       = int(os.environ.get('D',       576))
H_Q     = int(os.environ.get('H_Q',     9))
H_KV    = int(os.environ.get('H_KV',    3))
HD      = int(os.environ.get('HD',      64))
FFN     = int(os.environ.get('FFN',     1536))
NL      = int(os.environ.get('NL',      30))
MAX_CTX = int(os.environ.get('MAX_CTX', 64))
PROMPT  = os.environ.get('PROMPT',  'Once upon a time')
N_GEN   = int(os.environ.get('N_GEN',   15))
PREFIX  = os.environ.get('PREFIX',  'lbfp_full_')
OUT     = os.environ.get('OUT',     'generated')
LANES   = 16

# Cap: NPROMPT_MAX + N_GEN must fit the legacy 32-word result-token
# regmap window at 0x1D0..0x1EF (= 64 × 16-bit slots).  48 leaves 1
# slot of margin and is plenty for the demo (~12× the old 4-token
# limit).  Override via env if you've already widened the regmap.
NPROMPT_MAX = int(os.environ.get('NPROMPT_MAX', 48))


def main():
    os.makedirs(OUT, exist_ok=True)

    print(f"[cfg] loading config + tokenizer from {MODEL} ...", file=sys.stderr)
    config = AutoConfig.from_pretrained(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)

    vocab_size = config.vocab_size
    pad = (LANES - vocab_size % LANES) % LANES
    ids = tok(PROMPT, return_tensors='pt').input_ids[0].tolist()
    if len(ids) > NPROMPT_MAX:
        sys.exit(f"[cfg] PROMPT tokenises to {len(ids)} > NPROMPT_MAX={NPROMPT_MAX}; "
                 f"shorten the prompt or bump NPROMPT_MAX.")

    cfg_path = os.path.join(OUT, f"{PREFIX}cfg.svh")
    with open(cfg_path, 'w') as f:
        f.write(f"`define LBFP_FULL_D           {D}\n")
        f.write(f"`define LBFP_FULL_HQ          {H_Q}\n")
        f.write(f"`define LBFP_FULL_HKV         {H_KV}\n")
        f.write(f"`define LBFP_FULL_HD          {HD}\n")
        f.write(f"`define LBFP_FULL_FFN         {FFN}\n")
        f.write(f"`define LBFP_FULL_NL          {NL}\n")
        f.write(f"`define LBFP_FULL_MAX_CTX     {MAX_CTX}\n")
        f.write(f"`define LBFP_FULL_VOCAB       {vocab_size + pad}\n")
        f.write(f"`define LBFP_FULL_NPROMPT     {len(ids)}\n")
        # NPROMPT_MAX sizes prompt_rom and the result-tokens buffer at
        # synthesis.  Host overrides the active length at runtime via
        # regmap 0x063 — any prompt 1..NPROMPT_MAX runs on the same
        # bitstream without a Vivado rebuild.
        f.write(f"`define LBFP_FULL_NPROMPT_MAX {NPROMPT_MAX}\n")
        f.write(f"`define LBFP_FULL_NGEN        {N_GEN}\n")

    prompt_path = os.path.join(OUT, f"{PREFIX}PROMPT_TOKENS.txt")
    with open(prompt_path, 'w') as f:
        for tid in ids:
            f.write(f"{tid}\n")

    print(f"[cfg] {cfg_path}  ({len(ids)} prompt tokens, vocab={vocab_size}+{pad}pad)",
          file=sys.stderr)
    print(f"[cfg] {prompt_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
