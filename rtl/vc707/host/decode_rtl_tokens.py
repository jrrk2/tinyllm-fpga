#!/usr/bin/env python3
"""Decode the "RTL_TOKENS: …" line from a tb_autoregress_bfp run via
the SmolLM2 tokenizer.  Renders the on-chip BFP autoregress output as
the prompt + continuation, since prefill-step predictions are discarded
in the autoregress feedback (the model "sees" each prompt token, predicts
something, then we force-feed the next prompt token regardless — those
predictions only become part of the chain via step 3's output bridging
into step 4's input).

Usage:  python3 decode_rtl_tokens.py <run.log>
"""
import sys, re, os
from transformers import AutoTokenizer

if len(sys.argv) != 2:
    sys.exit(f"usage: {sys.argv[0]} <run.log>")
log = open(sys.argv[1]).read()
m = re.search(r'^RTL_TOKENS:\s*(.+)$', log, re.MULTILINE)
if not m:
    sys.exit("no RTL_TOKENS: line found in log")
rtl = [int(x) for x in m.group(1).split()]

# Read prompt tokens for proper prefix rendering.
prompt_path = os.path.join(os.path.dirname(__file__),
                           '..', 'generated', 'lbfp_full_PROMPT_TOKENS.txt')
prompt = [int(x) for x in open(prompt_path).read().split()] if os.path.exists(prompt_path) else []
N_PROMPT = len(prompt)

tok = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')

# Two views of the story:
#   raw      : just all RTL outputs (prefill predictions + autoregress)
#   complete : prompt + autoregress chain (step 3's output onward)
print(f"# {len(rtl)} RTL output tokens, {N_PROMPT}-token prompt\n")
print("Raw (every RTL output, including prefill predictions):")
print(f"  {tok.decode(rtl)!r}\n")
if N_PROMPT > 0 and len(rtl) >= N_PROMPT:
    autoregress_chain = prompt + rtl[N_PROMPT - 1:]
    print("Story (prompt + autoregress continuation):")
    print(f"  {tok.decode(autoregress_chain)!r}")
    print()
    print("Rendered:")
    print(tok.decode(autoregress_chain))
