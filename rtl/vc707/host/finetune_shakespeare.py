#!/usr/bin/env python3
"""Fine-tune SmolLM2-135M on a Shakespeare corpus for FPGA inference.

The bitstream is architecture-fixed (D=576, NL=30, H_Q=9, H_KV=3, HD=64,
FFN=1536, VOCAB=49152) but the per-model BRAMs and DDR3 weights are
host-loadable, so swapping the model is a pure software step — no
Vivado rebuild.

End-to-end flow:

    # 1. Fine-tune (one-time, ~10–30 min on a single GPU).
    python host/finetune_shakespeare.py --out generated/shakespeare-smollm

    # 2. Pack for FPGA — TWO scripts, both honour MODEL + PREFIX:
    #       _full.py  emits the host-loaded BRAM .hex files (gammas,
    #                 norm_w, prompt, embed, KV init).
    #       _ddr.py   bakes the wide-packed weight matrices into the
    #                 DDR3 .bin image that the FPGA streams over AXI.
    export MODEL=generated/shakespeare-smollm
    export PREFIX=shake_
    export PROMPT="Hark"        # keep prompt <= 4 tokens until N_PROMPT lifted in bitstream
    export N_GEN=32
    python host/gen_smollm_blockfp_full.py
    python host/gen_smollm_blockfp_ddr.py

    # 3. Upload to FPGA — no Vivado rebuild needed; same bitstream.
    cd host
    MGRT_PREFIX=shake_ ./bfp_client load-roms ../generated
    ./bfp_client all ../generated/shake_DDR3.bin

The corpus defaults to Karpathy's "tinyshakespeare" (1 MB, downloaded
to generated/tinyshakespeare.txt on first run).  Pass --corpus to
supply a different text file.
"""
import argparse
import os
import sys
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

TINYSHAKES_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


class TextBlocks(Dataset):
    """Tokenise once, slice into fixed-length blocks for causal-LM training."""

    def __init__(self, text: str, tokenizer, block_size: int):
        # Bypass the tokenizer's model_max_length warning — we tokenise the
        # whole corpus in one shot and slice into blocks ourselves; the
        # 8192-token model limit only applies to what we feed at training
        # time (one `block_size` slice per example), not to the tokenise
        # call itself.
        saved_max = tokenizer.model_max_length
        tokenizer.model_max_length = 10**12
        try:
            ids = tokenizer(text, return_tensors="pt").input_ids[0]
        finally:
            tokenizer.model_max_length = saved_max
        n_blocks = len(ids) // block_size
        if n_blocks == 0:
            raise ValueError(
                f"corpus too short for block_size={block_size} "
                f"(got {len(ids)} tokens)"
            )
        self.blocks = ids[: n_blocks * block_size].view(n_blocks, block_size)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return self.blocks[i]


def fetch_corpus(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"[finetune] downloading tinyshakespeare → {path}", file=sys.stderr)
        urllib.request.urlretrieve(TINYSHAKES_URL, path)
    return open(path, "r", encoding="utf-8").read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None,
                    help="Path to a plain-text corpus.  Defaults to tinyshakespeare "
                         "(auto-downloaded to generated/tinyshakespeare.txt).")
    ap.add_argument("--out", default="generated/shakespeare-smollm",
                    help="Directory to save the fine-tuned model (HuggingFace format).")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M",
                    help="HuggingFace base model to fine-tune.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--block-size", type=int, default=256,
                    help="Token block length per training example.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="AdamW learning rate.  3e-5 is conservative — bump to "
                         "1e-4 for a stronger style shift if loss plateaus.")
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0xc0ffee)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    corpus_path = args.corpus or "generated/tinyshakespeare.txt"
    text = fetch_corpus(corpus_path)
    print(f"[finetune] corpus: {corpus_path}  ({len(text):,} chars)", file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[finetune] device = {device}  base = {args.base_model}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32
    ).to(device)
    model.train()

    ds = TextBlocks(text, tok, args.block_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    n_steps = args.epochs * len(dl)
    print(
        f"[finetune] {len(ds)} blocks × {args.block_size} tokens; "
        f"{len(dl)} batches × {args.epochs} epochs = {n_steps} steps",
        file=sys.stderr,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=args.warmup_steps, num_training_steps=n_steps
    )

    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            batch = batch.to(device)
            out = model(input_ids=batch, labels=batch)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % args.log_every == 0:
                print(
                    f"  epoch {epoch + 1}/{args.epochs}  "
                    f"step {step:5d}/{n_steps}  "
                    f"loss = {out.loss.item():.3f}",
                    file=sys.stderr,
                )
            step += 1

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\n[finetune] saved {args.out}", file=sys.stderr)
    print("\nNext steps:", file=sys.stderr)
    print(
        f'  export MODEL={args.out} PREFIX=shake_ PROMPT="Hark" N_GEN=32\n'
        f"  python host/gen_smollm_blockfp_full.py     # .hex files\n"
        f"  python host/gen_smollm_blockfp_ddr.py      # DDR3 .bin\n"
        f"  MGRT_PREFIX=shake_ ./bfp_client load-roms ../generated\n"
        f"  ./bfp_client all ../generated/shake_DDR3.bin",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
