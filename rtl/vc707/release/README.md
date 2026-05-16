# release/ — snapshot of a known-good VC707 BFP build

This directory holds a captured bitstream + DDR3 weight image + vocab +
Vivado reports for a single, verified build of the VC707 SmolLM2-135M
BFP autoregress.  `make clean` deliberately does **not** touch it; use
`make distclean` to remove.

## Contents

| File                            | What it is                                         |
| ------------------------------- | -------------------------------------------------- |
| `vc707_microgpt_eth.bit`        | JTAG-loadable bitstream                            |
| `vc707_microgpt_eth.mcs`        | BPI x16 flash image (16 MB)                        |
| `lbfp_full_DDR3.bin`            | 337 MB packed BFP weight image — uploads to DDR3   |
| `bfp_vocab.bin`                 | Baked GPT-2 byte-level vocab for the C++ decoder   |
| `lbfp_full_cfg.svh`             | `D, HQ, HKV, HD, FFN, NL, MAX_CTX, VOCAB, …`        |
| `lbfp_full_PROMPT_TOKENS.txt`   | Input prompt token IDs (4 tokens)                  |
| `lbfp_full_GOLDEN_TOKENS.txt`   | FP32-reference expected output (regression target) |
| `build_version.svh`             | BUILD_VERSION constant baked into the bitstream    |
| `build_info.txt`                | git hash, dims, sizes, host, date                  |
| `reports/`                      | Vivado timing/utilisation/DRC reports               |

## Usage

From `rtl/vc707/`:

```
# One-shot capture from the current build:
make release

# Re-program / re-flash the FPGA from the captured artifacts:
make program-release      # JTAG (volatile — lost on power-cycle)
make flash-release        # BPI flash (persists across power cycles)

# Full end-to-end demo (builds host/bfp_client, then runs everything):
make demo
```

## Regression workflow

1. After any non-trivial change, run `make distclean && make && make release`
   to capture the new baseline.
2. Tag the commit so `git checkout <tag>; make release` can rebuild the
   same baseline from source.
3. To check a new feature hasn't broken the demo path: `make demo`
   should still produce the golden token sequence in
   `lbfp_full_GOLDEN_TOKENS.txt` (modulo FP32-vs-BFP divergence — the
   BFP-sim golden is what matters; see memory note
   `project_smollm_verilator_path`).

## DDR3 retention

JTAG reprogram loses MIG calibration, so DDR3 contents must be re-
uploaded **every** time after `program-release` (and the first time
after a power-on following `flash-release`).  `make demo` handles this
automatically — it always uploads the weight image before pulsing
restart.
