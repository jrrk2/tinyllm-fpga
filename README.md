# tinyllm-fpga

A fully on-chip block-floating-point inference engine for **SmolLM2-135M**
running on a Xilinx VC707 (Virtex-7 XC7VX485T).  Weights stream from
DDR3-1600 via MIG; per-model BRAMs are host-loadable at runtime over
1 G Ethernet, so swapping a fine-tuned model (e.g. Shakespeare style)
is a software step — no Vivado rebuild.

```
Once upon a time, in a far-off land, there lived a young...
```

The above is generated **on the FPGA** from the SmolLM2-135M
checkpoint, quantised to 16-bit mantissa + 8-bit per-tile exponent
(block-FP), with no host involvement other than uploading the weight
image and reading the result tokens.

## Status

| Component | State |
|-----------|-------|
| SmolLM2-135M canonical baseline | ✅ matches BFP-sim golden tokens bit-for-bit |
| Fresh-clone reproduce (clone → curl release → `make run` → golden) | ✅ verified at v0.2.0 |
| Runtime prompt swap (8-token prompt, no Vivado rebuild) | ✅ `n_prompt_active` auto-set by load-roms |
| Shakespeare-style fine-tune (swap model at runtime, same bitstream) | ✅ |
| Throughput | ~2-3 tokens/sec at 40 MHz core clock |
| Timing | WNS = +0.070 ns, all constraints met |
| Build target | Vivado 2020.1+, VC707 board |

## Quick start

From a fresh clone, with a VC707 wired up and on `192.168.1.x/24`:

```sh
git clone https://github.com/jrrk2/tinyllm-fpga
cd tinyllm-fpga/rtl/vc707/release

# Pull the 4 large binaries from the v0.2.0 GitHub release (~397 MB).
# The repo tracks the small text artifacts (.hex / .prm / cfg / reports);
# only the .bit / .mcs / .bin / vocab live in Releases.
BASE=https://github.com/jrrk2/tinyllm-fpga/releases/download/v0.2.0
for f in vc707_microgpt_eth.bit vc707_microgpt_eth.mcs \
         lbfp_full_DDR3.bin bfp_vocab.bin; do
    curl -LO $BASE/$f
done

# (optional) verify
md5sum --check <<'EOF'
3577e1840ccf60512befe520e19ac30b  vc707_microgpt_eth.bit
a11ca7ded50d023c8dac194becd71918  vc707_microgpt_eth.mcs
c4d5010afda27fb551ff3f4c09a50ac6  lbfp_full_DDR3.bin
705e773b7a60775541444d7db8061b30  bfp_vocab.bin
EOF

cd ..                       # back to rtl/vc707
make program-release        # JTAG-load the bitstream (~30 s; needs Vivado)
make run                    # build host, load BRAMs, upload DDR3, decode (~3 min)
```

You should see:

```
RTL_TOKENS: 712 9612 3102 645 260 905 436 441 2408 281 624 198 198 18 504 905 436 441 805
DECODED:
##â town when the world was not yet in its

"The world was not only
```

Building host/bfp_client needs only `g++ -std=c++17`.  No Vivado is
needed after `program-release` — that's a one-time JTAG load (or use
`make flash-release` for the BPI flash version that persists across
power cycles).

See [`rtl/vc707/README.md`](rtl/vc707/README.md) for the full
reproduction guide, including:

- changing the prompt at runtime (any length up to NPROMPT_MAX=48)
- fine-tuning SmolLM2 on a custom corpus and swapping models
- the FastTrans wire protocol over raw Ethernet
- the split bitstream-release vs model-release layout
- building the bitstream from source

## Architecture (one-paragraph version)

A single `microgpt_eth_ctrl` parses raw Ethernet frames (EtherType
`0x4D47`) into an Avalon-MM master that drives a regmap.  The regmap
controls a single-shot `autoregress_bfp_top` which time-multiplexes
30 SmolLM2 layers through a shared BFP matvec engine, an RMSNorm /
RoPE-CORDIC / SwiGLU / softmax leaf-op set, and a final decode-head.
Weights stream from a 337 MB DDR3 image (`lbfp_full_DDR3.bin`) over
an AXI master to MIG; per-model gammas, final norm, and prompt
tokens live in true-dual-port BRAMs whose write port is on `eth_clk`
(host-loaded at runtime via regmap 0x060/0x061) and whose read port
is on `core_clk` (FSM-side).  Output tokens land in a 32-word regmap
window the host reads via `FT_REG_READ`.

```
Host ── SGMII/SFP ──▶ MAC ──▶ FastTrans parser ──▶ regmap
                                                      │
                                  ┌───────────────────┼──────────────┐
                                  ▼                                  ▼
                  eth_clk: BRAM port-A write           core_clk: autoregress FSM
                  (gammas / norm / prompt)             (RMSNorm → GQA → RoPE
                                                       → SwiGLU → softmax)
                                                              │
                                                              ▼
                                              AXI master ──▶ MIG ──▶ DDR3 (weights)
```

## Repository layout

```text
rtl/src/             Synthesizable RTL (shared core, includes)
rtl/vc707/           VC707 board build + host client
    src/             VC707 top, SmolLM2 BFP engine, FastTrans parser
    eth/             SGMII MAC vendored from cva6 (lowRISC/Solderpad)
    host/            C++ Ethernet client (bfp_client), Python packers,
                     fine-tune scripts
    sim/             Verilator testbenches + golden generators
    scripts/         Vivado synth / program / flash TCL
    constraints/     XDC pin & timing constraints
    release/         Captured bitstream + model artifacts (gitignored)
    README.md        Full reproduction guide
rtl/generated/       Packer outputs — gitignored, regenerated by Python
rtl/docs/            Design notes, archived writeups
```

## Provenance

Originally based on
[Luthiraa's microGPT-RTL](https://github.com/Luthiraa) work on a
DE1-SoC + Cyclone V port of Karpathy's microGPT.  This fork ports
the inference engine to Virtex-7 / VC707, scales it up from a small
character-level model to the full **SmolLM2-135M** (Llama-arch:
RMSNorm + GQA + RoPE + SwiGLU, D=576, NL=30, FFN=1536, vocab=49152),
adds block-floating-point quantisation with per-tensor calibrated
scales, DDR3 weight streaming via MIG, and a runtime model-swap path
through host-writable BRAMs.

The original DE1-SoC int8 microGPT path still lives under `rtl/src/`
but is not actively maintained — the active codepath is the BFP
engine under `rtl/vc707/src/smollm/`.

## License

RTL and host code: Apache-2.0.
Vendored SGMII stack under `rtl/vc707/eth/`: lowRISC / Solderpad
(see `rtl/vc707/eth/LICENSE`).
SmolLM2-135M weights: Apache-2.0, courtesy of HuggingFaceTB.
