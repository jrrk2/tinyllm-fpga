# SmolLM2-135M on VC707 (Virtex-7) over raw Ethernet

A fully on-chip block-floating-point (BFP) inference engine for
HuggingFace **SmolLM2-135M** running on a Xilinx VC707 (XC7VX485T).
Weights stream from DDR3-1600 via MIG; per-model BRAMs (gammas, final
norm, prompt) are host-writable at runtime over 1 G SGMII Ethernet,
so swapping a fine-tuned model (e.g. a Shakespeare variant) is a
pure software step — no Vivado rebuild.

```
Once upon a time, in a far-off land, there lived a young...
```

That output is generated **on the FPGA** from the SmolLM2-135M
checkpoint quantised to 16-bit mantissa + 8-bit per-tile exponent.

## Reproduce the demo from a released bitstream

This is the no-Vivado path.  You need:

- A VC707 with the released bitstream loaded (JTAG-programmed or
  BPI-flashed; see "Programming the FPGA" below if you need to do it).
- The `release/` directory checked out (see "Getting the release artifacts").
- Linux host with `g++` (C++17), `iproute2`/`tcpdump` for diagnostics,
  and Ethernet wired between the host NIC and the VC707 SFP port.
- **Host NIC at `192.168.1.x/24`, MTU 1500** (no jumbo frames needed).
  FPGA's hardcoded IP is `192.168.1.42`, MAC `02:00:00:4d:47:31`.

### 1. Build the C++ host

```sh
cd rtl/vc707
make host/bfp_client       # ~2 s; needs only g++ + libstdc++
```

No PyTorch, no Python, no torch dependency — single C++17 source file.

### 2. Verify the FPGA is reachable

```sh
host/bfp_client read-crc
```

Expected output (the FPGA is alive, the BFP master may or may not have
consumed AXI beats yet):

```
[vocab] loaded 49152 tokens, 332367 byte blob (host/bfp_vocab.bin)
has_run = 0|1
rdata_hash = 0x........
```

If you instead see a timeout, check the cable and confirm the FPGA's
Ethernet link LED is on.

### 3. Run the canonical demo

```sh
make run                    # default model (lbfp_full_)
make run PREFIX=shake_      # any captured fine-tune
```

Single command, existing release → decoded text.  Auto-detects layout
(split release/bitstream + release/models/<PREFIX>/, or legacy flat
release/), conditionally loads per-model BRAMs (only if .hex files
are present — older $readmemh bitstreams skip this step), uploads
DDR3, restarts inference, reads tokens, decodes through the baked
vocab.  Expect ~3 minutes total (rate-limited DDR3 upload at ~2 MB/s).

The legacy `make demo` is still available for direct backward compat
(no auto-detect, only the flat layout, only the lbfp_full_ default).

Output ends with the golden token sequence:

```
RTL_TOKENS: 712 9612 3102 645 260 905 436 441 2408 281 624 198 198 18 504 905 436 441 805
DECODED:
##â town when the world was not yet in its

"The world was not only
```

If `DECODED` matches `release/lbfp_full_GOLDEN_TOKENS.txt`, the FPGA
is producing bit-exact output against the BFP-sim reference.

### 4. What `make demo` does, step by step

You can run these manually for finer control / debugging:

```sh
# 4a. Upload the 337 MB BFP weight image into DDR3.  Rate-limited to
#     ~2 MB/s — exceeding ~3.2 MB/s silently corrupts data while the
#     done_count still increments (project_ddr_upload_rate_limit.md).
#     Self-healing: any missing chunks are resent in passes 2/3.
host/bfp_client -v release/bfp_vocab.bin upload release/lbfp_full_DDR3.bin

# 4b. Pulse restart (single-shot autoregress), poll for completion,
#     read the 19 output tokens, decode via the baked vocab.
host/bfp_client tokens

# 4c. Or combine upload + restart + tokens + decode in one go:
host/bfp_client -v release/bfp_vocab.bin all release/lbfp_full_DDR3.bin
```

### 5. Swap in a different fine-tuned model (Shakespeare example)

This is the headline feature of the per-model BRAM design: any
SmolLM2-135M-class checkpoint can run on the **same bitstream** by
re-packing the weights and re-uploading.

```sh
# (i) Fine-tune (one-time, ~10–30 min on a single GPU).
python host/finetune_shakespeare.py --out generated/shakespeare-smollm

# (ii) Pack the model for the FPGA.  TWO scripts, both honour
#      MODEL + PREFIX + PROMPT.
export MODEL=generated/shakespeare-smollm
export PREFIX=shake_
export PROMPT="Hark"          # ≤4 tokens until N_PROMPT is lifted in the bitstream
export N_GEN=32
python host/gen_smollm_blockfp_full.py    # → shake_*.hex   (~30 s)
python host/gen_smollm_blockfp_ddr.py     # → shake_DDR3.bin (~2 min, 320 MB)

# (iii) Upload + run, no Vivado rebuild.
cd host
MGRT_PREFIX=shake_ ./bfp_client load-roms ../../generated   # ~10 s, self-healing
./bfp_client all ../../generated/shake_DDR3.bin             # ~3 min
```

`load-roms` writes the per-model BRAMs (G1/G2 gammas, final norm,
prompt) through regmap 0x060/0x061 to the eth_clk-domain write port
of true-dual-port BRAMs, then verifies them via regmap 0x062 and
re-writes any UDP-dropped entries up to 3 rounds.  A `FAIL` line
listing the first 8 stuck addresses means a hardware fault, not a
transient drop.

### 6. Verify BRAM contents without re-uploading

```sh
MGRT_PREFIX=shake_ host/bfp_client verify-roms ../generated
```

Reads every BRAM entry back through 0x062 and compares to the .hex
files (~10 s for the full set).  Useful after a long run to confirm
nothing has bit-flipped.

## Programming the FPGA

Skip this if you already have the bitstream loaded.

```sh
make program-release       # JTAG (volatile — DDR3 contents lost; re-run make demo)
make flash-release         # BPI x16 flash (persists across power cycles)
```

## Release layout: bitstream-release vs model-release

A bitstream and a model have independent lifecycles — one bitstream
serves many fine-tuned models (host-loadable BRAMs), and one fine-tune
targets many bitstream revisions.  `make release-bitstream` and
`make release-model` separate the two:

```
release/bitstream/             one per Vivado build
    vc707_microgpt_eth.bit
    vc707_microgpt_eth.mcs
    bfp_vocab.bin
    build_info.txt
    reports/

release/models/lbfp_full_/     one per fine-tune (default: SmolLM2 baseline)
release/models/shake_/         one per fine-tune (Shakespeare variant)
    <PREFIX>DDR3.bin
    <PREFIX>cfg.svh
    <PREFIX>{G1,G2,NORM_W}_{m,e}.hex
    <PREFIX>PROMPT.hex
    <PREFIX>{PROMPT,GOLDEN}_TOKENS.txt
    model_info.txt
```

Capture commands:

```sh
make release-bitstream                   # after a successful Vivado build
make release-model                       # default PREFIX=lbfp_full_
make release-model PREFIX=shake_         # Shakespeare variant
```

Demo using the split layout:

```sh
make program-release-bitstream           # one-time JTAG load (or use flash variant)
make demo-model                          # default model (lbfp_full_)
make demo-model PREFIX=shake_            # Shakespeare model on the same bitstream
```

The legacy flat `make release` / `make demo` still work for backward
compatibility (writes into `release/` root rather than the split
subdirs), but new releases should use the split form.

After any JTAG reprogram, the MIG refresh controller resets and DDR3
contents are lost within 64 ms (the refresh is soft fabric on Virtex-7,
not hard MIG silicon).  `make demo` always re-uploads, so this is
transparent — but bear it in mind if running `tokens` standalone.

## Getting the release artifacts

The `release/` directory holds a captured bitstream + DDR3 weight
image + vocab + Vivado reports for a single verified build.  It is
**not** included in the git repo for size reasons — fetch the latest
tagged release tarball:

```sh
# Tag: v0.1-golden (commit 41da092) — SmolLM2-135M canonical baseline
gh release download v0.1-golden -p 'release-*.tar.gz' -D rtl/vc707/
tar xf rtl/vc707/release-v0.1-golden.tar.gz -C rtl/vc707/
```

Or rebuild from source — see "Build from source" below.

## Build from source

Needs Vivado 2020.1+ and the standard SmolLM2 quantization pipeline.

```sh
export XILINX_PART=xc7vx485tffg1761-2
export XILINX_BOARD=xilinx.com:vc707:part0:1.4
export BOARD=vc707

# Generate the cfg the Vivado build needs (~2 s; tokenizer + config only).
python host/gen_smollm_blockfp_cfg.py  # cfg.svh + PROMPT_TOKENS.txt

# Generate the BFP weight artifacts (Python; ~3 min).  These are NOT
# Vivado build deps any more — gammas/norm_w/prompt are host-loaded
# at runtime via 0x060/0x061, weights stream from DDR3.  Run these
# when you want to capture a release-model.
python host/calibrate_smollm.py        # one-time; produces calibration scales
python host/gen_smollm_blockfp_full.py # .hex files for BRAMs (host-loaded)
python host/gen_smollm_blockfp_ddr.py  # 337 MB DDR3 .bin image

# Build the bitstream (Vivado; ~25 min on a modern desktop).
make                                   # → work/vc707_microgpt_eth.bit
make mcs                               # → work/vc707_microgpt_eth.mcs

# Capture a release snapshot that survives `make clean`.
make release
```

## Architecture overview

```
                ┌────────────────────────────────────────────┐
 SGMII (SFP) ──▶│ gig_ethernet_pcs_pma_0  (Xilinx IP)        │
                │   sgmii_soc → eth_mac_1g → framing_top     │
                └──────────────────┬─────────────────────────┘
                                   │ 17-bit addr / 64-bit data, 2-cycle latency
                                   ▼
                ┌────────────────────────────────────────────┐
                │ microgpt_eth_ctrl  (eth_clk 125 MHz)       │
                │   FastTrans frame parser (FT_REG_WRITE,    │
                │   FT_REG_READ, FT_DDR_WRITE)               │
                │   → Avalon-MM master → regmap              │
                └──────────────────┬─────────────────────────┘
                                   │
                                   │  ┌─── 0x010/0x011 → DDR3 stream ──┐
                                   │  │                                │
                                   │  │  ┌─── 0x060/0x061 → BRAM TDP ──┤
                                   ▼  ▼  ▼                             │
                ┌────────────────────────────────────────────┐         │
                │ vc707_microgpt_eth (top, eth_clk domain)   │         │
                │   regmap + CDC into core_clk where needed  │         │
                └──────┬───────────────────────────┬─────────┘         │
                       │                           │                   │
                       │ eth_clk (port A write+read of BRAMs)          │
                       │                                               │
                       ▼                                               │
                ┌────────────────────────────────────────────┐         │
                │ autoregress_bfp_top  (core_clk 40 MHz)     │         │
                │   ┌─────────────────────────────────┐      │         │
                │   │ smollm_multilayer_tm_bfp        │      │         │
                │   │   per-layer time-multiplex of:  │      │         │
                │   │   RMSNorm → GQA → RoPE → SwiGLU │◀─────┼─────────┤ AXI master
                │   │   weights stream from DDR3      │      │         │ to MIG
                │   │   gammas in TDP BRAM (port B)   │      │         │ (1 GB DDR3)
                │   └─────────────────────────────────┘      │         │
                │   ┌─────────────────────────────────┐      │         │
                │   │ smollm_decode_head_bfp          │◀─────┼─────────┘
                │   │   final RMSNorm + lm_head       │      │
                │   └─────────────────────────────────┘      │
                │   single-shot autoregress FSM, has_run latch│
                └────────────────────────────────────────────┘
```

Per-token compute: 30 layers × {attn + MLP} BFP matvecs streamed
weight-by-chunk from DDR3, ~2-3 tokens/sec at the conservative
40 MHz core clock.  The DDR3 image is 337 MB; the on-chip BRAM
holds only gammas (~70 KB), final norm (~1 KB), prompt (8 B), KV
cache (~50 KB), and per-stage hidden buffers.

## Frame protocol (FastTrans over raw Ethernet)

EtherType **0x4D47** ("MG").  All multi-byte fields little-endian.

| Type | Direction   | Layout / purpose                                   |
|------|-------------|----------------------------------------------------|
| 0x01 | host→FPGA   | REG_WRITE — pack up to 16 (addr16, data32, pad16)  |
| 0x02 | host→FPGA   | REG_READ  — start_addr (LE16) + nwords (1..32)     |
| 0x03 | FPGA→host   | REG_RSP  — start_addr + nwords + nwords × 32-bit   |
| 0x04 | FPGA→host   | HEARTBEAT (~10 Hz): state, last_token, out_len     |
| 0x05 | FPGA→host   | NAK (bad FCS / unknown type)                       |
| 0x06 | FPGA→host   | ACK (REG_WRITE acknowledged)                       |
| 0x07 | host→FPGA   | DDR_WRITE — 30-bit byte addr + 64-byte chunk       |

Key regmap addresses:

- `0x000` magic `MGRT`,  `0x001` version,  `0x04A` BUILD_VERSION
- `0x002` start/clear,  `0x049` has_run,  `0x04A` rdata_crc
- `0x010/0x011` DDR3 load triggers
- `0x060` BRAM target `{inc[31], kind[22:18], addr[17:0]}`
- `0x061` BRAM write — 16-bit data, pulses dual-port write at eth_clk
- `0x062` BRAM readback (port-A read on the same TDP BRAM)
- `0x1D0..0x1E1` result tokens (19 × 16-bit)
- `0x1F0` lay_done_latched,  `0x1F1` restart pulse

## Layout

```
rtl/vc707/
├── Makefile                      build flow + demo/release targets
├── README.md                     (you are here)
├── constraints/
│   └── microgpt_eth.xdc          pin/timing constraints
├── eth/                          vendored from cva6 (lowRISC/Solderpad)
│   └── *.sv, LICENSE, README.md
├── ip/                           Xilinx IP makefiles (component dirs gitignored)
│   ├── common.mk
│   ├── gig_ethernet_pcs_pma_0/   SGMII PCS/PMA — tcl/run.tcl regenerates
│   └── xlnx_mig_7_ddr3/          DDR3 controller — tcl/run.tcl regenerates
├── scripts/
│   ├── prologue.tcl  run.tcl  program.tcl  flash.tcl  write_cfgmem.tcl
├── src/
│   ├── vc707_microgpt_eth.sv          top-level
│   ├── microgpt_eth_ctrl.sv           FastTrans parser → Avalon-MM master
│   ├── smollm/                        SmolLM2 BFP engine
│   │   ├── autoregress_bfp_top.sv
│   │   ├── smollm_multilayer_tm_bfp.sv
│   │   ├── smollm_layer_bfp.sv
│   │   ├── smollm_decode_head_bfp.sv
│   │   └── README.md
│   └── …leaf ops: matvec / rmsnorm / rope / swiglu / softmax …
├── sim/                          Verilator testbenches + golden generators
│   ├── tb_autoregress_bfp_stream.{sv,cpp}
│   ├── gen_*.py                  per-op golden / LUT generators
│   └── …probe TCLs, mock_axi_slave, lint stubs…
├── host/
│   ├── bfp_client.cpp            no-torch C++ client (compiled to host/bfp_client)
│   ├── bfp_vocab.bin             baked GPT-2 byte-level vocab (49152 tokens)
│   ├── finetune_shakespeare.py   SFT script for swap-in fine-tunes
│   ├── gen_smollm_blockfp_full.py  → BRAM .hex set
│   ├── gen_smollm_blockfp_ddr.py   → DDR3 .bin image
│   ├── calibrate_smollm.py         → per-tensor BFP scales
│   └── smollm/                     int8 reference path (sim_int8.py, …)
└── release/                      captured snapshot (see release/README.md)
```

## Provenance

- `eth/*.sv`, `eth/LICENSE`, `eth/README.md` — copied from
  `cva6/corev_apu/fpga/src/cva6-ethernet/` (lowRISC / Solderpad).
- `ip/common.mk`, `ip/gig_ethernet_pcs_pma_0/{Makefile,tcl/run.tcl}` —
  adapted from `cva6/corev_apu/fpga/xilinx/`.
- `scripts/{prologue,run,program,flash}.tcl`, `Makefile`,
  `constraints/microgpt_eth.xdc` — adapted from
  `cva6/corev_apu/fpga/{scripts,constraints}/`.
- `src/vc707.svh` — copied from `cva6/corev_apu/fpga/src/vc707.svh`.
- SmolLM2-135M weights — Apache-2.0, HuggingFaceTB/SmolLM2-135M.
- BFP engine, host client, packers, fine-tune wrapper — this repo.
