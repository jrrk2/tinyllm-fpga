# tinyllm-fpga — repository layout

Block-floating-point inference engine for SmolLM2-135M on Xilinx VC707.
See [`README.md`](README.md) for the project overview and
[`rtl/vc707/README.md`](rtl/vc707/README.md) for the full reproduction
guide.

## Active codepath (VC707 + SmolLM2-135M, BFP, DDR3-streamed)

- `rtl/vc707/`: board build, host client, Vivado scripts, release area
  - `rtl/vc707/src/smollm/`: BFP engine (autoregress, multilayer, layer,
    decode head, matvec, RMSNorm, RoPE, SwiGLU, softmax)
  - `rtl/vc707/src/vc707_microgpt_eth.sv`: top-level
  - `rtl/vc707/src/microgpt_eth_ctrl.sv`: FastTrans parser → Avalon-MM
  - `rtl/vc707/eth/`: SGMII MAC vendored from cva6
  - `rtl/vc707/host/bfp_client.cpp`: no-torch C++ Ethernet client
  - `rtl/vc707/host/gen_smollm_blockfp_{cfg,full,ddr}.py`: weight packers
  - `rtl/vc707/host/finetune_shakespeare.py`: SFT wrapper
  - `rtl/vc707/sim/`: Verilator testbenches + golden generators
- `rtl/src/include/`: SystemVerilog include fragments

## Legacy / archived (DE1-SoC, int8 microGPT)

- `rtl/src/`: int8 microGPT core (DE1-SoC port; not actively maintained)
- `rtl/python/`: JTAG host + reference scripts for the int8 path
- `rtl/tcl/`: System Console / Quartus TCL helpers
- `rtl/sim/`: ModelSim testbenches
- Repository-root `.bat` files: DE1-SoC build / program / inference wrappers

## Common commands (VC707 path)

From `rtl/vc707/`:

```sh
make                        # full Vivado build (~25 min)
make release-bitstream      # capture .bit + .mcs + vocab + reports
make release-model          # capture .bin + .hex set + golden tokens
make program-release        # JTAG-program the captured bitstream
make flash                  # write BPI flash (persists across reboots)
make run                    # load BRAMs → upload DDR3 → run → print tokens
```

See `rtl/vc707/README.md` for the full reproduce-from-release walk-through.
