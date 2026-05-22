## Makefile — VC707 microgpt over raw Ethernet (replaces JTAG bridge)
##
## Usage:  make
##         make program
##         make flash
##         make clean
##
## Env (mirrors cva6 sourceme.sh) — defaults supplied below, override on the
## command line or in the environment if your install differs:
##   XILINX_VIVADO_VERSION=2020.1
##   XILINX_VIVADO_ROOT=/opt/Xilinx/Vivado/$(XILINX_VIVADO_VERSION)
##   XILINX_PART=xc7vx485tffg1761-2
##   XILINX_BOARD=xilinx.com:vc707:part0:1.4
##   BOARD=vc707
##
## The IPs were generated with Vivado 2020.1 (see ip/*/.../2020.1.1 cache
## dirs) — newer versions auto-upgrade on first parse but may emit warnings.

# ----------------------------------------------------------------------
# Vivado install + version defaults.
#
# Picks $(XILINX_VIVADO_ROOT)/bin/vivado when present, otherwise falls
# back to whatever `vivado` is on PATH (so a `source settings64.sh`
# workflow still works untouched).  Prepending the bin dir to PATH also
# lets the per-IP sub-makes (ip/common.mk → bare `vivado`) and any
# Vivado-launched helpers resolve the same executable.
# ----------------------------------------------------------------------
XILINX_VIVADO_VERSION ?= 2020.1
XILINX_VIVADO_ROOT    ?= /opt/Xilinx/Vivado/$(XILINX_VIVADO_VERSION)

ifneq ($(wildcard $(XILINX_VIVADO_ROOT)/bin/vivado),)
  VIVADO ?= $(XILINX_VIVADO_ROOT)/bin/vivado
  export PATH := $(XILINX_VIVADO_ROOT)/bin:$(PATH)
  export XILINX_VIVADO ?= $(XILINX_VIVADO_ROOT)
else
  VIVADO ?= vivado
endif

# Board/part triple consumed by scripts/prologue.tcl and ip/*/tcl/run.tcl
# via $::env(...).  Must be exported so the recursive `make` in the IP
# rule and the Vivado-launched Tcl both see them.
XILINX_PART  ?= xc7vx485tffg1761-2
XILINX_BOARD ?= xilinx.com:vc707:part0:1.4
BOARD        ?= vc707
export XILINX_PART XILINX_BOARD BOARD

VIVADOFLAGS ?= -nojournal -mode batch -source scripts/prologue.tcl

work-dir := work
bit := $(work-dir)/vc707_microgpt_eth.bit
mcs := $(work-dir)/vc707_microgpt_eth.mcs
ip-dir := ip

ips := gig_ethernet_pcs_pma_0.xci xlnx_mig_7_ddr3.xci microgpt_ila_core.xci microgpt_ila_axi.xci
ips := $(addprefix $(work-dir)/, $(ips))

# ----------------------------------------------------------------------
# Per-layer test data needed by the synth flow
#
# smollm_layer.sv $readmemh's the gamma/scale/KV-cache .hex files at
# elaboration; smollm_multilayer_tm_selftest.sv `includes the
# hidden_in case-statement ROM and the DDR3 base offsets.  Stale or
# missing files cause silent BRAM-init / address mismatches, so make
# the bitstream depend on a sentinel that re-runs both generators.
#
# Override on the command line to test other configs:
#   make NLAYERS=2 LAYER_CONFIG=smollm
#   make NLAYERS=3 LAYER_CONFIG=small
# ----------------------------------------------------------------------
GENERATED_DIR := ../generated
NLAYERS       ?= 30
LAYER_CONFIG  ?= smollm

tm_stamp := $(GENERATED_DIR)/.tm_layer_NL$(NLAYERS)_$(LAYER_CONFIG).stamp

# Brom names whose contents must track the matching tm_layer_<NAME>.hex.
BROM_NAMES := GAMMA1 GAMMA2 SCALE_Q SCALE_K SCALE_V SCALE_O \
              SCALE_GATE SCALE_UP SCALE_DOWN
BROM_SV    := $(foreach n,$(BROM_NAMES),$(GENERATED_DIR)/brom_$(n).sv)

# `all` must be the first non-pattern target so it stays the default
# goal (otherwise `make` would build $(tm_stamp) and stop).
all: $(bit)

# Regenerate the per-layer .hex files when the gen scripts change OR the
# requested NL/config no longer matches the previous stamp.  The
# `rm -f .tm_layer_*.stamp` line clears stamps for OTHER configs so a
# switch from NL=3 small to NL=30 smollm forces a full regen.
#
# smollm config: uses gen_smollm_blockfp.py which extracts REAL SmolLM2-135M
# weights + per-tensor calibration + scale-aware SwiGLU/Attn factors
# (tm_layer_swiglu_attn.svh) consumed by swiglu.sv and smollm_layer.sv.
# small config: random weights for Verilator regression — gen_multilayer_test.py.
ifeq ($(LAYER_CONFIG),smollm)
TM_HEX_GEN := host/gen_smollm_blockfp.py
# NL is fixed by the model (model.config.num_hidden_layers = 30 for SmolLM2-135M);
# the Makefile NLAYERS variable is for the small-config flow.
TM_HEX_CMD := python3 host/gen_smollm_blockfp.py
else
TM_HEX_GEN := host/gen_multilayer_test.py
TM_HEX_CMD := python3 host/gen_multilayer_test.py --mode tm --nlayers $(NLAYERS) --config $(LAYER_CONFIG)
endif

$(tm_stamp): $(TM_HEX_GEN)
	@echo "=== regenerating tm_layer_*  (NL=$(NLAYERS) config=$(LAYER_CONFIG)) ==="
	@rm -f $(GENERATED_DIR)/.tm_layer_*.stamp
	$(TM_HEX_CMD)
	@touch $@

# SiLU LUT for swiglu.sv.  Regen when SILU_LUT_SCALE in gen_swiglu_lut.py
# changes (it's the constant at the top of the file, encoded into every
# entry of the 65 K-row hex output).
$(GENERATED_DIR)/silu_lut.hex: sim/gen_swiglu_lut.py
	python3 sim/gen_swiglu_lut.py

# Exp LUT for softmax_bfp.sv / softmax_q15.sv.  Architecture LUT,
# model-independent.  Was previously assumed-present (only the sim
# Makefile had a rule); now it's an explicit synth-time dep so a clean
# checkout builds without surprises.
$(GENERATED_DIR)/exp_lut.hex: sim/gen_exp_lut.py
	python3 sim/gen_exp_lut.py

# ----------------------------------------------------------------------
# Block-FP weight bake for smollm_layer_bfp (small-dim selftest config).
# Emits all weight / gamma / KV-init mantissa+exponent hex files plus a
# hidden-in LUT SVH used by smollm_layer_bfp_selftest.sv.  Even when the
# int8 path is the active top-level instantiation, Vivado parses the BFP
# modules listed in run.tcl and elaborates their $readmemh initializers
# — so the hex files must exist (else: warnings + zero-initialized BRAMs).
#
# Override on the command line:
#   make BFP_D=576 BFP_HD=64 BFP_FFN=1536 BFP_MAX_CTX=4 ...
# ----------------------------------------------------------------------
BFP_D       ?= 64
BFP_H_Q     ?= 1
BFP_H_KV    ?= 1
BFP_HD      ?= 64
BFP_FFN     ?= 128
BFP_MAX_CTX ?= 4
BFP_POS     ?= 3
BFP_KV_POS  ?= 3

lbfp_stamp := $(GENERATED_DIR)/.lbfp_D$(BFP_D)_H$(BFP_H_Q)x$(BFP_H_KV)_HD$(BFP_HD)_F$(BFP_FFN)_C$(BFP_MAX_CTX).stamp

$(lbfp_stamp): host/gen_smollm_blockfp_bfp.py
	@echo "=== baking block-FP weights (D=$(BFP_D) H_Q=$(BFP_H_Q) H_KV=$(BFP_H_KV) HD=$(BFP_HD) FFN=$(BFP_FFN) MAX_CTX=$(BFP_MAX_CTX)) ==="
	@rm -f $(GENERATED_DIR)/.lbfp_*.stamp
	D=$(BFP_D) H_Q=$(BFP_H_Q) H_KV=$(BFP_H_KV) HD=$(BFP_HD) FFN=$(BFP_FFN) \
	MAX_CTX=$(BFP_MAX_CTX) POS=$(BFP_POS) KV_POS=$(BFP_KV_POS) \
	  python3 host/gen_smollm_blockfp_bfp.py
	@touch $@

# Pattern rule: each brom_<NAME>.sv depends on its tm_layer_<NAME>.hex
# AND on the gen_brom_sv.py script.  This catches the case where some
# other flow (e.g. the sim Verilator regression) overwrites the .hex
# files at a different config — make will then rebuild the brom .sv
# before letting Vivado run.
$(BROM_SV): $(GENERATED_DIR)/brom_%.sv: $(GENERATED_DIR)/tm_layer_%.hex \
                                        host/gen_brom_sv.py | $(tm_stamp)
	python3 host/gen_brom_sv.py $< $@ brom_$*

# Toggle the top-level layer instance.
#   USE_BFP=1    : swap the int8 multilayer path for the BFP autoregress
#                  stack (autoregress_bfp_top + 30-layer SmolLM2-135M).
#                  Adds MICROGPT_USE_BFP to VIVADO_DEFINES.
#   BFP_STREAM=1 : (only meaningful with USE_BFP=1) the BFP path streams
#                  W?_m/e + EMBED_m/e + EMBED_LU_m/e from DDR3 via the
#                  weight_streamer_bfp_mt AXI master instead of $readmemh
#                  BRAMs.  Adds MICROGPT_BFP_STREAM; also bakes the DDR3
#                  image (.bin) that the host has to upload before pulsing
#                  `start`.
USE_BFP    ?= 1
BFP_STREAM ?= 1
VIVADO_DEFINES :=
ifeq ($(USE_BFP),1)
  VIVADO_DEFINES += MICROGPT_USE_BFP
endif
ifeq ($(BFP_STREAM),1)
  VIVADO_DEFINES += MICROGPT_BFP_STREAM
endif
ifneq ($(strip $(VIVADO_DEFINES)),)
  export VIVADO_DEFINES
endif

# Synthesis-time hex deps for the BFP path.
#
# After the dual-port BRAM rework (gammas / norm_w / prompt now host-
# loaded via 0x060/0x061) and STREAM_LOOKUP=1 (embed table streamed
# from DDR3), the active BFP+STREAM path no longer `$readmemh`s any
# per-model data.  Only the cfg.svh defines (D, NL, FFN, VOCAB,
# NPROMPT_MAX, ...) and the architecture LUTs (silu_lut, exp_lut) are
# needed at synth.  The heavy hex bake (lbfp_full_stamp) is now a
# runtime artifact, captured by `make release-model`.
#
#   - lbfp_full_cfg_stamp  (gen_smollm_blockfp_cfg.py): emits
#       lbfp_full_cfg.svh + PROMPT_TOKENS.txt.  Loads tokenizer+config
#       only, ~2 s.
#   - lbfp_full_stamp  (gen_smollm_blockfp_full.py): full per-matrix
#       hex bake — needed by `release-model` and by the Verilator
#       autoregress regression, NOT by Vivado synth in the BFP+STREAM
#       path.
#   - lbfp_full_DDR3_stamp  (gen_smollm_blockfp_ddr.py): .bin DDR3
#       image.  Pure runtime artifact (uploaded by host) — not a synth
#       dep.  Listed under release-model.
lbfp_full_cfg_stamp  := $(GENERATED_DIR)/.lbfp_full_cfg.stamp
lbfp_full_stamp      := $(GENERATED_DIR)/.lbfp_full.stamp
lbfp_full_DDR3_stamp := $(GENERATED_DIR)/.lbfp_full_DDR3.stamp

# ----------------------------------------------------------------------
# VARIANT selects which SmolLM2 model the bitstream is dimensioned for.
# Same RTL, same LANES=16 engine, no throughput change — just bigger
# parameters baked into cfg.svh so the synth produces the right BRAM /
# state-buffer / KV-cache widths.
#
#   make                       (default: smollm135 = canonical 135M)
#   make VARIANT=smollm360     360M (D=960, NL=32, FFN=2560, ~870 MB DDR3)
#
# 1.7B would need >1 GB DDR3, so no smollm1700 variant — board change
# required there.
# ----------------------------------------------------------------------
# Build stamp — captured once at make-parse time, used to date the
# archive snapshot below.  Override BUILD_STAMP=manual or similar if
# you want a deterministic suffix for a specific session.
BUILD_STAMP ?= $(shell date -u +%Y%m%d-%H%M%S)

VARIANT ?= smollm360

## Per-VARIANT defaults.  MAX_CTX/NPROMPT_MAX/N_GEN use `?=` so a command
## line `make MAX_CTX=64 ...` still wins (command-line vars > makefile
## assignments, regardless of operator).  The smollm360 defaults are
## tuned to fit XC7VX485T's 1030 BRAM36 budget: at MAX_CTX=64 the design
## overflows by ~85 tiles (the KV cache is the dominant scaling consumer
## and is linear in MAX_CTX); 32 leaves comfortable margin.
ifeq ($(VARIANT),smollm135)
  SMOLLM_MODEL := HuggingFaceTB/SmolLM2-135M
  SMOLLM_D     := 576
  SMOLLM_H_Q   := 9
  SMOLLM_H_KV  := 3
  SMOLLM_HD    := 64
  SMOLLM_FFN   := 1536
  SMOLLM_NL    := 30
  MAX_CTX      ?= 64
  NPROMPT_MAX  ?= 48
  N_GEN        ?= 15
else ifeq ($(VARIANT),smollm360)
  SMOLLM_MODEL := HuggingFaceTB/SmolLM2-360M
  SMOLLM_D     := 960
  SMOLLM_H_Q   := 15
  SMOLLM_H_KV  := 5
  SMOLLM_HD    := 64
  SMOLLM_FFN   := 2560
  SMOLLM_NL    := 32
  MAX_CTX      ?= 32
  NPROMPT_MAX  ?= 24
  N_GEN        ?= 8
else
  $(error VARIANT=$(VARIANT) not recognised — use smollm135 or smollm360)
endif

## OUT pins the packer output dir to $(GENERATED_DIR) (= ../generated, i.e.
## rtl/generated/).  gen_smollm_blockfp_{cfg,full,ddr}.py default
## OUT='generated' relative to CWD = rtl/vc707/, which wrongly drops files
## in rtl/vc707/generated/ where Vivado's include-path can't find them.
##
## MAX_CTX / NPROMPT_MAX / N_GEN flow from the per-VARIANT block above.
## Without forwarding them through SMOLLM_ENV, `make MAX_CTX=32` etc.
## silently has no effect — make doesn't export command-line variables
## to recipe subshells by default.
SMOLLM_ENV := MODEL=$(SMOLLM_MODEL) D=$(SMOLLM_D) H_Q=$(SMOLLM_H_Q) \
              H_KV=$(SMOLLM_H_KV) HD=$(SMOLLM_HD) FFN=$(SMOLLM_FFN) \
              NL=$(SMOLLM_NL) OUT=$(GENERATED_DIR) \
              MAX_CTX=$(MAX_CTX) NPROMPT_MAX=$(NPROMPT_MAX) N_GEN=$(N_GEN)

## Target the real artifact (lbfp_full_cfg.svh) rather than a side-effect
## stamp.  A previous stamp-only rule could pass make's up-to-date check
## even when the .svh was deleted or never written, leaving Vivado to
## fail with "cannot open include file lbfp_full_cfg.svh".  Variant-marker
## ensures `make VARIANT=smollm360` after a smollm135 build regenerates.
lbfp_full_cfg_svh    := $(GENERATED_DIR)/lbfp_full_cfg.svh
## Marker name encodes every knob that changes cfg.svh — VARIANT plus
## MAX_CTX/NPROMPT_MAX/N_GEN — so changing any of them forces a regen.
## (Otherwise `make VARIANT=smollm360 MAX_CTX=32` after a prior smollm360
##  build would skip cfg regen and synth the wrong-sized KV cache.)
lbfp_full_cfg_marker := $(GENERATED_DIR)/.lbfp_full_cfg_$(VARIANT)_C$(MAX_CTX)_P$(NPROMPT_MAX)_G$(N_GEN).marker

$(lbfp_full_cfg_marker):
	@rm -f $(GENERATED_DIR)/.lbfp_full_cfg_*.marker
	@touch $@

$(lbfp_full_cfg_svh): host/gen_smollm_blockfp_cfg.py $(lbfp_full_cfg_marker)
	@echo "=== generating lbfp_full_cfg.svh + PROMPT_TOKENS.txt (VARIANT=$(VARIANT)) ==="
	$(SMOLLM_ENV) python3 host/gen_smollm_blockfp_cfg.py

BFP_BUILD_DEPS :=
ifeq ($(USE_BFP),1)
  BFP_BUILD_DEPS += $(lbfp_full_cfg_svh)
endif

# Date-stamped archive — each successful synth gets snapshotted into
# archive/<VARIANT>-<BUILD_STAMP>/ so historical builds aren't
# overwritten when the next `make` runs.  Vivado's project dir stays
# canonical (microgpt_eth.runs/) so incremental synth still works.
# A symlink `archive/latest` always points at the newest snapshot.
archive-dir := archive/$(VARIANT)-$(BUILD_STAMP)

# build_version.svh = git HEAD short hash baked into BUILD_VERSION (reg 0x10F)
# so the host can map the running bitstream to an exact commit.  Driven by
# FORCE so the clean-tree gate in host/gen_build_version.py runs on EVERY
# make (refusing to build with uncommitted Verilog; ALLOW_DIRTY=1 to override).
# The script is content-stable — it rewrites the file ONLY when the hash
# changes — so depending on it does not force a needless full re-synth.
.PHONY: FORCE
FORCE:

$(GENERATED_DIR)/build_version.svh: FORCE
	@python3 host/gen_build_version.py

$(bit): $(ips) $(tm_stamp) $(BROM_SV) $(GENERATED_DIR)/silu_lut.hex $(GENERATED_DIR)/exp_lut.hex $(lbfp_stamp) $(GENERATED_DIR)/build_version.svh scripts/run.tcl $(BFP_BUILD_DEPS)
	@# No auto-wipe of microgpt_eth.* here — prologue.tcl reopens the
	@# existing .xpr so synth_1/impl_1 reuse what they can (incremental
	@# synth/impl saves 30-70% on small RTL edits).  When switching
	@# VARIANT or after Vivado gets confused, run `make wipe-project`
	@# (or `make wipe-project all`) for a from-scratch rebuild.
	mkdir -p $(work-dir)
	$(VIVADO) $(VIVADOFLAGS) -source scripts/run.tcl
	cp microgpt_eth.runs/impl_1/vc707_microgpt_eth.bit ./$(work-dir)/ 2>/dev/null || \
	cp microgpt_eth.runs/impl_1/vc707_microgpt_eth*.bit ./$(work-dir)/ 2>/dev/null || true
	@# Snapshot bit + reports + cfg into a dated archive so this build
	@# survives the next `make`.  Cheap: a few MB of copies, ~1 s.
	@mkdir -p $(archive-dir)/reports
	@cp microgpt_eth.runs/impl_1/vc707_microgpt_eth.bit       $(archive-dir)/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/*timing_summary_routed.rpt   $(archive-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/*utilization*.rpt            $(archive-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/synth_1/*utilization_synth.rpt      $(archive-dir)/reports/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/lbfp_full_cfg.svh                    $(archive-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/build_version.svh                    $(archive-dir)/ 2>/dev/null || true
	@{ \
	    echo "build_date: $$(date -u --iso-8601=seconds)"; \
	    echo "variant:    $(VARIANT)"; \
	    echo "git_hash:   $$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	    echo "git_dirty:  $$(git diff --shortstat HEAD 2>/dev/null | head -1)"; \
	    grep -h '^`define LBFP_FULL_' $(GENERATED_DIR)/lbfp_full_cfg.svh 2>/dev/null; \
	} > $(archive-dir)/build_info.txt
	@ln -sfn $(VARIANT)-$(BUILD_STAMP) archive/latest
	@echo "=== snapshot in $(archive-dir)/ (and archive/latest) ==="

# ---------------------------------------------------------------------------
# PicoSoC bring-up shell (branch-local).  `make picosoc` builds the SoC boot
# ROM from src/soc/fw_shell.c then runs synth/impl with PICOSOC=1, which makes
# run.tcl select vc707_picosoc_shell (eth + DDR + dummy engine) instead of the
# inference engine.  NOTE: shares the microgpt_eth project dir — run
# `make wipe-project` when switching between the engine and the shell build.
# ---------------------------------------------------------------------------
RISCV_CROSS ?= riscv64-unknown-elf-
SOC_DIR     := src/soc

# Firmware build: C + asm -> .bin -> .mem.  The .mem both inits the progmem BRAM
# at synth (PROGMEM_HEX) and is the `updatemem -data` payload for fw-update.
$(SOC_DIR)/firmware_shell.mem: $(SOC_DIR)/fw_shell.c $(SOC_DIR)/start.s $(SOC_DIR)/soc.lds $(SOC_DIR)/bin2mem.py
	$(RISCV_CROSS)gcc -Os -march=rv32im -mabi=ilp32 -ffreestanding -nostdlib -nostartfiles \
	    -Wl,-T,$(SOC_DIR)/soc.lds -o $(SOC_DIR)/fw_shell.elf $(SOC_DIR)/start.s $(SOC_DIR)/fw_shell.c -lgcc
	$(RISCV_CROSS)objcopy -O binary $(SOC_DIR)/fw_shell.elf $(SOC_DIR)/fw_shell.bin
	python3 $(SOC_DIR)/bin2mem.py $(SOC_DIR)/fw_shell.bin > $@
	@echo "=== picosoc firmware: $$(stat -c%s $(SOC_DIR)/fw_shell.bin) bytes -> $@ ==="

# Stub firmware for the PicoSoC + real-engine fit build.
$(SOC_DIR)/firmware_engine.mem: $(SOC_DIR)/fw_engine.c $(SOC_DIR)/start.s $(SOC_DIR)/soc.lds $(SOC_DIR)/bin2mem.py
	$(RISCV_CROSS)gcc -Os -march=rv32im -mabi=ilp32 -ffreestanding -nostdlib -nostartfiles \
	    -Wl,-T,$(SOC_DIR)/soc.lds -o $(SOC_DIR)/fw_engine.elf $(SOC_DIR)/start.s $(SOC_DIR)/fw_engine.c -lgcc
	$(RISCV_CROSS)objcopy -O binary $(SOC_DIR)/fw_engine.elf $(SOC_DIR)/fw_engine.bin
	python3 $(SOC_DIR)/bin2mem.py $(SOC_DIR)/fw_engine.bin > $@
	@echo "=== picosoc-engine firmware: $$(stat -c%s $(SOC_DIR)/fw_engine.bin) bytes -> $@ ==="

# picosoc_ila IP — generated from its committed source tcl (same pattern as the
# engine ILAs); regenerated only when the tcl changes.  Defined here (before the
# picosoc-engine / picosoc targets that use it) so Make expands it at parse time.
# Integrated ILA default ON; PICOSOC_ILA=0 for a clean bring-up without it.
PICOSOC_ILA ?= 1
picosoc-ila-xci := $(ip-dir)/picosoc_ila/picosoc_ila.srcs/sources_1/ip/picosoc_ila/picosoc_ila.xci
$(picosoc-ila-xci): $(ip-dir)/picosoc_ila/tcl/run.tcl
	@echo "=== generating picosoc_ila IP ==="
	@cd $(ip-dir)/picosoc_ila && $(MAKE) clean && $(MAKE)

# Smoke firmware: minimal UART-only test, used by picosoc-engine-smoke below.
$(SOC_DIR)/firmware_smoke.mem: $(SOC_DIR)/fw_smoke.c $(SOC_DIR)/start.s $(SOC_DIR)/soc.lds $(SOC_DIR)/bin2mem.py
	$(RISCV_CROSS)gcc -Os -march=rv32im -mabi=ilp32 -ffreestanding -nostdlib -nostartfiles \
	    -Wl,-T,$(SOC_DIR)/soc.lds -o $(SOC_DIR)/fw_smoke.elf $(SOC_DIR)/start.s $(SOC_DIR)/fw_smoke.c -lgcc
	$(RISCV_CROSS)objcopy -O binary $(SOC_DIR)/fw_smoke.elf $(SOC_DIR)/fw_smoke.bin
	python3 $(SOC_DIR)/bin2mem.py $(SOC_DIR)/fw_smoke.bin > $@
	@echo "=== picosoc-engine smoke firmware: $$(stat -c%s $(SOC_DIR)/fw_smoke.bin) bytes -> $@ ==="

# Real engine + PicoSoC front-end.  Builds the firmware, then the engine
# bitstream ($(bit)) with PICOSOC_ENGINE exported so run.tcl swaps eth_ctrl ->
# picosoc_eth_bridge.  ILA on the SoC trace/iomem bus is ON by default
# (PICOSOC_ILA=0 to disable).  Baked firmware is selectable via ENGINE_FW
# (default the menu firmware; firmware_smoke = the Hello/echo smoke test).
# Source set / firmware changes -> wipe first:
#   make wipe-project picosoc-engine                      (menu firmware)
#   make wipe-project picosoc-engine ENGINE_FW=firmware_smoke   (Hello/echo)
ENGINE_FW ?= firmware_engine

# progmem INIT (src/soc/progmem_init.svh) is baked by run.tcl via bin2init.py from
# the firmware .bin just before synth, so it always matches the baked firmware and
# exists before the `include in progmem_bram.v.  The .mem prereq guarantees the
# .bin is present for run.tcl to read.
.PHONY: picosoc-engine
picosoc-engine: export PICOSOC_ENGINE := 1
picosoc-engine: export PICOSOC_ILA := $(PICOSOC_ILA)
picosoc-engine: export ENGINE_FW_MEM := $(abspath $(SOC_DIR)/$(ENGINE_FW).mem)
ifneq ($(PICOSOC_ILA),0)
picosoc-engine: $(picosoc-ila-xci)
endif
picosoc-engine: $(SOC_DIR)/$(ENGINE_FW).mem $(bit)
	@mkdir -p $(work-dir)
	@cp microgpt_eth.runs/impl_1/vc707_microgpt_eth.bit ./$(work-dir)/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/vc707_microgpt_eth_routed.dcp ./$(work-dir)/ 2>/dev/null || true
	@echo "=== picosoc-engine built: $(ENGINE_FW), ILA=$(PICOSOC_ILA), progmem=RAMB16 -> $(work-dir)/ ==="

# Smoke build: wipe + rebuild with the minimal Hello/echo firmware AND the ILA,
# so we get a clean proof-of-life and on-chip visibility in one bitstream.
# (updatemem can't patch the existing .bit — the progmem ROM is partly LUTROM,
# not BRAM — so a rebuild is the only way to change firmware.)
.PHONY: picosoc-engine-smoke
picosoc-engine-smoke:
	$(MAKE) wipe-project picosoc-engine ENGINE_FW=firmware_smoke

.PHONY: picosoc picosoc-fw fw-swap
picosoc-fw: $(SOC_DIR)/firmware_shell.mem

# Full shell build (synth/impl).  Integrated ILA is ON by default; disable with
# `make picosoc PICOSOC_ILA=0` (the shell's mark_debug nets still allow inserting
# an ILA via Vivado "Set Up Debug").
ifneq ($(PICOSOC_ILA),0)
picosoc: $(picosoc-ila-xci)
endif
picosoc: picosoc-fw
	mkdir -p $(work-dir)
	PICOSOC=1 PICOSOC_ILA=$(PICOSOC_ILA) $(VIVADO) $(VIVADOFLAGS) -source scripts/run.tcl
	@cp microgpt_eth.runs/impl_1/vc707_picosoc_shell.bit ./$(work-dir)/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/vc707_picosoc_shell_routed.dcp ./$(work-dir)/ 2>/dev/null || true
	@echo "=== picosoc shell built (top vc707_picosoc_shell, progmem=RAMB16) -> $(work-dir)/ ==="

# Firmware update WITHOUT re-synthesis — the ONLY reliable method.  Rewrites the
# directly-instantiated progmem RAMB INIT in the routed checkpoint and re-emits
# the bitstream (write_bitstream only, ~2-3 min, no place/route).
# DO NOT use updatemem here: with a hand-built .mmi it is a SILENT NO-OP (reports
# success, patches nothing — the .bit differs by ~4 header bytes and the FPGA runs
# the OLD firmware).  write_mem_info can't make a real .mmi (no PicoRV32 address
# space) and data2mem is retired.
#
# Needs the routed .dcp.  Builds archive it to work/ (shell & engine share
# impl_1/, so building one overwrites the other's dcp there).  Pick the top:
#   make fw-swap SWAP_FW=firmware_smoke                                 # shell
#   make fw-swap SWAP_TOP=vc707_microgpt_eth SWAP_FW=firmware_engine    # engine
SWAP_TOP     ?= vc707_picosoc_shell
SWAP_FW      ?= firmware_smoke
SWAP_DCP     ?= $(firstword $(wildcard $(work-dir)/$(SWAP_TOP)_routed.dcp microgpt_eth.runs/impl_1/$(SWAP_TOP)_routed.dcp))
SWAP_BIT_OUT ?= $(work-dir)/$(SWAP_TOP)_$(SWAP_FW:firmware_%=%).bit
fw-swap: $(SOC_DIR)/$(SWAP_FW).mem
	@test -n "$(SWAP_DCP)" && test -f "$(SWAP_DCP)" || { echo "ERROR: routed .dcp for $(SWAP_TOP) not found — (re)build it first"; exit 1; }
	@mkdir -p $(work-dir)
	DCP_IN=$(SWAP_DCP) FW_BIN=$(SOC_DIR)/$(SWAP_FW:firmware_%=fw_%).bin \
	  BIT_OUT=$(abspath $(SWAP_BIT_OUT)) \
	  $(VIVADO) -nojournal -mode batch -source scripts/swap_fw.tcl
	@echo "=== firmware swapped ($(SWAP_TOP) <- $(SWAP_FW)) -> $(SWAP_BIT_OUT) (reflash) ==="

# Fast JTAG download via openFPGALoader (SRAM, volatile).  ~12 s.
#   * Needs the SOURCE build of openFPGALoader — the packaged 0.12.0 lacks the
#     xc7vx485t IDCODE.  Set OFL=, or `sudo make install` in its build dir.
#   * 15 MHz is the safe rate; 30 MHz garbles the JTAG IDCODE on this board
#     ("Unknown device 0x67d0e027"), so OLOAD_FREQ is fixed low here.
#   * Only works on MATCH_CYCLE-NoWait bitstreams (picosoc.xdc) — otherwise the
#     design stalls on DCI match and never starts.
OFL        ?= $(HOME)/openFPGALoader/build/openFPGALoader
OLOAD_FREQ ?= 15000000
OLOAD_BIT  ?= work/vc707_picosoc_shell_jtag.bit
.PHONY: oload
oload:
	@test -f $(OLOAD_BIT) || { echo "ERROR: $(OLOAD_BIT) missing"; exit 1; }
	$(OFL) -c digilent --freq $(OLOAD_FREQ) $(OLOAD_BIT)

# Quick convenience target: bake BFP hex + run the Verilator layer test.
# Useful for sanity-checking precision after RTL edits before launching a
# full Vivado synth (≈45 min on VC707).
bfp-sim: $(lbfp_stamp)
	$(MAKE) -C sim bfp-layer

# Full-scale: bake real SmolLM2-135M weights + run autoregressive token
# loop in Verilator.  Pulls weights from HuggingFace (one-time download,
# ~270 MB).  Bake-and-run takes ~10 min wall, then compares 18 generated
# tokens against the Python BFP golden.
$(lbfp_full_stamp): host/gen_smollm_blockfp_full.py
	@echo "=== baking $(SMOLLM_MODEL) block-FP weights ($(SMOLLM_NL) layers, D=$(SMOLLM_D)) ==="
	$(SMOLLM_ENV) python3 host/gen_smollm_blockfp_full.py
	@touch $@

bfp-full: $(lbfp_full_stamp)
	$(MAKE) -C sim bfp-full

# Self-driven on-chip autoregress: same model, but the token-step loop is
# inside the RTL (autoregress_bfp_top.sv).  Pulses `go`, runs all 19 token
# steps internally, asserts `done`, exposes generated tokens via a packed
# bus.  Demonstrates the FPGA-target flow end-to-end in Verilator.
bfp-autoregress: $(lbfp_full_stamp)
	$(MAKE) -C sim bfp-autoregress

# Bake the DDR3 image (.bin) consumed by the FPGA streaming path AND
# (optionally) the matching .hex consumed by the Verilator stream tb.
# The .bin alone is ≈ 320 MB, the .hex doubles that — keep it gated.
$(lbfp_full_DDR3_stamp): host/gen_smollm_blockfp_ddr.py $(lbfp_full_stamp)
	@echo "=== baking lbfp_full_DDR3.bin ($(SMOLLM_MODEL)) ==="
	$(SMOLLM_ENV) python3 host/gen_smollm_blockfp_ddr.py
	@touch $@

# Same end-to-end demo but with weights streamed from a DDR3 image via
# mock_axi_slave (STREAM_WEIGHTS=1 / STREAM_LOOKUP=1).  Validates the
# AXI fetch + streamer pipeline against the BRAM-mode golden tokens.
# Requires the (~670 MB) hex image — generated with LBFP_EMIT_HEX=1.
bfp-autoregress-stream: $(lbfp_full_stamp)
	$(SMOLLM_ENV) LBFP_EMIT_HEX=1 python3 host/gen_smollm_blockfp_ddr.py
	$(MAKE) -C sim bfp-autoregress-stream

.PHONY: bfp-sim bfp-full bfp-autoregress bfp-autoregress-stream

$(ips): %.xci :
	mkdir -p $(work-dir)
	@echo Generating $(@F)
	@cd $(ip-dir)/$(basename $(@F)) && make clean && make
	@cp $(ip-dir)/$(basename $(@F))/$(basename $(@F)).srcs/sources_1/ip/$(basename $(@F))/$(@F) $@

# ----------------------------------------------------------------------
# program / flash / mcs — capture-style.  Never re-trigger a full Vivado
# synth: if a bitstream exists anywhere reasonable, use it directly.
# The previous `flash: $(mcs)` chain walked the synthesis dep graph all
# the way back, so a stale work/ dir or a touched .hex dep would launch
# a 25-min rebuild on what should be a 2-min flash.
# ----------------------------------------------------------------------
# Resolved at make-parse time — the freshest existing bit/mcs, if any.
# Search order: live Vivado runs/ first (just-finished build), then work/
# copy, then archive/latest (most recent successful build, surviving
# `make clean`), then arbitrary archive dirs.
EXISTING_BIT := $(firstword $(wildcard \
    microgpt_eth.runs/impl_1/vc707_microgpt_eth.bit \
    $(work-dir)/vc707_microgpt_eth.bit \
    archive/latest/vc707_microgpt_eth.bit))
EXISTING_MCS := $(firstword $(wildcard \
    microgpt_eth.runs/impl_1/vc707_microgpt_eth.mcs \
    $(work-dir)/vc707_microgpt_eth.mcs \
    archive/latest/vc707_microgpt_eth.mcs))

# Re-bake the BPI image from the current bit.  Falls back to a full
# Vivado synth only if no bitstream exists anywhere.
mcs:
	@if [ -n "$(EXISTING_BIT)" ]; then \
	    echo "[mcs] from $(EXISTING_BIT)"; \
	    $(VIVADO) $(VIVADOFLAGS) -source scripts/write_cfgmem.tcl \
	        -tclargs $(EXISTING_BIT:.bit=.mcs) $(EXISTING_BIT); \
	else \
	    echo "[mcs] no bitstream found — running full build"; \
	    $(MAKE) $(bit); \
	    $(VIVADO) $(VIVADOFLAGS) -source scripts/write_cfgmem.tcl \
	        -tclargs $(mcs) $(bit); \
	fi

# JTAG-load (volatile SRAM) the freshest bit via openFPGALoader — seconds, no
# Vivado hw_server.  Needs the openFPGALoader SOURCE build (OFL; packaged 0.12.0
# lacks xc7vx485t) and a MATCH_CYCLE-NoWait bit (now default via microgpt_eth.xdc;
# rebuild any pre-NoWait bit or it'll stall on DCI match).  15 MHz — 30 MHz garbles
# the IDCODE.  Falls back: `make program-vivado` for the old hw_server path.
program:
	@if [ -z "$(EXISTING_BIT)" ]; then \
	    echo "ERROR: no bitstream found in microgpt_eth.runs/ or $(work-dir)/"; \
	    echo "       Run 'make all' first, or 'make program-release' for release/."; \
	    exit 1; \
	fi
	@echo "[program] $(EXISTING_BIT) via openFPGALoader @ $(OLOAD_FREQ)"
	$(OFL) -c digilent --freq $(OLOAD_FREQ) $(EXISTING_BIT)

# Old Vivado hw_server path — fallback (e.g. a pre-NoWait bit, or openFPGALoader
# trouble).  Slower (Vivado startup) but handles the DCI-match startup itself.
program-vivado:
	@if [ -z "$(EXISTING_BIT)" ]; then \
	    echo "ERROR: no bitstream found in microgpt_eth.runs/ or $(work-dir)/"; \
	    exit 1; \
	fi
	@echo "[program-vivado] from $(EXISTING_BIT)"
	BITFILE=$(EXISTING_BIT) $(VIVADO) $(VIVADOFLAGS) -source scripts/program.tcl
	@# Vivado's cs_server/hw_server sometimes survive and idle-poll; reap them.
	@pkill -f 'cs_server|hw_server' 2>/dev/null || true

# Write BPI flash from the freshest mcs.  Bakes mcs on the fly from
# an existing bit if needed; never re-runs Vivado synth.
flash:
	@set -e; \
	if [ -n "$(EXISTING_MCS)" ]; then \
	    mcs_path="$(EXISTING_MCS)"; \
	elif [ -n "$(EXISTING_BIT)" ]; then \
	    mcs_path="$(EXISTING_BIT:.bit=.mcs)"; \
	    echo "[flash] generating $$mcs_path from $(EXISTING_BIT)"; \
	    $(VIVADO) $(VIVADOFLAGS) -source scripts/write_cfgmem.tcl \
	        -tclargs $$mcs_path $(EXISTING_BIT); \
	else \
	    echo "ERROR: no bitstream found in microgpt_eth.runs/ or $(work-dir)/"; \
	    echo "       Run 'make all' first, or 'make flash-release' for release/."; \
	    exit 1; \
	fi; \
	echo "[flash] writing $$mcs_path to BPI flash"; \
	MCSFILE=$$mcs_path $(VIVADO) $(VIVADOFLAGS) -source scripts/flash.tcl
	@# Same reap as `program:` — Vivado's cs_server/hw_server idle-poll
	@# at high CPU after JTAG flash if not killed.
	@pkill -f 'cs_server|hw_server' 2>/dev/null || true

# ----------------------------------------------------------------------
# Release: capture the current build into release/ so it survives
# `make clean` and can be re-programmed / flashed / demo'd later
# without rebuilding.  Stores the bitstream, the .mcs (BPI image),
# the DDR3 weight image, the baked vocab, the Vivado reports, and a
# small build_info.txt with build_version + git hash + dims.
#
# Targets:
#   make release         capture the current build into release/
#   make program-release JTAG-program release/*.bit  (DDR3 contents lost)
#   make flash-release   write release/*.mcs to BPI flash (persists)
#   make demo            build + run bfp_client against release/ artifacts
# ----------------------------------------------------------------------
release-dir := release
release-bit := $(release-dir)/vc707_microgpt_eth.bit
release-mcs := $(release-dir)/vc707_microgpt_eth.mcs
release-ddr := $(release-dir)/lbfp_full_DDR3.bin
release-voc := $(release-dir)/bfp_vocab.bin

# Capture a release snapshot from whatever is currently built — never
# triggers a fresh Vivado synth or hex/DDR3 bake.  Resolves the source
# bitstream from microgpt_eth.runs/ → work/ → release/bitstream/ at
# parse time (same EXISTING_BIT used by flash/program).  If nothing
# exists, fails with explicit instructions instead of silently kicking
# off a 25-min synth (the old behaviour).
release: host/bfp_vocab.bin
	@set -e; \
	if [ -z "$(EXISTING_BIT)" ]; then \
	    echo "ERROR: no bitstream found.  Run 'make all' first."; exit 1; \
	fi; \
	if [ ! -f "$(GENERATED_DIR)/lbfp_full_DDR3.bin" ]; then \
	    echo "ERROR: $(GENERATED_DIR)/lbfp_full_DDR3.bin missing."; \
	    echo "       Bake first with: python host/gen_smollm_blockfp_ddr.py"; \
	    exit 1; \
	fi
	@mkdir -p $(release-dir)/reports
	@cp $(EXISTING_BIT)                     $(release-bit)
	@if [ -n "$(EXISTING_MCS)" ]; then \
	    cp $(EXISTING_MCS) $(release-mcs); \
	else \
	    echo "[release] generating $(release-mcs) from $(EXISTING_BIT)"; \
	    $(VIVADO) $(VIVADOFLAGS) -source scripts/write_cfgmem.tcl \
	        -tclargs $(release-mcs) $(EXISTING_BIT); \
	fi
	@cp $(GENERATED_DIR)/lbfp_full_DDR3.bin $(release-ddr)
	@cp host/bfp_vocab.bin                  $(release-voc)
	@cp $(GENERATED_DIR)/lbfp_full_cfg.svh             $(release-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/lbfp_full_PROMPT_TOKENS.txt   $(release-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/lbfp_full_GOLDEN_TOKENS.txt   $(release-dir)/ 2>/dev/null || true
	@# Per-model BRAM .hex files for the host-loaded dual-port BRAMs
	@# (gammas, final norm, prompt) — `make run` loads these via 0x060/0x061.
	@cp $(GENERATED_DIR)/lbfp_full_G1_m.hex     $(release-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/lbfp_full_G1_e.hex     $(release-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/lbfp_full_G2_m.hex     $(release-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/lbfp_full_G2_e.hex     $(release-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/lbfp_full_NORM_W_m.hex $(release-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/lbfp_full_NORM_W_e.hex $(release-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/lbfp_full_PROMPT.hex   $(release-dir)/ 2>/dev/null || true
	@# Vivado reports — pull from both reports/ (run.tcl-emitted) and
	@# microgpt_eth.runs/impl_1 (per-step reports).  Missing files are OK.
	@cp -r reports/.                                          $(release-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/*timing_summary_routed.rpt   $(release-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/*utilization*.rpt            $(release-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/*drc_routed.rpt              $(release-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/*methodology_drc_routed.rpt  $(release-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/synth_1/*utilization_synth.rpt      $(release-dir)/reports/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/build_version.svh                    $(release-dir)/ 2>/dev/null || true
	@{ \
	    echo "build_date: $$(date -u --iso-8601=seconds)"; \
	    echo "git_hash:   $$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	    echo "git_status: $$(git diff --shortstat HEAD 2>/dev/null || echo unknown)"; \
	    echo "host:       $$(hostname)"; \
	    grep -h '^`define LBFP_FULL_' $(GENERATED_DIR)/lbfp_full_cfg.svh 2>/dev/null; \
	    grep -h 'BUILD_VERSION' $(GENERATED_DIR)/build_version.svh 2>/dev/null; \
	    echo "bitstream:  $$(du -h $(release-bit) | cut -f1)"; \
	    echo "mcs:        $$(du -h $(release-mcs) | cut -f1)"; \
	    echo "ddr3 image: $$(du -h $(release-ddr) | cut -f1)"; \
	} > $(release-dir)/build_info.txt
	@echo "=== release captured into $(release-dir)/ ==="
	@cat $(release-dir)/build_info.txt

program-release: $(release-bit)
	BITFILE=$(release-bit) $(VIVADO) $(VIVADOFLAGS) -source scripts/program.tcl
	@pkill -f 'cs_server|hw_server' 2>/dev/null || true

flash-release: $(release-mcs)
	MCSFILE=$(release-mcs) $(VIVADO) $(VIVADOFLAGS) -source scripts/flash.tcl
	@pkill -f 'cs_server|hw_server' 2>/dev/null || true

# ----------------------------------------------------------------------
# release-bitstream + release-model — split the legacy `make release`
# into the two artifact lifecycles it conflates:
#
#   release/bitstream/             one per Vivado build (model-indep.)
#     .bit, .mcs, bfp_vocab.bin, reports/, build_info.txt
#
#   release/models/<PREFIX>/       one per fine-tune (bitstream-indep.)
#     <PREFIX>DDR3.bin, <PREFIX>cfg.svh, <PREFIX>{G1,G2,NORM_W}_*.hex,
#     <PREFIX>PROMPT.hex, <PREFIX>{PROMPT,GOLDEN}_TOKENS.txt,
#     model_info.txt
#
# Why split: a single bitstream serves many fine-tuned models (the
# per-model BRAMs are host-loaded at runtime), and a single fine-tune
# can target many bitstream revisions.  The legacy `make release`
# target still works (writes flat into release/) but is layered on
# top of these for backward compatibility.
#
# Default PREFIX is lbfp_full_ (matches the rest of the build).  For a
# Shakespeare or other fine-tune variant:
#   make release-model PREFIX=shake_
# ----------------------------------------------------------------------
PREFIX ?= lbfp_full_

release-bs-dir := release/bitstream
release-bs-bit := $(release-bs-dir)/vc707_microgpt_eth.bit
release-bs-mcs := $(release-bs-dir)/vc707_microgpt_eth.mcs
release-bs-voc := $(release-bs-dir)/bfp_vocab.bin

release-bitstream: host/bfp_vocab.bin
	@if [ ! -f "$(bit)" ] || [ ! -f "$(mcs)" ]; then \
	    echo "ERROR: $(bit) or $(mcs) missing — release-bitstream is a"; \
	    echo "       capture-only target.  Run 'make' / 'make mcs' first,"; \
	    echo "       or use 'make release' (legacy) for build-and-capture."; \
	    exit 1; \
	fi
	@mkdir -p $(release-bs-dir)/reports
	@cp $(bit)             $(release-bs-bit)
	@cp $(mcs)             $(release-bs-mcs)
	@cp host/bfp_vocab.bin $(release-bs-voc)
	@cp -r reports/.                                          $(release-bs-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/*timing_summary_routed.rpt   $(release-bs-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/*utilization*.rpt            $(release-bs-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/*drc_routed.rpt              $(release-bs-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/impl_1/*methodology_drc_routed.rpt  $(release-bs-dir)/reports/ 2>/dev/null || true
	@cp microgpt_eth.runs/synth_1/*utilization_synth.rpt      $(release-bs-dir)/reports/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/build_version.svh                    $(release-bs-dir)/ 2>/dev/null || true
	@{ \
	    echo "build_date: $$(date -u --iso-8601=seconds)"; \
	    echo "git_hash:   $$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	    echo "git_status: $$(git diff --shortstat HEAD 2>/dev/null || echo unknown)"; \
	    echo "host:       $$(hostname)"; \
	    grep -h 'BUILD_VERSION' $(GENERATED_DIR)/build_version.svh 2>/dev/null; \
	    echo "bitstream:  $$(du -h $(release-bs-bit) | cut -f1)"; \
	    echo "mcs:        $$(du -h $(release-bs-mcs) | cut -f1)"; \
	} > $(release-bs-dir)/build_info.txt
	@echo "=== bitstream release captured into $(release-bs-dir)/ ==="
	@cat $(release-bs-dir)/build_info.txt

release-model-dir := release/models/$(PREFIX)
release-model-ddr := $(release-model-dir)/$(PREFIX)DDR3.bin

release-model:
	@if [ ! -f "$(GENERATED_DIR)/$(PREFIX)DDR3.bin" ] || [ ! -f "$(GENERATED_DIR)/$(PREFIX)G1_m.hex" ]; then \
	    echo "ERROR: $(GENERATED_DIR)/$(PREFIX)DDR3.bin or .hex set missing"; \
	    echo "       — capture-only target.  Bake first with:"; \
	    echo "         PREFIX=$(PREFIX) python host/gen_smollm_blockfp_full.py"; \
	    echo "         PREFIX=$(PREFIX) python host/gen_smollm_blockfp_ddr.py"; \
	    echo "       (the .hex files were previously baked as a Vivado synth"; \
	    echo "       side-effect; after the dual-port BRAM rework they're"; \
	    echo "       runtime-only artifacts, so make release-model is the"; \
	    echo "       trigger to bake them now.)"; \
	    exit 1; \
	fi
	@mkdir -p $(release-model-dir)
	@cp $(GENERATED_DIR)/$(PREFIX)DDR3.bin              $(release-model-ddr)
	@cp $(GENERATED_DIR)/$(PREFIX)cfg.svh               $(release-model-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/$(PREFIX)PROMPT_TOKENS.txt     $(release-model-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/$(PREFIX)GOLDEN_TOKENS.txt     $(release-model-dir)/ 2>/dev/null || true
	@# Per-model BRAM .hex files (gammas, final norm, prompt) — host
	@# loads these into BRAMs at runtime via load-roms.
	@cp $(GENERATED_DIR)/$(PREFIX)G1_m.hex     $(release-model-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/$(PREFIX)G1_e.hex     $(release-model-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/$(PREFIX)G2_m.hex     $(release-model-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/$(PREFIX)G2_e.hex     $(release-model-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/$(PREFIX)NORM_W_m.hex $(release-model-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/$(PREFIX)NORM_W_e.hex $(release-model-dir)/ 2>/dev/null || true
	@cp $(GENERATED_DIR)/$(PREFIX)PROMPT.hex   $(release-model-dir)/ 2>/dev/null || true
	@{ \
	    echo "build_date: $$(date -u --iso-8601=seconds)"; \
	    echo "prefix:     $(PREFIX)"; \
	    echo "git_hash:   $$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	    grep -h '^`define LBFP_FULL_' $(GENERATED_DIR)/$(PREFIX)cfg.svh 2>/dev/null; \
	    echo "ddr3 image: $$(du -h $(release-model-ddr) | cut -f1)"; \
	} > $(release-model-dir)/model_info.txt
	@echo "=== model release ($(PREFIX)) captured into $(release-model-dir)/ ==="
	@cat $(release-model-dir)/model_info.txt

# Convenience: program/flash from the split bitstream dir.
program-release-bitstream: $(release-bs-bit)
	BITFILE=$(release-bs-bit) $(VIVADO) $(VIVADOFLAGS) -source scripts/program.tcl
	@pkill -f 'cs_server|hw_server' 2>/dev/null || true

flash-release-bitstream: $(release-bs-mcs)
	MCSFILE=$(release-bs-mcs) $(VIVADO) $(VIVADOFLAGS) -source scripts/flash.tcl
	@pkill -f 'cs_server|hw_server' 2>/dev/null || true

# ----------------------------------------------------------------------
# Shakespeare fine-tune flow — convenience wrappers around the generic
# PREFIX-parameterised targets, so the "swap-a-fine-tuned-model" headline
# feature is one make invocation per stage instead of a wall of
# python+env-var commands.
#
#   make shakespeare-train     fine-tune SmolLM2-135M on tinyshakespeare
#                              (~10-30 min on a single GPU; one-time)
#   make shakespeare-pack      bake .hex set + DDR3.bin from the
#                              checkpoint (cfg + full + ddr packers)
#   make shakespeare-release   capture release/models/shake_/
#   make shakespeare-run       upload + run + decode on the FPGA
#   make shakespeare           pack + release + run (skip train; assumes
#                              checkpoint already exists)
#
# Override SHAKES_PROMPT to change the demo prompt without re-training.
# ----------------------------------------------------------------------
SHAKES_PREFIX := shake_
SHAKES_MODEL  := $(GENERATED_DIR)/shakespeare-smollm
SHAKES_PROMPT ?= To be, or not to be,
SHAKES_NGEN   ?= 32

shakespeare-train:
	@echo "=== fine-tuning SmolLM2-135M → $(SHAKES_MODEL) ==="
	python3 host/finetune_shakespeare.py --out $(SHAKES_MODEL)

shakespeare-pack:
	@if [ ! -f "$(SHAKES_MODEL)/config.json" ]; then \
	    echo "ERROR: $(SHAKES_MODEL)/config.json missing."; \
	    echo "       Run 'make shakespeare-train' first (one-time, ~10-30 min on GPU)."; \
	    exit 1; \
	fi
	@echo "=== packing shake_ artifacts from $(SHAKES_MODEL) (PROMPT=$(SHAKES_PROMPT)) ==="
	MODEL=$(SHAKES_MODEL) PREFIX=$(SHAKES_PREFIX) OUT=$(GENERATED_DIR) \
	  PROMPT="$(SHAKES_PROMPT)" N_GEN=$(SHAKES_NGEN) \
	  python3 host/gen_smollm_blockfp_cfg.py
	MODEL=$(SHAKES_MODEL) PREFIX=$(SHAKES_PREFIX) OUT=$(GENERATED_DIR) \
	  PROMPT="$(SHAKES_PROMPT)" N_GEN=$(SHAKES_NGEN) \
	  python3 host/gen_smollm_blockfp_full.py
	MODEL=$(SHAKES_MODEL) PREFIX=$(SHAKES_PREFIX) OUT=$(GENERATED_DIR) \
	  PROMPT="$(SHAKES_PROMPT)" N_GEN=$(SHAKES_NGEN) \
	  python3 host/gen_smollm_blockfp_ddr.py
	@echo "=== shake_ artifacts in $(GENERATED_DIR)/$(SHAKES_PREFIX)* ==="

shakespeare-release:
	$(MAKE) release-model PREFIX=$(SHAKES_PREFIX)

shakespeare-run:
	$(MAKE) run PREFIX=$(SHAKES_PREFIX)

shakespeare: shakespeare-pack shakespeare-release shakespeare-run

# Full demo against the split layout — uploads the model's DDR3 image,
# loads its BRAMs through the host-write path, then runs inference.
# Use PREFIX= to pick the model; bitstream comes from release/bitstream/.
demo-model: host/bfp_client
	MGRT_PREFIX=$(PREFIX) ./host/bfp_client load-roms release/models/$(PREFIX)
	MGRT_PREFIX=$(PREFIX) ./host/bfp_client -v $(release-bs-voc) \
	    all release/models/$(PREFIX)/$(PREFIX)DDR3.bin

# ----------------------------------------------------------------------
# make run — single command: existing release → decoded text.
#
# Auto-detects layout (prefers split release/bitstream + release/models/
# over legacy flat release/), conditionally loads BRAMs (only if .hex
# files are present in the layout — older bitstreams use $readmemh and
# need no load), then uploads DDR3 + restarts + reads + decodes.
#
#   make run                    # default model (lbfp_full_)
#   make run PREFIX=shake_      # any captured fine-tune
#
# Assumes the FPGA already has the matching bitstream loaded.  For a
# freshly power-cycled board:
#   make program-release-bitstream   # split layout, or
#   make program-release             # legacy flat
# then `make run`.
# ----------------------------------------------------------------------
run: host/bfp_client
	@set -e; \
	if [ -d release/bitstream ] && [ -d release/models/$(PREFIX) ]; then \
	    echo "[run] split layout — release/bitstream + release/models/$(PREFIX)/"; \
	    if ls release/models/$(PREFIX)/$(PREFIX)G1_m.hex >/dev/null 2>&1; then \
	        MGRT_PREFIX=$(PREFIX) ./host/bfp_client load-roms release/models/$(PREFIX); \
	    fi; \
	    ./host/bfp_client -v release/bitstream/bfp_vocab.bin \
	        all release/models/$(PREFIX)/$(PREFIX)DDR3.bin; \
	elif [ -f release/$(PREFIX)DDR3.bin ] && [ -f release/bfp_vocab.bin ]; then \
	    echo "[run] legacy flat layout — release/"; \
	    if ls release/$(PREFIX)G1_m.hex >/dev/null 2>&1; then \
	        MGRT_PREFIX=$(PREFIX) ./host/bfp_client load-roms release; \
	    fi; \
	    ./host/bfp_client -v release/bfp_vocab.bin all release/$(PREFIX)DDR3.bin; \
	else \
	    echo "ERROR: no release artifacts found for PREFIX=$(PREFIX)."; \
	    echo "       Expected one of:"; \
	    echo "         release/bitstream/bfp_vocab.bin + release/models/$(PREFIX)/$(PREFIX)DDR3.bin (split)"; \
	    echo "         release/bfp_vocab.bin + release/$(PREFIX)DDR3.bin (legacy)"; \
	    echo "       Capture a release first (see 'release-bitstream' / 'release-model' / legacy 'release')."; \
	    exit 1; \
	fi

# Single demo target: builds host/bfp_client (no-torch C++ client),
# then runs the full upload → restart → tokens → vocab-decode pipeline
# against the release-area artifacts.  Defaults are conservative
# (-L 1024 -P 20 = ~3.3 MB/s, verified byte-perfect).  Assumes the
# bitstream is already on the FPGA (via program-release / flash-release
# + power-cycle).  DDR3 contents are lost on any JTAG reprogram, so the
# demo always re-uploads the weight image.
host/bfp_client: host/bfp_client.cpp
	g++ -std=c++17 -O2 -Wall -Wextra -pthread -o $@ $<

demo: host/bfp_client $(release-ddr) $(release-voc)
	./host/bfp_client -v $(release-voc) all $(release-ddr)

# ----------------------------------------------------------------------
# make smoke-test — end-to-end baseline verification against a published
# GitHub release.  Pulls the 4 large binary artifacts (bit/mcs/DDR3/vocab)
# from the tagged release into release/ (only if absent — rm to force
# re-fetch), JTAG-programs the FPGA, runs inference, and diffs the
# RTL_TOKENS line against a known-good golden baked into this Makefile.
# Exits non-zero on mismatch.
#
# Per-model BRAM .hex files (G1, G2, NORM_W, PROMPT) are NOT in the
# GitHub release — they're tracked in-repo under release/ and must
# match the release tag.  If smoke-test fails on a non-tag commit,
# `git checkout $(SMOKE_RELEASE_TAG) -- release/*.hex` may be needed.
#
# Default tag: v0.2.0 (the canonical SmolLM2-135M baseline whose RTL
# token output is published in the top-level README).  Override with
# SMOKE_RELEASE_TAG=... AND a matching SMOKE_EXPECTED_TOKENS_<tag>
# entry (or pass SMOKE_EXPECTED_TOKENS=... explicitly).
# ----------------------------------------------------------------------
SMOKE_RELEASE_TAG ?= v0.2.0
SMOKE_BASE_URL    := https://github.com/jrrk2/tinyllm-fpga/releases/download/$(SMOKE_RELEASE_TAG)
SMOKE_ARTIFACTS   := vc707_microgpt_eth.bit vc707_microgpt_eth.mcs \
                     lbfp_full_DDR3.bin bfp_vocab.bin

# Per-model BRAM .hex files that load-roms uploads at runtime.  These
# are tracked in git (not in the GitHub-release tarball), so the smoke
# test pulls them from the tag's tree via `git show` into a dedicated
# per-tag dir.  This makes the test deterministic regardless of any
# local edits to release/*.hex (e.g. a modified PROMPT.hex for testing
# a different prompt would otherwise silently corrupt the baseline run).
SMOKE_HEX_FILES   := lbfp_full_G1_m.hex     lbfp_full_G1_e.hex \
                     lbfp_full_G2_m.hex     lbfp_full_G2_e.hex \
                     lbfp_full_NORM_W_m.hex lbfp_full_NORM_W_e.hex \
                     lbfp_full_PROMPT.hex
SMOKE_DIR         := release-smoke/$(SMOKE_RELEASE_TAG)

# Known-good 19-token output for the v0.2.0 baseline (SmolLM2-135M
# canonical, default prompt baked into release/lbfp_full_PROMPT.hex).
# Source: top-level README.md Quick start section.
SMOKE_EXPECTED_TOKENS_v0.2.0 := 712 9612 3102 645 260 905 436 441 2408 281 624 198 198 18 504 905 436 441 805
SMOKE_EXPECTED_TOKENS        ?= $(SMOKE_EXPECTED_TOKENS_$(SMOKE_RELEASE_TAG))

smoke-test: host/bfp_client
	@if [ -z "$(strip $(SMOKE_EXPECTED_TOKENS))" ]; then \
	    echo "ERROR: no golden tokens recorded for SMOKE_RELEASE_TAG=$(SMOKE_RELEASE_TAG)."; \
	    echo "       Add SMOKE_EXPECTED_TOKENS_$(SMOKE_RELEASE_TAG) := ... to Makefile,"; \
	    echo "       or pass SMOKE_EXPECTED_TOKENS=\"<tokens>\" on the command line."; \
	    exit 1; \
	fi
	@echo "=== smoke-test against GitHub release $(SMOKE_RELEASE_TAG) ==="
	@mkdir -p release $(SMOKE_DIR)
	@# Step 1: download the 4 large binaries into release/ (cached across
	@# runs).  Symlink them into the per-tag dir so bfp_client reads
	@# everything from one place.
	@for f in $(SMOKE_ARTIFACTS); do \
	    if [ ! -s release/$$f ]; then \
	        echo "[fetch] release/$$f"; \
	        curl -fsSL -o release/$$f $(SMOKE_BASE_URL)/$$f || \
	            { echo "ERROR: failed to fetch $$f from $(SMOKE_BASE_URL)"; exit 1; }; \
	    else \
	        echo "[skip]  release/$$f already present"; \
	    fi; \
	    ln -sf ../../release/$$f $(SMOKE_DIR)/$$f; \
	done
	@# Step 2: materialize the .hex set from the tagged revision (not the
	@# working tree) into the per-tag dir.  Works even if the working
	@# tree's release/*.hex are dirty or absent.
	@echo "[git ] materializing $(SMOKE_RELEASE_TAG) .hex files into $(SMOKE_DIR)/"
	@for f in $(SMOKE_HEX_FILES); do \
	    git show $(SMOKE_RELEASE_TAG):rtl/vc707/release/$$f > $(SMOKE_DIR)/$$f 2>/dev/null || \
	        { echo "ERROR: $$f not present at tag $(SMOKE_RELEASE_TAG)"; exit 1; }; \
	done
	@echo "=== programming FPGA from $(SMOKE_DIR)/vc707_microgpt_eth.bit ==="
	@# `-` prefix: ignore exit code.  scripts/program.tcl reliably loads
	@# the bitstream but Vivado often returns non-zero on cs_server /
	@# hw_server teardown even after a successful JTAG program; without
	@# this prefix smoke-test would FAIL on every run despite a good load.
	-BITFILE=$(SMOKE_DIR)/vc707_microgpt_eth.bit $(VIVADO) $(VIVADOFLAGS) -source scripts/program.tcl
	@# Bracketed first char dodges pkill's self-match: regex still resolves
	@# to cs_server/hw_server against real proc cmdlines, but the LITERAL
	@# pattern in pkill's own /proc/PID/cmdline is "[c]s_server|[h]w_server"
	@# which doesn't contain the substring "cs_server" (the '[' breaks it).
	@# Without this trick pkill would SIGTERM its own shell, killing the
	@# recipe even before `|| true` can swallow the failure.
	@pkill -f '[c]s_server|[h]w_server' 2>/dev/null || true
	@# After JTAG-load, the FPGA's SGMII PHY takes several seconds to
	@# come up and re-negotiate, during which any stale /proc/net/arp
	@# entry is meaningless — checking the ARP table will report the
	@# FPGA "up" while it's actually still negotiating, and the very
	@# first FT_REG_WRITE in load-roms then times out (REG_RSP timeout
	@# for addr 0x98).  Probe with a real FastTrans round-trip
	@# (peek-tokens — tiny FT_REG_READ) wrapped in `timeout` so a
	@# hung attempt during PHY bring-up doesn't eat the whole budget.
	@echo "=== waiting for FPGA on Ethernet (FastTrans probe loop) ==="
	@deadline=$$(( $$(date +%s) + 60 )); \
	attempt=0; \
	while :; do \
	    attempt=$$((attempt+1)); \
	    if timeout 3 ./host/bfp_client peek-tokens >/dev/null 2>&1; then \
	        echo "[probe] FPGA responsive after $$attempt attempt(s)"; \
	        break; \
	    fi; \
	    if [ $$(date +%s) -ge $$deadline ]; then \
	        echo "ERROR: FPGA did not respond to FastTrans probe within 60s"; \
	        echo "       bitstream loaded but no SGMII link?  Check:"; \
	        echo "       - host is on 192.168.1.x/24 (FPGA MAC 02:00:00:4d:47:31)"; \
	        echo "       - SFP cable seated"; \
	        echo "       - 'ip link' shows your NIC UP"; \
	        exit 1; \
	    fi; \
	    sleep 2; \
	done
	@echo "=== running inference (inputs from $(SMOKE_DIR)/) ==="
	@MGRT_PREFIX=lbfp_full_ ./host/bfp_client load-roms $(SMOKE_DIR)
	@# -r 0 = auto-probe: ramp 2/4/6/8/10 MB/s, pick min projected total.
	@./host/bfp_client -r 0 -v $(SMOKE_DIR)/bfp_vocab.bin all $(SMOKE_DIR)/lbfp_full_DDR3.bin \
	    | tee $(SMOKE_DIR)/smoke-test.out
	@echo "=== comparing RTL_TOKENS to $(SMOKE_RELEASE_TAG) baseline ==="
	@actual=$$(grep '^RTL_TOKENS:' $(SMOKE_DIR)/smoke-test.out | sed 's/^RTL_TOKENS: *//'); \
	expected="$(SMOKE_EXPECTED_TOKENS)"; \
	if [ "$$actual" = "$$expected" ]; then \
	    echo "PASS — tokens match $(SMOKE_RELEASE_TAG) baseline"; \
	else \
	    echo "FAIL — tokens diverged from $(SMOKE_RELEASE_TAG) baseline"; \
	    echo "  expected: $$expected"; \
	    echo "  actual:   $$actual"; \
	    exit 1; \
	fi

# ----------------------------------------------------------------------
# make smoke-test-shakespeare — same end-to-end harness as smoke-test
# but drives the Shakespeare fine-tune.  Bitstream + vocab still come
# from the $(SMOKE_RELEASE_TAG) GitHub release (the fine-tune is
# model-only — same RTL/dims as canonical SmolLM2-135M).  The
# Shakespeare BRAMs + DDR3 image are sourced from a local pack at
# $(SMOKE_SHAKES_DIR) (default release/models/shake_/), produced by
# `make shakespeare-pack && make shakespeare-release`.
#
# First run: leave SHAKES_EXPECTED_TOKENS empty — the target prints the
# actual tokens and exits 0 ("capture mode").  Paste those tokens into
# SHAKES_EXPECTED_TOKENS := ... (or commit a Makefile fragment) and
# subsequent runs become a real PASS/FAIL diff.  Token output depends on
# the fine-tune checkpoint *and* the prompt baked into shake_PROMPT.hex,
# so a re-train or prompt change requires a fresh capture.
# ----------------------------------------------------------------------
SMOKE_SHAKES_DIR        ?= release/models/shake_
## Captured from the v0.2.0 bitstream + the Shakespeare fine-tune as
## packed at SHAKES_PROMPT="To be, or not to be," / N_GEN=32 / git
## 6aa7a8d.  Re-train or change the prompt → recapture by running with
## SHAKES_EXPECTED_TOKENS= (empty) and pasting the printed list back here.
SHAKES_EXPECTED_TOKENS  ?= 15681 253 407 441 47 325 253 314 441 253 2138 198 1710 15511 42 198 198 86 42433 94 355 441 285

## Depend on shakespeare-release so artifacts in $(GENERATED_DIR)/shake_*
## (left there by `make shakespeare-pack`) are auto-copied into
## $(SMOKE_SHAKES_DIR)/.  shakespeare-release itself errors with a clear
## message if pack hasn't been run.  Idempotent — re-copies harmlessly
## if the release dir is already populated.
smoke-test-shakespeare: host/bfp_client shakespeare-release
	@echo "=== smoke-test-shakespeare: bitstream from $(SMOKE_RELEASE_TAG), model from $(SMOKE_SHAKES_DIR)/ ==="
	@mkdir -p release $(SMOKE_DIR)
	@# Step 1: fetch the model-independent bitstream + vocab (same as
	@# smoke-test); .mcs and lbfp_full_DDR3.bin are not needed here.
	@for f in vc707_microgpt_eth.bit bfp_vocab.bin; do \
	    if [ ! -s release/$$f ]; then \
	        echo "[fetch] release/$$f"; \
	        curl -fsSL -o release/$$f $(SMOKE_BASE_URL)/$$f || \
	            { echo "ERROR: failed to fetch $$f from $(SMOKE_BASE_URL)"; exit 1; }; \
	    else \
	        echo "[skip]  release/$$f already present"; \
	    fi; \
	    ln -sf ../../release/$$f $(SMOKE_DIR)/$$f; \
	done
	@echo "=== programming FPGA from $(SMOKE_DIR)/vc707_microgpt_eth.bit ==="
	-BITFILE=$(SMOKE_DIR)/vc707_microgpt_eth.bit $(VIVADO) $(VIVADOFLAGS) -source scripts/program.tcl
	@pkill -f '[c]s_server|[h]w_server' 2>/dev/null || true
	@echo "=== waiting for FPGA on Ethernet (FastTrans probe loop) ==="
	@deadline=$$(( $$(date +%s) + 60 )); \
	attempt=0; \
	while :; do \
	    attempt=$$((attempt+1)); \
	    if timeout 3 ./host/bfp_client peek-tokens >/dev/null 2>&1; then \
	        echo "[probe] FPGA responsive after $$attempt attempt(s)"; \
	        break; \
	    fi; \
	    if [ $$(date +%s) -ge $$deadline ]; then \
	        echo "ERROR: FPGA did not respond to FastTrans probe within 60s"; exit 1; \
	    fi; \
	    sleep 2; \
	done
	@echo "=== running inference (shake_ from $(SMOKE_SHAKES_DIR)/) ==="
	@MGRT_PREFIX=shake_ ./host/bfp_client load-roms $(SMOKE_SHAKES_DIR)
	@# -r 0 = auto-probe: ramp 2/4/6/8/10 MB/s, pick min projected total.
	@./host/bfp_client -r 0 -v $(SMOKE_DIR)/bfp_vocab.bin all $(SMOKE_SHAKES_DIR)/shake_DDR3.bin \
	    | tee $(SMOKE_SHAKES_DIR)/smoke-test.out
	@echo "=== comparing RTL_TOKENS to Shakespeare baseline ==="
	@actual=$$(grep '^RTL_TOKENS:' $(SMOKE_SHAKES_DIR)/smoke-test.out | sed 's/^RTL_TOKENS: *//'); \
	expected="$(SHAKES_EXPECTED_TOKENS)"; \
	if [ -z "$$expected" ]; then \
	    echo "INFO: SHAKES_EXPECTED_TOKENS not set — capture mode."; \
	    echo "      Pin the current output by adding to Makefile:"; \
	    echo "        SHAKES_EXPECTED_TOKENS := $$actual"; \
	    echo "      (or pass it on the command line for one-off runs.)"; \
	elif [ "$$actual" = "$$expected" ]; then \
	    echo "PASS — tokens match Shakespeare baseline"; \
	else \
	    echo "FAIL — tokens diverged from Shakespeare baseline"; \
	    echo "  expected: $$expected"; \
	    echo "  actual:   $$actual"; \
	    exit 1; \
	fi

# ----------------------------------------------------------------------
# `make clean` deliberately preserves release/ — the whole point is that
# a captured release survives a fresh build.  Use `make distclean` to
# blow it away.
# ----------------------------------------------------------------------
# `make clean` preserves both release/ AND archive/ — the whole point
# of the dated snapshots is that they survive a fresh build.
# `make distclean` wipes everything.
clean:
	rm -rf *.log *.jou *.str *.mif $(work-dir) microgpt_eth.cache microgpt_eth.hw \
	       microgpt_eth.ip_user_files microgpt_eth.runs microgpt_eth.srcs \
	       microgpt_eth.sim scripts/vivado*

distclean: clean
	rm -rf $(release-dir) archive host/bfp_client

# Opt-in: nuke the Vivado project so the next `make` runs synth + impl
# from scratch.  Use after switching VARIANT, changing cfg.svh dims, or
# when Vivado's run-status tracking gets out of sync with the sources.
# Day-to-day RTL iteration should NOT need this — incremental synth and
# impl reuse work whenever inputs are unchanged.
wipe-project:
	rm -rf microgpt_eth.xpr microgpt_eth.cache microgpt_eth.hw \
	       microgpt_eth.ip_user_files microgpt_eth.hbs \
	       microgpt_eth.runs microgpt_eth.srcs microgpt_eth.sim

.PHONY: all mcs program flash release program-release flash-release demo clean distclean \
        release-bitstream release-model program-release-bitstream flash-release-bitstream demo-model run \
        shakespeare shakespeare-train shakespeare-pack shakespeare-release shakespeare-run \
        smoke-test smoke-test-shakespeare wipe-project
