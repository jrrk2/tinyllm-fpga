## Makefile — VC707 microgpt over raw Ethernet (replaces JTAG bridge)
##
## Usage:  make
##         make program
##         make flash
##         make clean
##
## Required env (mirrors cva6 sourceme.sh):
##   XILINX_PART=xc7vx485tffg1761-2
##   XILINX_BOARD=xilinx.com:vc707:part0:1.4
##   BOARD=vc707

VIVADO ?= vivado
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

$(tm_stamp): $(TM_HEX_GEN) host/gen_layer_ddr3.py
	@echo "=== regenerating tm_layer_*  (NL=$(NLAYERS) config=$(LAYER_CONFIG)) ==="
	@rm -f $(GENERATED_DIR)/.tm_layer_*.stamp
	$(TM_HEX_CMD)
	python3 host/gen_layer_ddr3.py      --mode tm --nlayers $(NLAYERS)
	@touch $@

# SiLU LUT for swiglu.sv.  Regen when SILU_LUT_SCALE in gen_swiglu_lut.py
# changes (it's the constant at the top of the file, encoded into every
# entry of the 65 K-row hex output).
$(GENERATED_DIR)/silu_lut.hex: sim/gen_swiglu_lut.py
	python3 sim/gen_swiglu_lut.py

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

# Synthesis-time hex deps for the BFP path:
#   - lbfp_full_stamp  (gen_smollm_blockfp_full.py): bakes the per-matrix
#       lbfp_full_*.hex files that smollm_layer_bfp / decode_head /
#       embed_lookup $readmemh-load at elaboration.  Even in streaming
#       mode the gamma + KV-init + norm_w hex files are still BRAM-loaded
#       (the streamer only replaces the large W?_* + EMBED matrices).
#   - lbfp_full_DDR3_stamp  (gen_smollm_blockfp_ddr.py): bakes the .bin
#       DDR3 image.  Only the .bin is needed for synth (the layout svh
#       references baked-in offsets); the matching .hex is generated
#       on-demand for the Verilator streaming testbench (LBFP_EMIT_HEX=1).
lbfp_full_stamp      := $(GENERATED_DIR)/.lbfp_full.stamp
lbfp_full_DDR3_stamp := $(GENERATED_DIR)/.lbfp_full_DDR3.stamp

BFP_BUILD_DEPS :=
ifeq ($(USE_BFP),1)
  BFP_BUILD_DEPS += $(lbfp_full_stamp)
  ifeq ($(BFP_STREAM),1)
    BFP_BUILD_DEPS += $(lbfp_full_DDR3_stamp)
  endif
endif

$(bit): $(ips) $(tm_stamp) $(BROM_SV) $(GENERATED_DIR)/silu_lut.hex $(lbfp_stamp) $(BFP_BUILD_DEPS)
	mkdir -p $(work-dir)
	$(VIVADO) $(VIVADOFLAGS) -source scripts/run.tcl
	cp microgpt_eth.runs/impl_1/vc707_microgpt_eth.bit ./$(work-dir)/ 2>/dev/null || \
	cp microgpt_eth.runs/impl_1/vc707_microgpt_eth*.bit ./$(work-dir)/ 2>/dev/null || true

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
	@echo "=== baking SmolLM2-135M block-FP weights (30 layers, full dims) ==="
	python3 host/gen_smollm_blockfp_full.py
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
	@echo "=== baking lbfp_full_DDR3.bin ==="
	python3 host/gen_smollm_blockfp_ddr.py
	@touch $@

# Same end-to-end demo but with weights streamed from a DDR3 image via
# mock_axi_slave (STREAM_WEIGHTS=1 / STREAM_LOOKUP=1).  Validates the
# AXI fetch + streamer pipeline against the BRAM-mode golden tokens.
# Requires the (~670 MB) hex image — generated with LBFP_EMIT_HEX=1.
bfp-autoregress-stream: $(lbfp_full_stamp)
	LBFP_EMIT_HEX=1 python3 host/gen_smollm_blockfp_ddr.py
	$(MAKE) -C sim bfp-autoregress-stream

.PHONY: bfp-sim bfp-full bfp-autoregress bfp-autoregress-stream

$(ips): %.xci :
	mkdir -p $(work-dir)
	@echo Generating $(@F)
	@cd $(ip-dir)/$(basename $(@F)) && make clean && make
	@cp $(ip-dir)/$(basename $(@F))/$(basename $(@F)).srcs/sources_1/ip/$(basename $(@F))/$(@F) $@

$(mcs): $(bit)
	$(VIVADO) $(VIVADOFLAGS) -source scripts/write_cfgmem.tcl -tclargs $@ $^

mcs: $(mcs)

program:
	$(VIVADO) $(VIVADOFLAGS) -source scripts/program.tcl
	@# program.tcl cleanly closes the hw_manager, but Vivado's
	@# cs_server (ChipScope) and hw_server (JTAG) sometimes survive
	@# as background processes and idle-poll at high CPU.  Reap them.
	@pkill -f 'cs_server|hw_server' 2>/dev/null || true

flash: $(mcs)
	$(VIVADO) $(VIVADOFLAGS) -source scripts/flash.tcl
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

# Build a release snapshot from whatever is currently in work/, the
# generated/ tree, and the Vivado reports.  Requires `make` to have
# completed first (we don't trigger a fresh synth from here — that's
# what `make` itself is for).
release: $(bit) $(mcs) $(GENERATED_DIR)/lbfp_full_DDR3.bin host/bfp_vocab.bin
	@mkdir -p $(release-dir)/reports
	@cp $(bit)                              $(release-bit)
	@cp $(mcs)                              $(release-mcs)
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
	@if [ ! -f "$(GENERATED_DIR)/$(PREFIX)DDR3.bin" ]; then \
	    echo "ERROR: $(GENERATED_DIR)/$(PREFIX)DDR3.bin missing — capture-only target."; \
	    echo "       Run 'PREFIX=$(PREFIX) python host/gen_smollm_blockfp_ddr.py' first."; \
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
# `make clean` deliberately preserves release/ — the whole point is that
# a captured release survives a fresh build.  Use `make distclean` to
# blow it away.
# ----------------------------------------------------------------------
clean:
	rm -rf *.log *.jou *.str *.mif $(work-dir) microgpt_eth.cache microgpt_eth.hw \
	       microgpt_eth.ip_user_files microgpt_eth.runs microgpt_eth.srcs \
	       microgpt_eth.sim scripts/vivado*

distclean: clean
	rm -rf $(release-dir) host/bfp_client

.PHONY: all mcs program flash release program-release flash-release demo clean distclean \
        release-bitstream release-model program-release-bitstream flash-release-bitstream demo-model run
