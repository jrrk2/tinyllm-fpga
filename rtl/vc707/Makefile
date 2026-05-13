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

# Toggle the top-level layer instance.  USE_BFP=1 swaps the int8
# smollm_multilayer_tm_selftest for smollm_layer_bfp_selftest (small-dim
# single-layer block-FP selftest).  Passed through to Vivado via
# scripts/run.tcl's MICROGPT_USE_BFP define.
USE_BFP ?= 0
ifeq ($(USE_BFP),1)
  VIVADO_DEFINES := MICROGPT_USE_BFP
  export VIVADO_DEFINES
endif

$(bit): $(ips) $(tm_stamp) $(BROM_SV) $(GENERATED_DIR)/silu_lut.hex $(lbfp_stamp)
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
lbfp_full_stamp := $(GENERATED_DIR)/.lbfp_full.stamp
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

.PHONY: bfp-sim bfp-full bfp-autoregress

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

clean:
	rm -rf *.log *.jou *.str *.mif $(work-dir) microgpt_eth.cache microgpt_eth.hw \
	       microgpt_eth.ip_user_files microgpt_eth.runs microgpt_eth.srcs \
	       microgpt_eth.sim scripts/vivado*

.PHONY: all mcs program flash clean
