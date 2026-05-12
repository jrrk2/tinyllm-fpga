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

# Pattern rule: each brom_<NAME>.sv depends on its tm_layer_<NAME>.hex
# AND on the gen_brom_sv.py script.  This catches the case where some
# other flow (e.g. the sim Verilator regression) overwrites the .hex
# files at a different config — make will then rebuild the brom .sv
# before letting Vivado run.
$(BROM_SV): $(GENERATED_DIR)/brom_%.sv: $(GENERATED_DIR)/tm_layer_%.hex \
                                        host/gen_brom_sv.py | $(tm_stamp)
	python3 host/gen_brom_sv.py $< $@ brom_$*

$(bit): $(ips) $(tm_stamp) $(BROM_SV) $(GENERATED_DIR)/silu_lut.hex
	mkdir -p $(work-dir)
	$(VIVADO) $(VIVADOFLAGS) -source scripts/run.tcl
	cp microgpt_eth.runs/impl_1/vc707_microgpt_eth.bit ./$(work-dir)/ 2>/dev/null || \
	cp microgpt_eth.runs/impl_1/vc707_microgpt_eth*.bit ./$(work-dir)/ 2>/dev/null || true

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
