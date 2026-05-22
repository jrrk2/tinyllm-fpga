# Verilator simulation makefile for SmolLM2 building blocks.
#
# Targets:
#   make matvec       — build + run matvec_int8_engine testbench
#   make clean

VERILATOR := verilator
VFLAGS    := -Wall -Wno-fatal -cc --exe --build -j 0 \
             --x-assign 0 --x-initial 0 \
             -CFLAGS "-O2 -std=c++17"

SMOLLM_SRC := ../src/smollm

.PHONY: all matvec rmsnorm rope swiglu swiglu_selftest softmax persist softmax_selftest weight_streamer idle_scan_crc smollm_layer_ddr3 multilayer_tm bfp-layer bfp-full bfp-autoregress bfp-autoregress-stream clean

all: matvec rmsnorm rope swiglu softmax persist softmax_selftest

# ---------------------- matvec_int8_engine ----------------------
matvec: obj_dir/Vmatvec_int8_engine test_matvec.bin
	@echo "=== running matvec sim ==="
	@./obj_dir/Vmatvec_int8_engine

obj_dir/Vmatvec_int8_engine: $(SMOLLM_SRC)/matvec_int8_engine.sv tb_matvec.cpp
	$(VERILATOR) $(VFLAGS) --top-module matvec_int8_engine \
	    $(SMOLLM_SRC)/matvec_int8_engine.sv tb_matvec.cpp \
	    -o Vmatvec_int8_engine

test_matvec.bin: gen_matvec_test.py
	python3 gen_matvec_test.py

# ---------------------- rmsnorm ----------------------
rmsnorm: obj_dir/Vrmsnorm test_rmsnorm.bin
	@echo "=== running rmsnorm sim ==="
	@./obj_dir/Vrmsnorm

obj_dir/Vrmsnorm: $(SMOLLM_SRC)/rmsnorm.sv tb_rmsnorm.cpp
	$(VERILATOR) $(VFLAGS) --top-module rmsnorm \
	    $(SMOLLM_SRC)/rmsnorm.sv tb_rmsnorm.cpp \
	    -o Vrmsnorm

test_rmsnorm.bin: gen_rmsnorm_test.py
	python3 gen_rmsnorm_test.py

# ---------------------- rope ----------------------
# ROPE_LUTS no longer needed — CORDIC replaces the 256 KB LUT BRAMs.
# Kept as a variable for rope_selftest which still references the name but no
# longer needs the files (the variable is now empty).
ROPE_LUTS :=

ROPE_FREQ_TABLE := ../generated/rope_freq_turns.svh

rope: obj_dir/Vrope test_rope.bin
	@echo "=== running rope sim ==="
	@./obj_dir/Vrope

obj_dir/Vrope: $(SMOLLM_SRC)/rope.sv $(SMOLLM_SRC)/cordic_sincos.sv \
               tb_rope.cpp $(ROPE_FREQ_TABLE)
	$(VERILATOR) $(VFLAGS) --top-module rope \
	    -I../generated \
	    $(SMOLLM_SRC)/rope.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    tb_rope.cpp \
	    -o Vrope

test_rope.bin: gen_rope_test.py
	python3 gen_rope_test.py

$(ROPE_FREQ_TABLE): gen_rope_freq_table.py
	python3 gen_rope_freq_table.py

# ---------------------- swiglu ----------------------
swiglu: obj_dir/Vswiglu test_swiglu.bin
	@echo "=== running swiglu sim ==="
	@./obj_dir/Vswiglu

obj_dir/Vswiglu: $(SMOLLM_SRC)/swiglu.sv tb_swiglu.cpp
	$(VERILATOR) $(VFLAGS) --top-module swiglu \
	    $(SMOLLM_SRC)/swiglu.sv tb_swiglu.cpp \
	    -o Vswiglu

test_swiglu.bin: gen_swiglu_test.py
	python3 gen_swiglu_test.py

# ---------------------- swiglu_selftest (full FSM + ROMs) ----------------------
SWIGLU_SELFTEST_DEPS := ../generated/swiglu_selftest_data.svh \
                        ../generated/swiglu_selftest_expected.txt

SILU_LUT := ../generated/silu_lut.hex

swiglu_selftest: obj_dir/Vswiglu_selftest $(SWIGLU_SELFTEST_DEPS) $(SILU_LUT)
	@echo "=== running swiglu_selftest sim ==="
	@./obj_dir/Vswiglu_selftest

obj_dir/Vswiglu_selftest: $(SMOLLM_SRC)/swiglu_selftest.sv \
                          $(SMOLLM_SRC)/swiglu.sv \
                          tb_swiglu_selftest.cpp \
                          $(SWIGLU_SELFTEST_DEPS) $(SILU_LUT)
	$(VERILATOR) $(VFLAGS) --top-module swiglu_selftest \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    $(SMOLLM_SRC)/swiglu_selftest.sv \
	    $(SMOLLM_SRC)/swiglu.sv \
	    tb_swiglu_selftest.cpp \
	    -o Vswiglu_selftest

$(SWIGLU_SELFTEST_DEPS): ../host/gen_swiglu_selftest.py
	python3 ../host/gen_swiglu_selftest.py

# ---------------------- softmax_q15 ----------------------
EXP_LUT := ../generated/exp_lut.hex

softmax: obj_dir/Vsoftmax_q15 test_softmax.bin $(EXP_LUT)
	@echo "=== running softmax sim ==="
	@./obj_dir/Vsoftmax_q15

obj_dir/Vsoftmax_q15: $(SMOLLM_SRC)/softmax_q15.sv tb_softmax.cpp
	$(VERILATOR) $(VFLAGS) --top-module softmax_q15 \
	    $(SMOLLM_SRC)/softmax_q15.sv tb_softmax.cpp \
	    -o Vsoftmax_q15

$(EXP_LUT): gen_exp_lut.py
	OUT=../generated python3 gen_exp_lut.py

test_softmax.bin: gen_softmax_test.py
	python3 gen_softmax_test.py

# ---------------------- rmsnorm_selftest (full FSM + ROMs) ----------------------
RMSNORM_SELFTEST_DEPS := ../generated/rmsnorm_selftest_data.svh \
                         ../generated/rmsnorm_selftest_expected.txt

rmsnorm_selftest: obj_dir/Vrmsnorm_selftest $(RMSNORM_SELFTEST_DEPS)
	@echo "=== running rmsnorm_selftest sim ==="
	@./obj_dir/Vrmsnorm_selftest

obj_dir/Vrmsnorm_selftest: $(SMOLLM_SRC)/rmsnorm_selftest.sv \
                           $(SMOLLM_SRC)/rmsnorm.sv \
                           tb_rmsnorm_selftest.cpp \
                           $(RMSNORM_SELFTEST_DEPS)
	$(VERILATOR) $(VFLAGS) --top-module rmsnorm_selftest \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    $(SMOLLM_SRC)/rmsnorm_selftest.sv \
	    $(SMOLLM_SRC)/rmsnorm.sv \
	    tb_rmsnorm_selftest.cpp \
	    -o Vrmsnorm_selftest

$(RMSNORM_SELFTEST_DEPS): ../host/gen_rmsnorm_selftest.py
	python3 ../host/gen_rmsnorm_selftest.py

# ---------------------- matvec_selftest (full FSM + ROMs) ----------------------
SELFTEST_DEPS := ../generated/matvec_selftest_act.hex \
                 ../generated/matvec_selftest_weights.hex \
                 ../generated/matvec_selftest_scale.svh \
                 ../generated/matvec_selftest_expected.txt

selftest: obj_dir/Vmatvec_selftest $(SELFTEST_DEPS)
	@echo "=== running matvec_selftest sim ==="
	@./obj_dir/Vmatvec_selftest

obj_dir/Vmatvec_selftest: $(SMOLLM_SRC)/matvec_selftest.sv \
                          $(SMOLLM_SRC)/matvec_int8_engine.sv \
                          tb_matvec_selftest.cpp \
                          $(SELFTEST_DEPS)
	$(VERILATOR) $(VFLAGS) --top-module matvec_selftest \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    $(SMOLLM_SRC)/matvec_selftest.sv \
	    $(SMOLLM_SRC)/matvec_int8_engine.sv \
	    tb_matvec_selftest.cpp \
	    -o Vmatvec_selftest

$(SELFTEST_DEPS): ../host/gen_matvec_selftest.py
	python3 ../host/gen_matvec_selftest.py

# ---------------------- smollm_full (end-to-end token in / token out) ----------------
FULL_DEPS := ../generated/full_EMBED.hex \
             ../generated/full_data.svh \
             ../generated/full_expected.txt

full: obj_dir/Vsmollm_full $(FULL_DEPS)
	@echo "=== running smollm_full sim ==="
	@./obj_dir/Vsmollm_full

obj_dir/Vsmollm_full: $(SMOLLM_SRC)/smollm_full.sv \
                      $(SMOLLM_SRC)/smollm_layer.sv \
                      $(SMOLLM_SRC)/smollm_decode_head.sv \
                      $(SMOLLM_SRC)/weight_streamer_brom.sv \
                      $(SMOLLM_SRC)/matvec_int8_engine.sv \
                      $(SMOLLM_SRC)/rmsnorm.sv \
                      $(SMOLLM_SRC)/rope.sv \
                      $(SMOLLM_SRC)/cordic_sincos.sv \
                      $(SMOLLM_SRC)/swiglu.sv \
                      $(SMOLLM_SRC)/softmax_q15.sv \
                      tb_full.cpp \
                      $(FULL_DEPS)
	$(VERILATOR) $(VFLAGS) --top-module smollm_full \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    $(SMOLLM_SRC)/smollm_full.sv \
	    $(SMOLLM_SRC)/smollm_layer.sv \
	    $(SMOLLM_SRC)/smollm_decode_head.sv \
	    $(SMOLLM_SRC)/weight_streamer_brom.sv \
	    $(SMOLLM_SRC)/matvec_int8_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm.sv \
	    $(SMOLLM_SRC)/rope.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    $(SMOLLM_SRC)/swiglu.sv \
	    $(SMOLLM_SRC)/softmax_q15.sv \
	    tb_full.cpp \
	    -o Vsmollm_full

$(FULL_DEPS): ../host/gen_full_test.py
	python3 ../host/gen_full_test.py

# ---------------------- smollm_decode_head (final-norm + LM head + argmax) ----------
DECODEHEAD_DEPS := ../generated/decodehead_HIDDEN_IN.hex \
                   ../generated/decodehead_GAMMA.hex \
                   ../generated/decodehead_W_LM.hex \
                   ../generated/decodehead_SCALE_LM.hex \
                   ../generated/decodehead_expected.txt

decode_head: obj_dir/Vsmollm_decode_head $(DECODEHEAD_DEPS)
	@echo "=== running smollm_decode_head sim ==="
	@./obj_dir/Vsmollm_decode_head

obj_dir/Vsmollm_decode_head: $(SMOLLM_SRC)/smollm_decode_head.sv \
                             $(SMOLLM_SRC)/matvec_int8_engine.sv \
                             $(SMOLLM_SRC)/rmsnorm.sv \
                             tb_decode_head.cpp \
                             $(DECODEHEAD_DEPS)
	$(VERILATOR) $(VFLAGS) --top-module smollm_decode_head \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    $(SMOLLM_SRC)/smollm_decode_head.sv \
	    $(SMOLLM_SRC)/matvec_int8_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm.sv \
	    tb_decode_head.cpp \
	    -o Vsmollm_decode_head

$(DECODEHEAD_DEPS): ../host/gen_decode_head_test.py
	python3 ../host/gen_decode_head_test.py

# ---------------------- smollm_multilayer (NL-layer chain) ----------------------
MULTILAYER_DEPS := ../generated/multilayer_HIDDEN_IN.hex \
                   ../generated/multilayer_expected.txt \
                   ../generated/multilayer_data.svh

multilayer: obj_dir/Vsmollm_multilayer $(MULTILAYER_DEPS)
	@echo "=== running smollm_multilayer sim ==="
	@./obj_dir/Vsmollm_multilayer

obj_dir/Vsmollm_multilayer: $(SMOLLM_SRC)/smollm_multilayer.sv \
                            $(SMOLLM_SRC)/smollm_layer.sv \
                            $(SMOLLM_SRC)/weight_streamer_brom.sv \
                            $(SMOLLM_SRC)/matvec_int8_engine.sv \
                            $(SMOLLM_SRC)/rmsnorm.sv \
                            $(SMOLLM_SRC)/rope.sv \
                            $(SMOLLM_SRC)/cordic_sincos.sv \
                            $(SMOLLM_SRC)/swiglu.sv \
                            $(SMOLLM_SRC)/softmax_q15.sv \
                            tb_multilayer.cpp \
                            $(MULTILAYER_DEPS)
	$(VERILATOR) $(VFLAGS) --top-module smollm_multilayer \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    $(SMOLLM_SRC)/smollm_multilayer.sv \
	    $(SMOLLM_SRC)/smollm_layer.sv \
	    $(SMOLLM_SRC)/weight_streamer_brom.sv \
	    $(SMOLLM_SRC)/matvec_int8_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm.sv \
	    $(SMOLLM_SRC)/rope.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    $(SMOLLM_SRC)/swiglu.sv \
	    $(SMOLLM_SRC)/softmax_q15.sv \
	    tb_multilayer.cpp \
	    -o Vsmollm_multilayer

$(MULTILAYER_DEPS): ../host/gen_multilayer_test.py
	python3 ../host/gen_multilayer_test.py

# ---------------------- smollm_layer (full-layer integration test) ----------------------
LAYER_DEPS := ../generated/layer_test_data.svh \
              ../generated/layer_test_expected.txt

smollm_layer: obj_dir/Vsmollm_layer $(LAYER_DEPS)
	@echo "=== running smollm_layer sim ==="
	@./obj_dir/Vsmollm_layer

obj_dir/Vsmollm_layer: $(SMOLLM_SRC)/smollm_layer.sv \
                       $(SMOLLM_SRC)/weight_streamer_brom.sv \
                       $(SMOLLM_SRC)/matvec_int8_engine.sv \
                       $(SMOLLM_SRC)/rmsnorm.sv \
                       $(SMOLLM_SRC)/rope.sv \
                       $(SMOLLM_SRC)/cordic_sincos.sv \
                       $(SMOLLM_SRC)/swiglu.sv \
                       $(SMOLLM_SRC)/softmax_q15.sv \
                       tb_smollm_layer.cpp \
                       $(LAYER_DEPS)
	$(VERILATOR) $(VFLAGS) --top-module smollm_layer \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    +define+DEBUG \
	    $(SMOLLM_SRC)/smollm_layer.sv \
	    $(SMOLLM_SRC)/weight_streamer_brom.sv \
	    $(SMOLLM_SRC)/matvec_int8_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm.sv \
	    $(SMOLLM_SRC)/rope.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    $(SMOLLM_SRC)/swiglu.sv \
	    $(SMOLLM_SRC)/softmax_q15.sv \
	    tb_smollm_layer.cpp \
	    -o Vsmollm_layer

$(LAYER_DEPS): ../host/gen_layer_test.py
	python3 ../host/gen_layer_test.py

# ---------------------- cordic_sincos accuracy sweep ----------------------
cordic: obj_dir/Vcordic_sincos
	@echo "=== running cordic accuracy test ==="
	@./obj_dir/Vcordic_sincos

obj_dir/Vcordic_sincos: $(SMOLLM_SRC)/cordic_sincos.sv tb_cordic.cpp
	$(VERILATOR) $(VFLAGS) --top-module cordic_sincos \
	    $(SMOLLM_SRC)/cordic_sincos.sv tb_cordic.cpp \
	    -o Vcordic_sincos

# ---------------------- persistent matvec backend ----------------------
persist: obj_dir/Vpersist
	@echo "built obj_dir/Vpersist (persistent matvec backend)"

obj_dir/Vpersist: $(SMOLLM_SRC)/matvec_int8_engine.sv tb_persist.cpp
	$(VERILATOR) $(VFLAGS) --top-module matvec_int8_engine \
	    $(SMOLLM_SRC)/matvec_int8_engine.sv tb_persist.cpp \
	    -o Vpersist

# ---------------------- rope_selftest (full FSM + LUTs + ROMs) ----------------------
ROPE_SELFTEST_DEPS := ../generated/rope_selftest_data.svh \
                      ../generated/rope_selftest_expected.txt

rope_selftest: obj_dir/Vrope_selftest $(ROPE_SELFTEST_DEPS)
	@echo "=== running rope_selftest sim ==="
	@./obj_dir/Vrope_selftest

obj_dir/Vrope_selftest: $(SMOLLM_SRC)/rope_selftest.sv \
                        $(SMOLLM_SRC)/rope.sv \
                        $(SMOLLM_SRC)/cordic_sincos.sv \
                        tb_rope_selftest.cpp \
                        $(ROPE_SELFTEST_DEPS) \
                        $(ROPE_FREQ_TABLE)
	$(VERILATOR) $(VFLAGS) --top-module rope_selftest \
	    -I../generated \
	    $(SMOLLM_SRC)/rope_selftest.sv \
	    $(SMOLLM_SRC)/rope.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    tb_rope_selftest.cpp \
	    -o Vrope_selftest

$(ROPE_SELFTEST_DEPS): ../host/gen_rope_selftest.py
	python3 ../host/gen_rope_selftest.py

# ---------------------- softmax_selftest (full FSM + ROM + exp LUT) ----------------------
SOFTMAX_SELFTEST_DEPS := ../generated/softmax_selftest_data.svh \
                         ../generated/softmax_selftest_expected.txt

softmax_selftest: obj_dir/Vsoftmax_selftest $(SOFTMAX_SELFTEST_DEPS) $(EXP_LUT)
	@echo "=== running softmax_selftest sim ==="
	@./obj_dir/Vsoftmax_selftest

obj_dir/Vsoftmax_selftest: $(SMOLLM_SRC)/softmax_selftest.sv \
                           $(SMOLLM_SRC)/softmax_q15.sv \
                           tb_softmax_selftest.cpp \
                           $(SOFTMAX_SELFTEST_DEPS) $(EXP_LUT)
	$(VERILATOR) $(VFLAGS) --top-module softmax_selftest \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    $(SMOLLM_SRC)/softmax_selftest.sv \
	    $(SMOLLM_SRC)/softmax_q15.sv \
	    tb_softmax_selftest.cpp \
	    -o Vsoftmax_selftest

$(SOFTMAX_SELFTEST_DEPS): ../host/gen_softmax_selftest.py
	python3 ../host/gen_softmax_selftest.py

# ---------------------- weight_streamer idle_scan_crc (FPGA-side, mock AXI slave) ----------
weight_streamer: obj_dir/Vtb_weight_streamer_dut
	@echo "=== running weight_streamer idle_scan_crc sim ==="
	@./obj_dir/Vtb_weight_streamer_dut

idle_scan_crc: obj_dir/Vtb_idle_scan_crc_dut
	@echo "=== running idle_scan_crc sim ==="
	@./obj_dir/Vtb_idle_scan_crc_dut

obj_dir/Vtb_idle_scan_crc_dut: tb_idle_scan_crc_dut.sv \
                               mock_axi_slave.sv \
                               ../src/idle_scan_crc.sv \
                               tb_idle_scan_crc.cpp
	$(VERILATOR) $(VFLAGS) --top-module tb_idle_scan_crc_dut \
	    tb_idle_scan_crc_dut.sv \
	    mock_axi_slave.sv \
	    ../src/idle_scan_crc.sv \
	    tb_idle_scan_crc.cpp \
	    -o Vtb_idle_scan_crc_dut

obj_dir/Vtb_weight_streamer_dut: tb_weight_streamer_dut.sv \
                                 mock_axi_slave.sv \
                                 $(SMOLLM_SRC)/weight_streamer_mt.sv \
                                 tb_weight_streamer.cpp
	$(VERILATOR) $(VFLAGS) --top-module tb_weight_streamer_dut \
	    tb_weight_streamer_dut.sv \
	    mock_axi_slave.sv \
	    $(SMOLLM_SRC)/weight_streamer_mt.sv \
	    tb_weight_streamer.cpp \
	    -o Vtb_weight_streamer_dut

# ---------------------- smollm_layer DDR3 streamer integration test ----------
LAYER_DDR3_DEPS := ../generated/layer_DDR3.hex \
                   ../generated/layer_ddr3_bases.svh \
                   ../generated/layer_test_data.svh \
                   ../generated/layer_test_expected.txt

smollm_layer_ddr3: obj_dir/Vtb_smollm_layer_ddr3_dut $(LAYER_DDR3_DEPS)
	@echo "=== running smollm_layer DDR3-streamer sim ==="
	@./obj_dir/Vtb_smollm_layer_ddr3_dut

obj_dir/Vtb_smollm_layer_ddr3_dut: tb_smollm_layer_ddr3_dut.sv \
                                   mock_axi_slave.sv \
                                   $(SMOLLM_SRC)/smollm_layer.sv \
                                   $(SMOLLM_SRC)/weight_streamer_mt.sv \
                                   $(SMOLLM_SRC)/matvec_int8_engine.sv \
                                   $(SMOLLM_SRC)/rmsnorm.sv \
                                   $(SMOLLM_SRC)/rope.sv \
                                   $(SMOLLM_SRC)/cordic_sincos.sv \
                                   $(SMOLLM_SRC)/swiglu.sv \
                                   $(SMOLLM_SRC)/softmax_q15.sv \
                                   tb_smollm_layer_ddr3.cpp \
                                   $(LAYER_DDR3_DEPS)
	$(VERILATOR) $(VFLAGS) --top-module tb_smollm_layer_ddr3_dut \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    +define+MICROGPT_DDR3_WEIGHTS \
	    +define+DEBUG \
	    tb_smollm_layer_ddr3_dut.sv \
	    mock_axi_slave.sv \
	    $(SMOLLM_SRC)/smollm_layer.sv \
	    $(SMOLLM_SRC)/weight_streamer_mt.sv \
	    $(SMOLLM_SRC)/matvec_int8_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm.sv \
	    $(SMOLLM_SRC)/rope.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    $(SMOLLM_SRC)/swiglu.sv \
	    $(SMOLLM_SRC)/softmax_q15.sv \
	    tb_smollm_layer_ddr3.cpp \
	    -o Vtb_smollm_layer_ddr3_dut

../generated/layer_DDR3.hex ../generated/layer_ddr3_bases.svh: $(LAYER_DEPS) \
                                                               ../host/gen_layer_ddr3.py
	python3 ../host/gen_layer_ddr3.py

# ---------------------- smollm_multilayer_tm (time-multiplex NL layers) -----
TM_DEPS := ../generated/tm_layer_DDR3.hex \
           ../generated/tm_layer_ddr3_bases.svh \
           ../generated/tm_layer_HIDDEN_IN.hex \
           ../generated/tm_layer_expected.txt \
           ../generated/tm_layer_data.svh \
           ../generated/tm_layer_GAMMA1.hex \
           ../generated/tm_layer_GAMMA2.hex \
           ../generated/tm_layer_SCALE_Q.hex \
           ../generated/tm_layer_SCALE_K.hex \
           ../generated/tm_layer_SCALE_V.hex \
           ../generated/tm_layer_SCALE_O.hex \
           ../generated/tm_layer_SCALE_GATE.hex \
           ../generated/tm_layer_SCALE_UP.hex \
           ../generated/tm_layer_SCALE_DOWN.hex \
           ../generated/tm_layer_K_CACHE_INIT.hex \
           ../generated/tm_layer_V_CACHE_INIT.hex

multilayer_tm: obj_dir/Vtb_smollm_multilayer_tm_dut $(TM_DEPS)
	@echo "=== running smollm_multilayer_tm sim ==="
	@./obj_dir/Vtb_smollm_multilayer_tm_dut

# ----------------------------------------------------------------------
# multilayer_tm_smollm — Verilator at the EXACT FPGA config
# (D=576 H_Q=9 H_KV=3 FFN=1536 MAX_CTX=4 NL=30) using the same
# generated/tm_layer_*.hex and brom_*.sv files Vivado baked into the
# bitstream.  Used to bisect FPGA-vs-Verilator behaviour discrepancies.
# ----------------------------------------------------------------------
SMOLLM_PVALUES := \
    -GD=576 -GH_Q=9 -GH_KV=3 -GHD=64 \
    -GFFN=1536 -GMAX_CTX=4 -GNL=30

multilayer_tm_smollm: obj_dir_smollm/Vtb_smollm_multilayer_tm_dut
	@echo "=== running smollm_multilayer_tm_smollm sim (D=576 NL=30) ==="
	@./obj_dir_smollm/Vtb_smollm_multilayer_tm_dut

# Build from CWD = obj_dir_smollm so cpp/sv lookups resolve via VPATH-like
# `..`-relative paths.  Separate Mdir from the small-config build so the
# two coexist; same cpp is parameterised by -DDUT_D / -DDUT_NL.
obj_dir_smollm/Vtb_smollm_multilayer_tm_dut: tb_smollm_multilayer_tm_dut.sv \
                                              mock_axi_slave.sv \
                                              ramb36e1_stub.sv \
                                              $(SMOLLM_SRC)/factor_ram.sv $(SMOLLM_SRC)/smollm_multilayer_tm.sv \
                                              $(SMOLLM_SRC)/smollm_layer.sv \
                                              $(SMOLLM_SRC)/weight_streamer_mt.sv \
                                              $(SMOLLM_SRC)/matvec_int8_engine.sv \
                                              $(SMOLLM_SRC)/rmsnorm.sv \
                                              $(SMOLLM_SRC)/rope.sv \
                                              $(SMOLLM_SRC)/cordic_sincos.sv \
                                              $(SMOLLM_SRC)/swiglu.sv \
                                              $(SMOLLM_SRC)/softmax_q15.sv \
                                              $(BROM_SV) \
                                              tb_smollm_multilayer_tm.cpp
	mkdir -p obj_dir_smollm
	$(VERILATOR) $(VFLAGS) --top-module tb_smollm_multilayer_tm_dut \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    +define+MICROGPT_DDR3_WEIGHTS \
	    $(EXTRA_DEFINES) \
	    $(SMOLLM_PVALUES) \
	    -CFLAGS "-DDUT_D=576 -DDUT_NL=30" \
	    --Mdir obj_dir_smollm \
	    $(abspath tb_smollm_multilayer_tm_dut.sv) \
	    $(abspath mock_axi_slave.sv) \
	    $(abspath ramb36e1_stub.sv) \
	    $(SMOLLM_SRC)/factor_ram.sv $(SMOLLM_SRC)/smollm_multilayer_tm.sv \
	    $(SMOLLM_SRC)/smollm_layer.sv \
	    $(SMOLLM_SRC)/weight_streamer_mt.sv \
	    $(SMOLLM_SRC)/matvec_int8_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm.sv \
	    $(SMOLLM_SRC)/rope.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    $(SMOLLM_SRC)/swiglu.sv \
	    $(SMOLLM_SRC)/softmax_q15.sv \
	    $(BROM_SV) \
	    $(abspath tb_smollm_multilayer_tm.cpp) \
	    -o Vtb_smollm_multilayer_tm_dut

# Declared early so the obj_dir rule below sees a non-empty $(BROM_SV)
# during prerequisite expansion.
BROM_NAMES := GAMMA1 GAMMA2 SCALE_Q SCALE_K SCALE_V SCALE_O \
              SCALE_GATE SCALE_UP SCALE_DOWN
BROM_SV    := $(foreach n,$(BROM_NAMES),../generated/brom_$(n).sv)

obj_dir/Vtb_smollm_multilayer_tm_dut: tb_smollm_multilayer_tm_dut.sv \
                                       mock_axi_slave.sv \
                                       ramb36e1_stub.sv \
                                       $(SMOLLM_SRC)/factor_ram.sv $(SMOLLM_SRC)/smollm_multilayer_tm.sv \
                                       $(SMOLLM_SRC)/smollm_layer.sv \
                                       $(SMOLLM_SRC)/weight_streamer_mt.sv \
                                       $(SMOLLM_SRC)/matvec_int8_engine.sv \
                                       $(SMOLLM_SRC)/rmsnorm.sv \
                                       $(SMOLLM_SRC)/rope.sv \
                                       $(SMOLLM_SRC)/cordic_sincos.sv \
                                       $(SMOLLM_SRC)/swiglu.sv \
                                       $(SMOLLM_SRC)/softmax_q15.sv \
                                       $(BROM_SV) \
                                       tb_smollm_multilayer_tm.cpp \
                                       $(TM_DEPS)
	$(VERILATOR) $(VFLAGS) --top-module tb_smollm_multilayer_tm_dut \
	    -I../generated \
	    +define+MICROGPT_WEIGHT_DIR=\"../generated\" \
	    +define+MICROGPT_DDR3_WEIGHTS \
	    tb_smollm_multilayer_tm_dut.sv \
	    mock_axi_slave.sv \
	    ramb36e1_stub.sv \
	    $(SMOLLM_SRC)/factor_ram.sv $(SMOLLM_SRC)/smollm_multilayer_tm.sv \
	    $(SMOLLM_SRC)/smollm_layer.sv \
	    $(SMOLLM_SRC)/weight_streamer_mt.sv \
	    $(SMOLLM_SRC)/matvec_int8_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm.sv \
	    $(SMOLLM_SRC)/rope.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    $(SMOLLM_SRC)/swiglu.sv \
	    $(SMOLLM_SRC)/softmax_q15.sv \
	    $(BROM_SV) \
	    tb_smollm_multilayer_tm.cpp \
	    -o Vtb_smollm_multilayer_tm_dut

../generated/tm_layer_DDR3.hex ../generated/tm_layer_ddr3_bases.svh: \
        ../generated/tm_layer_HIDDEN_IN.hex \
        ../host/gen_layer_ddr3.py
	python3 ../host/gen_layer_ddr3.py --mode tm --nlayers 3

# brom pattern rule (BROM_NAMES/BROM_SV defined further up so the
# obj_dir rule's prereq expansion sees them).
$(BROM_SV): ../generated/brom_%.sv: ../generated/tm_layer_%.hex \
                                    ../host/gen_brom_sv.py
	python3 ../host/gen_brom_sv.py $< $@ brom_$*

../generated/tm_layer_HIDDEN_IN.hex ../generated/tm_layer_expected.txt \
../generated/tm_layer_data.svh ../generated/tm_layer_GAMMA1.hex \
../generated/tm_layer_GAMMA2.hex ../generated/tm_layer_SCALE_Q.hex \
../generated/tm_layer_SCALE_K.hex ../generated/tm_layer_SCALE_V.hex \
../generated/tm_layer_SCALE_O.hex ../generated/tm_layer_SCALE_GATE.hex \
../generated/tm_layer_SCALE_UP.hex ../generated/tm_layer_SCALE_DOWN.hex \
../generated/tm_layer_K_CACHE_INIT.hex \
../generated/tm_layer_V_CACHE_INIT.hex: ../host/gen_multilayer_test.py
	python3 ../host/gen_multilayer_test.py --mode tm --nlayers 3

# ---------------------- smollm_layer_bfp (full block-FP layer) ----------------------
# Boots from the lbfp_*.hex baked by host/gen_smollm_blockfp_bfp.py
# (run via `make` in vc707/ — see its lbfp_stamp target).  PASSes when
# hidden_out matches the per-stage BFP golden within ±256 LSB mantissa
# (sub-1% relative error across 14 op stages).
bfp-layer: obj_dir/Vtb_smollm_layer_bfp
	@echo "=== running smollm_layer_bfp sim ==="
	@./obj_dir/Vtb_smollm_layer_bfp

# ---------------------- bfp-sdpram-compare ----------------------
# Constrained-random equivalence check between the inferred-array RAM
# pattern (the OLD style smollm_layer_bfp used for mlp_m etc.) and
# bfp_sdpram.  Drives identical {we,wr_addr,wr_data,rd_addr} stimulus
# into both and asserts cycle-by-cycle bit equality.  Fast (~1 s) and
# self-contained — doesn't need the smollm weight bake or DDR3 hex.
bfp-sdpram-compare: obj_dir/Vtb_bfp_sdpram_compare
	@echo "=== running tb_bfp_sdpram_compare ==="
	@./obj_dir/Vtb_bfp_sdpram_compare

obj_dir/Vtb_bfp_sdpram_compare: tb_bfp_sdpram_compare.sv tb_bfp_sdpram_compare.cpp \
                                $(SMOLLM_SRC)/bfp_sdpram.sv
	$(VERILATOR) -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal -cc --exe --build \
	    -j 0 --x-assign 0 --x-initial 0 \
	    -CFLAGS "-O2 -std=c++17" \
	    --top-module tb_bfp_sdpram_compare \
	    -I$(SMOLLM_SRC) \
	    tb_bfp_sdpram_compare.sv tb_bfp_sdpram_compare.cpp \
	    $(SMOLLM_SRC)/bfp_sdpram.sv \
	    -o Vtb_bfp_sdpram_compare

# ---------------------- bfp-freeze (logic-analyser freeze demo) ----------------------
# Drives smollm_multilayer_tm_bfp through tb_freeze_bfp and demonstrates the
# LA freeze: counter-match (snap_layer_sel/step) or absolute-cycle (trig_cyc).
# Weights come from the sim_shadow $readmemh set (LBFP_STREAMER_SELFTEST); note
# sim_shadow holds one layer's weights, so this validates the freeze CONTROL
# (halts at the right layer/cycle, dbg_frozen/dbg_cyc/dbg_cur_layer) — a golden
# per-layer VALUE compare needs the stream path (tb_autoregress_bfp_stream) and
# the lbfp_freeze_*.hex goldens (task #8).
#   BFP_FREEZE_MODE=counter FREEZE_LAYER=3 FREEZE_POS=0 make bfp-freeze
#   BFP_FREEZE_MODE=cycle   FREEZE_CYC=50000           make bfp-freeze
bfp-freeze: obj_dir/Vtb_freeze_bfp
	@echo "=== running tb_freeze_bfp (LA freeze demo) ==="
	@./obj_dir/Vtb_freeze_bfp

obj_dir/Vtb_freeze_bfp: tb_freeze_bfp.sv tb_freeze_bfp.cpp mock_axi_slave.sv \
                        $(SMOLLM_SRC)/smollm_multilayer_tm_bfp.sv \
                        $(SMOLLM_SRC)/smollm_layer_bfp.sv \
                        $(SMOLLM_SRC)/weight_streamer_bfp_mt.sv \
                        $(SMOLLM_SRC)/matvec_bfp_engine.sv \
                        $(SMOLLM_SRC)/rmsnorm_bfp.sv \
                        $(SMOLLM_SRC)/rope_bfp.sv \
                        $(SMOLLM_SRC)/swiglu_bfp.sv \
                        $(SMOLLM_SRC)/softmax_bfp.sv \
                        $(SMOLLM_SRC)/residual_bfp.sv \
                        $(SMOLLM_SRC)/cordic_sincos.sv
	$(VERILATOR) -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal -cc --exe --build \
	    -j 0 --x-assign 0 --x-initial 0 \
	    -CFLAGS "-O2 -std=c++17" \
	    --top-module tb_freeze_bfp \
	    -I../generated -I$(SMOLLM_SRC) \
	    +define+LBFP_WBEAT_DUMP \
	    tb_freeze_bfp.sv tb_freeze_bfp.cpp mock_axi_slave.sv \
	    $(SMOLLM_SRC)/smollm_multilayer_tm_bfp.sv \
	    $(SMOLLM_SRC)/smollm_layer_bfp.sv \
	    $(SMOLLM_SRC)/weight_streamer_bfp_mt.sv \
	    $(SMOLLM_SRC)/matvec_bfp_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm_bfp.sv \
	    $(SMOLLM_SRC)/rope_bfp.sv \
	    $(SMOLLM_SRC)/swiglu_bfp.sv \
	    $(SMOLLM_SRC)/softmax_bfp.sv \
	    $(SMOLLM_SRC)/residual_bfp.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    -o Vtb_freeze_bfp

obj_dir/Vtb_smollm_layer_bfp: tb_smollm_layer_bfp.sv tb_smollm_layer_bfp.cpp \
                              $(SMOLLM_SRC)/smollm_layer_bfp.sv \
                              $(SMOLLM_SRC)/matvec_bfp_engine.sv \
                              $(SMOLLM_SRC)/rmsnorm_bfp.sv \
                              $(SMOLLM_SRC)/rope_bfp.sv \
                              $(SMOLLM_SRC)/swiglu_bfp.sv \
                              $(SMOLLM_SRC)/softmax_bfp.sv \
                              $(SMOLLM_SRC)/residual_bfp.sv \
                              $(SMOLLM_SRC)/cordic_sincos.sv
	$(VERILATOR) -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal -cc --exe --build \
	    -j 0 --x-assign 0 --x-initial 0 \
	    -CFLAGS "-O2 -std=c++17" \
	    --top-module tb_smollm_layer_bfp \
	    --trace \
	    -I../generated -I$(SMOLLM_SRC) \
	    +define+LBFP_STAGE_DUMP \
	    +define+LBFP_STREAMER_SELFTEST \
	    tb_smollm_layer_bfp.sv tb_smollm_layer_bfp.cpp \
	    $(SMOLLM_SRC)/smollm_layer_bfp.sv \
	    $(SMOLLM_SRC)/matvec_bfp_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm_bfp.sv \
	    $(SMOLLM_SRC)/rope_bfp.sv \
	    $(SMOLLM_SRC)/swiglu_bfp.sv \
	    $(SMOLLM_SRC)/softmax_bfp.sv \
	    $(SMOLLM_SRC)/residual_bfp.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    -o Vtb_smollm_layer_bfp

# ---------------------- tb_full_bfp (full SmolLM2 BFP token loop) ----------------------
# 30 layers, real HuggingFace weights, decode head, embed lookup —
# autoregressive token generation in Verilator.  Compares each emitted
# token to the Python BFP golden (lbfp_full_GOLDEN_TOKENS.txt baked by
# host/gen_smollm_blockfp_full.py).  Loads ~600 MB of $readmemh data;
# expect ~30-90 seconds init + ~30-60 seconds per token step.
bfp-full: obj_dir/Vtb_full_bfp
	@echo "=== running tb_full_bfp (SmolLM2-135M BFP autoregress) ==="
	@./obj_dir/Vtb_full_bfp

obj_dir/Vtb_full_bfp: tb_full_bfp.sv tb_full_bfp.cpp \
                     $(SMOLLM_SRC)/embed_lookup_bfp.sv \
                     $(SMOLLM_SRC)/smollm_multilayer_tm_bfp.sv \
                     $(SMOLLM_SRC)/smollm_decode_head_bfp.sv \
                     $(SMOLLM_SRC)/smollm_layer_bfp.sv \
                     $(SMOLLM_SRC)/matvec_bfp_engine.sv \
                     $(SMOLLM_SRC)/rmsnorm_bfp.sv \
                     $(SMOLLM_SRC)/rope_bfp.sv \
                     $(SMOLLM_SRC)/swiglu_bfp.sv \
                     $(SMOLLM_SRC)/softmax_bfp.sv \
                     $(SMOLLM_SRC)/residual_bfp.sv \
                     $(SMOLLM_SRC)/cordic_sincos.sv \
                     $(SMOLLM_SRC)/bfp_sdpram.sv \
                     $(SMOLLM_SRC)/bfp_sdpram_packed.sv \
                     $(SMOLLM_SRC)/bfp_sdpram_packed_pr.sv \
                     $(SMOLLM_SRC)/bfp_sdpram_striped.sv
	$(VERILATOR) -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal -cc --exe --build \
	    -j 0 --x-assign 0 --x-initial 0 \
	    -CFLAGS "-O2 -std=c++17" \
	    --top-module tb_full_bfp \
	    -I../generated -I$(SMOLLM_SRC) \
	    tb_full_bfp.sv tb_full_bfp.cpp \
	    $(SMOLLM_SRC)/embed_lookup_bfp.sv \
	    $(SMOLLM_SRC)/smollm_multilayer_tm_bfp.sv \
	    $(SMOLLM_SRC)/smollm_decode_head_bfp.sv \
	    $(SMOLLM_SRC)/smollm_layer_bfp.sv \
	    $(SMOLLM_SRC)/matvec_bfp_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm_bfp.sv \
	    $(SMOLLM_SRC)/rope_bfp.sv \
	    $(SMOLLM_SRC)/swiglu_bfp.sv \
	    $(SMOLLM_SRC)/softmax_bfp.sv \
	    $(SMOLLM_SRC)/residual_bfp.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    $(SMOLLM_SRC)/bfp_sdpram.sv \
	    $(SMOLLM_SRC)/bfp_sdpram_packed.sv \
	    $(SMOLLM_SRC)/bfp_sdpram_packed_pr.sv \
	    $(SMOLLM_SRC)/bfp_sdpram_striped.sv \
	    -o Vtb_full_bfp

# ---------------------- tb_autoregress_bfp -> alias to streaming -----------------------
# The BRAM-mode tb was retired with STREAM_WEIGHTS=0; the streaming
# testbench (mock_axi_slave-backed) is the only autoregress harness.
bfp-autoregress: bfp-autoregress-stream

# ---------------------- tb_autoregress_bfp_stream (streaming end-to-end) ----------------------
# Same DUT but with STREAM_WEIGHTS=1 / STREAM_LOOKUP=1; weights served
# from ../generated/lbfp_full_DDR3.hex via mock_axi_slave.  Tokens must
# match the BRAM-mode bfp-autoregress run.  Prereq:
#   LBFP_EMIT_HEX=1 python3 ../host/gen_smollm_blockfp_ddr.py
# (≈ 670 MB hex; not generated by default).
bfp-autoregress-stream: obj_dir/Vtb_autoregress_bfp_stream ../generated/lbfp_full_DDR3.hex
	@echo "=== running tb_autoregress_bfp_stream ==="
	@./obj_dir/Vtb_autoregress_bfp_stream | tee /tmp/vbfp_autoregress_stream.out
	@echo "=== decoded text ==="
	@python3 ../host/decode_rtl_tokens.py /tmp/vbfp_autoregress_stream.out

../generated/lbfp_full_DDR3.hex:
	@echo "ERROR: $@ missing.  Run:" >&2
	@echo "  LBFP_EMIT_HEX=1 python3 ../host/gen_smollm_blockfp_ddr.py" >&2
	@false

obj_dir/Vtb_autoregress_bfp_stream: tb_autoregress_bfp_stream.sv tb_autoregress_bfp_stream.cpp \
                                    mock_axi_slave.sv \
                                    $(SMOLLM_SRC)/autoregress_bfp_top.sv \
                                    $(SMOLLM_SRC)/embed_lookup_bfp.sv \
                                    $(SMOLLM_SRC)/smollm_multilayer_tm_bfp.sv \
                                    $(SMOLLM_SRC)/smollm_decode_head_bfp.sv \
                                    $(SMOLLM_SRC)/smollm_layer_bfp.sv \
                                    $(SMOLLM_SRC)/weight_streamer_bfp_mt.sv \
                                    $(SMOLLM_SRC)/matvec_bfp_engine.sv \
                                    $(SMOLLM_SRC)/rmsnorm_bfp.sv \
                                    $(SMOLLM_SRC)/rope_bfp.sv \
                                    $(SMOLLM_SRC)/swiglu_bfp.sv \
                                    $(SMOLLM_SRC)/softmax_bfp.sv \
                                    $(SMOLLM_SRC)/residual_bfp.sv \
                                    $(SMOLLM_SRC)/cordic_sincos.sv \
                                    $(SMOLLM_SRC)/bfp_sdpram.sv \
                                    $(SMOLLM_SRC)/bfp_sdpram_packed.sv \
                                    $(SMOLLM_SRC)/bfp_sdpram_packed_pr.sv \
                                    $(SMOLLM_SRC)/bfp_sdpram_striped.sv
	$(VERILATOR) -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal -cc --exe --build \
	    -j 0 --x-assign 0 --x-initial 0 \
	    -CFLAGS "-O2 -std=c++17" \
	    --top-module tb_autoregress_bfp_stream \
	    -I../generated -I$(SMOLLM_SRC) \
	    $(if $(LBFP_DUMP_LAYER),+define+LBFP_STAGE_DUMP +define+LBFP_DUMP_LAYER=$(LBFP_DUMP_LAYER),) \
	    tb_autoregress_bfp_stream.sv tb_autoregress_bfp_stream.cpp \
	    mock_axi_slave.sv \
	    $(SMOLLM_SRC)/autoregress_bfp_top.sv \
	    $(SMOLLM_SRC)/embed_lookup_bfp.sv \
	    $(SMOLLM_SRC)/smollm_multilayer_tm_bfp.sv \
	    $(SMOLLM_SRC)/smollm_decode_head_bfp.sv \
	    $(SMOLLM_SRC)/smollm_layer_bfp.sv \
	    $(SMOLLM_SRC)/weight_streamer_bfp_mt.sv \
	    $(SMOLLM_SRC)/matvec_bfp_engine.sv \
	    $(SMOLLM_SRC)/rmsnorm_bfp.sv \
	    $(SMOLLM_SRC)/rope_bfp.sv \
	    $(SMOLLM_SRC)/swiglu_bfp.sv \
	    $(SMOLLM_SRC)/softmax_bfp.sv \
	    $(SMOLLM_SRC)/residual_bfp.sv \
	    $(SMOLLM_SRC)/cordic_sincos.sv \
	    $(SMOLLM_SRC)/bfp_sdpram.sv \
	    $(SMOLLM_SRC)/bfp_sdpram_packed.sv \
	    $(SMOLLM_SRC)/bfp_sdpram_packed_pr.sv \
	    $(SMOLLM_SRC)/bfp_sdpram_striped.sv \
	    -o Vtb_autoregress_bfp_stream

clean:
	rm -rf obj_dir test_matvec.bin test_rmsnorm.bin test_rope.bin test_swiglu.bin test_softmax.bin
