#!/bin/bash
# run_xsim.sh — compile + run multilayer_tm under Vivado xsim with the
# REAL Xilinx unisim primitives (RAMB36E1 incl. all init/timing
# semantics, DSP48E1, etc.).  Validates that the FPGA-vs-Verilator
# divergence is/isn't a stub bug.

set -e
cd "$(dirname "$0")"

# Skip the heavy settings64.sh — just put Xilinx tools on PATH directly.
export PATH=/home/Xilinx/Vivado/2020.1/bin:$PATH
export XILINX_VIVADO=/home/Xilinx/Vivado/2020.1
export LD_LIBRARY_PATH=$XILINX_VIVADO/lib/lnx64.o:$XILINX_VIVADO/tps/lnx64/python-3.8.3/lib:${LD_LIBRARY_PATH:-}

SMOLLM_SRC=../src/smollm
GEN=../../generated

# Common flags
XVLOG_FLAGS="-sv -i $GEN -d MICROGPT_WEIGHT_DIR=\"$GEN\" -d MICROGPT_DDR3_WEIGHTS"

# Compile sources (DUT + helpers + brom_*.sv generated wrappers)
echo "=== xvlog compile ==="
xvlog /home/Xilinx/Vivado/2020.1/data/verilog/src/glbl.v
xvlog $XVLOG_FLAGS \
    tb_xsim_top.sv \
    mock_axi_slave.sv \
    $SMOLLM_SRC/factor_ram.sv \
    $SMOLLM_SRC/smollm_multilayer_tm.sv \
    $SMOLLM_SRC/smollm_layer.sv \
    $SMOLLM_SRC/weight_streamer_mt.sv \
    $SMOLLM_SRC/matvec_int8_engine.sv \
    $SMOLLM_SRC/rmsnorm.sv \
    $SMOLLM_SRC/rope.sv \
    $SMOLLM_SRC/cordic_sincos.sv \
    $SMOLLM_SRC/swiglu.sv \
    $SMOLLM_SRC/softmax_q15.sv \
    $GEN/brom_GAMMA1.sv $GEN/brom_GAMMA2.sv \
    $GEN/brom_SCALE_Q.sv $GEN/brom_SCALE_K.sv $GEN/brom_SCALE_V.sv \
    $GEN/brom_SCALE_O.sv $GEN/brom_SCALE_GATE.sv $GEN/brom_SCALE_UP.sv \
    $GEN/brom_SCALE_DOWN.sv

# Elaborate with the Xilinx libs so RAMB36E1 / DSP48E1 resolve to real
# unisim cells (not Verilator stubs).
echo "=== xelab ==="
xelab -L unisims_ver -L secureip -L unimacro_ver \
      --debug typical -timescale 1ns/1ps \
      tb_xsim_top glbl -s tb_xsim_top.exe

# Run
echo "=== xsim run ==="
xsim tb_xsim_top.exe -runall
