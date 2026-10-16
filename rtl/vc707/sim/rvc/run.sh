#!/usr/bin/env bash
# Build a compressed (rv32imc) and a control (rv32im) image of the same test.c,
# run both on the SAME COMPRESSED_ISA=1 PicoRV32 in iverilog.  If rv32im passes
# the milestones but rv32imc hangs/traps -> the bug is in compressed decode.
# If BOTH pass -> the CPU logic is fine and the FPGA failure is hardware-specific
# (the RAMB36 progmem or the firmware-load path), not the core.
set -e
cd "$(dirname "$0")"
GCC=riscv64-unknown-elf-gcc
OC=riscv64-unknown-elf-objcopy
CF="-mabi=ilp32 -Os -ffreestanding -nostdlib -nostartfiles -T link.ld"
PICORV=../../src/soc/picorv32.v

build() {  # $1 = march, $2 = tag
    $GCC -march=$1 $CF -o test_$2.elf start.S test.c
    $OC -O binary test_$2.elf test_$2.bin
    python3 ../../src/soc/bin2mem.py test_$2.bin > test_$2.mem
    echo "built test_$2.mem ($(wc -l < test_$2.mem) words) from -march=$1"
}

build rv32im  im
build rv32imc imc

iverilog -g2012 -o tb_rvc.vvp tb_rvc.v "$PICORV"

echo; echo "================= CONTROL: rv32im (uncompressed) ================="
vvp tb_rvc.vvp +mem=test_im.mem  | grep -vE "^loaded|VCD" | head -60
echo; echo "================= TEST: rv32imc (compressed) ====================="
vvp tb_rvc.vvp +mem=test_imc.mem | grep -vE "^loaded|VCD" | head -60
