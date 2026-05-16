# write_cfgmem.tcl — generate a flash-image (.mcs) from a bitstream (.bit).
#
# Usage:  vivado -mode batch -source scripts/write_cfgmem.tcl \
#               -tclargs work/vc707_microgpt_eth.mcs work/vc707_microgpt_eth.bit
#
# Modelled on cva6/corev_apu/fpga/scripts/write_cfgmem.tcl.
# VC707 carries a 128 MB BPI x16 flash (Micron MT28GU01GAAX1E).
#
# Reference: https://scholar.princeton.edu/jbalkind/blog/programming-vc707-virtex-7-bpi-flash

if {$argc < 2} {
    puts "Error: usage: write_cfgmem.tcl <mcsfile> <bitfile>"
    exit 1
}

lassign $argv mcsfile bitfile

if {![file exists $bitfile]} {
    puts "Error: bitfile not found: $bitfile"
    exit 1
}

# VC707 BPI x16, 128 Mbit (16 MB).  -size is in megabits.
write_cfgmem -format mcs -interface bpix16 -size 128 \
             -loadbit "up 0x0 $bitfile" -file $mcsfile -force

puts "[write_cfgmem] wrote $mcsfile from $bitfile"
