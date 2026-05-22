# swap_fw.tcl — swap PicoSoC firmware in a BUILT bitstream WITHOUT re-synthesis.
# Because progmem is a directly-instantiated RAMB (clean BRAM tile), we rewrite
# its INIT in the routed checkpoint and re-emit the bitstream (write_bitstream
# only, ~minutes — no place/route).  Verified: the placed RAMB36E1 INIT_00..3F
# equals bin2init.py's packing (identity), so bin2init values drop straight in.
# This is the reliable replacement for the retired data2mem / the picky
# updatemem .mmi flow.
#
# Env:
#   DCP_IN   routed checkpoint
#   FW_BIN   new firmware .bin
#   BIT_OUT  output bitstream
#   CELL     progmem RAMB cell (default: auto-find *progmem*)
set dcp  $::env(DCP_IN)
set fwb  $::env(FW_BIN)
set bito $::env(BIT_OUT)

# pack the new firmware into INIT params (python3 -E: Vivado pollutes PYTHONHOME)
exec python3 -E src/soc/bin2init.py $fwb > /tmp/swap_init.svh

open_checkpoint $dcp
set cell [expr {[info exists ::env(CELL)] ? $::env(CELL) \
               : [lindex [get_cells -hier -filter {REF_NAME =~ RAMB* && NAME =~ *progmem*}] 0]}]
puts "swap_fw: cell=$cell ref=[get_property REF_NAME $cell] fw=$fwb"

set n 0
set fh [open /tmp/swap_init.svh r]
while {[gets $fh line] >= 0} {
    if {[regexp {\.INIT_([0-9A-Fa-f]{2})\((256'h[0-9A-Fa-f]+)\)} $line -> idx val]} {
        set_property INIT_$idx $val $cell
        incr n
    }
}
close $fh
puts "swap_fw: set $n INIT params on $cell"

# read-back check: INIT_00 must now equal the new firmware's first 8 words
puts "swap_fw: INIT_00 now = [get_property INIT_00 $cell]"

write_bitstream -force $bito
puts "swap_fw: wrote $bito"
