# swap_fw.tcl — swap PicoSoC firmware in a BUILT bitstream WITHOUT re-synthesis.
# progmem is 4 directly-instantiated RAMB16_S9_S9 byte lanes (retargeted to
# RAMB18E1, 2048 x 8 each, see src/soc/progmem_bram.v).  We rewrite each lane's
# INIT_00..INIT_3F in the routed checkpoint from bin2init.py's per-lane
# fragments and re-emit the bitstream (write_bitstream only, ~minutes — no
# place/route).  bin2init.py's packing is exactly the RAMB18E1 INIT layout for
# an 8-bit port (byte address xx*32+k at INIT_xx[k*8 +: 8]), identical to what
# synthesis bakes from the same fragments.
#
# Env:
#   DCP_IN   routed checkpoint
#   FW_BIN   new firmware .bin
#   BIT_OUT  output bitstream
#   LANE_DIR where to put the generated lane fragments (default /tmp/swap_fw)
set dcp  $::env(DCP_IN)
set fwb  $::env(FW_BIN)
set bito $::env(BIT_OUT)
set ldir [expr {[info exists ::env(LANE_DIR)] ? $::env(LANE_DIR) : "/tmp/swap_fw"}]
file mkdir $ldir

# pack the new firmware into per-lane INIT params (python3 -E: Vivado pollutes
# PYTHONHOME; bin2init reports on stderr, which must not abort exec)
exec python3 -E src/soc/bin2init.py $fwb $ldir 2>@stderr

open_checkpoint $dcp

set total 0
for {set lane 0} {$lane < 4} {incr lane} {
    set cells [get_cells -hier -filter "REF_NAME =~ RAMB18* && NAME =~ *progmem*lane$lane*"]
    if {[llength $cells] != 1} {
        error "swap_fw: expected exactly one RAMB18 cell for progmem lane$lane, got: $cells"
    }
    set cell [lindex $cells 0]
    puts "swap_fw: lane$lane cell=$cell ref=[get_property REF_NAME $cell]"
    puts "swap_fw:   INIT_00 before = [get_property INIT_00 $cell]"
    set n 0
    set fh [open [file join $ldir "progmem_init_lane$lane.svh"] r]
    while {[gets $fh line] >= 0} {
        if {[regexp {\.INIT_([0-9A-Fa-f]{2})\((256'h[0-9A-Fa-f]+)\)} $line -> idx val]} {
            set_property INIT_$idx $val $cell
            incr n
        }
    }
    close $fh
    puts "swap_fw:   set $n INIT params; INIT_00 after = [get_property INIT_00 $cell]"
    incr total $n
}
if {$total != 256} { error "swap_fw: expected 256 INIT params in total, set $total" }

write_bitstream -force $bito
puts "swap_fw: wrote $bito (fw=$fwb)"
