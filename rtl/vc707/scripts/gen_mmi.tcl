# gen_mmi.tcl — build an updatemem .mmi for the directly-instantiated PicoSoC
# progmem by LISTING its BRAM tile(s) in a routed checkpoint.  No processor
# address space is needed (that is what blocked write_mem_info); this is the
# data2mem/bmm_gen replacement: a hand-built memory map referencing the RAMB by
# placement.  Geometry is known from the RTL: RAMB16_S36_S36 -> one RAMB36 tile,
# 32 data bits (0..31), 512 words (0..511).
#
# Env:
#   DCP_IN   routed checkpoint (default: shell impl_1 routed dcp)
#   MMI_OUT  output .mmi        (default: picosoc_shell.mmi)
#   PART     device part        (default: queried from the design)
set dcp  [expr {[info exists ::env(DCP_IN)]  ? $::env(DCP_IN)  : "microgpt_eth.runs/impl_1/vc707_picosoc_shell_routed.dcp"}]
set mmi  [expr {[info exists ::env(MMI_OUT)] ? $::env(MMI_OUT) : "picosoc_shell.mmi"}]

open_checkpoint $dcp
set part [get_property PART [current_design]]

# Find the progmem block-RAM tile(s).
set cells [get_cells -hier -filter {REF_NAME =~ RAMB* && NAME =~ *progmem*}]
if {[llength $cells] == 0} { error "gen_mmi: no progmem RAMB cells found in $dcp" }

# Use the parent of the RAMB as the InstPath (updatemem identifies the memory by
# this path); take the first tile's hierarchy.
set first [lindex $cells 0]
set instpath [regsub {/[^/]+$} $first ""]   ;# strip the leaf cell name

set fh [open $mmi w]
puts $fh "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
puts $fh "<MemInfo Version=\"1\" Minor=\"0\">"
puts $fh "  <Processor Endianness=\"Little\" InstPath=\"$instpath\">"
puts $fh "    <AddressSpace Name=\"progmem\" Begin=\"0\" End=\"2047\">"
puts $fh "      <BusBlock>"
foreach c [lsort $cells] {
    set loc [get_property LOC $c]
    regexp {RAMB(18|36)_(X\d+Y\d+)} $loc -> kind site
    set memtype "RAMB$kind"
    puts $fh "        <BitLane MemType=\"$memtype\" Placement=\"$site\">"
    puts $fh "          <DataWidth MSB=\"31\" LSB=\"0\"/>"
    puts $fh "          <AddressRange Begin=\"0\" End=\"511\"/>"
    puts $fh "          <Parity ON=\"false\" NumBits=\"0\"/>"
    puts $fh "        </BitLane>"
    puts "gen_mmi: $c  $memtype  $site"
}
puts $fh "      </BusBlock>"
puts $fh "    </AddressSpace>"
puts $fh "  </Processor>"
puts $fh "  <Config>"
puts $fh "    <Option Name=\"Part\" Value=\"$part\"/>"
puts $fh "  </Config>"
puts $fh "</MemInfo>"
close $fh
puts "gen_mmi: wrote $mmi (InstPath=$instpath, part=$part)"
