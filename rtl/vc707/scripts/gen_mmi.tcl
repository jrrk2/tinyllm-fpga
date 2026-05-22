# gen_mmi.tcl — emit an updatemem .mmi for the directly-instantiated PicoSoC
# progmem by LISTING its BRAM tile in a routed checkpoint.  No write_mem_info /
# processor address space needed (that is what blocked the auto path); this is
# the hand-built memory map.  Vivado 2020.1 format (verified working):
#   * <Option>/<Rule> use Val= (NOT Value=)
#   * NO <AddressSpaceRange>/<BitLayout> (those are Versal-era and are rejected)
#   * AddressSpace End in BYTES; BitLane AddressRange in WORDS
# The matching `updatemem -data` file needs a leading "@0" address line.
#
# Env:  DCP_IN (routed checkpoint), MMI_OUT (.mmi path)
set dcp [expr {[info exists ::env(DCP_IN)]  ? $::env(DCP_IN)  : "microgpt_eth.runs/impl_1/vc707_picosoc_shell_routed.dcp"}]
set mmi [expr {[info exists ::env(MMI_OUT)] ? $::env(MMI_OUT) : "picosoc_shell.mmi"}]

open_checkpoint $dcp
set part [get_property PART [current_design]]
set cells [get_cells -hier -filter {REF_NAME =~ RAMB* && NAME =~ *progmem*}]
if {[llength $cells] == 0} { error "gen_mmi: no progmem RAMB cells found in $dcp" }
set first    [lindex $cells 0]
set instpath [regsub {/[^/]+$} $first ""]   ;# strip leaf -> e.g. soc/progmem

set fh [open $mmi w]
puts $fh "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
puts $fh "<MemInfo Version=\"1\" Minor=\"9\">"
puts $fh "  <Processor Endianness=\"Little\" InstPath=\"$instpath\">"
puts $fh "    <AddressSpace Name=\"$instpath\" Begin=\"0\" End=\"2047\">"
puts $fh "      <BusBlock>"
foreach c [lsort $cells] {
    regexp {RAMB(18|36)_(X\d+Y\d+)} [get_property LOC $c] -> kind site
    puts $fh "        <BitLane MemType=\"RAMB$kind\" Placement=\"$site\">"
    puts $fh "          <DataWidth MSB=\"31\" LSB=\"0\"/>"
    puts $fh "          <AddressRange Begin=\"0\" End=\"511\"/>"
    puts $fh "          <Parity ON=\"false\" NumBits=\"0\"/>"
    puts $fh "        </BitLane>"
    puts "gen_mmi: $c  RAMB$kind  $site"
}
puts $fh "      </BusBlock>"
puts $fh "    </AddressSpace>"
puts $fh "  </Processor>"
puts $fh "  <Config><Option Name=\"Part\" Val=\"$part\"/></Config>"
puts $fh "  <DRC><Rule Name=\"RDADDRCHANGE\" Val=\"false\"/></DRC>"
puts $fh "</MemInfo>"
close $fh
puts "gen_mmi: wrote $mmi (InstPath=$instpath, part=$part)"
