# engine_mmi.tcl — extract the progmem BRAM memory-map (.mmi) from an ALREADY
# routed checkpoint, so firmware can be spliced into the matching .bit via
# updatemem WITHOUT re-running synthesis/implementation.  Use this for a
# bitstream that was built before run.tcl emitted the .mmi itself.
#
# Env:
#   DCP_IN   routed checkpoint (default: impl_1 routed dcp)
#   MMI_OUT  output .mmi        (default: picosoc_engine.mmi)
set dcp [expr {[info exists ::env(DCP_IN)]  ? $::env(DCP_IN)  : "microgpt_eth.runs/impl_1/vc707_microgpt_eth_routed.dcp"}]
set mmi [expr {[info exists ::env(MMI_OUT)] ? $::env(MMI_OUT) : "picosoc_engine.mmi"}]
if {![file exists $dcp]} { error "ERROR: routed checkpoint not found: $dcp" }
open_checkpoint $dcp
write_mem_info -force $mmi
puts "INFO: wrote $mmi from $dcp"
close_design
