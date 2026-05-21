# updatemem.tcl — swap the PicoSoC firmware in a BUILT bitstream without
# re-synthesis (seconds vs ~20 min).  Patches the progmem BRAM contents using
# the .mmi emitted by run.tcl (write_mem_info) and a fresh firmware .mem.
#
# Driven by env (set by the `make fw-update` target):
#   FW_MMI     memory-map info (picosoc_shell.mmi)
#   FW_MEM     new firmware .mem
#   FW_BIT_IN  input bitstream
#   FW_BIT_OUT patched bitstream
#   FW_PROC    (optional) BRAM/address-space instance name from the .mmi; if the
#              default invocation fails, set this from the .mmi <Processor>/<MemoryArray>
#              instance path (e.g. FW_PROC=soc/progmem).

set cmd [list updatemem -force \
    -meminfo $::env(FW_MMI) \
    -data    $::env(FW_MEM) \
    -bit     $::env(FW_BIT_IN) \
    -out     $::env(FW_BIT_OUT)]
if {[info exists ::env(FW_PROC)] && [string trim $::env(FW_PROC)] ne ""} {
    lappend cmd -proc $::env(FW_PROC)
}
puts "INFO: $cmd"
eval $cmd
puts "INFO: firmware patched -> $::env(FW_BIT_OUT)"
