# Program configuration memory (BPI x16 flash MT28GU01GAAX1E) on VC707.

open_hw_manager
connect_hw_server -url localhost:3121

set device_name xc7vx485t_0
set flash_type mt28gu01gaax1e-bpi-x16

foreach target [get_hw_targets] {
    open_hw_target $target
    if {[llength [get_hw_devices $device_name]] > 0} {
        break
    }
    close_hw_target
}

current_hw_device [get_hw_devices $device_name]
refresh_hw_device -update_hw_probes false [current_hw_device]

# Override via env MCSFILE=<path> (used by `make flash-release`).
set mcsfile [expr {[info exists ::env(MCSFILE)] ? $::env(MCSFILE)
                                                : "work/vc707_microgpt_eth.mcs"}]
puts "flash.tcl: mcsfile = $mcsfile"
create_hw_cfgmem -hw_device [current_hw_device] [lindex [get_cfgmem_parts $flash_type] 0]
set cfgmem [get_property PROGRAM.HW_CFGMEM [current_hw_device]]
set_property PROGRAM.FILES [list $mcsfile] $cfgmem
set_property PROGRAM.PRM_FILE {} $cfgmem
set_property PROGRAM.ADDRESS_RANGE {use_file} $cfgmem
set_property PROGRAM.BLANK_CHECK 0 $cfgmem
set_property PROGRAM.ERASE 1 $cfgmem
set_property PROGRAM.CFG_PROGRAM 1 $cfgmem
set_property PROGRAM.VERIFY 1 $cfgmem
set_property PROGRAM.CHECKSUM 0 $cfgmem
set_property PROGRAM.BPI_RS_PINS {none} $cfgmem
set_property PROGRAM.UNUSED_PIN_TERMINATION {pull-none} $cfgmem

create_hw_bitstream -hw_device [current_hw_device] [get_property PROGRAM.HW_CFGMEM_BITFILE [current_hw_device]]
program_hw_devices [current_hw_device]
refresh_hw_device [current_hw_device]
program_hw_cfgmem -hw_cfgmem $cfgmem

# Clean teardown so cs_server/hw_server don't linger as CPU hogs.
close_hw_target
disconnect_hw_server
close_hw_manager
