open_hw_manager
connect_hw_server -url localhost:3121

set device_name xc7vx485t_0

foreach target [get_hw_targets] {
  open_hw_target $target
  if {[llength [get_hw_devices $device_name]] > 0} {
    break
  }
  close_hw_target
}

current_hw_device [get_hw_devices $device_name]
set_property PROGRAM.FILE {work/vc707_microgpt_eth.bit} [current_hw_device]
program_hw_devices [current_hw_device]
refresh_hw_device [current_hw_device]

# Clean teardown so cs_server/hw_server don't linger as CPU hogs.
close_hw_target
disconnect_hw_server
close_hw_manager
