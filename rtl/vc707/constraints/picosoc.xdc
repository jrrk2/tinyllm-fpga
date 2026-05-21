# picosoc.xdc — constraints specific to vc707_picosoc_shell, added ONLY by the
# PICOSOC=1 build (run.tcl).  These ports always exist in the shell, so the
# constraints are unconditional — no fragile conditional-XDC logic.

## USB-UART (CP2103) debug console.  tx = FPGA->host, rx = host->FPGA.
set_property -dict {PACKAGE_PIN AU36 IOSTANDARD LVCMOS18} [get_ports usb_uart_tx]
set_property -dict {PACKAGE_PIN AU33 IOSTANDARD LVCMOS18} [get_ports usb_uart_rx]
