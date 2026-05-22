# picosoc.xdc — constraints for the PicoSoC builds, added by run.tcl for both the
# shell (PICOSOC=1) and the engine front-end (PICOSOC_ENGINE=1).  These ports
# always exist in those tops, so the constraints are unconditional.

## USB-UART (CP2103) debug console.  tx = FPGA->host, rx = host->FPGA.
set_property -dict {PACKAGE_PIN AU36 IOSTANDARD LVCMOS18} [get_ports usb_uart_tx]
set_property -dict {PACKAGE_PIN AU33 IOSTANDARD LVCMOS18} [get_ports usb_uart_rx]
## (BITSTREAM.STARTUP.MATCH_CYCLE NoWait — for openFPGALoader — now lives in
##  microgpt_eth.xdc so it applies to every build, not just the PicoSoC ones.)
