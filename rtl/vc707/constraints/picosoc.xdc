# picosoc.xdc — constraints for the PicoSoC builds, added by run.tcl for both the
# shell (PICOSOC=1) and the engine front-end (PICOSOC_ENGINE=1).  These ports
# always exist in those tops, so the constraints are unconditional.

## USB-UART (CP2103) debug console.  tx = FPGA->host, rx = host->FPGA.
set_property -dict {PACKAGE_PIN AU36 IOSTANDARD LVCMOS18} [get_ports usb_uart_tx]
set_property -dict {PACKAGE_PIN AU33 IOSTANDARD LVCMOS18} [get_ports usb_uart_rx]

## Don't gate DONE on DCI match at startup.  The DDR3 banks use DCI, and the
## default startup waits for DCI match before EOS/DONE — which Vivado's programmer
## satisfies but a generic JTAG loader (openFPGALoader) does not, so the design
## never starts (EOS=0, silent UART).  NoWait lets the SoC come up immediately;
## DCI still calibrates and the MIG's init_calib_complete still guards DDR3.
set_property BITSTREAM.STARTUP.MATCH_CYCLE NoWait [current_design]
