# picosoc_uart.xdc — VC707 USB-UART (CP2103) pins for the PicoSoC debug console.
#
# !!! VERIFY these PACKAGE_PINs against your VC707 master XDC (UG885 / the
#     board files) before building — wrong pins = no UART and a wasted build.
#     The names below are FPGA-direction (tx = FPGA->host, rx = host->FPGA).
#     On the VC707 these are the bank-VADJ 1.8 V CP2103 lines.
set_property -dict {PACKAGE_PIN AU36 IOSTANDARD LVCMOS18} [get_ports usb_uart_tx]
set_property -dict {PACKAGE_PIN AU33 IOSTANDARD LVCMOS18} [get_ports usb_uart_rx]
