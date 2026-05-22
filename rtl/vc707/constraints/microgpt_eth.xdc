## microgpt_eth.xdc — Constraints for vc707_microgpt_eth on VC707
## Two clock domains:
##   eth_clk  (125 MHz) — SGMII PCS/PMA userclk2, runs framing + bridge
##   core_clk (40 MHz) — derived from eth_clk via MMCME2_BASE, runs
##                          microgpt_exact_core. Toggle CDC between domains.

## Buttons
set_property -dict {PACKAGE_PIN AV40 IOSTANDARD LVCMOS18} [get_ports cpu_reset]

## (USB-UART console pins are at the bottom of this file — one shared XDC for
##  every build, no separate conditional picosoc.xdc.)

## LEDs
set_property -dict {PACKAGE_PIN AM39 IOSTANDARD LVCMOS18} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN AN39 IOSTANDARD LVCMOS18} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN AR37 IOSTANDARD LVCMOS18} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN AT37 IOSTANDARD LVCMOS18} [get_ports {led[3]}]
set_property -dict {PACKAGE_PIN AR35 IOSTANDARD LVCMOS18} [get_ports {led[4]}]
set_property -dict {PACKAGE_PIN AP41 IOSTANDARD LVCMOS18} [get_ports {led[5]}]
set_property -dict {PACKAGE_PIN AP42 IOSTANDARD LVCMOS18} [get_ports {led[6]}]
set_property -dict {PACKAGE_PIN AU39 IOSTANDARD LVCMOS18} [get_ports {led[7]}]

## Switches  (sw[0]=enable, sw[1]=reset, others unused)
set_property -dict {PACKAGE_PIN AV30 IOSTANDARD LVCMOS18} [get_ports {sw[0]}]
set_property -dict {PACKAGE_PIN AY33 IOSTANDARD LVCMOS18} [get_ports {sw[1]}]
set_property -dict {PACKAGE_PIN BA31 IOSTANDARD LVCMOS18} [get_ports {sw[2]}]
set_property -dict {PACKAGE_PIN BA32 IOSTANDARD LVCMOS18} [get_ports {sw[3]}]
set_property -dict {PACKAGE_PIN AW30 IOSTANDARD LVCMOS18} [get_ports {sw[4]}]
set_property -dict {PACKAGE_PIN AY30 IOSTANDARD LVCMOS18} [get_ports {sw[5]}]
set_property -dict {PACKAGE_PIN BA30 IOSTANDARD LVCMOS18} [get_ports {sw[6]}]
set_property -dict {PACKAGE_PIN BB31 IOSTANDARD LVCMOS18} [get_ports {sw[7]}]

## Fan
set_property -dict {PACKAGE_PIN BA37 IOSTANDARD LVCMOS18} [get_ports fan_pwm]

## SGMII Ethernet (Marvell 88E1111 PHY)
set_property PACKAGE_PIN AH8 [get_ports sgmii_refclk_p]
set_property PACKAGE_PIN AH7 [get_ports sgmii_refclk_n]
create_clock -period 8.000 -name sgmii_refclk [get_ports sgmii_refclk_p]

set_property PACKAGE_PIN AN2 [get_ports sgmii_txp]
set_property PACKAGE_PIN AN1 [get_ports sgmii_txn]
set_property PACKAGE_PIN AM8 [get_ports sgmii_rxp]
set_property PACKAGE_PIN AM7 [get_ports sgmii_rxn]

## PHY reset + MDIO
set_property -dict {PACKAGE_PIN AJ33 IOSTANDARD LVCMOS18} [get_ports eth_rst_n]
set_false_path -to [get_ports eth_rst_n]
set_property -dict {PACKAGE_PIN AH33 IOSTANDARD LVCMOS18} [get_ports eth_mdc]
set_property -dict {PACKAGE_PIN AK33 IOSTANDARD LVCMOS18} [get_ports eth_mdio]
set_false_path -to [get_ports eth_mdc]

## 200 MHz board oscillator pins are owned by the MIG IP, which generates
## its own pin LOC + IOSTANDARD + create_clock in the IP's internal XDC.
## Don't redeclare them here.

## Clock groups
## - sys_clk (200 MHz) is async to the PCS/PMA-derived clocks.
## - clkout0 = userclk2 = eth_clk (125 MHz), generated inside the PCS/PMA IP.
## - The core MMCM (i_core_mmcm) divides eth_clk down to core_clk (40 MHz);
##   the bridge ↔ core handshake is toggle-CDC, so the two are async.
set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks sgmii_refclk] \
  -group [get_clocks -include_generated_clocks clkout0]

set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks -of_objects [get_pins i_core_mmcm/CLKOUT0]] \
  -group [get_clocks -include_generated_clocks clkout0]

# MIG's ui_clk (clk_pll_i) is async to the SGMII userclk2 (clkout0): we cross
# only via toggle CDC + handshake-stable values, so declare async groups.
set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks clkout0] \
  -group [get_clocks -include_generated_clocks clk_pll_i]

## Configuration bank voltage
set_property CFGBVS GND [current_design]
set_property CONFIG_VOLTAGE 1.8 [current_design]

## Don't gate DONE on DCI match at startup.  The DDR3 banks use DCI and the
## default startup waits for DCI match before EOS/DONE — Vivado's programmer
## satisfies that, but a generic JTAG loader (openFPGALoader, used by `make
## program` / `make oload`) does not, so the design never starts (silent UART).
## NoWait lets DONE assert immediately; DCI still calibrates and the MIG's
## init_calib_complete still guards the DDR3 path.  Applies to ALL builds.
set_property BITSTREAM.STARTUP.MATCH_CYCLE NoWait [current_design]

## USB-UART (CP2103) debug console — tx = FPGA->host, rx = host->FPGA.  Only the
## PicoSoC tops expose these ports; on the plain engine they don't exist, so the
## -quiet get_ports just returns empty and these lines no-op (one shared XDC for
## every build, instead of a conditionally-added picosoc.xdc).  No UCIO-1: that
## only fires for ports that exist but are unconstrained.
set_property -dict {PACKAGE_PIN AU36 IOSTANDARD LVCMOS18} [get_ports -quiet usb_uart_tx]
set_property -dict {PACKAGE_PIN AU33 IOSTANDARD LVCMOS18} [get_ports -quiet usb_uart_rx]
