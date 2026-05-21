# run_picosoc.tcl — Vivado build for the PicoSoC bring-up shell
# (vc707_picosoc_shell).  Slimmed from run.tcl: keeps the SGMII ethernet stack
# + MIG DDR3, drops the inference engine + microgpt core, and uses the PicoSoC
# as top with a dummy engine.  Lets us iterate the SoC/eth/DDR plumbing fast.
#
# Usage:  vivado -mode batch -source scripts/run_picosoc.tcl
# (Build the firmware first: see src/soc/  ->  progmem_shell.v)

set project picosoc_shell

set is_fresh [expr {[llength [get_files -quiet *microgpt_eth.xdc]] == 0}]

if {$is_fresh} {
    add_files -fileset constrs_1 -norecurse constraints/microgpt_eth.xdc

    # IPs: SGMII PCS/PMA (needed by framing) + MIG DDR3.  No ILA IP — trace is
    # exposed via (* mark_debug *) nets; insert an ILA with Vivado "Set Up Debug"
    # (or add create_debug_core here) after synth.
    read_ip { \
        "ip/gig_ethernet_pcs_pma_0/gig_ethernet_pcs_pma_0.srcs/sources_1/ip/gig_ethernet_pcs_pma_0/gig_ethernet_pcs_pma_0.xci" \
    }
    read_ip { \
        "ip/xlnx_mig_7_ddr3/xlnx_mig_7_ddr3.srcs/sources_1/ip/xlnx_mig_7_ddr3/xlnx_mig_7_ddr3.xci" \
    }

    set_property include_dirs [list "../src/include" "src" "src/soc"] [current_fileset]

    # Ethernet MAC + framing (reused unchanged).
    read_verilog -sv { \
        eth/axis_gmii_rx.sv \
        eth/axis_gmii_tx.sv \
        eth/dualmem_widen8.sv \
        eth/dualmem_widen.sv \
        eth/eth_mac_1g.sv \
        eth/framing_top_sgmii.sv \
        eth/rgmii_lfsr.sv \
        eth/sgmii_soc.sv \
    }

    # PicoSoC + shell + firmware ROM.
    read_verilog { \
        src/soc/picorv32.v \
        src/soc/simpleuart.v \
        src/soc/picosoc_noflash.v \
        src/soc/progmem_shell.v \
    }
    read_verilog -sv { \
        src/soc/soc_ddr_bridge.sv \
        src/soc/vc707_picosoc_shell.sv \
    }

    # Board header (`VC707 for the dualmem BRAM config used by framing).
    read_verilog -sv {src/vc707.svh}
    set file_obj [get_files -of_objects [get_filesets sources_1] [list "*src/vc707.svh"]]
    set_property -dict {file_type {Verilog Header} is_global_include 1} -objects $file_obj

    set_property top vc707_picosoc_shell [current_fileset]
}

update_compile_order -fileset sources_1

foreach run_name {synth_1 impl_1} {
    if {[llength [get_runs -quiet $run_name]] > 0} {
        set st [get_property STATUS [get_runs $run_name]]
        if {!([string match -nocase "*Complete*" $st] && ![string match -nocase "*Error*" $st])} {
            puts "INFO: resetting $run_name (was: $st)"
            reset_run $run_name
        }
    }
}

launch_runs synth_1
wait_on_run synth_1
open_run synth_1
exec mkdir -p reports/
report_utilization -hierarchical -file reports/${project}.utilization.rpt
report_timing -nworst 1 -delay_type max -sort_by group -file reports/${project}.timing.rpt

launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
puts "INFO: picosoc_shell bitstream written."
