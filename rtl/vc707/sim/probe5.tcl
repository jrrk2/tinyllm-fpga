open_checkpoint ../microgpt_eth.runs/synth_1/vc707_microgpt_eth.dcp

# What is `Q[3]` inside factor_ram?  Find its driver.
set q_net [get_nets -hier -filter "NAME == i_lay_st/i_ml/i_factor_ram/Q\[3\]"]
puts "Q\[3\] net: $q_net"
set drv [get_pins -of $q_net -leaf -filter {DIRECTION == OUT}]
puts "  driver pin: $drv"
foreach d $drv {
    set c [get_cells -of $d]
    puts "  driver cell: $c  ref: [get_property REF_NAME $c]"
}

# Show some context — all Q[*] nets in factor_ram
set qs [get_nets -hier -filter "NAME =~ i_lay_st/i_ml/i_factor_ram/Q\\\[*\\\]"]
puts "\nQ\[*\] nets in factor_ram: [llength $qs]"
foreach q [lsort $qs] {
    set drv [get_pins -of $q -leaf -filter {DIRECTION == OUT}]
    set drv_cells {}
    foreach d $drv { lappend drv_cells [get_cells -of $d] }
    puts "  $q ← $drv_cells"
}

# Count FDREs that have name suggesting mlp_out_factor_ram (per-layer per-bit storage)
set mlp_ffs [get_cells -hier -filter {NAME =~ "*mlp_out_factor_ram*" && REF_NAME == FDRE}]
puts "\nmlp_out_factor_ram FDREs: [llength $mlp_ffs]"
# Expected: 30 layers * 24 bits = 720 if Vivado kept them all
