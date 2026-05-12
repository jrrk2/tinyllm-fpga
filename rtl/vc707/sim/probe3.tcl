open_checkpoint ../microgpt_eth.runs/synth_1/vc707_microgpt_eth.dcp

# Inspect each bit of i_lay_st/i_ml/i_layer/mlp_out_factor[*] — what drives it.
for {set i 0} {$i < 24} {incr i} {
    set net [get_nets -hier -filter "NAME == i_lay_st/i_ml/i_layer/mlp_out_factor\[$i\]"]
    if {[llength $net] == 0} {
        puts "bit $i : no net"
        continue
    }
    set drv_pins [get_pins -of $net -leaf -filter {DIRECTION == OUT}]
    set drv_ports [get_ports -of $net]
    set type [get_property TYPE $net]
    set route [get_property ROUTE_STATUS $net]
    puts "bit $i : net=$net  TYPE=$type  drivers=[concat $drv_pins $drv_ports]"
}

# Also check the equivalent net at the multilayer's cur_sg_mlp_out_factor output
puts "\n--- cur_sg_mlp_out_factor (multilayer side) ---"
set ml_nets [get_nets -hier -filter {NAME =~ "i_lay_st/i_ml/cur_sg_mlp_out_factor*"}]
foreach n $ml_nets {
    set drv_pins [get_pins -of $n -leaf -filter {DIRECTION == OUT}]
    set type [get_property TYPE $n]
    puts "  $n  TYPE=$type  drivers=$drv_pins"
}
