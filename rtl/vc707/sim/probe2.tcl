open_checkpoint ../microgpt_eth.runs/synth_1/vc707_microgpt_eth.dcp

set pins [get_pins -hier -filter {NAME =~ "*i_sg/mlp_out_factor[*"}]
puts "swiglu mlp_out_factor pins: [llength $pins]"

# What's driving each — could be a constant net.
foreach p $pins {
    set n [get_nets -of $p]
    set type [get_property TYPE $n]
    set name [get_property NAME $n]
    puts "  $p ← net=$name  TYPE=$type"
}

# Check whether they connect to ground/vcc
puts "\nTie-offs:"
set gnd_nets [get_nets -filter {TYPE == GROUND}]
set vcc_nets [get_nets -filter {TYPE == POWER}]
foreach p $pins {
    set n [get_nets -of $p]
    set name [get_property NAME $n]
    if {$name == "GND" || $name == "vcc"} {
        puts "  $p tied to $name"
    }
}
