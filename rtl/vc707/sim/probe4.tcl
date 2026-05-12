open_checkpoint ../microgpt_eth.runs/synth_1/vc707_microgpt_eth.dcp

# Inspect result_w_i_24 — the LUT driving mlp_out_factor[0].
set lut [get_cells -hier -filter {NAME == i_lay_st/i_ml/i_factor_ram/result_w_i_24}]
puts "LUT cell: $lut"
puts "REF_NAME: [get_property REF_NAME $lut]"
puts "INIT:     [get_property INIT $lut]"

# Its input pins
foreach p [get_pins -of $lut -filter {DIRECTION == IN}] {
    set n [get_nets -of $p]
    set nname [get_property NAME $n]
    puts "  $p ← $nname"
}

# Same for a couple more
foreach n {result_w_i_23 result_w_i_22 result_w_i_1 result_w_i_2} {
    set lut [get_cells -hier -filter "NAME == i_lay_st/i_ml/i_factor_ram/$n"]
    if {[llength $lut] == 0} continue
    puts "\n$n:"
    puts "  INIT: [get_property INIT $lut]"
    foreach p [get_pins -of $lut -filter {DIRECTION == IN}] {
        set net [get_nets -of $p]
        puts "  $p ← [get_property NAME $net]"
    }
}
