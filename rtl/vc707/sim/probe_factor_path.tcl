# Open the post-synth DCP and inspect the cur_sg_mlp_out_factor path.
open_checkpoint ../microgpt_eth.runs/synth_1/vc707_microgpt_eth.dcp

puts "=========================================================="
puts " Probe cur_sg_mlp_out_factor path in synthesized design"
puts "=========================================================="

# 1. Does factor_ram instance exist?
set fram_inst [get_cells -hier -filter {NAME =~ "*i_factor_ram*"}]
puts "factor_ram instances: [llength $fram_inst]"
foreach c $fram_inst { puts "  $c" }

# 2. cur_sg_mlp_out_factor_r — is it kept?  Who's driving it?
set mlp_nets [get_nets -hier -filter {NAME =~ "*cur_sg_mlp_out_factor*"}]
puts "\ncur_sg_mlp_out_factor* nets: [llength $mlp_nets]"
foreach n $mlp_nets {
    set drv [get_pins -of $n -filter {DIRECTION == OUT}]
    set ld [get_pins -of $n -filter {DIRECTION == IN}]
    puts "  $n  driver=$drv  loads=[llength $ld]"
}

# 3. Loads of cur_sg_mlp_out_factor at the multilayer→layer→swiglu boundary
set swig_mlp_pins [get_pins -hier -filter {NAME =~ "*swiglu*mlp_out_factor*" || NAME =~ "*i_sg*mlp_out_factor*"}]
puts "\nswiglu mlp_out_factor pins: [llength $swig_mlp_pins]"
foreach p $swig_mlp_pins { puts "  $p" }

# 4. Is the mlp_out_factor input to swiglu connected to a constant?
foreach p $swig_mlp_pins {
    set net [get_nets -of $p]
    set drv [get_pins -of $net -filter {DIRECTION == OUT}]
    set drv_cell [get_cells -of $drv]
    set ref [get_property REF_NAME $drv_cell]
    puts "  $p ← $drv  (cell ref: $ref)"
}

puts "=========================================================="
