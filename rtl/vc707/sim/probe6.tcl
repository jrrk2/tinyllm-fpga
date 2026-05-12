open_checkpoint ../microgpt_eth.runs/synth_1/vc707_microgpt_eth.dcp

# Count FFs in i_factor_ram subhierarchy regardless of name.
set ffs [get_cells -hier -filter {NAME =~ "i_lay_st/i_ml/i_factor_ram/*" && REF_NAME == FDRE}]
puts "All FDREs under i_factor_ram: [llength $ffs]"

set lutrams [get_cells -hier -filter {NAME =~ "i_lay_st/i_ml/i_factor_ram/*" && REF_NAME =~ "RAM*"}]
puts "RAM cells under i_factor_ram: [llength $lutrams]"

set luts [get_cells -hier -filter {NAME =~ "i_lay_st/i_ml/i_factor_ram/*" && REF_NAME =~ "LUT*"}]
puts "LUT cells under i_factor_ram: [llength $luts]"

puts "---"
puts "Expected: 30 layers * (16+16+24+24) = 2400 FFs storage + 80 output regs = 2480"

# Sample FF names so I can see naming convention
puts "\nFirst 10 FF names:"
foreach f [lrange $ffs 0 9] { puts "  $f" }
