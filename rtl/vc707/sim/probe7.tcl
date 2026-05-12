open_checkpoint ../microgpt_eth.runs/synth_1/vc707_microgpt_eth.dcp
set ffs [get_cells -hier -filter {NAME =~ "i_lay_st/i_ml/i_factor_ram/*" && REF_NAME == FDRE}]
puts "FDREs under i_factor_ram: [llength $ffs]  (expected 2480)"
