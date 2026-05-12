# Isolated synth of factor_ram.sv — count flip-flops and check we have
# 30*(16+16+24+24) = 2400 storage FFs + 80 output regs = 2480 total.
set_part xc7vx485tffg1761-2
read_verilog -sv ../src/smollm/factor_ram.sv
set_property top factor_ram [current_fileset]
synth_design -top factor_ram -part xc7vx485tffg1761-2 \
             -include_dirs ../../generated -mode out_of_context
report_utilization -file synth_factor_ram_util.rpt

# Detailed per-bit FF counts to detect bit-level optimization.
set total 0
foreach arr {gate_in_factor_ram up_in_factor_ram mlp_out_factor_ram attn_factor_ram} {
    set ffs [get_cells -hier -filter "NAME =~ \"*${arr}*\" && REF_NAME == FDRE"]
    puts "  $arr : [llength $ffs] FDREs"
    incr total [llength $ffs]
}
puts "  total ram FDREs: $total (expected 30*(16+16+24+24) = 2400)"

set cur_ffs [get_cells -hier -filter {NAME =~ "*cur_*_factor_r*" && REF_NAME == FDRE}]
puts "  cur_*_factor_r : [llength $cur_ffs] FDREs (expected 16+16+24+24 = 80)"
