# picosoc_ila — ILA for the PicoSoC bring-up shell (vc707_picosoc_shell).
# Clock: eth_clk (125 MHz).  Probe layout matches i_ila in vc707_picosoc_shell:
#   probe0 : 1  bit   trace_valid
#   probe1 : 36 bits  trace_data    — PicoRV32 execution trace
#   probe2 : 1  bit   iomem_valid
#   probe3 : 1  bit   iomem_ready
#   probe4 : 4  bits  iomem_wstrb   — 0 = read, else write byte-enables
#   probe5 : 32 bits  iomem_addr    — 0x03 LED / 0x10 dummy / 0x20 eth / 0x30 DDR
#   probe6 : 32 bits  iomem_wdata
#   probe7 : 32 bits  iomem_rdata
#
# Same pattern as ip/microgpt_ila_core/tcl/run.tcl: a dedicated IP project where
# create_ip works reliably; the main run.tcl read_ip's the generated .xci.

set partNumber $::env(XILINX_PART)
set boardName  $::env(XILINX_BOARD)

set ipName picosoc_ila

create_project $ipName . -force -part $partNumber
set_property board_part $boardName [current_project]

create_ip -name ila -vendor xilinx.com -library ip -module_name $ipName

set_property -dict [list \
    CONFIG.C_NUM_OF_PROBES   {8}   \
    CONFIG.C_PROBE0_WIDTH    {1}   \
    CONFIG.C_PROBE1_WIDTH    {36}  \
    CONFIG.C_PROBE2_WIDTH    {1}   \
    CONFIG.C_PROBE3_WIDTH    {1}   \
    CONFIG.C_PROBE4_WIDTH    {4}   \
    CONFIG.C_PROBE5_WIDTH    {32}  \
    CONFIG.C_PROBE6_WIDTH    {32}  \
    CONFIG.C_PROBE7_WIDTH    {32}  \
    CONFIG.C_DATA_DEPTH      {4096} \
    CONFIG.C_INPUT_PIPE_STAGES {1}  \
] [get_ips $ipName]

generate_target {instantiation_template} [get_files ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
generate_target all                      [get_files ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
create_ip_run [get_files -of_objects [get_fileset sources_1] ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
launch_run -jobs 8 ${ipName}_synth_1
wait_on_run ${ipName}_synth_1
