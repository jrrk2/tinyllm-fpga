# microgpt_ila_axi — ILA IP for the streamer ↔ MIG AXI read channel.
# Clock: ui_clk (200 MHz).
#
# Probe layout (matches the .probeN connections in vc707_microgpt_eth.sv):
#   probe0 : 1  bit   m_axi_arvalid
#   probe1 : 1  bit   m_axi_arready
#   probe2 : 30 bits  m_axi_araddr
#   probe3 : 1  bit   m_axi_rvalid
#   probe4 : 1  bit   m_axi_rlast
#   probe5 : 128 bits m_axi_rdata[127:0]   (low slice = entry 0 of 4 in beat)
#   probe6 : 3  bits  ws_state             (streamer FSM, axi side)
#   probe7 : 1  bit   start_load_axi
#   probe8 : 2  bits  tile_idx
#   probe9 : 7  bits  beat_idx
#
# Adapted from cva6's mining_ila/tcl/run.tcl.

set partNumber $::env(XILINX_PART)
set boardName  $::env(XILINX_BOARD)

set ipName microgpt_ila_axi

create_project $ipName . -force -part $partNumber
set_property board_part $boardName [current_project]

create_ip -name ila -vendor xilinx.com -library ip -module_name $ipName

set_property -dict [list \
    CONFIG.C_NUM_OF_PROBES   {10}  \
    CONFIG.C_PROBE0_WIDTH    {1}   \
    CONFIG.C_PROBE1_WIDTH    {1}   \
    CONFIG.C_PROBE2_WIDTH    {30}  \
    CONFIG.C_PROBE3_WIDTH    {1}   \
    CONFIG.C_PROBE4_WIDTH    {1}   \
    CONFIG.C_PROBE5_WIDTH    {128} \
    CONFIG.C_PROBE6_WIDTH    {3}   \
    CONFIG.C_PROBE7_WIDTH    {1}   \
    CONFIG.C_PROBE8_WIDTH    {2}   \
    CONFIG.C_PROBE9_WIDTH    {7}   \
    CONFIG.C_DATA_DEPTH      {8192} \
    CONFIG.C_INPUT_PIPE_STAGES {1}  \
] [get_ips $ipName]

generate_target {instantiation_template} [get_files ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
generate_target all                      [get_files ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
create_ip_run [get_files -of_objects [get_fileset sources_1] ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
launch_run -jobs 8 ${ipName}_synth_1
wait_on_run ${ipName}_synth_1
