# microgpt_ila_core — ILA IP for the smollm_layer matvec datapath.
# Clock: core_clk (50 MHz).
#
# Probe layout (matches the .probeN connections in vc707_microgpt_eth.sv):
#   probe0  : 5  bits  state           — layer top-level FSM
#   probe1  : 3  bits  mv_phase        — matvec sub-FSM
#   probe2  : 11 bits  cnt             — MV_DRIVE drive index
#   probe3  : 7  bits  chunk           — current 16-lane chunk
#   probe4  : 3  bits  ws_matvec_id    — which matrix (0=Q..6=DOWN)
#   probe5  : 1  bit   ws_load_req
#   probe6  : 1  bit   ws_ready
#   probe7  : 11 bits  ws_rd_addr
#   probe8  : 128 bits ws_weight_data
#   probe9  : 128 bits eng_w
#   probe10 : 16 bits  eng_in_value
#   probe11 : 1  bit   eng_in_valid
#   probe12 : 1  bit   eng_in_last
#   probe13 : 1  bit   eng_acc_clear
#   probe14 : 1  bit   eng_out_valid
#
# Adapted from cva6's mining_ila/tcl/run.tcl.

set partNumber $::env(XILINX_PART)
set boardName  $::env(XILINX_BOARD)

set ipName microgpt_ila_core

create_project $ipName . -force -part $partNumber
set_property board_part $boardName [current_project]

create_ip -name ila -vendor xilinx.com -library ip -module_name $ipName

set_property -dict [list \
    CONFIG.C_NUM_OF_PROBES   {15}  \
    CONFIG.C_PROBE0_WIDTH    {5}   \
    CONFIG.C_PROBE1_WIDTH    {3}   \
    CONFIG.C_PROBE2_WIDTH    {11}  \
    CONFIG.C_PROBE3_WIDTH    {7}   \
    CONFIG.C_PROBE4_WIDTH    {3}   \
    CONFIG.C_PROBE5_WIDTH    {1}   \
    CONFIG.C_PROBE6_WIDTH    {1}   \
    CONFIG.C_PROBE7_WIDTH    {11}  \
    CONFIG.C_PROBE8_WIDTH    {128} \
    CONFIG.C_PROBE9_WIDTH    {128} \
    CONFIG.C_PROBE10_WIDTH   {16}  \
    CONFIG.C_PROBE11_WIDTH   {1}   \
    CONFIG.C_PROBE12_WIDTH   {1}   \
    CONFIG.C_PROBE13_WIDTH   {1}   \
    CONFIG.C_PROBE14_WIDTH   {1}   \
    CONFIG.C_DATA_DEPTH      {8192} \
    CONFIG.C_INPUT_PIPE_STAGES {1}  \
] [get_ips $ipName]

generate_target {instantiation_template} [get_files ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
generate_target all                      [get_files ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
create_ip_run [get_files -of_objects [get_fileset sources_1] ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
launch_run -jobs 8 ${ipName}_synth_1
wait_on_run ${ipName}_synth_1
