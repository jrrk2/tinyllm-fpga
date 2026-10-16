# microgpt_ila_weights — ILA IP capturing the FULL 512-bit weight beat as it is
# delivered from the MIG read FIFO into the engine (i_lay_st/i_emb beat_buf).
# Clock: ui_clk (200 MHz).
#
# Two diagnostic upgrades over the first cut:
#   1. rdata (512 b) and araddr (30 b) are DATA-ONLY probes (C_PROBE_TYPE 1).
#      The original DATA_AND_TRIGGER on the 512-b probe built a 512-wide trigger
#      comparator (the U_TM/probeDelay path) — that was the -53 ps net.  As a
#      data-only probe there is no comparator on it, so timing closes with margin
#      and the trigger fabric shrinks.  araddr stays correct per beat (the
#      streamer holds m_axi_araddr stable across a whole burst — strictly
#      AR -> drain -> next AR, no AR pipelining).
#   2. arvalid / arready are now probed, so each AR handshake is visible and the
#      returned R beats can be aligned to the address that requested them.
#
# Capture control (storage qualification) is ENABLED (C_EN_STRG_QUAL 1).  In the
# HW manager set the capture condition to  rvalid==1 && rready==1  so the 2048-
# sample buffer is PACKED with real weight beats instead of idle ui_clk cycles
# (araddr travels with each beat because it is held stable across the burst).
# Trigger on probe7 (restart_edge_ui) rising to frame one inference run.
#
# Probe layout (matches the .probeN connections in vc707_microgpt_eth.sv):
#   probe0 : 512 bits  m_axi_rdata[511:0]   weight beat into the engine   (DATA)
#   probe1 : 1  bit    m_axi_rvalid                                    (DATA+TRIG)
#   probe2 : 1  bit    m_axi_rready                                    (DATA+TRIG)
#   probe3 : 1  bit    m_axi_rlast                                     (DATA+TRIG)
#   probe4 : 30 bits   m_axi_araddr   source addr (held over burst)        (DATA)
#   probe5 : 1  bit    m_axi_arvalid                                   (DATA+TRIG)
#   probe6 : 1  bit    m_axi_arready                                   (DATA+TRIG)
#   probe7 : 1  bit    restart_edge_ui  engine restart pulse           (DATA+TRIG)
#
# Same standalone-IP-project pattern as ip/microgpt_ila_axi/tcl/run.tcl.

set partNumber $::env(XILINX_PART)
set boardName  $::env(XILINX_BOARD)

set ipName microgpt_ila_weights

create_project $ipName . -force -part $partNumber
set_property board_part $boardName [current_project]

create_ip -name ila -vendor xilinx.com -library ip -module_name $ipName

set_property -dict [list \
    CONFIG.C_NUM_OF_PROBES   {8}    \
    CONFIG.C_PROBE0_WIDTH    {512}  \
    CONFIG.C_PROBE0_TYPE     {1}    \
    CONFIG.C_PROBE1_WIDTH    {1}    \
    CONFIG.C_PROBE2_WIDTH    {1}    \
    CONFIG.C_PROBE3_WIDTH    {1}    \
    CONFIG.C_PROBE4_WIDTH    {30}   \
    CONFIG.C_PROBE4_TYPE     {1}    \
    CONFIG.C_PROBE5_WIDTH    {1}    \
    CONFIG.C_PROBE6_WIDTH    {1}    \
    CONFIG.C_PROBE7_WIDTH    {1}    \
    CONFIG.C_DATA_DEPTH      {2048} \
    CONFIG.C_EN_STRG_QUAL    {1}    \
    CONFIG.C_INPUT_PIPE_STAGES {2}  \
] [get_ips $ipName]

generate_target {instantiation_template} [get_files ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
generate_target all                      [get_files ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
create_ip_run [get_files -of_objects [get_fileset sources_1] ./$ipName.srcs/sources_1/ip/$ipName/$ipName.xci]
launch_run -jobs 8 ${ipName}_synth_1
wait_on_run ${ipName}_synth_1
