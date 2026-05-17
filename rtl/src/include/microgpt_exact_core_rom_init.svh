`ifndef MICROGPT_WEIGHT_DIR
  `define MICROGPT_WEIGHT_DIR "generated"
`endif
initial begin
    $readmemh({`MICROGPT_WEIGHT_DIR, "/wte_q12.hex"}, wte_rom);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/wpe_q12.hex"}, wpe_rom);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/lm_head_q12.hex"}, lm_head_rom);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/layer0_attn_wq_q12.hex"}, attn_wq_rom);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/layer0_attn_wk_q12.hex"}, attn_wk_rom);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/layer0_attn_wv_q12.hex"}, attn_wv_rom);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/layer0_attn_wo_q12.hex"}, attn_wo_rom);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/layer0_mlp_fc1_q12.hex"}, mlp_fc1_rom);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/layer0_mlp_fc2_q12.hex"}, mlp_fc2_rom);
end
