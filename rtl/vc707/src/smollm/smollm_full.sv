// smollm_full.sv — end-to-end token-in / token-out inference (Phase F.2).
//
// Pipeline:
//   in_token → embed[in_token] (D-dim Q1.15) → 3× smollm_layer chain
//            → smollm_decode_head (final-norm + lm-head + argmax)
//            → next_token
//
// Embedding lookup: D-cycle FSM walks `embed_rom` packing one element per
// cycle into `hidden_in_reg`.  Chained start/done flow: when the embed
// fills, the first transformer layer leaves S_IDLE; layers cascade via
// done→start as in smollm_multilayer; finally the decode_head's start
// is the last layer's done.

`default_nettype none

module smollm_full #(
  parameter int D       = 128,
  parameter int H_Q     = 2,
  parameter int H_KV    = 1,
  parameter int HD      = 64,
  parameter int FFN     = 128,
  parameter int MAX_CTX = 4,
  parameter int VOCAB   = 128
)(
  input  wire                          clk,
  input  wire                          rst,
  input  wire                          start,
  input  wire [10:0]                   pos,
  input  wire [4:0]                    kv_pos,
  input  wire [$clog2(VOCAB)-1:0]      in_token,
  output logic [$clog2(VOCAB)-1:0]     next_token,
  output logic signed [15:0]           top_logit,
  output logic                         done
);

  // ----- Embedding ROM (loaded via $readmemh) -----
  // Layout: embed_rom[v*D + e] = embed[v, e] (Q1.15)
  logic signed [15:0]  rom_EMBED [0:VOCAB*D - 1];
`ifdef MICROGPT_WEIGHT_DIR
  initial $readmemh({`MICROGPT_WEIGHT_DIR, "/", "full_EMBED.hex"}, rom_EMBED);
`else
  initial $readmemh("full_EMBED.hex", rom_EMBED);
`endif

  // ----- Embedding FSM -----
  // D cycles to populate hidden_in_reg from rom_EMBED, then assert embed_done.
  reg [D*16-1:0]   hidden_in_reg;
  reg [10:0]       embed_cnt;
  reg              embed_done;

  always_ff @(posedge clk) begin
    if (rst) begin
      embed_cnt   <= '0;
      embed_done  <= 1'b0;
    end else if (start && !embed_done) begin
      hidden_in_reg[embed_cnt*16 +: 16] <= rom_EMBED[in_token * D + embed_cnt];
      embed_cnt <= embed_cnt + 1'b1;
      if (embed_cnt == D-1)
        embed_done <= 1'b1;
    end
  end

  // ----- 3-layer chain (same wiring as smollm_multilayer) -----
  wire signed [D*16-1:0] hout0, hout1, hout2;
  wire                   ldone0, ldone1, ldone2;

  smollm_layer #(.D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX),
                 .PREFIX("full_L0_")) i_layer0 (
    .clk(clk), .rst(rst), .start(embed_done),
    .pos(pos), .kv_pos(kv_pos),
    .hidden_in(hidden_in_reg),
    .hidden_out(hout0),
    .done(ldone0)
  );
  smollm_layer #(.D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX),
                 .PREFIX("full_L1_")) i_layer1 (
    .clk(clk), .rst(rst), .start(ldone0),
    .pos(pos), .kv_pos(kv_pos),
    .hidden_in(hout0),
    .hidden_out(hout1),
    .done(ldone1)
  );
  smollm_layer #(.D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX),
                 .PREFIX("full_L2_")) i_layer2 (
    .clk(clk), .rst(rst), .start(ldone1),
    .pos(pos), .kv_pos(kv_pos),
    .hidden_in(hout1),
    .hidden_out(hout2),
    .done(ldone2)
  );

  // ----- Decode head -----
  smollm_decode_head #(.D(D), .VOCAB(VOCAB), .PREFIX("full_")) i_dh (
    .clk(clk), .rst(rst), .start(ldone2),
    .hidden_state(hout2),
    .next_token(next_token),
    .top_logit(top_logit),
    .done(done)
  );

endmodule

`default_nettype wire
