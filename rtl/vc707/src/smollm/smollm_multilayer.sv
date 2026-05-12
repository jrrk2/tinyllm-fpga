// smollm_multilayer.sv — chains NL smollm_layer instances.
//
// Layer 0 takes hidden_in from the wrapper's port.  Each subsequent
// layer takes hidden_in from the previous layer's hidden_out (which
// stays valid in S_DONE).  Cascade is purely combinational since
// smollm_layer parks in S_DONE indefinitely; start_N is just done_{N-1}.
//
// For Phase E test config we hard-code NL=3 with multilayer_L{0,1,2}_*
// hex prefixes so SV doesn't need string formatting at elaboration.

`default_nettype none

module smollm_multilayer #(
  parameter int D       = 128,
  parameter int H_Q     = 2,
  parameter int H_KV    = 1,
  parameter int HD      = 64,
  parameter int FFN     = 128,
  parameter int MAX_CTX = 4
)(
  input  wire                    clk,
  input  wire                    rst,
  input  wire                    start,
  input  wire [10:0]             pos,
  input  wire [4:0]              kv_pos,
  input  wire signed [D*16-1:0]  hidden_in,
  output logic signed [D*16-1:0] hidden_out,
  output logic                   done
);

  // ----- 3 cascaded layers -----
  wire signed [D*16-1:0] hout0, hout1, hout2;
  wire                   done0, done1, done2;

  smollm_layer #(.D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX),
                 .PREFIX("multilayer_L0_")) i_layer0 (
    .clk(clk), .rst(rst), .start(start),
    .pos(pos), .kv_pos(kv_pos),
    .hidden_in(hidden_in),
    .hidden_out(hout0),
    .done(done0)
  );

  smollm_layer #(.D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX),
                 .PREFIX("multilayer_L1_")) i_layer1 (
    .clk(clk), .rst(rst), .start(done0),
    .pos(pos), .kv_pos(kv_pos),
    .hidden_in(hout0),
    .hidden_out(hout1),
    .done(done1)
  );

  smollm_layer #(.D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX),
                 .PREFIX("multilayer_L2_")) i_layer2 (
    .clk(clk), .rst(rst), .start(done1),
    .pos(pos), .kv_pos(kv_pos),
    .hidden_in(hout1),
    .hidden_out(hout2),
    .done(done2)
  );

  assign hidden_out = hout2;
  assign done       = done2;

endmodule

`default_nettype wire
