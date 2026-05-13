// embed_lookup_bfp.sv — BFP embedding-table row lookup.
//
//   in:  token_id (16-bit), start pulse
//   out: hidden_m (D mantissas packed), hidden_e (NT_D exps packed), done
//
// Internal BRAM: VOCAB*D mantissas + VOCAB*NT_D exponents, loaded via
// $readmemh from the lbfp_full_EMBED_LOOKUP_{m,e}.hex files baked by
// gen_smollm_blockfp_full.py.
//
// One D-element row read = D + NT_D BRAM-read cycles (sequential).  Output
// hidden_m / hidden_e drive in parallel after `done`.  This is the minimum-
// area path for Verilator simulation; on FPGA the same module can be
// re-cast as a single-cycle wide-read by cascading BRAMs.

`include "bfp_format.svh"

`default_nettype none

module embed_lookup_bfp #(
  parameter int D     = 576,
  parameter int VOCAB = 49152,
  parameter     PREFIX = "lbfp_full_"
)(
  input  wire                                       clk,
  input  wire                                       rst,
  input  wire                                       start,
  input  wire [15:0]                                token_id,
  output logic signed [D*BFP_MANT_W-1:0]            hidden_m,
  output logic signed [(D/BFP_TILE)*BFP_EXP_W-1:0]  hidden_e,
  output logic                                      done
);

  localparam int NT_D = D / BFP_TILE;

  (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] rom_m [0:VOCAB*D-1];
  (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] rom_e [0:VOCAB*NT_D-1];

`ifdef MICROGPT_WEIGHT_DIR
  initial begin
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "EMBED_LOOKUP_m.hex"}, rom_m);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "EMBED_LOOKUP_e.hex"}, rom_e);
  end
`else
  initial begin
    $readmemh({PREFIX, "EMBED_LOOKUP_m.hex"}, rom_m);
    $readmemh({PREFIX, "EMBED_LOOKUP_e.hex"}, rom_e);
  end
`endif

  typedef enum logic [2:0] { S_IDLE, S_READ_M, S_READ_E, S_DONE } st_t;
  st_t state;
  logic [$clog2(D)-1:0]  cnt_m;
  logic [$clog2(NT_D)-1:0] cnt_e;
  logic [15:0]           tok_r;

  // Output buffers (registered)
  logic signed [BFP_MANT_W-1:0] m_buf [0:D-1];
  logic signed [BFP_EXP_W -1:0] e_buf [0:NT_D-1];

  always_ff @(posedge clk) begin
    if (rst) begin
      state <= S_IDLE;
      cnt_m <= '0; cnt_e <= '0;
      tok_r <= '0;
      done  <= 1'b0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: if (start) begin
          tok_r <= token_id;
          cnt_m <= '0; cnt_e <= '0;
          state <= S_READ_M;
        end
        S_READ_M: begin
          m_buf[cnt_m] <= rom_m[tok_r * D + cnt_m];
          if (cnt_m == D-1) begin
            cnt_m <= '0; state <= S_READ_E;
          end else cnt_m <= cnt_m + 1'b1;
        end
        S_READ_E: begin
          e_buf[cnt_e] <= rom_e[tok_r * NT_D + cnt_e];
          if (cnt_e == NT_D-1) begin
            cnt_e <= '0; state <= S_DONE;
          end else cnt_e <= cnt_e + 1'b1;
        end
        S_DONE: begin
          done  <= 1'b1;
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  always_comb begin
    for (int i = 0; i < D; i++)
      hidden_m[i*BFP_MANT_W +: BFP_MANT_W] = m_buf[i];
    for (int t = 0; t < NT_D; t++)
      hidden_e[t*BFP_EXP_W  +: BFP_EXP_W ] = e_buf[t];
  end

endmodule

`default_nettype wire
