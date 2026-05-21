// soc_matvec_feed.sv — iomem peripheral that feeds a matvec_bfp_engine
// beat-by-beat from the PicoSoC.  This is the weight-feed DEBUG MODE in
// miniature: instead of the streamer pulling weights from MIG, the CPU writes
// each column's activation + 16-lane weight beat over iomem and pulses the
// engine's in_valid one beat at a time.  The engine's cnt only advances on
// in_valid, so it stalls between the CPU's (slow) writes — no MIG, fully
// deterministic.  Proves the feed path before the full-layer integration.
//
// iomem map (word-indexed at the peripheral base, addr[31:24]==0x10):
//   [0]      CTRL (W): bit0=pulse start_matvec, bit1=pulse in_valid,
//                      bit2=last_elem (assert with in_valid -> write 6)
//   [1]      X_MANT (W, 16b)        [2] X_EXP (W, 8b)
//   [4..11]  W_MANT words 0..7 (256b weight beat: 16 lanes x 16b)
//   [12..15] W_EXP  words 0..3 (128b: 16 lanes x 8b)
//   [16]     STATUS (R): bit0 = out captured
//   [20..27] OUT_MANT words 0..7 (R)   [28..31] OUT_EXP words 0..3 (R)

`include "bfp_format.svh"
`default_nettype none

module soc_matvec_feed #(
  parameter int LANES = 16
) (
  input  wire        clk,
  input  wire        resetn,
  // iomem (PicoSoC) slave
  input  wire        iomem_valid,
  output reg         iomem_ready,
  input  wire [3:0]  iomem_wstrb,
  input  wire [31:0] iomem_addr,
  input  wire [31:0] iomem_wdata,
  output reg  [31:0] iomem_rdata,
  // exposed to the TB for checking
  output wire                              out_captured,
  output wire signed [LANES*BFP_MANT_W-1:0] out_mant_o,
  output wire signed [LANES*BFP_EXP_W -1:0] out_exp_o
);

  wire sel   = iomem_valid && (iomem_addr[31:24] == 8'h10);
  wire wr    = sel && (|iomem_wstrb);
  wire [5:0] widx = iomem_addr[7:2];

  // Held engine inputs.
  reg signed [BFP_MANT_W-1:0]        x_mant;
  reg signed [BFP_EXP_W-1:0]         x_exp;
  reg signed [LANES*BFP_MANT_W-1:0]  w_mant;
  reg signed [LANES*BFP_EXP_W-1:0]   w_exp;

  // Engine control pulses (1 cycle, on the CTRL write).
  reg start_p, valid_p, last_p;

  // Captured output.
  reg                              cap;
  reg signed [LANES*BFP_MANT_W-1:0] out_m_cap;
  reg signed [LANES*BFP_EXP_W -1:0] out_e_cap;

  wire signed [LANES*BFP_MANT_W-1:0] mv_out_m;
  wire signed [LANES*BFP_EXP_W -1:0] mv_out_e;
  wire                               mv_out_valid;

  always_ff @(posedge clk) begin
    if (!resetn) begin
      iomem_ready <= 1'b0; iomem_rdata <= 32'h0;
      x_mant <= '0; x_exp <= '0; w_mant <= '0; w_exp <= '0;
      start_p <= 1'b0; valid_p <= 1'b0; last_p <= 1'b0;
      cap <= 1'b0; out_m_cap <= '0; out_e_cap <= '0;
    end else begin
      iomem_ready <= 1'b0;
      start_p <= 1'b0; valid_p <= 1'b0; last_p <= 1'b0;   // pulses default low

      if (sel && !iomem_ready) begin
        iomem_ready <= 1'b1;
        if (wr) begin
          case (widx)
            6'd0: begin start_p <= iomem_wdata[0]; valid_p <= iomem_wdata[1]; last_p <= iomem_wdata[2]; end
            6'd1: x_mant <= iomem_wdata[BFP_MANT_W-1:0];
            6'd2: x_exp  <= iomem_wdata[BFP_EXP_W-1:0];
            6'd4,6'd5,6'd6,6'd7,6'd8,6'd9,6'd10,6'd11:
                  w_mant[(widx-6'd4)*32 +: 32] <= iomem_wdata;
            6'd12,6'd13,6'd14,6'd15:
                  w_exp[(widx-6'd12)*32 +: 32] <= iomem_wdata;
            default: ;
          endcase
        end
        // reads
        case (widx)
          6'd16: iomem_rdata <= {31'h0, cap};
          6'd20,6'd21,6'd22,6'd23,6'd24,6'd25,6'd26,6'd27:
                 iomem_rdata <= out_m_cap[(widx-6'd20)*32 +: 32];
          6'd28,6'd29,6'd30,6'd31:
                 iomem_rdata <= out_e_cap[(widx-6'd28)*32 +: 32];
          default: iomem_rdata <= 32'h0;
        endcase
      end

      // Capture the engine result.
      if (mv_out_valid) begin
        cap <= 1'b1;
        out_m_cap <= mv_out_m;
        out_e_cap <= mv_out_e;
      end
    end
  end

  matvec_bfp_engine #(.LANES(LANES)) mv (
    .clk(clk), .rst(!resetn),
    .start_matvec(start_p),
    .in_x_mant(x_mant), .in_x_exp(x_exp),
    .in_valid(valid_p), .last_elem(last_p),
    .w_mant(w_mant), .w_exp(w_exp),
    .out_mant(mv_out_m), .out_exp(mv_out_e), .out_valid(mv_out_valid)
  );

  assign out_captured = cap;
  assign out_mant_o   = out_m_cap;
  assign out_exp_o    = out_e_cap;

endmodule

`default_nettype wire
