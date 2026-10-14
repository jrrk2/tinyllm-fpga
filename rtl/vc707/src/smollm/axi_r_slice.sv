// axi_r_slice.sv — full (2-deep skid) register slice on an AXI4 read-data
// channel.  Both sides handshake with valid/ready; beats come out in order with
// one extra cycle of latency and full 1-beat/cycle throughput.
//
// Why it exists: the MIG's AXI read-data FIFO is SRL-based, so s_axi_rdata is
// a combinational read of a 512-wide SRL array addressed by the FIFO's read
// pointer.  Fed straight into the three weight consumers (embed beat_buf, the
// layer and decode streamer banks) that spread across the die, the pointer ->
// SRL -> consumer path was the clk_pll_i (200 MHz) critical path and missed
// timing by ~0.1 ns, which showed up on chip as intermittently corrupted weight
// beats (non-deterministic tokens, run-to-run weight-hash changes).  Registering
// the channel once, right at the MIG, splits that path in two.
//
// s_rready is a plain register (~skid_v) so it adds no logic on the MIG side.
module axi_r_slice #(
  parameter int ID_W   = 5,
  parameter int DATA_W = 512
) (
  input  wire              clk,
  input  wire              rst,       // synchronous, active-high
  // slave side (from the MIG)
  input  wire              s_rvalid,
  output wire              s_rready,
  input  wire [ID_W-1:0]   s_rid,
  input  wire [DATA_W-1:0] s_rdata,
  input  wire [1:0]        s_rresp,
  input  wire              s_rlast,
  // master side (to the consumers)
  output wire              m_rvalid,
  input  wire              m_rready,
  output wire [ID_W-1:0]   m_rid,
  output wire [DATA_W-1:0] m_rdata,
  output wire [1:0]        m_rresp,
  output wire              m_rlast
);
  localparam int W = ID_W + DATA_W + 2 + 1;

  wire [W-1:0] s_pack = {s_rid, s_rdata, s_rresp, s_rlast};

  logic         out_v;
  logic         skid_v;
  logic [W-1:0] out_d;
  logic [W-1:0] skid_d;

  assign s_rready = ~skid_v;
  wire   s_fire   = s_rvalid & ~skid_v;

  always_ff @(posedge clk) begin
    if (rst) begin
      out_v  <= 1'b0;
      skid_v <= 1'b0;
    end else begin
      if (!out_v || m_rready) begin
        // Output slot is free (or is being consumed this cycle): refill it,
        // oldest beat first — the skid entry, else whatever arrives now.
        if (skid_v) begin
          out_d  <= skid_d;
          out_v  <= 1'b1;
          skid_v <= 1'b0;        // s_rready was low this cycle, nothing fired
        end else begin
          out_d  <= s_pack;
          out_v  <= s_fire;
        end
      end else if (s_fire) begin
        // Output stalled with a beat in it: park the incoming beat.  skid_v is
        // 0 here (s_fire implies it), so this never overwrites a parked beat.
        skid_d <= s_pack;
        skid_v <= 1'b1;
      end
    end
  end

  assign m_rvalid = out_v;
  assign {m_rid, m_rdata, m_rresp, m_rlast} = out_d;
endmodule
