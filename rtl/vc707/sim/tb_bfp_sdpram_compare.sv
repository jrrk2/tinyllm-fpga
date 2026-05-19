// tb_bfp_sdpram_compare.sv — constrained-random equivalence check between
// the inferred unpacked-array RAM pattern (the OLD style smollm_layer_bfp
// used for mlp_m and friends) and explicit bfp_sdpram.  Drives identical
// {we, wr_addr, wr_data, rd_addr} stimulus to both, asserts the registered
// read data is bit-identical every cycle.
//
// Purpose: certify the substitution is timing-correct before propagating
// the bfp_sdpram pattern to the other registered-read mantissa arrays
// (n1_m, n2_m, q_m, q_rot_m, k_rot_m, attn_m, o_m, d_m).
//
// Sized for smollm360's largest mantissa array (FFN=2560, 16-bit width)
// because that's the worst case for BRAM packing — but parametric so the
// same TB can be reused for any depth/width.

`default_nettype none

module tb_bfp_sdpram_compare #(
  parameter int DEPTH    = 2560,   // FFN at smollm360
  parameter int WIDTH    = 16,     // BFP_MANT_W
  parameter int N_CYCLES = 50000,  // number of random stimulus cycles
  parameter int SEED     = 32'hdeadbeef
)(
  input  wire clk,
  input  wire rst,
  output reg  done,
  output reg  pass,
  output int  cycle_count,
  output int  error_count
);
  localparam int AW = $clog2(DEPTH > 1 ? DEPTH : 2);

  // ------------------------------------------------------------------
  // Stimulus driver — LFSR-driven {we, wr_addr, wr_data, rd_addr}.
  // ------------------------------------------------------------------
  reg [31:0]      lfsr;
  reg             we;
  reg [AW-1:0]    wr_addr;
  reg [WIDTH-1:0] wr_data;
  reg [AW-1:0]    rd_addr;

  function automatic [31:0] lfsr_next(input [31:0] s);
    // 32-bit Galois LFSR (polynomial 0xEDB88320 — same one CRC32 uses).
    lfsr_next = (s >> 1) ^ ({32{s[0]}} & 32'hEDB88320);
  endfunction

  always_ff @(posedge clk) begin
    if (rst) begin
      lfsr        <= SEED;
      we          <= 1'b0;
      wr_addr     <= '0;
      wr_data     <= '0;
      rd_addr     <= '0;
      cycle_count <= 0;
      done        <= 1'b0;
    end else begin
      // Phase 1: warmup-fill — first DEPTH cycles, write every entry
      // sequentially so the golden array and DUT are populated before
      // the random-read phase starts.
      if (cycle_count < DEPTH) begin
        we      <= 1'b1;
        wr_addr <= AW'(cycle_count);
        wr_data <= WIDTH'(lfsr[WIDTH-1:0] ^ cycle_count[WIDTH-1:0]);
        rd_addr <= '0;
      end else begin
        // Phase 2: random mix.  ~30% writes, 100% reads (always reading).
        we      <= (lfsr[3:0] < 4'd5);                    // ~31%
        wr_addr <= lfsr[15:0] % AW'(DEPTH);
        wr_data <= WIDTH'(lfsr ^ 32'(cycle_count));
        rd_addr <= lfsr[31:16] % AW'(DEPTH);
      end
      lfsr        <= lfsr_next(lfsr);
      cycle_count <= cycle_count + 1;
      if (cycle_count == N_CYCLES) done <= 1'b1;
    end
  end

  // ------------------------------------------------------------------
  // GOLDEN: behavioural inferred unpacked-array RAM (the old style).
  // ------------------------------------------------------------------
  logic [WIDTH-1:0] golden_mem [0:DEPTH-1];
  logic [WIDTH-1:0] golden_rd_data;
  always_ff @(posedge clk) begin
    if (we) golden_mem[wr_addr] <= wr_data;
    golden_rd_data <= golden_mem[rd_addr];
  end

  // ------------------------------------------------------------------
  // DUT: explicit bfp_sdpram (the new style).
  // ------------------------------------------------------------------
  wire [WIDTH-1:0] dut_rd_data;
  bfp_sdpram #(.DEPTH(DEPTH), .WIDTH(WIDTH)) i_dut (
    .clk     (clk),
    .rst     (rst),
    .we      (we),
    .wr_addr (wr_addr),
    .wr_data (wr_data),
    .rd_addr (rd_addr),
    .rd_data (dut_rd_data)
  );

  // ------------------------------------------------------------------
  // Compare cycle-by-cycle.  golden_rd_data and dut_rd_data are both
  // registered outputs of identical SDPRAM semantics — they MUST match
  // every cycle after the first two (one for write to settle, one for
  // read-address-register to update).
  // ------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst) begin
      error_count <= 0;
      pass        <= 1'b1;
    end else if (cycle_count > 2 && !done) begin
      if (golden_rd_data !== dut_rd_data) begin
        if (error_count < 10)  // throttle log spam
          $display("[%0t] MISMATCH cycle=%0d rd_addr=%0d golden=%h dut=%h",
                   $time, cycle_count, rd_addr, golden_rd_data, dut_rd_data);
        error_count <= error_count + 1;
        pass        <= 1'b0;
      end
    end
  end

endmodule

`default_nettype wire
