// bfp_sdpram.sv — simple-dual-port BRAM wrapper.
//
// Explicit XPM_MEMORY_SDPRAM instantiation.  Vivado maps XPM macros
// directly to RAMB36E1 / RAMB18E1 primitives without running the
// generic memory-inference pass, eliminating the 1 h+ "deciding RAM
// shape" hangs we hit on awkward depths like 23040 × 256.
//
// Single clock domain.  Read latency = 1 cycle (BRAM output register).
// Write to addra; read from addrb; data on doutb one cycle after addrb
// is presented.  No byte-enable; full WIDTH bits written each cycle
// when wea is high.

`default_nettype none

module bfp_sdpram #(
  parameter int DEPTH        = 23040,
  parameter int WIDTH        = 256,
  parameter int ADDR_WIDTH   = $clog2(DEPTH > 1 ? DEPTH : 2),
  parameter     PRIMITIVE    = "block"
)(
  input  wire                    clk,
  input  wire                    rst,
  input  wire                    we,
  input  wire [ADDR_WIDTH-1:0]   wr_addr,
  input  wire [WIDTH-1:0]        wr_data,
  input  wire [ADDR_WIDTH-1:0]   rd_addr,
  output wire [WIDTH-1:0]        rd_data
);

`ifdef VERILATOR
  // Behavioural simple-dual-port BRAM for Verilator (xpm_memory_sdpram
  // is Vivado-only).  Same 1-cycle read latency the Vivado primitive
  // gives; consumer sees identical timing.
  logic [WIDTH-1:0] mem [0:DEPTH-1];
  logic [WIDTH-1:0] rd_data_r;
  always_ff @(posedge clk) begin
    if (we) mem[wr_addr] <= wr_data;
    rd_data_r <= mem[rd_addr];
  end
  assign rd_data = rd_data_r;
`else
  xpm_memory_sdpram #(
    .ADDR_WIDTH_A           (ADDR_WIDTH),
    .ADDR_WIDTH_B           (ADDR_WIDTH),
    .AUTO_SLEEP_TIME        (0),
    .BYTE_WRITE_WIDTH_A     (WIDTH),
    .CASCADE_HEIGHT         (0),
    .CLOCKING_MODE          ("common_clock"),
    .ECC_MODE               ("no_ecc"),
    .MEMORY_INIT_FILE       ("none"),
    .MEMORY_INIT_PARAM      ("0"),
    .MEMORY_OPTIMIZATION    ("true"),
    .MEMORY_PRIMITIVE       (PRIMITIVE),
    .MEMORY_SIZE            (DEPTH * WIDTH),
    .MESSAGE_CONTROL        (0),
    .READ_DATA_WIDTH_B      (WIDTH),
    .READ_LATENCY_B         (1),
    .READ_RESET_VALUE_B     ("0"),
    .RST_MODE_A             ("SYNC"),
    .RST_MODE_B             ("SYNC"),
    .SIM_ASSERT_CHK         (0),
    .USE_EMBEDDED_CONSTRAINT(0),
    .USE_MEM_INIT           (0),
    .USE_MEM_INIT_MMI       (0),
    .WAKEUP_TIME            ("disable_sleep"),
    .WRITE_DATA_WIDTH_A     (WIDTH),
    .WRITE_MODE_B           ("no_change"),
    .WRITE_PROTECT          (1)
  ) i_xpm (
    .clka          (clk),
    .clkb          (clk),
    .ena           (we),
    .enb           (1'b1),
    .wea           (we),
    .addra         (wr_addr),
    .addrb         (rd_addr),
    .dina          (wr_data),
    .doutb         (rd_data),
    .regceb        (1'b1),
    .injectsbiterra(1'b0),
    .injectdbiterra(1'b0),
    .rstb          (rst),
    .sbiterrb      (),
    .dbiterrb      (),
    .sleep         (1'b0)
  );
`endif

endmodule

`default_nettype wire
