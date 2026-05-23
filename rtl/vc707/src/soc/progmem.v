module progmem (
    // Clock & reset
    input wire clk,
    input wire rstn,

    // PicoRV32 bus interface
    input  wire        valid,
    output wire        ready,
    input  wire [31:0] addr,
    output wire [31:0] rdata,
    // Port-B CPU store interface (ignored here — this is a fixed test ROM).
    input  wire        we_b,
    input  wire [9:0]  addr_b,
    input  wire [31:0] data_b
);

  // ============================================================================

  localparam MEM_SIZE_BITS = 12;  // In 32-bit words
  localparam MEM_SIZE = 1 << MEM_SIZE_BITS;
  localparam MEM_ADDR_MASK = 32'h0010_0000;

  // ============================================================================

  wire [MEM_SIZE_BITS-1:0] mem_addr;
  reg  [             31:0] mem_data;

  // Memory implemented as synchronous case statement for better simulator compatibility
  always @(posedge clk) begin
    case (mem_addr)
      12'h000: mem_data <= 32'h008000ef;
      12'h001: mem_data <= 32'h0000006f;
      12'h002: mem_data <= 32'hcafe1737;
      12'h003: mem_data <= 32'h100007b7;
      12'h004: mem_data <= 32'h23470713;
      12'h005: mem_data <= 32'h00e7a223;
      12'h006: mem_data <= 32'h00478793;
      12'h007: mem_data <= 32'h0007a783;
      12'h008: mem_data <= 32'h10000737;
      12'h009: mem_data <= 32'h00f72423;
      12'h00a: mem_data <= 32'h00178793;
      12'h00b: mem_data <= 32'h10000737;
      12'h00c: mem_data <= 32'h00f72623;
      12'h00d: mem_data <= 32'h600dd737;
      12'h00e: mem_data <= 32'h100007b7;
      12'h00f: mem_data <= 32'hafe70713;
      12'h010: mem_data <= 32'h3ee7ae23;
      12'h011: mem_data <= 32'h0000006f;
      default: mem_data <= 32'h00000000;
    endcase
  end

  // ============================================================================

  reg o_ready;

  always @(posedge clk or negedge rstn)
    if (!rstn) o_ready <= 1'd0;
    else o_ready <= valid && ((addr & MEM_ADDR_MASK) != 0);

  // Output connectins
  assign ready    = o_ready;
  assign rdata    = mem_data;
  assign mem_addr = addr[MEM_SIZE_BITS+1:2];

endmodule
