module progmem (
    // Clock & reset
    input wire clk,
    input wire rstn,

    // PicoRV32 bus interface
    input  wire        valid,
    output wire        ready,
    input  wire [31:0] addr,
    output wire [31:0] rdata
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
      12'h002: mem_data <= 32'hff010113;
      12'h003: mem_data <= 32'h00112623;
      12'h004: mem_data <= 32'h00812423;
      12'h005: mem_data <= 32'h00912223;
      12'h006: mem_data <= 32'h01212023;
      12'h007: mem_data <= 32'h00100713;
      12'h008: mem_data <= 32'h100007b7;
      12'h009: mem_data <= 32'h10000537;
      12'h00a: mem_data <= 32'h10000837;
      12'h00b: mem_data <= 32'h100008b7;
      12'h00c: mem_data <= 32'h10000337;
      12'h00d: mem_data <= 32'h010106b7;
      12'h00e: mem_data <= 32'h10000e37;
      12'h00f: mem_data <= 32'h10000eb7;
      12'h010: mem_data <= 32'h10000f37;
      12'h011: mem_data <= 32'h00e7a023;
      12'h012: mem_data <= 32'h3e800413;
      12'h013: mem_data <= 32'h00000713;
      12'h014: mem_data <= 32'h00450513;
      12'h015: mem_data <= 32'h00880813;
      12'h016: mem_data <= 32'h00f00493;
      12'h017: mem_data <= 32'h00f00913;
      12'h018: mem_data <= 32'h01088893;
      12'h019: mem_data <= 32'h01000293;
      12'h01a: mem_data <= 32'h03030313;
      12'h01b: mem_data <= 32'h10168693;
      12'h01c: mem_data <= 32'h034e0e13;
      12'h01d: mem_data <= 32'h038e8e93;
      12'h01e: mem_data <= 32'h03cf0f13;
      12'h01f: mem_data <= 32'h100000b7;
      12'h020: mem_data <= 32'h00170393;
      12'h021: mem_data <= 32'h028387b3;
      12'h022: mem_data <= 32'h00000613;
      12'h023: mem_data <= 32'h00f52023;
      12'h024: mem_data <= 32'h01282023;
      12'h025: mem_data <= 32'h00160793;
      12'h026: mem_data <= 32'h40c705b3;
      12'h027: mem_data <= 32'h40e787b3;
      12'h028: mem_data <= 32'h0017b793;
      12'h029: mem_data <= 32'h0015b593;
      12'h02a: mem_data <= 32'h00161f93;
      12'h02b: mem_data <= 32'h01e79793;
      12'h02c: mem_data <= 32'h00e59593;
      12'h02d: mem_data <= 32'h011f8fb3;
      12'h02e: mem_data <= 32'h00b7e7b3;
      12'h02f: mem_data <= 32'h00ffa023;
      12'h030: mem_data <= 32'h00260613;
      12'h031: mem_data <= 32'hfc5618e3;
      12'h032: mem_data <= 32'h00d32023;
      12'h033: mem_data <= 32'h00de2023;
      12'h034: mem_data <= 32'h00dea023;
      12'h035: mem_data <= 32'h00df2023;
      12'h036: mem_data <= 32'h00200793;
      12'h037: mem_data <= 32'h00971463;
      12'h038: mem_data <= 32'h00600793;
      12'h039: mem_data <= 32'h00f0a023;
      12'h03a: mem_data <= 32'h00038713;
      12'h03b: mem_data <= 32'hf8539ae3;
      12'h03c: mem_data <= 32'h0000006f;
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
