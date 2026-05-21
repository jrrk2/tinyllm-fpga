// progmem_bram.v — BRAM-backed PicoSoC program ROM (drop-in for the bin2verilog
// case-statement progmem).  Implemented as an inferred block RAM initialized
// from PROGMEM_HEX so the firmware can be SWAPPED in a finished bitstream with
// Vivado `updatemem` (no re-synthesis).  Same module name + interface as the
// case-ROM, so picosoc_noflash is unchanged.
//
// PROGMEM_HEX must be set (absolute path to firmware_shell.mem) at synth, e.g.
//   set_property verilog_define {... PROGMEM_HEX="/abs/.../firmware_shell.mem"} ...

module progmem (
    input  wire        clk,
    input  wire        rstn,
    input  wire        valid,
    output wire        ready,
    input  wire [31:0] addr,
    output wire [31:0] rdata
);
    localparam MEM_SIZE_BITS = 12;            // 4096 x 32-bit = 16 KB
    localparam MEM_ADDR_MASK = 32'h0010_0000; // progmem region select bit

    (* ram_style = "block", keep_hierarchy = "yes" *)
    reg [31:0] mem [0:(1<<MEM_SIZE_BITS)-1];
`ifdef PROGMEM_HEX
    initial $readmemh(`PROGMEM_HEX, mem);
`endif

    reg [31:0] rdata_r;
    reg        ready_r;
    always @(posedge clk)
        rdata_r <= mem[addr[MEM_SIZE_BITS+1:2]];
    always @(posedge clk or negedge rstn)
        if (!rstn) ready_r <= 1'b0;
        else       ready_r <= valid && ((addr & MEM_ADDR_MASK) != 0);

    assign rdata = rdata_r;
    assign ready = ready_r;
endmodule
