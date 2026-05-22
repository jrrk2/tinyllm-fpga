// progmem_bram.v — PicoSoC program ROM forced into a real block RAM by DIRECTLY
// instantiating the RAMB16_S36_S36 primitive.  Inference collapsed the read-only,
// mostly-zero ROM to one RAMB18 + LUTROM, which updatemem cannot patch (it only
// rewrites BRAM).  A direct primitive instance is a clean, identifiable BRAM tile:
//   - firmware is baked into the INIT_xx params by bin2init.py (the PROGMEM_INIT
//     include) so the raw bitstream boots, AND
//   - the same tile can be re-patched in a built .bit with updatemem + a .mmi
//     generated from the RAM list (lowRISC bootrom pattern; data2mem -> updatemem).
//
// Same module name + interface as before, so picosoc_noflash is unchanged.
// Sim (no Xilinx unisim primitives) falls back to an inferred array + $readmemh.

module progmem (
    input  wire        clk,
    input  wire        rstn,
    input  wire        valid,
    output wire        ready,
    input  wire [31:0] addr,
    output wire [31:0] rdata
);
    localparam MEM_ADDR_MASK = 32'h0010_0000; // progmem region select bit

    reg ready_r;
    always @(posedge clk or negedge rstn)
        if (!rstn) ready_r <= 1'b0;
        else       ready_r <= valid && ((addr & MEM_ADDR_MASK) != 0);
    assign ready = ready_r;

`ifdef PROGMEM_RAMB
    // ---- synth: directly instantiated 512 x 36 block RAM (32 data bits used) --
    // Port A = synchronous read (1-cycle latency, matching the inferred rdata_r).
    // INIT_xx baked from firmware by bin2init.py; absent -> blank (updatemem load).
    wire [31:0] doa;
    RAMB16_S36_S36 #(
`ifdef PROGMEM_INIT
        `include `PROGMEM_INIT
`endif
    ) rom (
        .CLKA  (clk),          .ENA  (1'b1), .WEA (1'b0), .SSRA (1'b0),
        .ADDRA (addr[10:2]),   .DIA  (32'b0), .DIPA (4'b0),
        .DOA   (doa),          .DOPA (),
        .CLKB  (clk),          .ENB  (1'b0), .WEB (1'b0), .SSRB (1'b0),
        .ADDRB (9'b0),         .DIB  (32'b0), .DIPB (4'b0),
        .DOB   (),             .DOPB ()
    );
    assign rdata = doa;
`else
    // ---- sim: inferred array initialised from PROGMEM_HEX ----
    reg [31:0] mem [0:511];
`ifdef PROGMEM_HEX
    initial $readmemh(`PROGMEM_HEX, mem);
`endif
    reg [31:0] rdata_r;
    always @(posedge clk)
        rdata_r <= mem[addr[10:2]];
    assign rdata = rdata_r;
`endif
endmodule
