// progmem_bram.v — PicoSoC program RAM as 4 byte-lanes of RAMB16_S9_S9 (each
// retargets to a RAMB18E1, 2048 x 8+parity).  4 lanes -> 2048 words x 32 bits =
// 8 KB code space (2x the old single-RAMB36E1).  Byte lanes give per-byte write
// (lane i WE = mem_wstrb[i]) so sb/sh/sw to instruction memory are correct — for
// the self-modifying / ethernet-loaded overlay path.  Port A = sync fetch; port
// B = the CPU store window (0x50, decoded in picosoc_noflash).  INIT baked from
// firmware by bin2init.py -> progmem_init_lane{0..3}.svh (one per byte lane),
// `included into each RAMB16_S9_S9 instance.  Sim (no unisim) falls back to an
// inferred array + $readmemh.

module progmem (
    input  wire        clk,
    input  wire        rstn,
    input  wire        valid,
    output wire        ready,
    input  wire [31:0] addr,
    output wire [31:0] rdata,
    // Port-B CPU store interface — self-modifying / ethernet-loaded overlays.
    // we_b[i] is the byte-write-enable for lane i (= mem_wstrb[i] from the 0x50
    // window); addr_b is the 2048-word index; data_b the word.  Same clock as
    // the fetch port (no CDC).
    input  wire [3:0]  we_b,
    input  wire [10:0] addr_b,
    input  wire [31:0] data_b
);
    localparam MEM_ADDR_MASK = 32'h0010_0000; // progmem region select bit

    // ready must PULSE (one cycle per access), not stay high while valid+match:
    // the registered rdata lags addr by a cycle, so a continuously-high ready let
    // PicoRV32's compressed back-to-back word fetches (a 32-bit instr straddling a
    // word boundary) sample the STALE previous word -> mis-assembled instruction ->
    // hang.  The !ready_r gate forces a fresh handshake per word (same pattern as
    // picosoc_mem).  This is why rv32imc hung while rv32im (no straddles) worked.
    reg ready_r;
    always @(posedge clk or negedge rstn)
        if (!rstn) ready_r <= 1'b0;
        else       ready_r <= valid && !ready_r && ((addr & MEM_ADDR_MASK) != 0);
    assign ready = ready_r;

`ifdef PROGMEM_RAMB
    // ---- synth: 4 byte-lanes, each RAMB16_S9_S9 (2048 x 8, retarget->RAMB18E1).
    // Port A read-only (fetch), port B byte-write (CPU store).  Per-lane INIT
    // baked by bin2init.py.  9th (parity) bit unused: DIPB=0, DOP* open.
    wire [7:0] doa0, doa1, doa2, doa3;
    RAMB16_S9_S9 #( `include "progmem_init_lane0.svh" ) lane0 (
        .CLKA(clk), .ENA(1'b1), .WEA(1'b0), .SSRA(1'b0),
        .ADDRA(addr[12:2]), .DIA(8'b0), .DIPA(1'b0), .DOA(doa0), .DOPA(),
        .CLKB(clk), .ENB(we_b[0]), .WEB(we_b[0]), .SSRB(1'b0),
        .ADDRB(addr_b), .DIB(data_b[7:0]), .DIPB(1'b0), .DOB(), .DOPB());
    RAMB16_S9_S9 #( `include "progmem_init_lane1.svh" ) lane1 (
        .CLKA(clk), .ENA(1'b1), .WEA(1'b0), .SSRA(1'b0),
        .ADDRA(addr[12:2]), .DIA(8'b0), .DIPA(1'b0), .DOA(doa1), .DOPA(),
        .CLKB(clk), .ENB(we_b[1]), .WEB(we_b[1]), .SSRB(1'b0),
        .ADDRB(addr_b), .DIB(data_b[15:8]), .DIPB(1'b0), .DOB(), .DOPB());
    RAMB16_S9_S9 #( `include "progmem_init_lane2.svh" ) lane2 (
        .CLKA(clk), .ENA(1'b1), .WEA(1'b0), .SSRA(1'b0),
        .ADDRA(addr[12:2]), .DIA(8'b0), .DIPA(1'b0), .DOA(doa2), .DOPA(),
        .CLKB(clk), .ENB(we_b[2]), .WEB(we_b[2]), .SSRB(1'b0),
        .ADDRB(addr_b), .DIB(data_b[23:16]), .DIPB(1'b0), .DOB(), .DOPB());
    RAMB16_S9_S9 #( `include "progmem_init_lane3.svh" ) lane3 (
        .CLKA(clk), .ENA(1'b1), .WEA(1'b0), .SSRA(1'b0),
        .ADDRA(addr[12:2]), .DIA(8'b0), .DIPA(1'b0), .DOA(doa3), .DOPA(),
        .CLKB(clk), .ENB(we_b[3]), .WEB(we_b[3]), .SSRB(1'b0),
        .ADDRB(addr_b), .DIB(data_b[31:24]), .DIPB(1'b0), .DOB(), .DOPB());
    assign rdata = {doa3, doa2, doa1, doa0};
`else
    // ---- sim: inferred array initialised from PROGMEM_HEX ----
    reg [31:0] mem [0:2047];
`ifdef PROGMEM_HEX
    initial $readmemh(`PROGMEM_HEX, mem);
`endif
    always @(posedge clk) begin
        if (we_b[0]) mem[addr_b][7:0]   <= data_b[7:0];     // per-byte store
        if (we_b[1]) mem[addr_b][15:8]  <= data_b[15:8];
        if (we_b[2]) mem[addr_b][23:16] <= data_b[23:16];
        if (we_b[3]) mem[addr_b][31:24] <= data_b[31:24];
    end
    reg [31:0] rdata_r;
    always @(posedge clk)
        rdata_r <= mem[addr[12:2]];
    assign rdata = rdata_r;
`endif
endmodule
