// tb_rvc.v — minimal PicoRV32 + behavioral memory, to test compressed-ISA decode
// in isolation.  Loads a flat program image (+mem=<hex>) at address 0, runs it on
// a COMPRESSED_ISA=1 core (same option as picosoc_noflash), and prints each write
// to the magic MMIO address 0x10000000.  Catches `trap` (illegal/misaligned) and,
// on timeout, dumps the last instruction-fetch addresses so a hang is localised.
`timescale 1ns/1ps

module tb_rvc;
    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;          // 100 MHz

    wire        trap;
    wire        mem_valid, mem_instr;
    wire [31:0] mem_addr, mem_wdata;
    wire [3:0]  mem_wstrb;
    reg         mem_ready = 0;
    reg  [31:0] mem_rdata = 0;

    picorv32 #(
        .PROGADDR_RESET(32'h0000_0000),
        .STACKADDR(32'h0004_0000),
        .BARREL_SHIFTER(1),
        .COMPRESSED_ISA(1),
        .ENABLE_MUL(1),
        .ENABLE_DIV(1),
        .ENABLE_IRQ(0)
    ) dut (
        .clk(clk), .resetn(resetn), .trap(trap),
        .mem_valid(mem_valid), .mem_instr(mem_instr), .mem_ready(mem_ready),
        .mem_addr(mem_addr), .mem_wdata(mem_wdata), .mem_wstrb(mem_wstrb),
        .mem_rdata(mem_rdata),
        .pcpi_wr(1'b0), .pcpi_rd(32'b0), .pcpi_wait(1'b0), .pcpi_ready(1'b0),
        .irq(32'b0)
    );

    // 256 KB flat memory.
    reg [31:0] mem [0:65535];
    integer i;
    reg [1023:0] memfile;
    initial begin
        for (i = 0; i < 65536; i = i + 1) mem[i] = 32'h0;
        if (!$value$plusargs("mem=%s", memfile)) begin
            $display("FATAL: pass +mem=<hexfile>"); $finish;
        end
        $readmemh(memfile, mem);
        $display("loaded program: %0s", memfile);
    end

    // Back-pressure on the magic store (cycles), set via +bp=N — mimics the UART's
    // reg_dat_wait so we can test the straddle + stalled-store interaction.
    integer bp = 0, bpc = 0;
    initial void'($value$plusargs("bp=%d", bp));

    // Behavioral mem: registered 1-cycle ready + rdata (matches progmem latency).
    always @(posedge clk) begin
        mem_ready <= 1'b0;
        if (mem_valid && !mem_ready) begin
            if (mem_addr == 32'h1000_0000) begin
                if (mem_wstrb != 0 && bpc < bp) begin
                    bpc <= bpc + 1;                 // hold ready low (back-pressure)
                end else begin
                    if (mem_wstrb != 0)
                        $display("[%0t] MAGIC <= 0x%08x", $time, mem_wdata);
                    mem_ready <= 1'b1;
                    bpc <= 0;
                end
            end else if (mem_addr < 32'h0004_0000) begin
                if (mem_wstrb != 0) begin
                    if (mem_wstrb[0]) mem[mem_addr[17:2]][7:0]   <= mem_wdata[7:0];
                    if (mem_wstrb[1]) mem[mem_addr[17:2]][15:8]  <= mem_wdata[15:8];
                    if (mem_wstrb[2]) mem[mem_addr[17:2]][23:16] <= mem_wdata[23:16];
                    if (mem_wstrb[3]) mem[mem_addr[17:2]][31:24] <= mem_wdata[31:24];
                end else begin
                    mem_rdata <= mem[mem_addr[17:2]];
                end
                mem_ready <= 1'b1;
            end else begin
                mem_ready <= 1'b1;          // ack strays so the CPU never wedges on the bus
            end
        end
    end

    // Ring buffer of the last 24 instruction-fetch addresses.
    reg [31:0] fetchlog [0:23];
    integer fp = 0, nfetch = 0;
    always @(posedge clk)
        if (resetn && mem_valid && mem_ready && mem_instr) begin
            fetchlog[fp] <= mem_addr;
            fp = (fp + 1) % 24;
            nfetch = nfetch + 1;
        end

    task dump_fetches;
        integer k, idx;
        begin
            $display("--- last instruction fetches (oldest first), total=%0d ---", nfetch);
            for (k = 0; k < 24; k = k + 1) begin
                idx = (fp + k) % 24;
                if (fetchlog[idx] !== 32'hx) $display("    pc=0x%08x", fetchlog[idx]);
            end
        end
    endtask

    // trap watch
    always @(posedge clk)
        if (resetn && trap) begin
            $display("[%0t] *** TRAP asserted (illegal/misaligned) ***", $time);
            dump_fetches;
            $finish;
        end

    integer cyc = 0;
    initial begin
        repeat (8) @(posedge clk);
        resetn = 1;
        $display("reset released");
    end
    always @(posedge clk) begin
        cyc = cyc + 1;
        if (cyc > 200000) begin
            $display("[%0t] TIMEOUT after %0d cycles (likely hang)", $time, cyc);
            dump_fetches;
            $finish;
        end
    end
endmodule
