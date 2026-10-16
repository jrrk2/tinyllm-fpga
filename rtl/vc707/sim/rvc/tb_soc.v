// tb_soc.v — run the REAL fw_engine firmware on picosoc_noflash (real picorv32 +
// simpleuart + inferred progmem), the exact FPGA front-end minus the external
// eth/regmap peripherals (stubbed: acked, reads return 0 -> "no eth frame", just
// like an idle board).  We DON'T ack iomem 0x02 (UART) — that's internal to the
// SoC and governed by the UART's own back-pressure (acking it early is the known
// char-drop bug).  Every byte the CPU writes to reg_uart_data is printed, so the
// transcript shows whether compressed firmware boots the full banner or dies
// after the first newline (the reported FPGA symptom).
`timescale 1ns/1ps

module tb_soc;
    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;                 // 100 MHz

    wire        iomem_valid;
    reg         iomem_ready = 0;
    wire [3:0]  iomem_wstrb;
    wire [31:0] iomem_addr, iomem_wdata;
    reg  [31:0] iomem_rdata = 0;
    wire        ser_tx;

    picosoc_noflash dut (
        .clk(clk), .resetn(resetn),
        .iomem_valid(iomem_valid), .iomem_ready(iomem_ready),
        .iomem_wstrb(iomem_wstrb), .iomem_addr(iomem_addr),
        .iomem_wdata(iomem_wdata), .iomem_rdata(iomem_rdata),
        .irq_5(1'b0), .irq_6(1'b0), .irq_7(1'b0),
        .ser_tx(ser_tx), .ser_rx(1'b1),
        .trace_valid(), .trace_data()
    );

    // iomem stub: ack everything EXCEPT 0x02 (UART — internal back-pressure).
    // Reads return 0: eth RSR bit15=0 (no frame), regmap reads 0.
    always @(posedge clk) begin
        iomem_ready <= 1'b0;
        if (iomem_valid && !iomem_ready && iomem_addr[31:24] != 8'h02) begin
            iomem_ready <= 1'b1;
            iomem_rdata <= 32'h0;
        end
    end

    // Print every byte the CPU writes to reg_uart_data (0x02000008).
    always @(posedge clk)
        if (resetn && dut.mem_valid && dut.mem_ready &&
            (dut.mem_wstrb != 0) && dut.mem_addr == 32'h0200_0008)
            $write("%c", dut.mem_wdata[7:0]);

    // Hang/trap watch: ring buffer of the last 32 instruction-fetch PCs.
    reg [31:0] flog [0:31];
    integer fp = 0, nf = 0, k, idx;
    always @(posedge clk)
        if (resetn && dut.mem_valid && dut.mem_ready && dut.mem_instr) begin
            flog[fp] <= dut.mem_addr; fp = (fp + 1) % 32; nf = nf + 1;
        end
    task dump; begin
        $display("\n--- last fetch PCs (oldest first), total fetches=%0d ---", nf);
        for (k = 0; k < 32; k = k + 1) begin
            idx = (fp + k) % 32;
            if (flog[idx] !== 32'hx) $display("    pc=0x%08x", flog[idx]);
        end
    end endtask

    always @(posedge clk)
        if (resetn && dut.cpu.trap) begin
            $display("\n[%0t] *** picorv32 TRAP (illegal/misaligned) ***", $time);
            dump; $finish;
        end

    // Stall detector: mem access asserted but never acknowledged.
    integer stall = 0;
    always @(posedge clk) begin
        if (resetn && dut.mem_valid && !dut.mem_ready) stall = stall + 1;
        else stall = 0;
        if (stall == 60000) begin
            $display("\n[%0t] STALL: mem_valid held >%0d cyc with no ready", $time, stall);
            $display("    addr=0x%08x  wstrb=%b  instr=%b  wdata=0x%08x",
                     dut.mem_addr, dut.mem_wstrb, dut.mem_instr, dut.mem_wdata);
            $display("    UART: cfg_divider=%0d (0x%08x) send_bitcnt=%0d send_dummy=%b send_divcnt=%0d",
                     dut.simpleuart.cfg_divider, dut.simpleuart.cfg_divider,
                     dut.simpleuart.send_bitcnt, dut.simpleuart.send_dummy,
                     dut.simpleuart.send_divcnt);
            dump; $finish;
        end
    end

    integer cyc = 0;
    initial begin
        repeat (8) @(posedge clk); resetn = 1;
        $display("// reset released; firmware UART transcript follows:");
        $display("//----------------------------------------------------");
    end
    always @(posedge clk) begin
        cyc = cyc + 1;
        if (cyc > 4000000) begin
            $display("\n//---- end (%0d cycles) ----", cyc); dump; $finish;
        end
    end
endmodule
