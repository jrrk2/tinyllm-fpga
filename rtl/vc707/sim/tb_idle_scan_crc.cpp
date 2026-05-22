// tb_idle_scan_crc.cpp — drives idle_scan_crc over mock_axi_slave (mem[i]=i)
// and checks the rolling hash against a reference computed the same way the
// firmware's scan_ref() does, so a hash-convention bug (rotl direction, fold
// order, init value) is caught here before a 3-hour FPGA build.
//
// mock layout: a 512-bit beat at byte addr A = {mem[E+3],mem[E+2],mem[E+1],
// mem[E+0]}, E = A/16, mem[i]=i.  So beat k (base B) uses entries e=B/16+4k..+3,
// and (entries < 2^32) its sixteen 32-bit words XOR down to e^(e+1)^(e+2)^(e+3).
#include "Vtb_idle_scan_crc_dut.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

static Vtb_idle_scan_crc_dut* dut;
static uint64_t cycle = 0;
static void tick() { dut->clk = 0; dut->eval(); dut->clk = 1; dut->eval(); cycle++; }

static uint32_t rotl1(uint32_t x) { return (x << 1) | (x >> 31); }

static uint32_t ref_crc(uint32_t base_bytes, uint32_t nbeats) {
    uint32_t crc = 0xFFFFFFFFu, be = base_bytes >> 4;
    for (uint32_t k = 0; k < nbeats; k++) {
        uint32_t e = be + 4 * k;
        uint32_t fold = e ^ (e + 1) ^ (e + 2) ^ (e + 3);
        crc = rotl1(crc) ^ fold;
    }
    return crc;
}

static int run_scan(uint32_t base, uint32_t len) {
    dut->base = base; dut->len = len;
    dut->trig = 1; tick(); dut->trig = 0;
    uint64_t t0 = cycle;
    while (!dut->done && (cycle - t0) < 200000) tick();
    if (!dut->done) { fprintf(stderr, "FAIL: done never asserted (len=%u)\n", len); return 1; }
    uint32_t hw = dut->crc, rf = ref_crc(base, len);
    printf("  base=0x%06x len=%-4u  hw=0x%08x ref=0x%08x  %s\n",
           base, len, hw, rf, hw == rf ? "ok" : "MISMATCH");
    return hw == rf ? 0 : 1;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtb_idle_scan_crc_dut;
    dut->rst = 1; dut->base = 0; dut->len = 0; dut->trig = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0; tick();

    int e = 0;
    printf("\n--- idle_scan_crc vs mock_axi_slave (mem[i]=i) ---\n");
    // base=0 is degenerate for mem[i]=i (4-aligned folds XOR to 0 -> hash stays
    // at init); keep it as an init/FSM check but lean on offset bases (entry not
    // a multiple of 4) to get non-zero folds that exercise the rotate-accumulate.
    e += run_scan(0,  1);     // single beat (init + fold=0 path)
    e += run_scan(0,  16);
    e += run_scan(16, 1);     // entry 1: fold = 1^2^3^4 = 4 (non-zero)
    e += run_scan(16, 16);    // rolling rotate-accumulate over non-zero folds
    e += run_scan(48, 64);    // entry 3, longer scan
    e += run_scan(16, 255);   // many beats

    if (e == 0) { printf("\nPASS: scan-CRC matches reference across all scans\n"); delete dut; return 0; }
    printf("\nFAIL: %d mismatches\n", e); delete dut; return 1;
}
