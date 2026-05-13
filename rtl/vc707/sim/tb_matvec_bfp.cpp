// tb_matvec_bfp.cpp — Verilator driver for the matvec_bfp engine unit test.
//
// Strobes go, ticks clock, waits for done, reports fail/pass.

#include "Vtb_matvec_bfp.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>

static Vtb_matvec_bfp* dut;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtb_matvec_bfp;

    dut->clk = 0; dut->rst = 1; dut->go = 0; dut->eval();
    for (int i = 0; i < 5; i++) tick();
    dut->rst = 0; dut->eval();
    tick();

    dut->go = 1;
    tick();
    dut->go = 0;

    const uint64_t TIMEOUT = 10000;
    while (!dut->done && cycle < TIMEOUT) tick();

    if (cycle >= TIMEOUT) {
        std::fprintf(stderr, "TIMEOUT: done never asserted (%lu cycles)\n", (unsigned long)cycle);
        delete dut; return 1;
    }

    if (dut->fail) {
        std::fprintf(stderr, "FAIL at cycle %lu (see lane-level $display)\n", (unsigned long)cycle);
        delete dut; return 1;
    }
    std::printf("PASS: matvec_bfp_engine done at cycle %lu\n", (unsigned long)cycle);
    delete dut; return 0;
}
