#include "Vtb_rmsnorm_bfp.h"
#include "verilated.h"
#include <cstdio>

static Vtb_rmsnorm_bfp* dut;
static uint64_t cycle = 0;
static void tick() { dut->clk=0; dut->eval(); dut->clk=1; dut->eval(); cycle++; }

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtb_rmsnorm_bfp;
    dut->clk=0; dut->rst=1; dut->go=0; dut->eval();
    for (int i=0;i<5;i++) tick();
    dut->rst=0; dut->eval(); tick();
    dut->go=1; tick(); dut->go=0;
    while (!dut->done && cycle < 10000) tick();
    if (cycle >= 10000) { std::fprintf(stderr, "TIMEOUT cycle=%lu\n", (unsigned long)cycle); delete dut; return 1; }
    if (dut->fail)      { std::fprintf(stderr, "FAIL cycle=%lu\n", (unsigned long)cycle); delete dut; return 1; }
    std::printf("PASS: rmsnorm_bfp done cycle %lu\n", (unsigned long)cycle);
    delete dut; return 0;
}
