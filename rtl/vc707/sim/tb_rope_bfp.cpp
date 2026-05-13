#include "Vtb_rope_bfp.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstdio>
static Vtb_rope_bfp* d;
static VerilatedVcdC* tfp;
static uint64_t cycle = 0;
static uint64_t sim_time = 0;
static void tick() {
    d->clk=0; d->eval(); if (tfp) tfp->dump(sim_time++);
    d->clk=1; d->eval(); if (tfp) tfp->dump(sim_time++);
    cycle++;
}
int main(int a, char** v){
    Verilated::commandArgs(a, v);
    Verilated::traceEverOn(true);
    d = new Vtb_rope_bfp;
    tfp = new VerilatedVcdC;
    d->trace(tfp, 99);
    tfp->open("rope_bfp.vcd");
    d->clk=0; d->rst=1; d->go=0; d->eval();
    for (int i=0;i<5;i++) tick();
    d->rst=0; tick();
    d->go=1; tick(); d->go=0;
    while (!d->done && cycle < 5000) tick();
    tfp->close();
    if (cycle >= 5000) { std::fprintf(stderr, "TIMEOUT\n"); delete d; return 1; }
    if (d->fail)       { std::fprintf(stderr, "FAIL (vcd: rope_bfp.vcd)\n"); delete d; return 1; }
    std::printf("PASS: rope_bfp cycle %lu (vcd: rope_bfp.vcd)\n", (unsigned long)cycle);
    delete d; return 0;
}
