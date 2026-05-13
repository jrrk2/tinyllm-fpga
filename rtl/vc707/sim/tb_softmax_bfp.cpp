#include "Vtb_softmax_bfp.h"
#include "verilated.h"
#include <cstdio>
static Vtb_softmax_bfp* d;
static uint64_t cycle = 0;
static void tick() { d->clk=0; d->eval(); d->clk=1; d->eval(); cycle++; }
int main(int a, char** v){
    Verilated::commandArgs(a, v);
    d = new Vtb_softmax_bfp;
    d->clk=0; d->rst=1; d->go=0; d->eval();
    for (int i=0;i<5;i++) tick();
    d->rst=0; tick();
    d->go=1; tick(); d->go=0;
    while (!d->done && cycle < 5000) tick();
    if (cycle >= 5000) { std::fprintf(stderr, "TIMEOUT\n"); delete d; return 1; }
    if (d->fail)       { std::fprintf(stderr, "FAIL\n"); delete d; return 1; }
    std::printf("PASS: softmax_bfp cycle %lu\n", (unsigned long)cycle);
    delete d; return 0;
}
