#include "Vtb_smollm_layer_bfp.h"
#include "verilated.h"
#include <cstdio>

static Vtb_smollm_layer_bfp* d;
static uint64_t cycle = 0;
static void tick() { d->clk = 0; d->eval(); d->clk = 1; d->eval(); cycle++; }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  d = new Vtb_smollm_layer_bfp;
  d->clk = 0; d->rst = 1; d->go = 0; d->eval();
  for (int i = 0; i < 8; i++) tick();
  d->rst = 0; tick();
  d->go = 1; tick();
  // Long limit — layer with D=64/FFN=128 + chunked LANES=16 takes ~30K cycles
  const uint64_t LIM = 2000000;
  while (!d->done && cycle < LIM) tick();
  if (cycle >= LIM) { std::fprintf(stderr, "TIMEOUT @ cycle %llu\n",
                                   (unsigned long long)cycle);
                      delete d; return 1; }
  if (d->fail) { std::fprintf(stderr, "FAIL @ cycle %llu\n",
                              (unsigned long long)cycle);
                 delete d; return 1; }
  std::printf("PASS: smollm_layer_bfp cycle %llu\n", (unsigned long long)cycle);
  delete d; return 0;
}
