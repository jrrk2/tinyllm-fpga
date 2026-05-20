#include "Vtb_smollm_layer_bfp.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>

#if VM_TRACE
# include "verilated_vcd_c.h"
static VerilatedVcdC* tfp = nullptr;
#endif

static Vtb_smollm_layer_bfp* d;
static uint64_t cycle    = 0;
static uint64_t sim_time = 0;  // 2 time-units per cycle for clean VCD waves
static void tick() {
  d->clk = 0; d->eval();
#if VM_TRACE
  if (tfp) tfp->dump(sim_time);
#endif
  sim_time++;
  d->clk = 1; d->eval();
#if VM_TRACE
  if (tfp) tfp->dump(sim_time);
#endif
  sim_time++;
  cycle++;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  d = new Vtb_smollm_layer_bfp;
#if VM_TRACE
  if (const char* p = std::getenv("LBFP_VCD")) {
    Verilated::traceEverOn(true);
    tfp = new VerilatedVcdC;
    d->trace(tfp, /*levels*/ 99);
    tfp->open(p);
    std::fprintf(stderr, "[tb] VCD dumping to %s\n", p);
  }
#endif
  d->clk = 0; d->rst = 1; d->go = 0; d->eval();
  for (int i = 0; i < 8; i++) tick();
  d->rst = 0; tick();
  d->go = 1; tick();
  // Long limit — layer with D=64/FFN=128 + chunked LANES=16 takes ~30K cycles
  const uint64_t LIM = 2000000;
  while (!d->done && cycle < LIM) tick();
  if (cycle >= LIM) { std::fprintf(stderr, "TIMEOUT @ cycle %llu\n",
                                   (unsigned long long)cycle);
#if VM_TRACE
                      if (tfp) { tfp->close(); delete tfp; }
#endif
                      delete d; return 1; }
  if (d->fail) { std::fprintf(stderr, "FAIL @ cycle %llu\n",
                              (unsigned long long)cycle);
#if VM_TRACE
                 if (tfp) { tfp->close(); delete tfp; }
#endif
                 delete d; return 1; }
  std::printf("PASS: smollm_layer_bfp cycle %llu\n", (unsigned long long)cycle);
#if VM_TRACE
  if (tfp) { tfp->close(); delete tfp; }
#endif
  delete d; return 0;
}
