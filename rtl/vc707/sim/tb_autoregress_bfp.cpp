// tb_autoregress_bfp.cpp — drives the on-chip autoregress demo to
// completion and dumps the resulting token sequence as "RTL_TOKENS:".
// A separate host script can pipe this through the SmolLM2 tokenizer to
// render the generated story.

#include "Vtb_autoregress_bfp.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

static Vtb_autoregress_bfp* d;
static uint64_t cycle = 0;
static void tick() { d->clk = 0; d->eval(); d->clk = 1; d->eval(); ++cycle; }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  d = new Vtb_autoregress_bfp;
  d->clk = 0; d->rst = 1; d->go = 0;
  d->eval();
  for (int i = 0; i < 8; ++i) tick();
  d->rst = 0; tick();
  d->go = 1; tick();
  d->go = 0;

  const uint64_t LIM = 1000000000ULL;
  std::fprintf(stderr, "[tb_autoregress_bfp] running on-chip autoregress…\n");
  while (!d->done && cycle < LIM) tick();
  if (cycle >= LIM) {
    std::fprintf(stderr, "TIMEOUT @ cycle %llu\n", (unsigned long long)cycle);
    delete d; return 1;
  }

  // Pull all N_STEPS tokens out of the packed result bus.  N_STEPS is
  // baked into the SV via `LBFP_FULL_NPROMPT + `LBFP_FULL_NGEN; we mirror
  // it here at compile time from a corresponding C++ header that the
  // baker emits alongside cfg.svh.
  std::printf("RTL_TOKENS:");
  const int N_STEPS = 19;            // matches `LBFP_FULL_NPROMPT (4) + `LBFP_FULL_NGEN (15)
  for (int i = 0; i < N_STEPS; ++i) {
    int word = (i * 16) / 32;
    int shift = (i * 16) % 32;
    uint16_t tok = (d->result_tokens[word] >> shift) & 0xFFFF;
    std::printf(" %u", (unsigned)tok);
  }
  std::printf("\n");
  std::fprintf(stderr, "[tb_autoregress_bfp] done @ cycle %llu\n",
               (unsigned long long)cycle);
  delete d; return 0;
}
