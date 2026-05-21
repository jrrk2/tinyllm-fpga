// tb_soc_matvec.cpp — driver for the PicoSoC weight-feed (matvec) proof.
// Boots the SoC, lets firmware feed an identity matvec beat-by-beat, waits for
// the captured result, then checks out[lane] ~= x[lane] = (lane+1)*1000.
#include "Vtb_soc_matvec.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cmath>

static int s16(uint32_t v) { v &= 0xFFFF; return (v & 0x8000) ? (int)v - 0x10000 : (int)v; }
static int s8 (uint32_t v) { v &= 0xFF;   return (v & 0x80)   ? (int)v - 0x100   : (int)v; }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vtb_soc_matvec* d = new Vtb_soc_matvec;

  d->clk = 0; d->resetn = 0; d->eval();
  for (int i = 0; i < 16; ++i) { d->clk = 0; d->eval(); d->clk = 1; d->eval(); }
  d->resetn = 1;

  const long budget = 3'000'000;
  long c = 0; bool cap = false;
  for (; c < budget && !cap; ++c) {
    d->clk = 0; d->eval();
    d->clk = 1; d->eval();
    if (d->out_captured) cap = true;
  }

  if (!cap) { std::printf("FAIL: matvec result never captured (SoC feed stalled)\n"); delete d; return 1; }

  // out_mant: 256b (16 lanes x 16b); out_exp: 128b (16 lanes x 8b).
  int fails = 0;
  std::printf("lane: got (decoded)  vs  expected\n");
  for (int lane = 0; lane < 16; ++lane) {
    uint32_t mw = d->out_mant[lane / 2];
    int mant = s16(mw >> ((lane & 1) * 16));
    uint32_t ew = d->out_exp[lane / 4];
    int expo = s8(ew >> ((lane & 3) * 8));
    double got = (double)mant * std::ldexp(1.0, expo - 15);
    double exp_val = (double)((lane + 1) * 1000);
    bool ok = std::fabs(got - exp_val) <= 0.01 * exp_val + 1.0;
    if (lane < 8 || !ok)
      std::printf("  %2d: %12.2f  vs  %8.0f  %s\n", lane, got, exp_val, ok ? "" : "<-- MISMATCH");
    if (!ok) ++fails;
  }

  std::printf("captured after %ld cycles, %d/16 lanes mismatch\n", c, fails);
  if (fails) { std::printf("SOC WEIGHT-FEED MATVEC: FAIL\n"); delete d; return 1; }
  std::printf("SOC WEIGHT-FEED MATVEC: PASS (engine computed correctly off SoC-fed weights)\n");
  delete d;
  return 0;
}
