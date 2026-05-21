// tb_soc_regmap.cpp — driver for the stage-1 PicoSoC->regmap proof.
// Boots the SoC, runs firmware, and checks the value it wrote/read-back/derived
// through the iomem bridge.
#include "Vtb_soc_regmap.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vtb_soc_regmap* d = new Vtb_soc_regmap;

  // Reset.
  d->clk = 0; d->resetn = 0; d->eval();
  for (int i = 0; i < 16; ++i) { d->clk = 0; d->eval(); d->clk = 1; d->eval(); }
  d->resetn = 1;

  const long budget = 3'000'000;
  bool done = false;
  long c = 0;
  for (; c < budget && !done; ++c) {
    d->clk = 0; d->eval();
    d->clk = 1; d->eval();
    if (d->dbg_done) done = true;
  }

  std::printf("after %ld cycles: done=%d  r1=0x%08x  r2=0x%08x  r3=0x%08x\n",
              c, d->dbg_done, d->dbg_r1, d->dbg_r2, d->dbg_r3);

  int rc = 0;
  if (!done) { std::printf("FAIL: firmware never raised the done sentinel\n"); rc = 1; }
  else if (d->dbg_r2 != 0xCAFE1234u) { std::printf("FAIL: readback r2 != 0xCAFE1234 (read path broken)\n"); rc = 1; }
  else if (d->dbg_r3 != 0xCAFE1235u) { std::printf("FAIL: derived r3 != 0xCAFE1235 (CPU/compute broken)\n"); rc = 1; }
  else std::printf("SOC->REGMAP BRIDGE: PASS\n");

  delete d;
  return rc;
}
