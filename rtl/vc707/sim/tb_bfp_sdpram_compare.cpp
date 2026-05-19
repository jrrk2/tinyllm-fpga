// tb_bfp_sdpram_compare.cpp — clock driver + pass/fail summary for the
// constrained-random bfp_sdpram equivalence harness.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include "Vtb_bfp_sdpram_compare.h"
#include "verilated.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    auto dut = std::make_unique<Vtb_bfp_sdpram_compare>();

    // Reset 5 cycles.
    dut->clk = 0; dut->rst = 1;
    for (int i = 0; i < 10; ++i) {
        dut->clk = !dut->clk;
        dut->eval();
    }
    dut->rst = 0;

    uint64_t cycles = 0;
    const uint64_t kMaxCycles = 200000;
    while (!dut->done && cycles < kMaxCycles) {
        dut->clk = !dut->clk;
        dut->eval();
        if (dut->clk) ++cycles;
    }

    if (cycles >= kMaxCycles) {
        std::fprintf(stderr, "[tb_bfp_sdpram_compare] TIMEOUT @ %lu cycles\n",
                     (unsigned long)cycles);
        return 2;
    }

    int errors = dut->error_count;
    int driven = dut->cycle_count;
    std::printf("[tb_bfp_sdpram_compare] %d cycles driven, %d mismatches\n",
                driven, errors);
    if (errors == 0 && dut->pass) {
        std::printf("PASS — bfp_sdpram is bit-identical to inferred SDPRAM\n");
        return 0;
    } else {
        std::printf("FAIL — bfp_sdpram diverges from inferred SDPRAM\n");
        return 1;
    }
}
