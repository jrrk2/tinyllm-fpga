// tb_rope_selftest.cpp — Verilator harness for rope_selftest.sv.
// Runs the selftest, then compares the HEAD_DIM-lane Q1.15 result bus against
// ../generated/rope_selftest_expected.txt (±1 LSB tolerance for rounding
// differences between the numpy reference and the fixed-point engine).

#include "Vrope_selftest.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

static const int HEAD_DIM = 64;
static Vrope_selftest* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vrope_selftest;

    std::vector<int16_t> expected;
    {
        std::ifstream f("../generated/rope_selftest_expected.txt");
        if (!f) { fprintf(stderr, "FAIL: missing rope_selftest_expected.txt\n"); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string hex; int dec;
            if (iss >> hex >> dec) expected.push_back((int16_t)dec);
        }
    }
    if ((int)expected.size() != HEAD_DIM) {
        fprintf(stderr, "FAIL: expected %d ref values, got %zu\n",
                HEAD_DIM, expected.size());
        return 1;
    }

    dut->rst = 1;
    dut->restart = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;

    // S_INIT (1) + rope S_IDLE→S_LOAD (1) + HEAD_DIM drive (64) +
    // CORDIC: HEAD_DIM/2 pairs × ~20 cycles each = ~640 cycles +
    // HEAD_DIM output (64) + S_DONE settle ≈ 710.
    // Allow generous headroom for CORDIC-based sin/cos (replaces LUT).
    uint64_t timeout = 800;
    while (!dut->done && cycle < timeout) tick();
    if (!dut->done) {
        fprintf(stderr, "FAIL: done never asserted (cycle=%lu)\n",
                (unsigned long)cycle);
        delete dut;
        return 1;
    }
    printf("done=1 at cycle %lu\n\n", (unsigned long)cycle);

    // Decode HEAD_DIM*16-bit packed result bus into HEAD_DIM lanes.
    // Verilator represents wide ports as uint32_t arrays, little-endian words.
    int16_t lanes[HEAD_DIM];
    {
        uint8_t* rb = reinterpret_cast<uint8_t*>(&dut->result);
        for (int l = 0; l < HEAD_DIM; l++) {
            uint16_t v = rb[l*2] | (uint16_t(rb[l*2+1]) << 8);
            lanes[l] = (int16_t)v;
        }
    }

    int fails = 0, worst = 0;
    for (int l = 0; l < HEAD_DIM; l++) {
        int diff = (int)lanes[l] - (int)expected[l];
        if (std::abs(diff) > std::abs(worst)) worst = diff;
        if (std::abs(diff) > 1) {
            printf("  lane %3d:  fpga=%+7d   ref=%+7d   diff=%+5d  FAIL\n",
                   l, lanes[l], expected[l], diff);
            fails++;
        }
    }
    if (fails) {
        printf("\nFAIL: %d/%d lanes mismatch (>1 LSB), worst diff %+d\n",
               fails, HEAD_DIM, worst);
        delete dut;
        return 1;
    }
    printf("PASS: rope_selftest produces all %d lanes within ±1 LSB "
           "(worst %+d) in Verilator\n", HEAD_DIM, worst);
    delete dut;
    return 0;
}
