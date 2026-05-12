// tb_swiglu_selftest.cpp — Verilator harness for swiglu_selftest.sv.
// Runs the selftest, then compares the N-lane Q1.15 result bus against
// ../generated/swiglu_selftest_expected.txt (±1 LSB tolerance for any
// rounding difference between numpy and the Q1.15 product shift).

#include "Vswiglu_selftest.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

static const int N = 64;
static Vswiglu_selftest* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vswiglu_selftest;

    std::vector<int16_t> expected;
    {
        std::ifstream f("../generated/swiglu_selftest_expected.txt");
        if (!f) { fprintf(stderr, "FAIL: missing swiglu_selftest_expected.txt\n"); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string hex; int dec;
            if (iss >> hex >> dec) expected.push_back((int16_t)dec);
        }
    }
    if ((int)expected.size() != N) {
        fprintf(stderr, "FAIL: expected %d ref values, got %zu\n", N, expected.size());
        return 1;
    }

    // Reset
    dut->rst = 1;
    dut->restart = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;

    // S_INIT (1) + N drive cycles + 1 latency cycle + done assertion ≈ 140.
    // Allow 4× headroom.
    uint64_t timeout = 600;
    while (!dut->done && cycle < timeout) tick();
    if (!dut->done) {
        fprintf(stderr, "FAIL: done never asserted (cycle=%lu)\n", (unsigned long)cycle);
        delete dut;
        return 1;
    }
    printf("done=1 at cycle %lu\n\n", (unsigned long)cycle);

    // Decode packed N×16-bit result bus into N lanes (lane 0 in [15:0]).
    int16_t lanes[N];
    {
        uint8_t* rb = reinterpret_cast<uint8_t*>(&dut->result);
        for (int l = 0; l < N; l++) {
            uint16_t v = rb[l*2] | (uint16_t(rb[l*2+1]) << 8);
            lanes[l] = (int16_t)v;
        }
    }

    int fails = 0, worst = 0;
    for (int l = 0; l < N; l++) {
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
               fails, N, worst);
        delete dut;
        return 1;
    }
    printf("PASS: swiglu_selftest produces all %d lanes within +-1 LSB "
           "(worst %+d) in Verilator\n", N, worst);
    delete dut;
    return 0;
}
