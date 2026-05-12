// tb_softmax_selftest.cpp — Verilator harness for softmax_selftest.sv.
// Runs the selftest, then compares the N-lane Q1.15 result bus against
// ../generated/softmax_selftest_expected.txt (±10 LSB tolerance for the
// combined error of the exp LUT step and the NR reciprocal).

#include "Vsoftmax_selftest.h"
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
static const int TOLERANCE = 10;   // ±10 LSB (wider than rmsnorm due to exp LUT + recip)
static Vsoftmax_selftest* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsoftmax_selftest;

    std::vector<int16_t> expected;
    {
        std::ifstream f("../generated/softmax_selftest_expected.txt");
        if (!f) { fprintf(stderr, "FAIL: missing softmax_selftest_expected.txt\n"); return 1; }
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

    dut->rst = 1;
    dut->restart = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;

    // Pipeline: S_INIT(1) + S_DRIVE(N) + S_EXP(N) + S_RECIP(6) + S_OUTPUT(N) + done(1)
    // = 1 + 64 + 64 + 6 + 64 + 1 = 200 cycles typical.
    // Allow 5000 cycles for the sim-only recompute path in S_OUTPUT.
    uint64_t timeout = 5000;
    while (!dut->done && cycle < timeout) tick();
    if (!dut->done) {
        fprintf(stderr, "FAIL: done never asserted (cycle=%lu)\n", (unsigned long)cycle);
        delete dut;
        return 1;
    }
    printf("done=1 at cycle %lu\n\n", (unsigned long)cycle);

    // Decode N*16-bit packed result bus into N lanes.
    // The result port is exposed as a flat array of 32-bit words by Verilator.
    int16_t lanes[N];
    {
        uint8_t* rb = reinterpret_cast<uint8_t*>(dut->result.data());
        for (int l = 0; l < N; l++) {
            uint16_t v = rb[l*2] | (uint16_t(rb[l*2+1]) << 8);
            lanes[l] = (int16_t)v;
        }
    }

    int fails = 0, worst = 0;
    for (int l = 0; l < N; l++) {
        int diff = (int)lanes[l] - (int)expected[l];
        if (std::abs(diff) > std::abs(worst)) worst = diff;
        if (std::abs(diff) > TOLERANCE) {
            printf("  lane %3d:  fpga=%+7d   ref=%+7d   diff=%+5d  FAIL\n",
                   l, lanes[l], expected[l], diff);
            fails++;
        }
    }
    if (fails) {
        printf("\nFAIL: %d/%d lanes mismatch (>%d LSB), worst diff %+d\n",
               fails, N, TOLERANCE, worst);
        delete dut;
        return 1;
    }
    printf("PASS: softmax_selftest produces all %d lanes within ±%d LSB "
           "(worst %+d) in Verilator\n", N, TOLERANCE, worst);
    delete dut;
    return 0;
}
