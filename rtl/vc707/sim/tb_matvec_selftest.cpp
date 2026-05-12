// tb_matvec_selftest.cpp — Verilator harness for matvec_selftest.sv.
//
// Resets, ticks until `done` goes high, then compares the 256-bit `result`
// bus to the per-lane expected values in ../generated/matvec_selftest_expected.txt.

#include "Vmatvec_selftest.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

static Vmatvec_selftest* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vmatvec_selftest;

    // Read the per-lane reference (signed decimal in column 2)
    std::vector<int16_t> expected;
    {
        std::ifstream f("../generated/matvec_selftest_expected.txt");
        if (!f) { fprintf(stderr, "FAIL: missing expected.txt\n"); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string hex; int dec;
            if (iss >> hex >> dec) expected.push_back((int16_t)dec);
        }
    }
    if (expected.size() != 16) {
        fprintf(stderr, "FAIL: expected 16 reference values, got %zu\n", expected.size());
        return 1;
    }

    // Reset
    dut->rst = 1;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;

    // Tick until done or timeout (FSM should finish in ~70 cycles)
    uint64_t timeout = 256;
    while (!dut->done && cycle < timeout) tick();

    if (!dut->done) {
        fprintf(stderr, "FAIL: done never asserted (cycle=%lu)\n", (unsigned long)cycle);
        delete dut;
        return 1;
    }
    printf("done=1 at cycle %lu\n\n", (unsigned long)cycle);

    // Decode the 256-bit result bus into 16 lanes.  Verilator stores it as
    // VlWide<8> (8 × 32-bit).  Cast to byte array, low byte = lane 0 [7:0].
    int16_t lanes[16];
    {
        uint8_t* rb = reinterpret_cast<uint8_t*>(&dut->result);
        for (int l = 0; l < 16; l++) {
            uint16_t v = rb[l*2] | (uint16_t(rb[l*2+1]) << 8);
            lanes[l] = (int16_t)v;
        }
    }

    int fails = 0;
    printf("lane :   dut       ref     diff\n");
    for (int l = 0; l < 16; l++) {
        int diff = (int)lanes[l] - (int)expected[l];
        const char* tag = (diff == 0) ? "ok" : "FAIL";
        printf("%4d : %+7d   %+7d   %+5d  %s\n",
               l, lanes[l], expected[l], diff, tag);
        if (diff != 0) fails++;
    }
    if (fails) {
        printf("\nFAIL: %d/16 lanes mismatched\n", fails);
        delete dut;
        return 1;
    }
    printf("\nPASS: matvec_selftest.sv reproduces all 16 lanes bit-exact in Verilator\n");
    delete dut;
    return 0;
}
