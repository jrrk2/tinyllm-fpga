// tb_full.cpp — Verilator harness for smollm_full.sv (end-to-end Phase F.2).
// Drives an `in_token` and verifies the produced `next_token` matches the
// PyTorch reference (which ran the same in_token through embed + NL layers
// + final-norm + lm-head + argmax).

#include "Vsmollm_full.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

static Vsmollm_full* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsmollm_full;

    // Parse expected next_token + in_token from generated files.
    int expected_token = -1;
    int in_token       = -1;
    {
        std::ifstream f("../generated/full_expected.txt");
        if (!f) { fprintf(stderr, "FAIL: missing full_expected.txt\n"); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            // Header comment "in_token=N" — extract N
            if (line[0] == '#') {
                auto p = line.find("in_token=");
                if (p != std::string::npos)
                    std::sscanf(line.c_str() + p + 9, "%d", &in_token);
                continue;
            }
            if (std::sscanf(line.c_str(), "%d", &expected_token) == 1) break;
        }
    }
    if (expected_token < 0 || in_token < 0) {
        fprintf(stderr, "FAIL: couldn't parse expected_token / in_token\n");
        return 1;
    }
    printf("driving in_token=%d, expecting next_token=%d\n", in_token, expected_token);

    dut->rst      = 1;
    dut->start    = 0;
    dut->pos      = 3;
    dut->kv_pos   = 3;
    dut->in_token = in_token;
    for (int i = 0; i < 4; i++) tick();
    dut->rst   = 0;
    dut->start = 1;     // hold high for the run

    // Embed (D=128) + 3 layers × ~11K + decode head ~1.3K ≈ 35K cycles.
    const uint64_t timeout = 100000;
    while (!dut->done && cycle < timeout) tick();
    if (!dut->done) {
        fprintf(stderr, "FAIL: done never asserted (cycle=%lu)\n", (unsigned long)cycle);
        delete dut;
        return 1;
    }

    int dut_token = (int)dut->next_token;
    int dut_logit = (int16_t)dut->top_logit;
    printf("done=1 at cycle %lu\n", (unsigned long)cycle);
    printf("  dut next_token = %d  top_logit = %+d\n", dut_token, dut_logit);
    printf("  ref next_token = %d\n", expected_token);

    if (dut_token == expected_token) {
        printf("\nPASS: smollm_full produces correct next_token from in_token=%d\n",
               in_token);
        delete dut;
        return 0;
    } else {
        printf("\nFAIL: token mismatch (dut=%d, ref=%d)\n", dut_token, expected_token);
        delete dut;
        return 1;
    }
}
