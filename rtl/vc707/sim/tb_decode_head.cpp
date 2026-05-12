// tb_decode_head.cpp — Verilator harness for smollm_decode_head.sv.
// Loads a hidden_state and verifies the head produces the expected
// argmax token from the PyTorch reference.

#include "Vsmollm_decode_head.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

static const int D = 128;

static Vsmollm_decode_head* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsmollm_decode_head;

    // Load hidden_state from hex
    std::vector<int16_t> hin;
    {
        std::ifstream f("../generated/decodehead_HIDDEN_IN.hex");
        if (!f) { fprintf(stderr, "FAIL: missing decodehead_HIDDEN_IN.hex\n"); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            unsigned v;
            if (std::sscanf(line.c_str(), "%x", &v) == 1)
                hin.push_back((int16_t)v);
        }
    }
    if ((int)hin.size() != D) {
        fprintf(stderr, "FAIL: hidden_state size %zu vs D=%d\n", hin.size(), D);
        return 1;
    }

    // Load expected token
    int expected_token = -1;
    {
        std::ifstream f("../generated/decodehead_expected.txt");
        if (!f) { fprintf(stderr, "FAIL: missing expected.txt\n"); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            int t;
            if (std::sscanf(line.c_str(), "%d", &t) == 1) {
                expected_token = t;
                break;
            }
        }
    }
    if (expected_token < 0) {
        fprintf(stderr, "FAIL: couldn't parse expected_token\n");
        return 1;
    }

    dut->rst   = 1;
    dut->start = 0;
    {
        uint8_t* hb = reinterpret_cast<uint8_t*>(&dut->hidden_state);
        for (int l = 0; l < D; l++) {
            uint16_t u = (uint16_t)hin[l];
            hb[l*2 + 0] = u & 0xFF;
            hb[l*2 + 1] = u >> 8;
        }
    }
    for (int i = 0; i < 4; i++) tick();
    dut->rst   = 0;
    dut->start = 1;
    tick();
    dut->start = 0;

    // Norm (~D + 5 + D ≈ 260) + LM head (~8 chunks × (D+4) ≈ 1100) + glue
    const uint64_t timeout = 5000;
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

    // Tolerate the case where two logits tie within ±1 LSB and argmax
    // could pick either — in such cases warn but PASS if dut top_logit
    // is within ±1 of ref top_logit.
    if (dut_token == expected_token) {
        printf("\nPASS: smollm_decode_head argmax matches reference (token %d)\n",
               dut_token);
        delete dut;
        return 0;
    } else {
        printf("\nFAIL: token mismatch (dut=%d, ref=%d)\n", dut_token, expected_token);
        delete dut;
        return 1;
    }
}
