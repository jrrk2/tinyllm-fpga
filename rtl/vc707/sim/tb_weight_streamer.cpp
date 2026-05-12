// tb_weight_streamer.cpp — exercises the FPGA-side weight_streamer.sv
// against mock_axi_slave (mem[i] = i pattern).  For each chunk c, in_dim k:
// after load completes, the streamer's weight_data at rd_addr=k must equal
// mem[(matrix_base/16) + c*in_dim + k] = (matrix_base/16) + c*in_dim + k.
//
// We test:
//   matrix_base = 0,    chunk = 0..3, in_dim = 64
//   matrix_base = 8192, chunk = 0..1, in_dim = 64    (offset matrix)
//
// Latencies: load_req → ready takes ~135 cycles (1 AR + 128 R beats + swap).
//            rd_addr → weight_data has 1-cycle registered latency.

#include "Vtb_weight_streamer_dut.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cinttypes>

static Vtb_weight_streamer_dut* dut = nullptr;
static uint64_t cycle = 0;

static void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycle++;
}

static int load_and_check(uint32_t matrix_base, uint32_t chunk, uint32_t in_dim) {
    // Pulse load_req
    dut->matrix_base = matrix_base;
    dut->chunk_idx   = chunk;
    dut->in_dim      = in_dim;
    dut->load_req    = 1;
    tick();
    dut->load_req    = 0;

    // Wait for ready (with timeout)
    uint64_t t0 = cycle;
    while (!dut->ready && (cycle - t0) < 1000) tick();
    if (!dut->ready) {
        fprintf(stderr, "FAIL: ready never asserted (chunk=%u, in_dim=%u)\n",
                chunk, in_dim);
        return 1;
    }
    printf("  chunk=%u: ready after %" PRIu64 " cycles\n", chunk, cycle - t0);

    // Probe each rd_addr.  weight_data is registered with 1-cycle latency,
    // so we drive rd_addr at cycle K and sample at cycle K+1.
    int errors = 0;
    for (uint32_t k = 0; k < in_dim; k++) {
        dut->rd_addr = k;
        tick();
        // After this tick weight_data reflects the previous rd_addr's read.
        // Probe one more tick to capture this addr's value.
        tick();
        uint64_t expected_lo = (uint64_t)(matrix_base >> 4) + (uint64_t)chunk * in_dim + k;
        // weight_data is 128-bit; low 64 bits should equal expected_lo
        // (mem[i] = 128'(i) — high bits zero for entries < 2^64).
        // Verilator exposes it as VlWide<4> for 128-bit signal: dut->weight_data is uint32_t[4]
        uint64_t got_lo = ((uint64_t)dut->weight_data[1] << 32) | dut->weight_data[0];
        uint64_t got_hi = ((uint64_t)dut->weight_data[3] << 32) | dut->weight_data[2];
        if (got_lo != expected_lo || got_hi != 0) {
            if (errors < 10) {
                fprintf(stderr,
                    "FAIL: chunk=%u rd_addr=%u  got=0x%016" PRIx64 "%016" PRIx64
                    "  expected=0x%016" PRIx64 "\n",
                    chunk, k, got_hi, got_lo, expected_lo);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("  chunk=%u: all %u rd_addr probes match\n", chunk, in_dim);
    }
    return errors;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vtb_weight_streamer_dut;

    dut->rst         = 1;
    dut->matrix_base = 0;
    dut->chunk_idx   = 0;
    dut->in_dim      = 0;
    dut->load_req    = 0;
    dut->rd_addr     = 0;
    for (int i = 0; i < 4; i++) tick();
    dut->rst = 0;
    tick();

    int total_errors = 0;

    // Test 1: in_dim=64, 4 chunks (single-tile path)
    printf("\n--- matrix_base=0, in_dim=64 (single-tile) ---\n");
    for (uint32_t c = 0; c < 4; c++) total_errors += load_and_check(0, c, 64);

    // Test 2: in_dim=576, 2 chunks (2 tiles per chunk, like SmolLM2 Q/K/V/O/GATE/UP)
    printf("\n--- matrix_base=0, in_dim=576 (2-tile) ---\n");
    for (uint32_t c = 0; c < 2; c++) total_errors += load_and_check(0, c, 576);

    // Test 3: in_dim=1536, 1 chunk (3 tiles, like SmolLM2 DOWN-proj)
    printf("\n--- matrix_base=0, in_dim=1536 (3-tile) ---\n");
    total_errors += load_and_check(0, 0, 1536);

    // Test 4: matrix_base offset (chunk 0 of "second matrix")
    printf("\n--- matrix_base=8192, in_dim=576 (2-tile, offset) ---\n");
    total_errors += load_and_check(8192, 0, 576);

    if (total_errors == 0) {
        printf("\nPASS: weight_streamer matches mem[i]=i across all probes\n");
        delete dut;
        return 0;
    } else {
        printf("\nFAIL: %d mismatches\n", total_errors);
        delete dut;
        return 1;
    }
}
