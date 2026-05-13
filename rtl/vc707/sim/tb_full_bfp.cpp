// tb_full_bfp.cpp — autoregressive token-loop harness for tb_full_bfp.sv.
//
// Reads the BFP-Python golden token list and the SmolLM2 prompt tokens
// baked by host/gen_smollm_blockfp_full.py.  For each step:
//   - drives token_in (prefill prompt token or last generated token)
//   - sets pos, kv_pos, pulses start
//   - waits for `done`, captures token_out
//   - compares to golden, prints PASS/FAIL per token
//
// No embed lookup needed in C++ — the RTL has embed_lookup_bfp inside.

#include "Vtb_full_bfp.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

static Vtb_full_bfp* d;
static uint64_t cycle_count = 0;
static void tick() { d->clk = 0; d->eval(); d->clk = 1; d->eval(); ++cycle_count; }

static std::vector<int> read_tokens(const std::string& path) {
  std::vector<int> out;
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "ERROR: can't open %s\n", path.c_str()); std::exit(1); }
  int t;
  while (f >> t) out.push_back(t);
  return out;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  d = new Vtb_full_bfp;

  auto prompt = read_tokens("../generated/lbfp_full_PROMPT_TOKENS.txt");
  auto golden = read_tokens("../generated/lbfp_full_GOLDEN_TOKENS.txt");
  std::fprintf(stderr, "[tb_full_bfp] prompt=%zu tokens, golden=%zu tokens\n",
               prompt.size(), golden.size());

  d->clk = 0; d->rst = 1; d->start = 0;
  d->token_in = 0; d->pos = 0; d->kv_pos = 0;
  d->eval();
  for (int i = 0; i < 8; ++i) tick();
  d->rst = 0; tick();

  // golden[] starts with the prompt tokens too — golden[0..len-1] == prompt.
  // We step (len(prompt) + N_GEN - 1) times total (prompt steps fill KV;
  // first NEW token comes from the LAST prompt step's output).
  const int N_STEPS = (int)golden.size() - 1;     // last golden has no follow-up
  std::vector<int> generated;
  int pass_count = 0, fail_count = 0;

  std::fprintf(stderr, "[tb_full_bfp] starting %d token steps\n", N_STEPS);
  for (int step = 0; step < N_STEPS; ++step) {
    int tid_in = (step < (int)prompt.size()) ? prompt[step]
                                              : generated.back();
    // Drive
    d->token_in = (uint16_t)tid_in;
    d->pos      = (uint32_t)step;
    d->kv_pos   = (uint32_t)step;
    d->start    = 1;
    tick();
    d->start = 0;

    // Wait for done
    const uint64_t LIM = cycle_count + 200000000ULL;
    while (!d->done && cycle_count < LIM) tick();
    if (cycle_count >= LIM) {
      std::fprintf(stderr, "TIMEOUT at step %d (cycle %llu)\n", step,
                   (unsigned long long)cycle_count);
      delete d; return 1;
    }
    int tid_out = (int)d->token_out;
    tick();   // clear done

    // Compare: token_out at step k should equal golden[k+1] (golden is the
    // input sequence at each step, so the model's output at step k matches
    // golden's NEXT entry).
    int expected = golden[step + 1];
    bool ok = (tid_out == expected);
    if (ok) ++pass_count; else ++fail_count;
    std::fprintf(stderr, "  step %2d  in=%5d  out=%5d  exp=%5d  %s\n",
                 step, tid_in, tid_out, expected, ok ? "PASS" : "FAIL");
    generated.push_back(tid_out);
  }

  std::printf("=== tb_full_bfp summary: %d PASS, %d FAIL out of %d steps ===\n",
              pass_count, fail_count, N_STEPS);
  delete d;
  return fail_count == 0 ? 0 : 1;
}
