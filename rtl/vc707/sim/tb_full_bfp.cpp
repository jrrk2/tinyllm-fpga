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

  auto prompt = read_tokens("../../generated/lbfp_full_PROMPT_TOKENS.txt");
  auto golden = read_tokens("../../generated/lbfp_full_GOLDEN_TOKENS.txt");
  std::fprintf(stderr, "[tb_full_bfp] prompt=%zu tokens, golden=%zu tokens\n",
               prompt.size(), golden.size());

  d->clk = 0; d->rst = 1; d->start = 0;
  d->token_in = 0; d->pos = 0; d->kv_pos = 0;
  d->eval();
  for (int i = 0; i < 8; ++i) tick();
  d->rst = 0; tick();

  // Two modes, selected by env var BFP_FEED:
  //   BFP_FEED=golden (default): force-feed Python BFP golden tokens as inputs.
  //     Each step is an independent test: compare RTL's argmax to golden[k].
  //     Diagnoses arithmetic divergence per-step.
  //   BFP_FEED=auto: feed RTL's previous output back as the next input
  //     (true autoregress).  No per-step PASS/FAIL — we just print the
  //     resulting text and judge coherence by eye.  Useful when RTL drifts
  //     from golden but might still produce sensible English on its own.
  const char* feed_mode = std::getenv("BFP_FEED");
  bool auto_feed = (feed_mode && std::string(feed_mode) == "auto");
  const int N_PROMPT = (int)prompt.size();
  const int N_STEPS  = (int)golden.size();
  int pass_count = 0, fail_count = 0, compared = 0;
  std::vector<int> rtl_chain;

  std::fprintf(stderr, "[tb_full_bfp] feed=%s, %d steps (prompt=%d, golden=%d)\n",
               auto_feed ? "auto" : "golden", N_STEPS, N_PROMPT, (int)golden.size());
  for (int step = 0; step < N_STEPS; ++step) {
    int tid_in;
    if (step < N_PROMPT) {
      tid_in = prompt[step];
    } else if (auto_feed) {
      tid_in = rtl_chain.back();
    } else {
      tid_in = golden[step - 1];
    }
    d->token_in = (uint16_t)tid_in;
    d->pos      = (uint32_t)step;
    d->kv_pos   = (uint32_t)step;
    d->start    = 1;
    tick();
    d->start = 0;

    const uint64_t LIM = cycle_count + 200000000ULL;
    while (!d->done && cycle_count < LIM) tick();
    if (cycle_count >= LIM) {
      std::fprintf(stderr, "TIMEOUT at step %d (cycle %llu)\n", step,
                   (unsigned long long)cycle_count);
      delete d; return 1;
    }
    int tid_out = (int)d->token_out;
    tick();

    rtl_chain.push_back(tid_out);

    // Only compare for steps where the baker actually wrote a golden nid.
    if (step >= N_PROMPT && !auto_feed) {
      int expected = golden[step];
      bool ok = (tid_out == expected);
      ++compared;
      if (ok) ++pass_count; else ++fail_count;
      std::fprintf(stderr, "  step %2d  in=%5d  out=%5d  exp=%5d  %s\n",
                   step, tid_in, tid_out, expected, ok ? "PASS" : "FAIL");
    } else {
      const char* tag = auto_feed ? "auto" :
                        (step < N_PROMPT ? "prefill — no golden" : "auto");
      std::fprintf(stderr, "  step %2d  in=%5d  out=%5d  (%s)\n",
                   step, tid_in, tid_out, tag);
    }
  }

  std::printf("=== tb_full_bfp summary: %d PASS, %d FAIL out of %d comparisons ===\n",
              pass_count, fail_count, compared);
  // Print the final RTL-generated chain (prompt + autoregress) for offline decoding.
  std::printf("RTL_TOKENS:");
  for (int t : rtl_chain) std::printf(" %d", t);
  std::printf("\n");
  delete d;
  return fail_count == 0 ? 0 : 1;
}
