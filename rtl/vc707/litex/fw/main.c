/* main.c -- tinyllm bare-metal firmware: console + Ethernet + SD card.
 *
 * Console (UART, 115200): single-character commands + Enter, or a line of
 * two or more characters = a prompt for the model (tokenised against the
 * vocab table in DDR3, answered on the console).  The Ethernet server
 * (net.c) keeps host/bfp_client working for weight upload / regmap access,
 * and 'l' loads the whole model setup from the SD card (sdload.c). */
#include <stdio.h>
#include <string.h>
#include <irq.h>
#include <libbase/uart.h>
#include <libbase/console.h>
#include "tinyllm.h"

#define LINE_MAX 160

static void menu(void) {
    printf("\ntinyllm_fw " __DATE__ " " __TIME__ "  (LiteX VC707, build 0x%08lx)\n",
           (unsigned long)tinyllm_build_version_read());
    printf("v ver  s status  c crc  r restart  t tokens  e eth  l load-sd  d dir-sd  R reboot  ? menu  (+Enter)\n");
    printf("or type a prompt (2+ chars) + Enter to run the model\n");
}

static void cmd_status(void) {
    uint32_t st = tinyllm_status_read();
    printf("status=0x%02lx done_latched=%u has_run=%u first_ar=%u first_r=%u  n_prompt=%lu  layer=%lu\n",
           (unsigned long)st, !!(st & ST_DONE_LATCHED), !!(st & ST_HAS_RUN),
           !!(st & ST_FIRST_AR), !!(st & ST_FIRST_R),
           (unsigned long)tinyllm_n_prompt_read(), (unsigned long)tinyllm_dbg_cur_layer_read());
    printf("model D=%d NL=%d FFN=%d ctx=%d prompt<=%d gen=%d  weights @main_ram+0 .. 0x%08x\n",
           TINYLLM_D, TINYLLM_NL, TINYLLM_FFN, TINYLLM_MAX_CTX, NPROMPT_MAX, N_GEN, TINYLLM_BASE_END);
}

static void cmd_tokens(void) {
    unsigned n = tinyllm_n_prompt_read();
    printf("RTL_TOKENS:");
    for (unsigned k = 0; k < N_STEPS; k++) printf(" %u", eng_result_token(k));
    printf("   (prompt %u + %u generated)\n", n, N_GEN);
}

/* Print token k of a run of `steps` steps; returns 0 once <|endoftext|>
 * (or the end of the run) is hit. */
static int emit_token(int k, int nt, int steps) {
    if (k < nt - 1) return 1;                    /* still feeding the prompt */
    if (k >= steps) return 0;
    uint16_t t = eng_result_token((unsigned)k);
    if (t == 0) return 0;                        /* <|endoftext|> */
    print_token(t);
    return 1;
}

/* Stream tokens while the engine runs.  The layer index resets to 0 on
 * restart, counts to NL-1 and holds there through the decode head, so a
 * drop from NL-1 means step k's token is in result_buf.  NL-1 must be seen
 * on two polls in a row: a multi-bit counter crossing the clock domain can
 * read as NL-1 for one sample mid-increment (e.g. 15 -> 16).
 * With the n_gen register the reply uses the whole context left after the
 * prompt; it goes back to 0 (= N_GEN) afterwards for the host's runs. */
void run_prompt_tokens(const uint16_t *toks, int nt) {
    int steps = nt + N_GEN;
#ifdef CSR_TINYLLM_N_GEN_ADDR
    steps = N_STEPS < TINYLLM_MAX_CTX ? N_STEPS : TINYLLM_MAX_CTX;
    tinyllm_n_gen_write((uint32_t)(steps - nt));
#endif
    for (int i = 0; i < nt; i++) eng_bram_write(BRAM_KIND_PROMPT, (uint32_t)i, toks[i]);
    tinyllm_n_prompt_write((uint32_t)nt);
    eng_restart();
    busy_wait(1);
    int k = 0, live = 1, last_seen = 0;
    for (unsigned long ms = 0; !(tinyllm_status_read() & ST_DONE_LATCHED); ms++) {
        if (ms >= 600000) { printf("\nengine timeout\n"); live = 0; break; }
        int last = tinyllm_dbg_cur_layer_read() == TINYLLM_NL - 1;
        if (last) last_seen++;
        else {
            if (last_seen >= 2 && live) live = emit_token(k++, nt, steps);
            last_seen = 0;
        }
        net_service();
        busy_wait(1);
    }
    while (live) live = emit_token(k++, nt, steps);   /* final step(s) */
#ifdef CSR_TINYLLM_N_GEN_ADDR
    tinyllm_n_gen_write(0);
#endif
    printf("\n");
}

static void run_prompt(const char *line, int n) {
    uint16_t toks[NPROMPT_MAX];
    int nt = tokenize(line, n, toks, NPROMPT_MAX);
    if (nt < 0) { printf("no vocab table at main_ram+0x%08x (SD: VOCAB.BIN, or bfp_client -a 0x%08x upload serial_vocab.bin)\n", VOCAB_OFFSET, VOCAB_OFFSET); return; }
    if (nt == 0) return;
    printf("tokens:");
    for (int i = 0; i < nt; i++) printf(" %u", toks[i]);
    printf("\n%s", line);
    run_prompt_tokens(toks, nt);
}

int main(void) {
#ifdef CONFIG_CPU_HAS_INTERRUPT
    irq_setmask(0);
    irq_setie(1);
#endif
    uart_init();
    printf("\n=== tinyllm on LiteX VC707 ===\n");
    menu();
    net_init();
    if (vocab_init()) printf("vocab table present\n");
    /* A card with the model bundle makes the SoC self-sufficient: load it now. */
    if (sd_card_present()) sd_load_all();
    printf("> ");
    static char line[LINE_MAX];
    int ll = 0;
    for (;;) {
        net_service();
        if (!readchar_nonblock()) continue;
        char ch = (char)getchar();
        if (ch == 0x7f || ch == 0x08) { if (ll) { ll--; fputs("\b \b", stdout); } continue; }
        if (ch != '\r' && ch != '\n') {
            if (ll < LINE_MAX - 1) { line[ll++] = ch; putchar(ch); }
            continue;
        }
        putchar('\n');
        if (ll == 1) {
            switch (line[0]) {
                case 'v': printf("BUILD_VERSION=%08lx\n", (unsigned long)tinyllm_build_version_read()); break;
                case 's': cmd_status(); break;
                case 'c': printf("weight hash=%08lx\n", (unsigned long)tinyllm_weight_hash_read()); break;
                case 'r': eng_restart(); printf("restart pulsed\n"); break;
                case 't': cmd_tokens(); break;
                case 'e': net_stats(); break;
                case 'l': sd_load_all(); break;
                case 'd': sd_ls(); break;
                case 'R': printf("rebooting (BIOS)\n"); uart_sync(); ctrl_reset_write(1); break;
                case '?': case 'h': menu(); break;
                default:  printf("? press ? for menu\n");
            }
        } else if (ll > 1) {
            line[ll] = 0;
            run_prompt(line, ll);
        }
        ll = 0;
        printf("> ");
    }
}
