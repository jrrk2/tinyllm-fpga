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
    printf("v ver  s status  c crc  r restart  t tokens  e eth  l load-sd  d dir-sd  ? menu  (+Enter)\n");
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

static int wait_done(unsigned long max_ms) {
    for (unsigned long ms = 0; ms < max_ms; ms++) {
        if (tinyllm_status_read() & ST_DONE_LATCHED) return 1;
        net_service();
        busy_wait(1);
    }
    return 0;
}

void run_prompt_tokens(const uint16_t *toks, int nt) {
    for (int i = 0; i < nt; i++) eng_bram_write(BRAM_KIND_PROMPT, (uint32_t)i, toks[i]);
    tinyllm_n_prompt_write((uint32_t)nt);
    eng_restart();
    if (!wait_done(60000)) { printf("engine timeout\n"); return; }
    for (int k = nt - 1; k < nt + N_GEN && k < (int)N_STEPS; k++) {
        uint16_t t = eng_result_token((unsigned)k);
        if (t == 0) break;                       /* <|endoftext|> */
        print_token(t);
    }
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
