/* fw_engine.c — PicoSoC + real-engine console (lean build).
 * UART is the only SoC-visible channel in this engine top (LEDs belong to the
 * engine; the eth LSU bus + ddr_wr upload path are tied off in this lean build).
 * It serves an interactive single-key menu that reads the engine regmap over the
 * Avalon master bridge (iomem 0x10) + the status snapshot (iomem 0x40), so every
 * command also proves the SoC<->engine register path works.  No globals. */
#include <stdint.h>

#define reg_uart_clkdiv (*(volatile uint32_t *)0x02000004)
#define reg_uart_data   (*(volatile uint32_t *)0x02000008)
#define UART_DIV 1085                 /* 125 MHz eth_clk / 115200 */

#define REG(n) (*(volatile uint32_t *)(0x10000000u + ((uint32_t)(n) << 2)))
#define HB     (*(volatile uint32_t *)0x40000000u)   /* engine status snapshot */

/* engine regmap word addresses (jtag_word_addr = master_address[11:2]) */
#define R_DBG_CALIB  0x012   /* [0] init_calib_complete + dbg snap bits        */
#define R_ML_STATE   0x017   /* [2:0] ml_state, [7:3] layer_idx, [8] lay_done  */
#define R_HAS_RUN    0x049   /* [0] autoregress has completed since restart     */
#define R_CRC        0x04A   /* rolling weight-hash over AXI rdata beats        */
#define R_MVST_R0    0x100   /* matvec selftest result, 8 words (16 Q1.15 lanes)*/
#define R_MVST_DONE  0x108   /* [0] matvec selftest done                        */
#define R_MVST_TRIG  0x109   /* write 1 -> retrigger matvec selftest            */
#define R_BUILDVER   0x10F   /* 32-bit BUILD_VERSION (git HEAD epoch low)       */
#define R_LAYER_DONE 0x1F0   /* [0] lay_done, [1] lay_done_latched              */
#define R_RESTART    0x1F1   /* write 1 -> restart pulse (clears CRC/has_run)   */

static void putc_(char c) { if (c == '\n') reg_uart_data = '\r'; reg_uart_data = c; }
static void print_(const char *s) { while (*s) putc_(*s++); }
static void hex_(uint32_t v, int nyb) {
    for (int i = nyb - 1; i >= 0; i--) putc_("0123456789abcdef"[(v >> (4 * i)) & 15]);
}

static void menu(void) {
    print_("\ncommands:\n");
    print_("  v  build version\n");
    print_("  c  weight CRC + has_run\n");
    print_("  s  engine status (FSM / token / flags)\n");
    print_("  m  matvec selftest  (compute proof-of-life)\n");
    print_("  r  restart engine\n");
    print_("  ?  this menu\n");
}

static void cmd_version(void) {
    print_("BUILD_VERSION="); hex_(REG(R_BUILDVER), 8); putc_('\n');
}

static void cmd_crc(void) {
    print_("weight CRC=");  hex_(REG(R_CRC), 8);
    print_("  has_run=");   hex_(REG(R_HAS_RUN) & 1, 1); putc_('\n');
}

static void cmd_status(void) {
    uint32_t hb = HB;
    print_("state=");      hex_(hb        & 0xFF, 2);
    print_(" last_tok=");  hex_((hb >>  8) & 0xFF, 2);
    print_(" out_len=");   hex_((hb >> 16) & 0xFF, 2);
    print_(" flags=");     hex_((hb >> 24) & 0xFF, 2);  /* [0]error [1]done */
    putc_('\n');
    uint32_t ml = REG(R_ML_STATE);
    print_("ml_state=");   hex_(ml & 7, 1);
    print_(" layer=");     hex_((ml >> 3) & 0x1F, 2);
    print_(" calib=");     hex_(REG(R_DBG_CALIB) & 1, 1);
    print_(" layer_done=");hex_(REG(R_LAYER_DONE) & 3, 1);
    putc_('\n');
}

static void cmd_matvec(void) {
    print_("matvec selftest ... ");
    REG(R_MVST_TRIG) = 1;
    uint32_t to = 4000000u;
    while (!(REG(R_MVST_DONE) & 1) && --to) { }
    if (!to) { print_("TIMEOUT (engine not responding)\n"); return; }
    print_("done:\n");
    for (int i = 0; i < 8; i++) {
        print_("  "); hex_(REG(R_MVST_R0 + i), 8);
        if ((i & 3) == 3) putc_('\n');
    }
}

static void cmd_restart(void) {
    REG(R_RESTART) = 1;
    print_("restart pulsed (0x1F1)\n");
}

void main(void)
{
    reg_uart_clkdiv = UART_DIV;
    print_("\n=== PicoSoC + engine console (lean) on VC707 ===\n");
    cmd_version();
    menu();
    print_("> ");
    for (;;) {
        int32_t c = reg_uart_data;
        if (c == -1) continue;
        char ch = (char)c;
        if (ch == '\r' || ch == '\n') { print_("> "); continue; }
        putc_(ch); putc_('\n');
        switch (ch) {
            case 'v': cmd_version(); break;
            case 'c': cmd_crc();     break;
            case 's': cmd_status();  break;
            case 'm': cmd_matvec();  break;
            case 'r': cmd_restart(); break;
            case '?':
            case 'h': menu();        break;
            default:  print_("? press ? for menu\n");
        }
        print_("> ");
    }
}
