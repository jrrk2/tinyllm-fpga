/* fw_engine.c — PicoSoC + real-engine LEAN bring-up firmware.
 * UART is the ONLY SoC-visible channel in this engine top (the LEDs belong to
 * the engine, the eth LSU bus + ddr_wr upload path are tied off in this lean
 * build), so it carries the boot banner and the CPU heartbeat.
 *
 * It also reads the engine regmap over the Avalon master bridge (iomem 0x10)
 * and the status snapshot (iomem 0x40), which proves both that the SoC booted
 * AND that the SoC<->engine register path works.  No globals (no data init). */
#include <stdint.h>

#define reg_uart_clkdiv (*(volatile uint32_t *)0x02000004)
#define reg_uart_data   (*(volatile uint32_t *)0x02000008)
#define UART_DIV 1085                 /* 125 MHz eth_clk / 115200 */

#define REG(n) (*(volatile uint32_t *)(0x10000000u + ((uint32_t)(n) << 2)))
#define HB     (*(volatile uint32_t *)0x40000000u)   /* engine status snapshot */
#define REG_BUILD_VERSION 0x10F

static void putc_(char c) { if (c == '\n') reg_uart_data = '\r'; reg_uart_data = c; }
static void print_(const char *s) { while (*s) putc_(*s++); }
static void hex_(uint32_t v, int nyb) {
    for (int i = nyb - 1; i >= 0; i--) putc_("0123456789abcdef"[(v >> (4 * i)) & 15]);
}

void main(void)
{
    reg_uart_clkdiv = UART_DIV;
    print_("\n=== PicoSoC + engine (lean) on VC707 ===\n");
    print_("engine BUILD_VERSION="); hex_(REG(REG_BUILD_VERSION), 8); putc_('\n');
    print_("UART heartbeat every ~1s (LEDs/eth/ddr belong to the engine).\n");

    uint32_t spin = 0, beat = 0;
    for (;;) {
        int32_t c = reg_uart_data;            /* UART echo, proves RX path too */
        if (c != -1) putc_((char)c);
        if (++spin >= 1000000u) {
            spin = 0;
            print_("hb "); hex_(beat++, 4);
            print_(" status="); hex_(HB, 8);
            print_(" bv=");     hex_(REG(REG_BUILD_VERSION), 8);
            putc_('\n');
        }
    }
}
