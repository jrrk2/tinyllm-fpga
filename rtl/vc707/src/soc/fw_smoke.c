/* fw_smoke.c — PicoSoC liveness + UART diagnostic (engine/shell smoke test).
 * Loops a banner continuously so it can't be missed (the menu's one-shot banner
 * was), and echoes input.  Two output paths per line to localise faults:
 *   - "PicoSoC alive #"  via print_()  -> tests multi-char strings from .rodata
 *   - the counter        via direct putc_ of hex digits -> multi-char, no .rodata
 * If the full line repeats: UART + print_ + .rodata all good (the menu commands
 * work too; you just missed the boot text / discounted the regmap values).
 * If only the first char of each burst shows: UART TX overrun.
 * If the text is missing but the counter shows: .rodata/string-read fault. */
#include <stdint.h>

#define reg_uart_clkdiv (*(volatile uint32_t *)0x02000004)
#define reg_uart_data   (*(volatile uint32_t *)0x02000008)
#define UART_DIV 1085                 /* 125 MHz eth_clk / 115200 */

static void putc_(char c) { if (c == '\n') reg_uart_data = '\r'; reg_uart_data = c; }
static void print_(const char *s) { while (*s) putc_(*s++); }

void main(void)
{
    reg_uart_clkdiv = UART_DIV;
    uint32_t n = 0;
    for (;;) {
        print_("smoke " __DATE__ " " __TIME__ " #");
        for (int i = 7; i >= 0; i--) putc_("0123456789abcdef"[(n >> (4 * i)) & 15]);
        putc_('\r'); putc_('\n');
        n++;
        for (volatile uint32_t d = 0; d < 1000000u; d++) {   /* ~pause + echo */
            int32_t c = (int32_t)reg_uart_data;
            if (c != -1) putc_((char)(c & 0xFF));
        }
    }
}
