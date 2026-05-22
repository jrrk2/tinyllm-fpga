/* fw_smoke.c — the simplest possible PicoSoC smoke test.
 * No dependencies, no regmap, no helper functions, no string/.data section
 * (chars are immediates).  It sets the UART baud divisor, prints "\nHello\n"
 * once, then sits in a loop echoing UART input back out.  This isolates the
 * bare path  eth_clk -> PicoRV32 fetch/execute -> simpleuart -> AU36/AU33 pins,
 * and exercises BOTH directions (Hello = TX, echo = RX->TX).
 *
 * Splice into the EXISTING bitstream with `make picosoc-engine-smoke`
 * (updatemem, no re-synth).  If "Hello" appears and typed keys echo, SoC + UART
 * + both pins are alive and the fault is in the menu firmware / regmap path.
 * If it is silent, the fault is upstream (eth_clk not toggling / SoC held in
 * reset) -> the ILA build is next. */
#include <stdint.h>

void main(void)
{
    volatile uint32_t *uart_clkdiv = (volatile uint32_t *)0x02000004;
    volatile uint32_t *uart_data   = (volatile uint32_t *)0x02000008;

    *uart_clkdiv = 1085;                 /* 125 MHz eth_clk / 115200 */

    *uart_data = '\r';
    *uart_data = '\n';
    *uart_data = 'H';
    *uart_data = 'e';
    *uart_data = 'l';
    *uart_data = 'l';
    *uart_data = 'o';
    *uart_data = '\r';
    *uart_data = '\n';

    for (;;) {                           /* echo UART input forever */
        int32_t c = (int32_t)*uart_data; /* -1 when no byte waiting */
        if (c != -1) {
            if (c == '\r') *uart_data = '\r';   /* Enter -> CR+LF */
            *uart_data = (c == '\r') ? '\n' : (uint32_t)(c & 0xFF);
        }
    }
}
