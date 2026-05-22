/* fw_smoke.c — the simplest possible PicoSoC smoke test.
 * No dependencies, no regmap, no helper functions, no string/.data section
 * (chars are immediates).  It only sets the UART baud divisor and prints
 * "\nHello\n" in a loop, so it isolates the bare path
 *   eth_clk -> PicoRV32 fetch/execute -> simpleuart -> AU36 pin.
 *
 * Splice into the EXISTING bitstream with `make picosoc-engine-smoke`
 * (updatemem, no re-synth).  If "Hello" appears, SoC + UART + pin are alive and
 * the fault is in the menu firmware / regmap path.  If it is still silent, the
 * fault is upstream (eth_clk not toggling / SoC held in reset) -> ILA build. */
#include <stdint.h>

void main(void)
{
    volatile uint32_t *uart_clkdiv = (volatile uint32_t *)0x02000004;
    volatile uint32_t *uart_data   = (volatile uint32_t *)0x02000008;

    *uart_clkdiv = 1085;                 /* 125 MHz eth_clk / 115200 */

    for (;;) {
        *uart_data = '\n';
        *uart_data = 'H';
        *uart_data = 'e';
        *uart_data = 'l';
        *uart_data = 'l';
        *uart_data = 'o';
        *uart_data = '\n';
        for (volatile uint32_t d = 0; d < 2000000u; d++) { }
    }
}
