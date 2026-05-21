/* fw_engine.c — stub firmware for the PicoSoC + real-engine LEAN fit build.
 * Just exercises the Avalon bridge to the engine regmap (iomem 0x10) and the
 * status window (0x40) so the SoC + bridge aren't optimized away.  Functional
 * eth/upload/control firmware comes after base fit + timing are confirmed.
 * No globals; no UART (this engine top has no UART pin — ser_tx dangles). */
#include <stdint.h>

#define REG(n) (*(volatile uint32_t *)(0x10000000u + ((uint32_t)(n) << 2)))
#define HB     (*(volatile uint32_t *)0x40000000u)   /* engine status snapshot */

#define REG_BUILD_VERSION 0x10F
#define REG_RESTART       0x1F1

void main(void)
{
    volatile uint32_t bv = REG(REG_BUILD_VERSION);   /* read a regmap reg */
    (void)bv;
    for (;;) {
        REG(REG_RESTART) = 1;                         /* write a regmap reg  */
        volatile uint32_t st = HB;                    /* read engine status  */
        (void)st;
        for (volatile uint32_t d = 0; d < 2000000u; d++) { }
    }
}
