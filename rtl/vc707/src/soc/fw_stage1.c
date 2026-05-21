/* Stage-1 firmware: prove the PicoSoC can drive the engine register map over
 * the iomem bus.  The TB maps a 256-word register stub at 0x1000_0000.
 * We write a known value, read it back through the bridge, echo it + a derived
 * value, then raise a done sentinel the TB polls.  No globals/.data so startup
 * needs no data/bss init. */
#include <stdint.h>

#define REG ((volatile uint32_t *)0x10000000u)   /* word-indexed regmap window */

void main(void)
{
    REG[1]   = 0xCAFE1234u;     /* write test register                     */
    uint32_t v = REG[1];        /* read it back through the iomem bridge    */
    REG[2]   = v;               /* echo the readback (proves read path)     */
    REG[3]   = v + 1u;          /* derived value (proves CPU executed)      */
    REG[255] = 0x600DCAFEu;     /* done sentinel the TB watches             */
    for (;;) { }
}
