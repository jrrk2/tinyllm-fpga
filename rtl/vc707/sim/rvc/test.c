/* Compressed-ISA smoke test.  Each REPORT() writes a milestone to a magic MMIO
 * address the testbench prints.  The LAST milestone seen before the sim times
 * out (or traps) tells us which compressed construct broke:
 *   0x100  reached main           (call/prologue: c.addi16sp, c.swsp)
 *   0x20f  loop sum 1..5 == 15    (compressed branches/arith in a loop)
 *   0x307  addfn(3,4) == 7        (compressed call + return: c.jal/c.jr)
 *   0xd09e all-passed sentinel
 * Built once -march=rv32imc (compressed) and once -march=rv32im (control). */
#include <stdint.h>
#define REPORT(x) (*(volatile uint32_t *)0x10000000u = (uint32_t)(x))

static int __attribute__((noinline)) addfn(int a, int b) { return a + b; }

void main(void) {
    REPORT(0x100);
    volatile int s = 0;
    for (int i = 1; i <= 5; i++) s += i;          /* 15 */
    REPORT(0x200 | (s & 0xff));                    /* 0x20f */
    int r = addfn(3, 4);                           /* 7 */
    REPORT(0x300 | (r & 0xff));                    /* 0x307 */
    REPORT(0xd09e);
    for (;;) { }
}
