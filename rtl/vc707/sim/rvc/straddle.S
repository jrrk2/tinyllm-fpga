/* Isolate a straddling LOAD instruction (the shape of fw_imc's lbu at 0x10000e):
 * a 32-bit load placed at a 2-byte-aligned address, so the instruction itself
 * straddles a word boundary AND performs a memory access.  This makes PicoRV32
 * juggle the compressed prefetch (2 fetches for the straddle) and the load's own
 * bus access at once.  Reports milestones to magic 0x10000000:
 *   0xAA   before
 *   a0     the straddling load's result (expect 0x0000c0de)
 *   0xBB   after
 * Hang before the a0 report => the straddling load is the trigger. */
.section .text
.global _start
_start:
    lui  sp, 0x10
    lui  t0, 0x10000          /* magic 0x10000000 */
    li   t1, 0xAA
    sw   t1, 0(t0)            /* 0xAA */
    li   t2, 0x200            /* a RAM scratch address */
    li   t3, 0xC0DE
    sw   t3, 0(t2)            /* RAM[0x200] = 0xC0DE */
    /* the c.lui above already left us 2-byte misaligned, so this 32-bit lw
     * lands at a 2-byte address -> the instruction straddles a word boundary */
.option norvc
    lw   a0, 0(t2)            /* STRADDLING LOAD */
    sw   a0, 0(t0)            /* report a0 (expect 0x0000c0de) */
    li   t1, 0xBB
    sw   t1, 0(t0)            /* 0xBB */
1:  j 1b
