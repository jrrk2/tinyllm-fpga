/* Uncompressed startup: set the stack pointer and call main.  Forcing this stub
 * to rv32i (no compression) means the very first instructions that bring up SP
 * are NOT compressed, so any failure isolates to the compressed body in test.c. */
.option norvc
.section .text.init
.global _start
_start:
    lui  sp, 0x40          /* sp = 0x40000 (top of the 256 KB test RAM) */
    call main
1:  j 1b
