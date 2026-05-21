/* Minimal PicoRV32 startup for the tinyllm SoC.
 * picorv32 initializes sp from the STACKADDR parameter on reset, and this
 * stage-1 firmware uses no .data/.bss/globals, so we just jump to main. */
.section .text
.global _start
_start:
    call main
1:  j 1b
