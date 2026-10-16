OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { RAM : ORIGIN = 0x00000000, LENGTH = 256K }
SECTIONS {
  .text : { *(.text.init) *(.text*) } > RAM
  .rodata : { *(.rodata*) } > RAM
  .data : { *(.data*) } > RAM
  .bss : { *(.bss*) *(COMMON) } > RAM
  . = ALIGN(4);
}
