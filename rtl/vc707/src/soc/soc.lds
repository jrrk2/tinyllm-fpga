/* Linker script for the tinyllm PicoSoC firmware.
 *   FLASH = progmem ROM, mapped at PROGADDR_RESET (0x0010_0000), 4 KB = the
 *           RAMB36E1 progmem (1024 x 32).  Linker errors if firmware exceeds it.
 *   RAM   = picosoc_mem data RAM at 0x0, 16 KB (STACKADDR = 0x4000). */
MEMORY
{
    FLASH (rx) : ORIGIN = 0x00100000, LENGTH = 0x00001000
    RAM (xrw)  : ORIGIN = 0x00000000, LENGTH = 0x00004000
}

SECTIONS {
    .text :
    {
        . = ALIGN(4);
        *(.text)
        *(.text*)
        *(.rodata)
        *(.rodata*)
        *(.srodata)
        *(.srodata*)
        . = ALIGN(4);
        _etext = .;
        _sidata = _etext;
    } >FLASH

    .data : AT ( _sidata )
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data*)
        *(.sdata)
        *(.sdata*)
        . = ALIGN(4);
        _edata = .;
    } >RAM

    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss)
        *(.bss*)
        *(.sbss)
        *(.sbss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } >RAM
}
