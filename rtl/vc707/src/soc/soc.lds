/* Linker script for the tinyllm PicoSoC firmware.
 *   FLASH = progmem, mapped at PROGADDR_RESET (0x0010_0000), 8 KB = the 4x
 *           RAMB16_S9_S9 byte-lane progmem (2048 x 32).  Linker errors if the
 *           firmware exceeds it.
 *   RAM   = picosoc_mem data RAM at 0x0, 16 KB (STACKADDR = 0x4000). */
MEMORY
{
    FLASH (rx) : ORIGIN = 0x00100000, LENGTH = 0x00002000
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
