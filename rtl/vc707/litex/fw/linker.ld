/* tinyllm firmware: lives at the TOP of the 1 GiB SODIMM (main_ram + 0x3F000000)
 * so the weight image (main_ram + 0, ~843 MB) and the vocab table
 * (main_ram + 0x3C000000) never overlap it.  The BIOS loads it there
 * (boot.json) and jumps to _start. */
INCLUDE generated/output_format.ld
ENTRY(_start)
__DYNAMIC = 0;

MEMORY {
	fwram : ORIGIN = 0x7F000000, LENGTH = 0x01000000
}

SECTIONS
{
	.text :
	{
		_ftext = .;
		*crt0*(.text)
		KEEP(*crt0*(.text))
		KEEP(*(.text.isr))
		*(.text .stub .text.* .gnu.linkonce.t.*)
		_etext = .;
	} > fwram

	.rodata :
	{
		. = ALIGN(8);
		_frodata = .;
		*(.rodata .rodata.* .gnu.linkonce.r.*)
		*(.rodata1)
		*(.srodata .srodata.*)
		*(.got .got.*)
		*(.toc .toc.*)
		FILL(0);
		. = ALIGN(8);
		_erodata = .;
	} > fwram

	.preinit_array : { . = ALIGN(8); PROVIDE_HIDDEN (__preinit_array_start = .); KEEP(*(SORT(.preinit_array.*))) KEEP(*(.preinit_array)) PROVIDE_HIDDEN (__preinit_array_end = .); } > fwram
	.init_array    : { . = ALIGN(8); PROVIDE_HIDDEN (__init_array_start = .); KEEP(*(SORT(.init_array.*))) KEEP(*(.init_array)) KEEP(*(SORT(.ctors.*))) KEEP(*(.ctors)) PROVIDE_HIDDEN (__init_array_end = .); } > fwram
	.fini_array    : { . = ALIGN(8); PROVIDE_HIDDEN (__fini_array_start = .); KEEP(*(SORT(.fini_array.*))) KEEP(*(.fini_array)) KEEP(*(SORT(.dtors.*))) KEEP(*(.dtors)) PROVIDE_HIDDEN (__fini_array_end = .); } > fwram

	.data :
	{
		. = ALIGN(8);
		_fdata = .;
		*(.data .data.* .gnu.linkonce.d.*)
		*(.data1)
		_gp = ALIGN(16);
		*(.sdata .sdata.* .gnu.linkonce.s.*)
		FILL(0);
		. = ALIGN(8);
		_edata = .;
	} > fwram

	.bss :
	{
		. = ALIGN(8);
		_fbss = .;
		*(.dynsbss)
		*(.sbss .sbss.* .gnu.linkonce.sb.*)
		*(.scommon)
		*(.dynbss)
		*(.bss .bss.* .gnu.linkonce.b.*)
		*(COMMON)
		. = ALIGN(8);
		_ebss = .;
		_end = .;
	} > fwram
}

PROVIDE(_fstack = ORIGIN(fwram) + LENGTH(fwram) - 16);
PROVIDE(_fdata_rom = LOADADDR(.data));
PROVIDE(_edata_rom = LOADADDR(.data) + SIZEOF(.data));
