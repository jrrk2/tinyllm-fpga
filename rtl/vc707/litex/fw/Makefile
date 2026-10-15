# tinyllm bare-metal firmware for the LiteX VC707 SoC (litex/tinyllm_vc707.py).
# Built against a LiteX build directory (software libs + generated headers):
#     make BUILD_DIR=../../build/litex
BUILD_DIR ?= ../../build/litex

include $(BUILD_DIR)/software/include/generated/variables.mak
include $(SOC_DIRECTORY)/software/common.mak

OBJECTS = crt0.o main.o net.o tok.o sdload.o

COMMA = ,
REGULAR_LIBS = $(filter-out $(ALWAYS_LINK_LIBS),$(LIBS))
ALWAYS_LINK_LIBFLAGS = $(if $(ALWAYS_LINK_LIBS),-Wl$(COMMA)--whole-archive $(ALWAYS_LINK_LIBS:lib%=-l%) -Wl$(COMMA)--no-whole-archive)

all: tinyllm_fw.bin

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@
	chmod -x $@
	@echo "=== tinyllm_fw.bin: $$(stat -c%s $@) bytes ==="

vpath %.a $(PACKAGES:%=../%)

tinyllm_fw.elf: $(OBJECTS) linker.ld
	$(CC) $(LDFLAGS) -T linker.ld -N -o $@ \
		$(OBJECTS) \
		$(PACKAGES:%=-L$(BUILD_DIR)/software/%) \
		-Wl,--start-group \
		$(REGULAR_LIBS:lib%=-l%) \
		$(ALWAYS_LINK_LIBFLAGS) \
		-Wl,--end-group \
		-Wl,--gc-sections \
		-Wl,-Map,$@.map
	chmod -x $@

-include $(OBJECTS:.o=.d)

VPATH = $(BIOS_DIRECTORY):$(BIOS_DIRECTORY)/cmds:$(CPU_DIRECTORY)

%.o: %.c
	$(compile)

%.o: %.S
	$(assemble)

clean:
	$(RM) $(OBJECTS) $(OBJECTS:.o=.d) tinyllm_fw.elf tinyllm_fw.elf.map tinyllm_fw.bin

.PHONY: all clean
