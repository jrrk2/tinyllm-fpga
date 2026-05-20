# Python-cosim shim build.  Produces libmv_cosim.so callable from host/
# cosim_matvec.py via ctypes.  Single rule: bring up matvec_bfp_engine
# under Verilator, wrap with mv_cosim.cpp (C ABI), link as a shared lib.

VINC      := /home/local/share/verilator/include
SMOLLM    := ../../src/smollm
TOP       := matvec_bfp_engine

OBJDIR    := obj_dir
DUT_LIB   := $(OBJDIR)/V$(TOP)__ALL.a

# Verilator generates the per-module C++ headers + .a archive.
$(DUT_LIB): $(SMOLLM)/$(TOP).sv $(SMOLLM)/bfp_format.svh
	verilator -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal \
	          -CFLAGS "-fPIC -O2 -std=c++17" \
	          --cc --build -j 0 --x-assign 0 --x-initial 0 \
	          -I../../generated -I$(SMOLLM) \
	          --top-module $(TOP) \
	          $(SMOLLM)/$(TOP).sv

# Verilator's runtime support (verilated.o + verilated_threads.o) needs to
# be -fPIC too.
$(OBJDIR)/verilated.o: $(VINC)/verilated.cpp
	g++ -fPIC -O2 -std=c++17 -I$(VINC) -I$(OBJDIR) -c $(VINC)/verilated.cpp -o $@
$(OBJDIR)/verilated_threads.o: $(VINC)/verilated_threads.cpp
	g++ -fPIC -O2 -std=c++17 -I$(VINC) -I$(OBJDIR) -c $(VINC)/verilated_threads.cpp -o $@

libmv_cosim.so: mv_cosim.cpp $(DUT_LIB) $(OBJDIR)/verilated.o $(OBJDIR)/verilated_threads.o
	g++ -fPIC -shared -O2 -std=c++17 \
	    -I$(VINC) -I$(VINC)/vltstd -I$(OBJDIR) \
	    mv_cosim.cpp \
	    $(OBJDIR)/verilated.o $(OBJDIR)/verilated_threads.o \
	    $(DUT_LIB) \
	    -lpthread -latomic \
	    -o libmv_cosim.so
	@echo "[cosim] $$(readlink -f libmv_cosim.so)"

clean:
	rm -rf obj_dir libmv_cosim.so

.PHONY: clean
