# Python-cosim shim build.  Produces libmv_cosim.so callable from host/
# cosim_matvec.py via ctypes.  Single rule: bring up matvec_bfp_engine
# under Verilator, wrap with mv_cosim.cpp (C ABI), link as a shared lib.

VINC      := /home/local/share/verilator/include
SMOLLM    := ../../src/smollm

# Per-module Verilator builds.  Each lives in obj_dir_<TOP>/ so they
# don't share state.
OBJDIR_MV := obj_dir_mv
OBJDIR_SW := obj_dir_sw
OBJDIR_RN := obj_dir_rn
DUT_MV    := $(OBJDIR_MV)/Vmatvec_bfp_engine__ALL.a
DUT_SW    := $(OBJDIR_SW)/Vswiglu_bfp__ALL.a
DUT_RN    := $(OBJDIR_RN)/Vrmsnorm_bfp__ALL.a

$(DUT_MV): $(SMOLLM)/matvec_bfp_engine.sv $(SMOLLM)/bfp_format.svh
	verilator -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal \
	          -CFLAGS "-fPIC -O2 -std=c++17" \
	          --cc --build -j 0 --x-assign 0 --x-initial 0 \
	          --Mdir $(OBJDIR_MV) \
	          -I../../generated -I$(SMOLLM) \
	          --top-module matvec_bfp_engine \
	          $(SMOLLM)/matvec_bfp_engine.sv

$(DUT_SW): $(SMOLLM)/swiglu_bfp.sv $(SMOLLM)/bfp_format.svh
	verilator -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal \
	          -CFLAGS "-fPIC -O2 -std=c++17" \
	          --cc --build -j 0 --x-assign 0 --x-initial 0 \
	          --Mdir $(OBJDIR_SW) \
	          -I../../generated -I$(SMOLLM) \
	          --top-module swiglu_bfp \
	          $(SMOLLM)/swiglu_bfp.sv

$(DUT_RN): $(SMOLLM)/rmsnorm_bfp.sv $(SMOLLM)/bfp_format.svh
	verilator -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal \
	          -CFLAGS "-fPIC -O2 -std=c++17" \
	          --cc --build -j 0 --x-assign 0 --x-initial 0 \
	          --Mdir $(OBJDIR_RN) \
	          -I../../generated -I$(SMOLLM) \
	          --top-module rmsnorm_bfp \
	          $(SMOLLM)/rmsnorm_bfp.sv

# Backward-compat (existing flow): point legacy OBJDIR/DUT_LIB at the
# matvec build so any consumer that hasn't moved over still works.
OBJDIR    := $(OBJDIR_MV)
DUT_LIB   := $(DUT_MV)

# Per-DUT verilated runtime objects (must be -fPIC for the .so).
$(OBJDIR_MV)/verilated.o: $(VINC)/verilated.cpp
	g++ -fPIC -O2 -std=c++17 -I$(VINC) -I$(OBJDIR_MV) -c $(VINC)/verilated.cpp -o $@
$(OBJDIR_MV)/verilated_threads.o: $(VINC)/verilated_threads.cpp
	g++ -fPIC -O2 -std=c++17 -I$(VINC) -I$(OBJDIR_MV) -c $(VINC)/verilated_threads.cpp -o $@
$(OBJDIR_SW)/verilated.o: $(VINC)/verilated.cpp
	g++ -fPIC -O2 -std=c++17 -I$(VINC) -I$(OBJDIR_SW) -c $(VINC)/verilated.cpp -o $@
$(OBJDIR_SW)/verilated_threads.o: $(VINC)/verilated_threads.cpp
	g++ -fPIC -O2 -std=c++17 -I$(VINC) -I$(OBJDIR_SW) -c $(VINC)/verilated_threads.cpp -o $@
$(OBJDIR_RN)/verilated.o: $(VINC)/verilated.cpp
	g++ -fPIC -O2 -std=c++17 -I$(VINC) -I$(OBJDIR_RN) -c $(VINC)/verilated.cpp -o $@
$(OBJDIR_RN)/verilated_threads.o: $(VINC)/verilated_threads.cpp
	g++ -fPIC -O2 -std=c++17 -I$(VINC) -I$(OBJDIR_RN) -c $(VINC)/verilated_threads.cpp -o $@

libmv_cosim.so: mv_cosim.cpp $(DUT_MV) $(OBJDIR_MV)/verilated.o $(OBJDIR_MV)/verilated_threads.o
	g++ -fPIC -shared -O2 -std=c++17 \
	    -I$(VINC) -I$(VINC)/vltstd -I$(OBJDIR_MV) \
	    mv_cosim.cpp \
	    $(OBJDIR_MV)/verilated.o $(OBJDIR_MV)/verilated_threads.o \
	    $(DUT_MV) \
	    -lpthread -latomic \
	    -o libmv_cosim.so
	@echo "[cosim] $$(readlink -f libmv_cosim.so)"

libsw_cosim.so: sw_cosim.cpp $(DUT_SW) $(OBJDIR_SW)/verilated.o $(OBJDIR_SW)/verilated_threads.o
	g++ -fPIC -shared -O2 -std=c++17 \
	    -I$(VINC) -I$(VINC)/vltstd -I$(OBJDIR_SW) \
	    sw_cosim.cpp \
	    $(OBJDIR_SW)/verilated.o $(OBJDIR_SW)/verilated_threads.o \
	    $(DUT_SW) \
	    -lpthread -latomic \
	    -o libsw_cosim.so
	@echo "[cosim] $$(readlink -f libsw_cosim.so)"

librn_cosim.so: rn_cosim.cpp $(DUT_RN) $(OBJDIR_RN)/verilated.o $(OBJDIR_RN)/verilated_threads.o
	g++ -fPIC -shared -O2 -std=c++17 \
	    -I$(VINC) -I$(VINC)/vltstd -I$(OBJDIR_RN) \
	    rn_cosim.cpp \
	    $(OBJDIR_RN)/verilated.o $(OBJDIR_RN)/verilated_threads.o \
	    $(DUT_RN) \
	    -lpthread -latomic \
	    -o librn_cosim.so
	@echo "[cosim] $$(readlink -f librn_cosim.so)"

clean:
	rm -rf obj_dir_mv obj_dir_sw obj_dir_rn libmv_cosim.so libsw_cosim.so librn_cosim.so

.PHONY: clean
