HPVM_INSTALL_DIR ?= /home/xrouth2/hdc/hpvm/hpvm
# HPVM_BUILD: debug (default, uses build/) or release (uses build-release/)
HPVM_BUILD ?= debug
ifeq ($(HPVM_BUILD),release)
HPVM_BUILD_DIR := $(HPVM_INSTALL_DIR)/build-release
else
HPVM_BUILD_DIR := $(HPVM_INSTALL_DIR)/build
endif

INCLUDES := -I$(HPVM_INSTALL_DIR)/include -I$(HPVM_INSTALL_DIR)/benchmarks/include -I$(HPVM_INSTALL_DIR)/projects/hpvm-hdc -I$(HPVM_INSTALL_DIR)/projects/hetero-c++/test/include -I./src

HPVM_LIB := $(HPVM_BUILD_DIR)/lib
HPVM_BIN := $(HPVM_BUILD_DIR)/bin

HPVM_DECLS_FILE := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hetero-c++/lib/HPVMCFunctionDeclarations/HPVMCFunctionDeclarations.bc

CUDA_PATH:=/software/cuda-9.1
OPENCL_PATH:=$(CUDA_PATH)
OPENCL_LIB_PATH:=$(OPENCL_PATH/lib64)
NVCC ?= "/usr/local/cuda"/bin/nvcc

CFLAGS :=  -O0 -Xclang -disable-O0-optnone -fenable-matrix -fno-discard-value-names -fno-exceptions -lm -lrt -pthread -lOpenCL
O1_FLAGS = -targetlibinfo -tti -tbaa -scoped-noalias-aa -assumption-cache-tracker -profile-summary-info -forceattrs -inferattrs -ipsccp -called-value-propagation -globalopt -domtree -mem2reg -deadargelim -basic-aa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg -basiccg -globals-aa -prune-eh -always-inline -function-attrs  -memoryssa -early-cse-memssa -speculative-execution -lazy-value-info -jump-threading -correlated-propagation -libcalls-shrinkwrap -branch-prob -block-freq -pgo-memop-opt -tailcallelim -reassociate -loop-simplify -lcssa-verification -lcssa -scalar-evolution -loop-rotate -licm -loop-unswitch -indvars -loop-idiom -loop-deletion -loop-unroll -memdep -memcpyopt -sccp -demanded-bits -bdce -dse -postdomtree -adce -barrier -rpo-function-attrs -globaldce -float2int -loop-accesses -loop-distribute -loop-vectorize -loop-load-elim -alignment-from-assumptions -strip-dead-prototypes -loop-sink -instsimplify -div-rem-pairs -verify -ee-instrument -early-cse -lower-expect

HDC_LIBS := -load $(HPVM_LIB)/LLVMHPVMHDC.so -load $(HPVM_LIB)/LLVMHDCBinProp.so  -load $(HPVM_LIB)/LLVMBinarize.so  -load $(HPVM_LIB)/LLVMHDCApproxKnobs.so -load $(HPVM_LIB)/LLVMHDCApprox.so

SRC_ROOT := /home/xrouth2/hdc/HDC-monorepo/applications/HD-Graph
CONFIG_IN_PATH ?= $(SRC_ROOT)/config/config_in.json
CONFIG_OUT_PATH ?= $(SRC_ROOT)/config/config_out.json

ENCODING_DIM ?= 4096
DATA_TYPE ?= float
HAMMING_DIST ?= 1
BINARIZE ?= 0
NO_APPROX_HINT ?= 0
RHO_SHIFT ?= 0
POLARIZE ?= 0

APPROX_MACRO_DEFINES := -DENCODING_DIM=$(ENCODING_DIM) -DDATA_TYPE=$(DATA_TYPE) -DHAMMING_DIST=$(HAMMING_DIST) -DBINARIZE=$(BINARIZE) -DNO_APPROX_HINT=$(NO_APPROX_HINT) -DRHO_SHIFT=$(RHO_SHIFT) -DPOLARIZE=$(POLARIZE)

### ---------- CPU ---------------

# Generate Executable
host-cpu: build/output.cpu.ll
	$(HPVM_BIN)/clang++ -lm -lrt -lOpenCL -pthread -O2 $< -o $@ 

# Clang Frontend to IR
build/host.cpu.ll: src/host.cpp src/DFG.hpp src/host.h
	$(HPVM_BIN)/clang++ -DHPVM $(APPROX_MACRO_DEFINES) -DDEVICE=hpvm::CPU_TARGET $(CFLAGS) $(INCLUDES) -S -emit-llvm $< -o $@ 

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.cpu.ll: build/host.cpu.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@

# Lower HDC Intrinsics
build/hdc.lowered.cpu.ll: build/hpvm.genhdc.cpu.ll # -enable-hcc
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so -vector-block-dim=1 -lower-hdc -dce -simplifycfg $< -S -o $@ 
	
# HCC -> HPVM
build/hpvmc.cpu.ll: build/hdc.lowered.cpu.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

# GenHPVM for the HPVM IR
build/hpvm.cpu.ll: build/hpvmc.cpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -dce -mem2reg -genhpvm $< -S -o $@

# Other HPVM Passes
build/hpvmcompiled.cpu.ll: build/hpvm.cpu.ll # -hpvm-timers-cpu
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMBuildDFG.so -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so -load $(HPVM_LIB)/HPVMClearDFG.so -load $(HPVM_LIB)/HPVMDFGTransformPasses.so -dfg2llvm-cpu -clearDFG $< -S -o $@

HPVM_RT := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc

# LINK HPVM RT and DCE
build/output.cpu.ll: build/hpvmcompiled.cpu.ll
	$(HPVM_BIN)/llvm-link $< $(HPVM_RT) -S -o $@ 
	$(HPVM_BIN)/opt -dce -globaldce $< -S -o $<

### ----------- CPU APPROX -----------
# Generate Executable
host-cpu-approx: build/output.2.ll 
	$(HPVM_BIN)/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O3 $^ -o $@ 

# Clang Frontend to IR
build/host.cpu2.ll: src/host.cpp src/DFG.hpp src/host.h
	$(HPVM_BIN)/clang++ -O0 -Xclang -disable-O0-optnone -DHPVM $(APPROX_MACRO_DEFINES) -DDEVICE=hpvm::CPU_TARGET $(CFLAGS) $(INCLUDES) -S -emit-llvm $< -o $@ 

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.cpu2.0.ll: build/host.cpu2.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@

# Get HDC Approx Knobs
build/hpvm.genhdc.cpu2.1.ll: build/hpvm.genhdc.cpu2.0.ll 
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -hdc-collect-approx --approx-config-path-collect=$(CONFIG_OUT_PATH) $< -S -o $@ 

# Set HDC Approx Knobs
build/hpvm.genhdc.cpu2.2.ll: build/hpvm.genhdc.cpu2.0.ll 
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -hdc-set-approx-knobs --approx-config-path=$(CONFIG_IN_PATH) $< -S -o $@ 

# Lower HDC Intrinsics
build/hdc.lowered.cpu2.ll: build/hpvm.genhdc.cpu2.2.ll #hpvm. # -enable-hcc 
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -vector-block-dim=1 -lower-hdc  -dce -simplifycfg $< -S -o $@ 
	
# HCC -> HPVM
build/hpvmc.cpu2.ll: build/hdc.lowered.cpu2.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs -skip-verify -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

# GenHPVM for the HPVM IR
build/hpvm.cpu2.ll: build/hpvmc.cpu2.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -dce -mem2reg -genhpvm $< -S -o $@

# Other HPVM Passes
build/hpvmcompiled.cpu2.ll: build/hpvm.cpu2.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMBuildDFG.so -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so -load $(HPVM_LIB)/HPVMClearDFG.so -load $(HPVM_LIB)/HPVMDFGTransformPasses.so -dfg2llvm-cpu -clearDFG $< -S -o $@

HPVM_RT := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc

# LINK HPVM RT and DCE
build/output.2.ll: build/hpvmcompiled.cpu2.ll # build/cpu2_test.hpvmcompiled.ll
	$(HPVM_BIN)/llvm-link $< $(HPVM_RT) -S -o $@ 
	$(HPVM_BIN)/opt -dce -globaldce $< -S -o $<

# Target to generate approximation knobs
collect-approx: build/hpvm.genhdc.cpu2.1.ll

### ---------- GPU ---------------

# Generate Executable
host-gpu: build/output.gpu.ll
	$(HPVM_BIN)/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O2 $< -o $@

# Clang Frontend to IR
build/host.gpu.ll: src/host.cpp
	$(HPVM_BIN)/clang++ -DHPVM -DDEVICE=hpvm::GPU_TARGET -DENCODING_DIM=$(ENCODING_DIM) $(CFLAGS) $(INCLUDES) -S -emit-llvm $^ -o $@ 
#$HPVM_BIN/clang++ -O1 -I$OPENCL_PATH/include/CL -L$OPENCL_LIB_PATH -I$HPVM_ROOT/projects/hetero-c++/test/include/ -lm -lrt -pthread -lOpenCL  $INCLUDE_FLAG -fenable-matrix    $PREFIX.cpp -S -emit-llvm -o $PREFIX.ll

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.gpu.ll: build/host.gpu.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@
# $OPT -mem2reg -enable-new-pm=0 -load $HPVM_LIB/HPVMGenHPVM.so -genhpvm  -hpvm-hdc-only $PREFIX.ll -S -o $PREFIX.genhdc.hpvm.ll

# Lower HDC Intrinsics
build/hdc.lowered.gpu.ll: build/hpvm.genhdc.gpu.ll #hpvm. -enable-hcc
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so -vector-block-dim=1 -lower-hdc  -dce -simplifycfg $< -S -o $@

# HCC -> HPVM
build/hpvmc.gpu.ll: build/hdc.lowered.gpu.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs --skip-verify -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

# GenHPVM for the HPVM IR
build/hpvm.gpu.ll: build/hpvmc.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -O1 -dce -mem2reg -genhpvm $< -S -o $@

build/hpvmcompiled.gpu.ll: build/hpvm.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMBuildDFG.so -load $(HPVM_LIB)/HPVMLocalMem.so -load $(HPVM_LIB)/HPVMDFG2LLVM_GPU_OCL.so -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so -load $(HPVM_LIB)/HPVMClearDFG.so -load $(HPVM_LIB)/HPVMDFGTransformPasses.so -localmem -dfg2llvm-gpu-ocl -dfg2llvm-cpu -clearDFG $< -S -o $@
	$(HPVM_BIN)/llvm-ocl build/hpvm.gpu.kernels.ll

HPVM_RT := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc

# LINK HPVM RT and DCE
build/output.gpu.ll: build/hpvmcompiled.gpu.ll # build/gpu_test.hpvmcompiled.ll
	$(HPVM_BIN)/llvm-link $< $(HPVM_RT) -S -o $@ 
	$(HPVM_BIN)/opt -dce -globaldce $< -S -o $<

### ---------- GPU APPROX ---------------

# Generate Executable
host-gpu-approx: build/output.approx.gpu.ll
	$(HPVM_BIN)/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O2 $< -o $@

# Clang Frontend to IR
build/host.approx.gpu.ll: src/host.cpp
	$(HPVM_BIN)/clang++ -DHPVM -DDEVICE=hpvm::GPU_TARGET $(APPROX_MACRO_DEFINES) $(CFLAGS) $(INCLUDES) -S -emit-llvm $^ -o $@ 
#$HPVM_BIN/clang++ -O1 -I$OPENCL_PATH/include/CL -L$OPENCL_LIB_PATH -I$HPVM_ROOT/projects/hetero-c++/test/include/ -lm -lrt -pthread -lOpenCL  $INCLUDE_FLAG -fenable-matrix    $PREFIX.cpp -S -emit-llvm -o $PREFIX.ll

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.approx.gpu.0.ll: build/host.approx.gpu.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@
# $OPT -mem2reg -enable-new-pm=0 -load $HPVM_LIB/HPVMGenHPVM.so -genhpvm  -hpvm-hdc-only $PREFIX.ll -S -o $PREFIX.genhdc.hpvm.ll

# Get HDC Approx Knobs
build/hpvm.genhdc.approx.gpu.1.ll: build/hpvm.genhdc.approx.gpu.0.ll 
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -hdc-collect-approx --approx-config-path-collect=$(CONFIG_OUT_PATH) $< -S -o $@ 

# Set HDC Approx Knobs
build/hpvm.genhdc.approx.gpu.2.ll: build/hpvm.genhdc.approx.gpu.0.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -hdc-set-approx-knobs --approx-config-path=$(CONFIG_IN_PATH) $< -S -o $@ 

# Lower HDC Intrinsics
build/hdc.lowered.approx.gpu.ll: build/hpvm.genhdc.approx.gpu.2.ll #hpvm. -enable-hcc 
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so -vector-block-dim=1 -lower-hdc -dce -simplifycfg $< -S -o $@

# HCC -> HPVM
build/hpvmc.approx.gpu.ll: build/hdc.lowered.approx.gpu.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs --skip-verify -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

# GenHPVM for the HPVM IR
build/hpvm.approx.gpu.ll: build/hpvmc.approx.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -O1 -dce -mem2reg -genhpvm $< -S -o $@

build/hpvmcompiled.approx.gpu.ll: build/hpvm.approx.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMBuildDFG.so -load $(HPVM_LIB)/HPVMLocalMem.so -load $(HPVM_LIB)/HPVMDFG2LLVM_GPU_OCL.so -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so -load $(HPVM_LIB)/HPVMClearDFG.so -load $(HPVM_LIB)/HPVMDFGTransformPasses.so -localmem -dfg2llvm-gpu-ocl -dfg2llvm-cpu -clearDFG $< -S -o $@
	$(HPVM_BIN)/llvm-ocl build/hpvm.approx.gpu.kernels.ll

HPVM_RT := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc

# LINK HPVM RT and DCE
build/output.approx.gpu.ll: build/hpvmcompiled.approx.gpu.ll # build/gpu_test.hpvmcompiled.ll
	$(HPVM_BIN)/llvm-link $< $(HPVM_RT) -S -o $@ 
	$(HPVM_BIN)/opt -dce -globaldce $< -S -o $<

.PHONY: clean

clean:
	rm -rf *.dot build/*.ll output/out.txt score.txt host-gpu host-gpu-approx host-cpu-approx host-cpu host-accel host-sim host-cublas
