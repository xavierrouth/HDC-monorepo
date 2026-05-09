HPVM_INSTALL_DIR ?= /home/arnoor2/hpvm_hdc/hpvm/hpvm/
HPVM_BUILD_DIR := $(HPVM_INSTALL_DIR)/build/
INCLUDES := -I$(HPVM_INSTALL_DIR)/include -I$(HPVM_INSTALL_DIR)/benchmarks/include -I$(HPVM_INSTALL_DIR)/projects/hpvm-hdc -I$(HPVM_INSTALL_DIR)/projects/hetero-c++/test/include -I./src

HPVM_LIB := $(HPVM_INSTALL_DIR)/build/lib
HPVM_BIN := $(HPVM_INSTALL_DIR)/build/bin

HPVM_DECLS_FILE := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hetero-c++/lib/HPVMCFunctionDeclarations/HPVMCFunctionDeclarations.bc
ACCELERATOR_API_PATH := accelerator_api

CUDA_PATH:=/software/cuda-9.1
OPENCL_PATH:=$(CUDA_PATH)
OPENCL_LIB_PATH:=$(OPENCL_PATH/lib64)

MLC ?= 1
WRITE_VERIFY ?= 1
ENCODING_DIM ?= 1024
CONFIG_IN_PATH ?= /home/xrouth2/hdc/HDC-Approximation-Tuner/tests/compiler_integration/Hetero-C++/config_in.json
CONFIG_OUT_PATH ?= /home/xrouth2/hdc/HDC-Approximation-Tuner/tests/compiler_integration/Hetero-C++/config_out.json
# HV_ELEMENT_TYPE ?= F32


CFLAGS := -fenable-matrix -fno-discard-value-names -fno-exceptions -pthread

HDC_LIBS := -load $(HPVM_LIB)/LLVMHPVMHDC.so -load $(HPVM_LIB)/LLVMHDCBinProp.so  -load $(HPVM_LIB)/LLVMBinarize.so  -load $(HPVM_LIB)/LLVMHDCApproxKnobs.so -load $(HPVM_LIB)/LLVMHDCApprox.so

NVCC := /usr/bin/nvcc
O1_FLAGS = -dce -targetlibinfo -tti -tbaa -scoped-noalias-aa -assumption-cache-tracker -profile-summary-info -forceattrs -inferattrs -ipsccp -called-value-propagation -globalopt -domtree -mem2reg -deadargelim -basic-aa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -simplifycfg -basiccg -globals-aa -prune-eh -always-inline -function-attrs  -memoryssa -early-cse-memssa -speculative-execution -lazy-value-info -jump-threading -correlated-propagation -libcalls-shrinkwrap -branch-prob -block-freq -pgo-memop-opt -tailcallelim -reassociate -loop-simplify -lcssa-verification -lcssa -scalar-evolution -loop-rotate -licm -loop-unswitch -indvars -loop-idiom -loop-deletion -loop-unroll -memdep -memcpyopt -sccp -demanded-bits -bdce -dse -postdomtree -adce -barrier -rpo-function-attrs -globaldce -float2int -loop-accesses -loop-distribute -loop-vectorize -loop-load-elim -alignment-from-assumptions -strip-dead-prototypes -loop-sink -instsimplify -div-rem-pairs -verify -ee-instrument -early-cse -lower-expect


# PARAMETERS := -DHPVM -DDEVICE=hpvm::CPU_TARGET
#
#
#

##### HOST CPU
# Generate Executable
host-cpu: build/output.ll
	$(HPVM_INSTALL_DIR)/build/bin/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O3 $^ -o $@ 

# Clang Frontend to IR
build/host.cpu.ll: src/host.cpp
	$(HPVM_BIN)/clang++ -O0 -Xclang -disable-O0-optnone -DENCODING_DIM=$(ENCODING_DIM) -DHPVM -DDEVICE=hpvm::CPU_TARGET $(CFLAGS) $(INCLUDES) -S -emit-llvm $< -o $@ 
#$HPVM_BIN/clang++ -O1 -I$OPENCL_PATH/include/CL -L$OPENCL_LIB_PATH -I$HPVM_ROOT/projects/hetero-c++/test/include/ -lm -lrt -pthread -lOpenCL  $INCLUDE_FLAG -fenable-matrix    $PREFIX.cpp -S -emit-llvm -o $PREFIX.ll

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.cpu.ll: build/host.cpu.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@
# $OPT -mem2reg -enable-new-pm=0 -load $HPVM_LIB/HPVMGenHPVM.so -genhpvm  -hpvm-hdc-only $PREFIX.ll -S -o $PREFIX.genhdc.hpvm.ll

# Lower HDC Intrinsics
build/hdc.lowered.cpu.ll: build/hpvm.genhdc.cpu.ll #hpvm. # -enable-hcc 
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -hdc-bin-prop -hdc-binarize -vector-block-dim=1 -lower-hdc  -dce -simplifycfg $< -S -o $@ 
	
# HCC -> HPVM
build/hpvmc.cpu.ll: build/hdc.lowered.cpu.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs --dot-dfg -skip-verify  -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

# GenHPVM for the HPVM IR
build/hpvm.cpu.ll: build/hpvmc.cpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -dce -mem2reg -genhpvm $< -S -o $@

# Other HPVM Passes
build/hpvmcompiled.cpu.ll: build/hpvm.cpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMBuildDFG.so -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so -load $(HPVM_LIB)/HPVMClearDFG.so -load $(HPVM_LIB)/HPVMDFGTransformPasses.so -dfg2llvm-cpu -clearDFG $< -S -o $@

HPVM_RT := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc

# LINK HPVM RT and DCE
build/output.ll: build/hpvmcompiled.cpu.ll # build/cpu_test.hpvmcompiled.ll
	$(HPVM_INSTALL_DIR)/build/bin/llvm-link $< $(HPVM_RT) -S -o $@ 
	$(HPVM_INSTALL_DIR)/build/bin/opt -dce -globaldce $< -S -o $<

### ----------- CPU APPROX -----------
# Generate Executable
host-cpu-approx: build/output.2.ll
	$(HPVM_INSTALL_DIR)/build/bin/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O3 $^ -o $@ 

# Clang Frontend to IR
build/host.cpu2.ll: src/host.cpp
	$(HPVM_BIN)/clang++ -O0 -Xclang -disable-O0-optnone -DHPVM -DENCODING_DIM=$(ENCODING_DIM) -DDEVICE=hpvm::CPU_TARGET $(CFLAGS) $(INCLUDES) -S -emit-llvm $< -o $@ 
#$HPVM_BIN/clang++ -O1 -I$OPENCL_PATH/include/CL -L$OPENCL_LIB_PATH -I$HPVM_ROOT/projects/hetero-c++/test/include/ -lm -lrt -pthread -lOpenCL  $INCLUDE_FLAG -fenable-matrix    $PREFIX.cpp -S -emit-llvm -o $PREFIX.ll

# GenHPVM for HDC only and mem2reg
build/hpvm.genhdc.cpu2.0.ll: build/host.cpu2.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@
# $OPT -mem2reg -enable-new-pm=0 -load $HPVM_LIB/HPVMGenHPVM.so -genhpvm  -hpvm-hdc-only $PREFIX.ll -S -o $PREFIX.genhdc.hpvm.ll

# Get HDC Approx Knobs
build/hpvm.genhdc.cpu2.1.ll: build/hpvm.genhdc.cpu2.0.ll 
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -hdc-collect-approx --debug --approx-config-path-collect=$(CONFIG_OUT_PATH) $< -S -o $@ 

# Set HDC Approx Knobs
build/hpvm.genhdc.cpu2.2.ll: build/hpvm.genhdc.cpu2.1.ll 
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -hdc-set-approx-knobs --debug --approx-config-path=$(CONFIG_IN_PATH) $< -S -o $@ 

# Lower HDC Intrinsics
build/hdc.lowered.cpu2.ll: build/hpvm.genhdc.cpu2.2.ll #hpvm. # -enable-hcc 
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -hdc-bin-prop -hdc-binarize -vector-block-dim=1 -lower-hdc  -dce -simplifycfg $< -S -o $@ 
	
# HCC -> HPVM
build/hpvmc.cpu2.ll: build/hdc.lowered.cpu2.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs -skip-verify  -S -o $@
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
	$(HPVM_INSTALL_DIR)/build/bin/llvm-link $< $(HPVM_RT) -S -o $@ 
	$(HPVM_INSTALL_DIR)/build/bin/opt -dce -globaldce $< -S -o $<


.PHONY: clean


collect-approx: build/hpvm.genhdc.cpu2.1.ll
	
clean:
	rm -rf *.dot build/*.ll build/*.o host-gpu host-cpu host-accel host-cpu-approx host-sim host-fpga host-cublas
