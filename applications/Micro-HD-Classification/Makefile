HPVM_INSTALL_DIR ?= /home/xrouth2/hdc/hpvm/hpvm
# HPVM_BUILD: debug (default, uses build/) or release (uses build-release/)
HPVM_BUILD ?= debug
ifeq ($(HPVM_BUILD),release)
HPVM_BUILD_DIR := $(HPVM_INSTALL_DIR)/build-release
else
HPVM_BUILD_DIR := $(HPVM_INSTALL_DIR)/build
endif

INCLUDES := -I$(HPVM_INSTALL_DIR)/include \
            -I$(HPVM_INSTALL_DIR)/benchmarks/include \
            -I$(HPVM_INSTALL_DIR)/projects/hpvm-hdc \
            -I$(HPVM_INSTALL_DIR)/projects/hetero-c++/test/include \
            -I./src

HPVM_LIB := $(HPVM_BUILD_DIR)/lib
HPVM_BIN := $(HPVM_BUILD_DIR)/bin

HPVM_DECLS_FILE := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hetero-c++/lib/HPVMCFunctionDeclarations/HPVMCFunctionDeclarations.bc

CUDA_PATH    := /software/cuda-9.1
OPENCL_PATH  := $(CUDA_PATH)
OPENCL_LIB_PATH := $(OPENCL_PATH)/lib64
NVCC ?= "/usr/local/cuda"/bin/nvcc

CFLAGS := -O0 -Xclang -disable-O0-optnone -fenable-matrix -fno-discard-value-names \
          -fno-exceptions -lm -lrt -pthread -lOpenCL

O1_FLAGS = -targetlibinfo -tti -tbaa -scoped-noalias-aa -assumption-cache-tracker \
           -profile-summary-info -forceattrs -inferattrs -ipsccp \
           -called-value-propagation -globalopt -domtree -mem2reg -deadargelim \
           -basic-aa -aa -loops -lazy-branch-prob -lazy-block-freq \
           -opt-remark-emitter -instcombine -simplifycfg -basiccg -globals-aa \
           -prune-eh -always-inline -function-attrs -memoryssa -early-cse-memssa \
           -speculative-execution -lazy-value-info -jump-threading \
           -correlated-propagation -libcalls-shrinkwrap -branch-prob -block-freq \
           -pgo-memop-opt -tailcallelim -reassociate -loop-simplify \
           -lcssa-verification -lcssa -scalar-evolution -loop-rotate -licm \
           -loop-unswitch -indvars -loop-idiom -loop-deletion -loop-unroll \
           -memdep -memcpyopt -sccp -demanded-bits -bdce -dse -postdomtree \
           -adce -barrier -rpo-function-attrs -globaldce -float2int -loop-accesses \
           -loop-distribute -loop-vectorize -loop-load-elim \
           -alignment-from-assumptions -strip-dead-prototypes -loop-sink \
           -instsimplify -div-rem-pairs -verify -ee-instrument -early-cse \
           -lower-expect

HDC_LIBS := -load $(HPVM_LIB)/LLVMHPVMHDC.so \
            -load $(HPVM_LIB)/LLVMHDCBinProp.so \
            -load $(HPVM_LIB)/LLVMBinarize.so \
            -load $(HPVM_LIB)/LLVMHDCApproxKnobs.so \
            -load $(HPVM_LIB)/LLVMHDCApprox.so

SRC_ROOT := /home/xrouth2/hdc/HDC-monorepo/applications/Micro-HD-Classification
CONFIG_IN_PATH  ?= $(SRC_ROOT)/config/config_in.json
CONFIG_OUT_PATH ?= $(SRC_ROOT)/config/config_out.json

# ---- Tunable parameters ----
ENCODING_DIM  ?= 1024
DATA_TYPE     ?= float
N_LEVELS      ?= 128
N_FEAT        ?= 617
N_CLASS       ?= 26
N_SAMPLE      ?= 6237
N_TEST        ?= 1560
ENCODER_TYPE  ?= 0   # 0 = IDLev (default), 1 = RP

APPROX_MACRO_DEFINES := \
    -DENCODING_DIM=$(ENCODING_DIM) \
    -DDATA_TYPE=$(DATA_TYPE) \
    -DN_LEVELS=$(N_LEVELS) \
    -DN_FEAT=$(N_FEAT) \
    -DN_CLASS=$(N_CLASS) \
    -DN_SAMPLE=$(N_SAMPLE) \
    -DN_TEST=$(N_TEST) \
    -DENCODER_TYPE=$(ENCODER_TYPE)

HPVM_RT := $(HPVM_BUILD_DIR)/tools/hpvm/projects/hpvm-rt/hpvm-rt.bc

### ---------- GPU-APPROX ----------

host-gpu-approx: build/output.approx.gpu.ll
	$(HPVM_BIN)/clang++ \
	    -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) \
	    -lm -lrt -pthread -lOpenCL -O2 $< -o $@

build/host.approx.gpu.ll: src/host.cpp src/DFG.hpp src/host.h
	$(HPVM_BIN)/clang++ -O0 -Xclang -disable-O0-optnone \
	    -DHPVM -DDEVICE=hpvm::GPU_TARGET \
	    $(APPROX_MACRO_DEFINES) $(CFLAGS) $(INCLUDES) \
	    -S -emit-llvm $< -o $@

build/hpvm.genhdc.approx.gpu.0.ll: build/host.approx.gpu.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 \
	    -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only \
	    $< -S -o $@

build/hpvm.genhdc.approx.gpu.1.ll: build/hpvm.genhdc.approx.gpu.0.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) \
	    -hdc-collect-approx --approx-config-path-collect=$(CONFIG_OUT_PATH) \
	    $< -S -o $@

build/hpvm.genhdc.approx.gpu.2.ll: build/hpvm.genhdc.approx.gpu.0.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) \
	    -hdc-set-approx-knobs --approx-config-path=$(CONFIG_IN_PATH) \
	    $< -S -o $@

build/hdc.lowered.approx.gpu.ll: build/hpvm.genhdc.approx.gpu.2.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 \
	    -load $(HPVM_LIB)/LLVMHPVMHDC.so \
	    -vector-block-dim=1 -lower-hdc -dce -simplifycfg \
	    $< -S -o $@

build/hpvmc.approx.gpu.ll: build/hdc.lowered.approx.gpu.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) \
	    -sanitise-funcs --skip-verify -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

build/hpvm.approx.gpu.ll: build/hpvmc.approx.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 \
	    -load $(HPVM_LIB)/HPVMGenHPVM.so \
	    -O1 -dce -mem2reg -genhpvm \
	    $< -S -o $@

build/hpvmcompiled.approx.gpu.ll: build/hpvm.approx.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 \
	    -load $(HPVM_LIB)/HPVMBuildDFG.so \
	    -load $(HPVM_LIB)/HPVMLocalMem.so \
	    -load $(HPVM_LIB)/HPVMDFG2LLVM_GPU_OCL.so \
	    -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so \
	    -load $(HPVM_LIB)/HPVMClearDFG.so \
	    -load $(HPVM_LIB)/HPVMDFGTransformPasses.so \
	    -localmem -dfg2llvm-gpu-ocl -dfg2llvm-cpu -clearDFG \
	    $< -S -o $@
	$(HPVM_BIN)/llvm-ocl build/hpvm.approx.gpu.kernels.ll

build/output.approx.gpu.ll: build/hpvmcompiled.approx.gpu.ll
	$(HPVM_BIN)/llvm-link $< $(HPVM_RT) -S -o $@
	$(HPVM_BIN)/opt -dce -globaldce $< -S -o $<

collect-approx: build/hpvm.genhdc.approx.gpu.1.ll

### ---------- CPU-APPROX ----------

host-cpu-approx: build/output.cpu2.ll
	$(HPVM_BIN)/clang++ \
	    -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) \
	    -lm -lrt -pthread -lOpenCL -O3 $^ -o $@

build/host.cpu2.ll: src/host.cpp src/DFG.hpp src/host.h
	$(HPVM_BIN)/clang++ -O0 -Xclang -disable-O0-optnone \
	    -DHPVM -DDEVICE=hpvm::CPU_TARGET \
	    $(APPROX_MACRO_DEFINES) $(CFLAGS) $(INCLUDES) \
	    -S -emit-llvm $< -o $@

build/hpvm.genhdc.cpu2.0.ll: build/host.cpu2.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 \
	    -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only \
	    $< -S -o $@

build/hpvm.genhdc.cpu2.1.ll: build/hpvm.genhdc.cpu2.0.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) \
	    -hdc-collect-approx --approx-config-path-collect=$(CONFIG_OUT_PATH) \
	    $< -S -o $@

build/hpvm.genhdc.cpu2.2.ll: build/hpvm.genhdc.cpu2.0.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) \
	    -hdc-set-approx-knobs --approx-config-path=$(CONFIG_IN_PATH) \
	    $< -S -o $@

build/hdc.lowered.cpu2.ll: build/hpvm.genhdc.cpu2.2.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) \
	    -vector-block-dim=1 -lower-hdc -dce -simplifycfg \
	    $< -S -o $@

build/hpvmc.cpu2.ll: build/hdc.lowered.cpu2.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) \
	    -sanitise-funcs -skip-verify -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

build/hpvm.cpu2.ll: build/hpvmc.cpu2.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 \
	    -load $(HPVM_LIB)/HPVMGenHPVM.so \
	    -dce -mem2reg -genhpvm \
	    $< -S -o $@

build/hpvmcompiled.cpu2.ll: build/hpvm.cpu2.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 \
	    -load $(HPVM_LIB)/HPVMBuildDFG.so \
	    -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so \
	    -load $(HPVM_LIB)/HPVMClearDFG.so \
	    -load $(HPVM_LIB)/HPVMDFGTransformPasses.so \
	    -dfg2llvm-cpu -clearDFG \
	    $< -S -o $@

build/output.cpu2.ll: build/hpvmcompiled.cpu2.ll
	$(HPVM_BIN)/llvm-link $< $(HPVM_RT) -S -o $@
	$(HPVM_BIN)/opt -dce -globaldce $< -S -o $<

### ---------- Phony ----------

.PHONY: clean
clean:
	rm -rf build/*.ll host-cpu-approx host-gpu-approx output/out.txt
