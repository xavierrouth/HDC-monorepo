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

SRC_ROOT := /home/xrouth2/hdc/HDC-monorepo/applications/HD-HashTable
CONFIG_IN_PATH ?= $(SRC_ROOT)/config/config_in.json
CONFIG_OUT_PATH ?= $(SRC_ROOT)/config/config_out.json

# Guard against silently applying a leftover config_in.json (written by a
# previous tuning run) to a build that did not ask for approximation.
# autotune.py passes APPROX_CONFIG=1; hand-run builds must opt in the same way
# or reset the file. See applications/util/check_approx_config.py.
APPROX_CONFIG ?= 0
CHECK_APPROX_CONFIG := python3 $(SRC_ROOT)/../util/check_approx_config.py

ENCODING_DIM ?= 4096
STRIDE ?= 1
QUERY_END ?= 1.0
KMER_STRIDE ?= 1
KMERS_PER_HV ?= 400
THRESHOLD_MULT ?= 1.0
N_TEST ?= 10000
DATA_TYPE ?= float
BINARIZE_BUCKETS ?= 0
APPROX_MACRO_DEFINES := -DENCODING_DIM=$(ENCODING_DIM) -DSTRIDE=$(STRIDE) -DQUERY_END=$(QUERY_END) -DKMER_STRIDE=$(KMER_STRIDE) -DKMERS_PER_HV=$(KMERS_PER_HV) -DTHRESHOLD_MULT=$(THRESHOLD_MULT) -DN_TEST=$(N_TEST)
TYPED_DEFINES := $(APPROX_MACRO_DEFINES) -DDATA_TYPE=$(DATA_TYPE) -DBINARIZE_BUCKETS=$(BINARIZE_BUCKETS)

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
	@$(CHECK_APPROX_CONFIG) $(CONFIG_IN_PATH) $(APPROX_CONFIG)
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

# Target to generate approximation knobs.
# When CLEAR_CONFIG_OUT=1 (used by the GPU-batched 4-series), write an empty
# config_out.json so no stale HPVM-level ops pollute the custom-knob search space.
CLEAR_CONFIG_OUT ?= 0
ifeq ($(CLEAR_CONFIG_OUT),1)
collect-approx:
	echo '[]' > $(CONFIG_OUT_PATH)
else
collect-approx: build/hpvm.genhdc.cpu2.1.ll
endif

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
	@$(CHECK_APPROX_CONFIG) $(CONFIG_IN_PATH) $(APPROX_CONFIG)
	$(HPVM_BIN)/opt -enable-new-pm=0 $(HDC_LIBS) -hdc-set-approx-knobs --approx-config-path=$(CONFIG_IN_PATH) $< -S -o $@ 

# Lower HDC Intrinsics
build/hdc.lowered.approx.gpu.ll: build/hpvm.genhdc.approx.gpu.0.ll #hpvm. -enable-hcc 
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

### ---------- GPU BATCHED (parallel_loop per-kmer) ---------------

host-gpu-batched: build/output.batched.gpu.ll
	$(HPVM_BIN)/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O2 $< -o $@

build/host.batched.gpu.ll: src/host_gpu_batched.cpp src/DFG_gpu_batched.hpp src/DFG.hpp src/host.h
	$(HPVM_BIN)/clang++ -DHPVM -DDEVICE=hpvm::GPU_TARGET $(APPROX_MACRO_DEFINES) $(CFLAGS) $(INCLUDES) -S -emit-llvm $< -o $@

build/hpvm.genhdc.batched.gpu.ll: build/host.batched.gpu.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@

build/hdc.lowered.batched.gpu.ll: build/hpvm.genhdc.batched.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so -vector-block-dim=1 -lower-hdc -dce -simplifycfg $< -S -o $@

build/hpvmc.batched.gpu.ll: build/hdc.lowered.batched.gpu.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs --skip-verify -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

build/hpvm.batched.gpu.ll: build/hpvmc.batched.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -O1 -dce -mem2reg -genhpvm $< -S -o $@

build/hpvmcompiled.batched.gpu.ll: build/hpvm.batched.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMBuildDFG.so -load $(HPVM_LIB)/HPVMLocalMem.so -load $(HPVM_LIB)/HPVMDFG2LLVM_GPU_OCL.so -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so -load $(HPVM_LIB)/HPVMClearDFG.so -load $(HPVM_LIB)/HPVMDFGTransformPasses.so -localmem -dfg2llvm-gpu-ocl -dfg2llvm-cpu -clearDFG $< -S -o $@
	$(HPVM_BIN)/llvm-ocl build/hpvm.batched.gpu.kernels.ll

build/output.batched.gpu.ll: build/hpvmcompiled.batched.gpu.ll
	$(HPVM_BIN)/llvm-link $< $(HPVM_RT) -S -o $@
	$(HPVM_BIN)/opt -dce -globaldce $< -S -o $<

### ---------- GPU BATCHED TYPED (DATA_TYPE + BINARIZE_BUCKETS knobs) ----------

host-gpu-batched-typed: build/output.batched.typed.gpu.ll
	$(HPVM_BIN)/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O2 $< -o $@

build/host.batched.typed.gpu.ll: src/host_gpu_batched_typed.cpp src/DFG_gpu_batched_typed.hpp src/host_typed.h
	$(HPVM_BIN)/clang++ -DHPVM -DDEVICE=hpvm::GPU_TARGET $(TYPED_DEFINES) $(CFLAGS) $(INCLUDES) -S -emit-llvm $< -o $@

build/hpvm.genhdc.batched.typed.gpu.ll: build/host.batched.typed.gpu.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@

build/hdc.lowered.batched.typed.gpu.ll: build/hpvm.genhdc.batched.typed.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so -vector-block-dim=1 -lower-hdc -dce -simplifycfg $< -S -o $@

build/hpvmc.batched.typed.gpu.ll: build/hdc.lowered.batched.typed.gpu.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs --skip-verify -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

build/hpvm.batched.typed.gpu.ll: build/hpvmc.batched.typed.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -O1 -dce -mem2reg -genhpvm $< -S -o $@

build/hpvmcompiled.batched.typed.gpu.ll: build/hpvm.batched.typed.gpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMBuildDFG.so -load $(HPVM_LIB)/HPVMLocalMem.so -load $(HPVM_LIB)/HPVMDFG2LLVM_GPU_OCL.so -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so -load $(HPVM_LIB)/HPVMClearDFG.so -load $(HPVM_LIB)/HPVMDFGTransformPasses.so -localmem -dfg2llvm-gpu-ocl -dfg2llvm-cpu -clearDFG $< -S -o $@
	$(HPVM_BIN)/llvm-ocl build/hpvm.batched.typed.gpu.kernels.ll

build/output.batched.typed.gpu.ll: build/hpvmcompiled.batched.typed.gpu.ll
	$(HPVM_BIN)/llvm-link $< $(HPVM_RT) -S -o $@
	$(HPVM_BIN)/opt -dce -globaldce $< -S -o $<


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

### ---------- CPU BATCHED TYPED (DATA_TYPE + BINARIZE_BUCKETS knobs) ----------

host-cpu-batched-typed: build/output.batched.typed.cpu.ll
	$(HPVM_BIN)/clang++ -I$(OPENCL_PATH)/include/CL -L$(OPENCL_LIB_PATH) -lm -lrt -pthread -lOpenCL -O2 $< -o $@

build/host.batched.typed.cpu.ll: src/host_gpu_batched_typed.cpp src/DFG_gpu_batched_typed.hpp src/host_typed.h
	$(HPVM_BIN)/clang++ -DHPVM -DDEVICE=hpvm::CPU_TARGET $(TYPED_DEFINES) $(CFLAGS) $(INCLUDES) -S -emit-llvm $< -o $@

build/hpvm.genhdc.batched.typed.cpu.ll: build/host.batched.typed.cpu.ll
	$(HPVM_BIN)/opt -mem2reg $(O1_FLAGS) -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -genhpvm -hpvm-hdc-only $< -S -o $@

build/hdc.lowered.batched.typed.cpu.ll: build/hpvm.genhdc.batched.typed.cpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/LLVMHPVMHDC.so -vector-block-dim=1 -lower-hdc -dce -simplifycfg $< -S -o $@

build/hpvmc.batched.typed.cpu.ll: build/hdc.lowered.batched.typed.cpu.ll
	$(HPVM_BIN)/hcc $< -declsfile $(HPVM_DECLS_FILE) -sanitise-funcs --skip-verify -S -o $@
	$(HPVM_BIN)/opt $@ -dce -S -o $@

build/hpvm.batched.typed.cpu.ll: build/hpvmc.batched.typed.cpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMGenHPVM.so -O1 -dce -mem2reg -genhpvm $< -S -o $@

build/hpvmcompiled.batched.typed.cpu.ll: build/hpvm.batched.typed.cpu.ll
	$(HPVM_BIN)/opt -enable-new-pm=0 -load $(HPVM_LIB)/HPVMBuildDFG.so -load $(HPVM_LIB)/HPVMLocalMem.so -load $(HPVM_LIB)/HPVMDFG2LLVM_CPU.so -load $(HPVM_LIB)/HPVMClearDFG.so -load $(HPVM_LIB)/HPVMDFGTransformPasses.so -localmem -dfg2llvm-cpu -clearDFG $< -S -o $@

build/output.batched.typed.cpu.ll: build/hpvmcompiled.batched.typed.cpu.ll
	$(HPVM_BIN)/llvm-link $< $(HPVM_RT) -S -o $@
	$(HPVM_BIN)/opt -dce -globaldce $< -S -o $<

.PHONY: clean

clean:
	rm -rf *.dot build/*.ll output/out.txt score.txt host-gpu host-gpu-approx host-cpu-approx host-cpu host-gpu-batched host-gpu-batched-typed host-cpu-batched-typed host-accel host-sim host-cublas
