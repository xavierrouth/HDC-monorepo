ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# ----- Tuning knob defaults (overridden by autotune.py on the command line) -----
HV_DIM             ?= 2048
EPOCHS             ?= 1
MLC                ?= 1
ADC_RES            ?= 8
ADC_STEP           ?= 1
ADC_FLOOR          ?= 0
ADC_CEIL           ?= 4
WRITE_VERIFY_CYCLE ?= 0

# ----- Build flavor -----
HPVM_BUILD ?= release
ifeq ($(HPVM_BUILD),release)
  OPT := -O3 -DNDEBUG
else
  OPT := -O0 -g
endif

CXX       := g++
CXXFLAGS  := $(OPT) -std=c++17 -DMAX_SIZE=1024 -DVERSION=\"1.0\"

DEFINES := \
  -DHV_DIM=$(HV_DIM) \
  -DEPOCHS=$(EPOCHS) \
  -DMLC=$(MLC) \
  -DADC_RES=$(ADC_RES) \
  -DADC_STEP=$(ADC_STEP) \
  -DADC_FLOOR=$(ADC_FLOOR) \
  -DADC_CEIL=$(ADC_CEIL) \
  -DWRITE_VERIFY_CYCLE=$(WRITE_VERIFY_CYCLE)

HDRS := indicators.hpp hpvm_interface.hpp api_hdnn_reram.hpp mnist_reader.hpp mnist_reader_common.hpp

.PHONY: all clean

all: test

test: test.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $(DEFINES) test.cpp -o test

minitest: minitest.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $(DEFINES) minitest.cpp -o minitest

clean:
	-rm -f *.o test minitest results.txt
