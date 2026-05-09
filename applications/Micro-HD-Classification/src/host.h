#pragma once

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <time.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>

// ---- Compile-time dataset / model parameters ----
#ifndef N_FEAT
#define N_FEAT      617     // ISOLET: 617, UCIHAR: 561
#endif
#ifndef N_CLASS
#define N_CLASS     26      // ISOLET: 26, UCIHAR: 12
#endif
#ifndef N_SAMPLE
#define N_SAMPLE    6237    // ISOLET train
#endif
#ifndef N_TEST
#define N_TEST      1560    // ISOLET test
#endif
#ifndef N_LEVELS
#define N_LEVELS    128     // number of level hypervectors (MicroHD default: 128)
#endif
#ifndef ENCODING_DIM
#define ENCODING_DIM 1024
#endif
#define Dhv ENCODING_DIM

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif
typedef DATA_TYPE hvtype;

#ifndef DEVICE
#define DEVICE 1
#endif
#ifndef ENCODER_TYPE
#define ENCODER_TYPE 0   // 0 = IDLev (default), 1 = RP
#endif
