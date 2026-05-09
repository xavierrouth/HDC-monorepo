#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <time.h>
#include <chrono>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>

// #define PARALLEL_UPDATE

#ifndef HAMMING_DIST
#define HAMMING_DIST 1
#endif

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef BINARIZE
#define BINARIZE 0
#endif

#ifndef RHO_SHIFT
#define RHO_SHIFT 0
#endif

#ifndef POLARIZE
#define POLARIZE 0
#endif

typedef DATA_TYPE hv_type;