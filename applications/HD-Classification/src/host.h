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

#ifndef N_FEAT
#define N_FEAT	        617  //  28 * 28  //feature per input (e.g., isolet: 617)
#endif
#ifndef N_CLASS
#define N_CLASS		    26 // 10	//number of classes. (e.g., isolet: 26, ucihar 12)
#endif
#define Dhv				ENCODING_DIM //hypervectors length
#ifndef N_SAMPLE
#define N_SAMPLE 		6237 // 60000 // FIXME: Make these parametesr variable.
#endif

#ifndef N_TEST
#define N_TEST 			1560 // 10000  // Needs to be constant for templated funciton.
#endif

#define COL				8 //number of columns of a matrix-vector multiplication window
#define ROW				32 //number of rows of a matrix-vector multiplication window (32, 64, 128, 256, 512)

#define PAD_			(N_FEAT & (COL - 1))

#if PAD_ == 0
	#define PAD 		0
#else
	#define PAD 		(COL - PAD_)
#endif

#ifndef N_FEAT_PAD
#define N_FEAT_PAD		N_FEAT // (N_FEAT + PAD)	//feature per input (e.g., isolet: 617, ucihar 568)
#endif

// #define HAMMING_DIST 1
