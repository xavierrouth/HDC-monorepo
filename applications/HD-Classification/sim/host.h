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

#define N_FEAT	        617  //  28 * 28  //feature per input (e.g., isolet: 617)
#define N_CLASS		    26 // 10	//number of classes. (e.g., isolet: 26, ucihar 12)
#define Dhv				32 //hypervectors length
#define N_SAMPLE 		6237 // 60000 // FIXME: Make these parametesr variable. 

#define N_TEST 			1560 // 10000  // Needs to be constant for templated funciton. 

#define COL				8 //number of columns of a matrix-vector multiplication window
#define ROW				32 //number of rows of a matrix-vector multiplication window (32, 64, 128, 256, 512)

#define PAD_			(N_FEAT & (COL - 1))

#if PAD_ == 0
	#define PAD 		0
#else
	#define PAD 		(COL - PAD_)
#endif

#define N_FEAT_PAD		617 // 28 * 28 // (N_FEAT + PAD)	//feature per input (e.g., isolet: 624, ucihar 568)

// #define HAMMING_DIST 1
#define QUANT

#ifdef QUANT
// TODO: Add these to directory.
std::string  X_train_path = "../../datasets/classification/isolet_X_train.csv";
std::string  y_train_path = "../../datasets/classification/isolet_y_train.csv";
std::string  X_test_path = "../../datasets/classification/isolet_X_test.csv";
std::string  y_test_path = "../../datasets/classification/isolet_y_test.csv";

#else

std::string X_train_path = "../../datasets/classification/isolet_nq_trainX.bin";
std::string  y_train_path = "../../datasets/classification/isolet_nq_trainY.bin";
std::string  X_test_path = "../../datasets/classification/isolet_nq_trainX.bin";
std::string  y_test_path = "../../datasets/classification/isolet_nq_trainY.bin";

#endif
