#pragma once

#include <fstream>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>

#ifndef N_CENTER
#define N_CENTER		26	//number of centers. (e.g., isolet: 26, mnist: 10)
#endif
#ifndef N_FEAT
#define N_FEAT			617//feature per input (e.g., isolet: 617, mnist: 784)
#endif
#define Dhv				ENCODING_DIM  //hypervectors length
#ifndef N_SAMPLE
#define N_SAMPLE 		6238
#endif

#define COL				8 //number of columns of a matrix-vector multiplication window
#define ROW				32 //number of rows of a matrix-vector multiplication window (32, 64, 128, 256, 512)

#define PAD_			(N_FEAT & (COL - 1))
#if PAD_ == 0
	#define PAD 		0
#else
	#define PAD 		(COL - PAD_)
#endif

#define N_FEAT_PAD		(N_FEAT + PAD)	//feature per input (e.g., isolet: 624, ucihar 568)

// float, int64, int32, int8?, 
typedef DATA_TYPE hvtype;

#if HAMMING_DIST == 1
	#define SCORES_TYPE hvtype
#else
	#define SCORES_TYPE float
#endif

#ifndef DEVICE
#define DEVICE 1
#endif