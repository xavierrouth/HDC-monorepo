#define Dhv ENCODING_DIM
#define K 500

#ifndef KMERS_PER_HV
#define KMERS_PER_HV 400
#endif

#define N_TRAIN 24829
#define N_HVS ((N_TRAIN + KMERS_PER_HV - 1) / KMERS_PER_HV)

#ifndef N_TEST
#define N_TEST 10000
#endif

#ifndef KMER_STRIDE
#define KMER_STRIDE 1
#endif

#ifndef THRESHOLD_MULT
#define THRESHOLD_MULT 1.0f
#endif

typedef float hvtype;
