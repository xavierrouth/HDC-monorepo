#define HPVM 1
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <chrono>

#ifdef HPVM
#include <heterocc.h>
#include <hpvm_hdc.h>
#include "DFG_gpu_batched.hpp"  // includes DFG.hpp which includes host.h + undefines K, D
#include "host.h"               // re-defines K=500, Dhv=ENCODING_DIM, N_TRAIN, N_HVS, etc.
#endif

template <typename T> T zero(size_t loop_index_var) { return 0; }
template <typename T> T one(size_t loop_index_var)  { return 1; }

int main() {
    __hpvm__init();
    srand(time(NULL));

    auto start_time = std::chrono::high_resolution_clock::now();

    // ---- Encoding scheme (same as serial version) ----
    __hypervector__<Dhv, hvtype> A_encoding = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
    __hypervector__<Dhv, hvtype> C_encoding = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
    __hypervector__<Dhv, hvtype> G_encoding = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
    __hypervector__<Dhv, hvtype> T_encoding = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);

    auto A_enc_handle = __hetero_hdc_get_handle(A_encoding);
    auto C_enc_handle = __hetero_hdc_get_handle(C_encoding);
    auto G_enc_handle = __hetero_hdc_get_handle(G_encoding);
    auto T_enc_handle = __hetero_hdc_get_handle(T_encoding);

    void* init_enc = __hetero_launch(
        (void*) generate_encoding_scheme<Dhv>, 4,
        A_enc_handle, sizeof(hvtype) * Dhv,
        C_enc_handle, sizeof(hvtype) * Dhv,
        G_enc_handle, sizeof(hvtype) * Dhv,
        T_enc_handle, sizeof(hvtype) * Dhv,
        4,
        A_enc_handle, sizeof(hvtype) * Dhv,
        C_enc_handle, sizeof(hvtype) * Dhv,
        G_enc_handle, sizeof(hvtype) * Dhv,
        T_enc_handle, sizeof(hvtype) * Dhv
    );
    __hetero_wait(init_enc);
    std::cout << "Created encoding scheme!" << std::endl;

    // ---- Read all training kmers into flat buffer ----
    char* train_kmers = new char[(size_t)N_TRAIN * K]();
    int   actual_train = 0;

    {
        std::fstream trainFile("./dataset/dataTrain.csv", std::ios_base::in);
        std::string kmer_str;
        while (trainFile >> kmer_str && actual_train < N_TRAIN) {
            if ((int)kmer_str.length() != K) continue;
            memcpy(&train_kmers[(size_t)actual_train * K], kmer_str.c_str(), K);
            actual_train++;
        }
    }
    std::cout << "Read " << actual_train << " training kmers." << std::endl;

    // ---- Batch encode training kmers on GPU ----
    // Use HPVM-managed allocation so the runtime tracks the buffer in its table.
    int    train_enc_count   = actual_train;
    size_t train_kmers_size  = sizeof(char) * (size_t)train_enc_count * K;
    size_t encoded_train_size = sizeof(hvtype) * (size_t)N_TRAIN * Dhv;

    __hypermatrix__<N_TRAIN, Dhv, hvtype> encoded_train_hm = __hetero_hdc_hypermatrix<N_TRAIN, Dhv, hvtype>();
    auto   encoded_train_handle = __hetero_hdc_get_handle(encoded_train_hm);

    void* enc_train = __hetero_launch(
        (void*) encode_all_node<N_TRAIN, K, Dhv>, 6,
        train_kmers,          train_kmers_size,
        A_enc_handle,         sizeof(hvtype) * Dhv,
        C_enc_handle,         sizeof(hvtype) * Dhv,
        G_enc_handle,         sizeof(hvtype) * Dhv,
        T_enc_handle,         sizeof(hvtype) * Dhv,
        encoded_train_handle, encoded_train_size,   // input+output: registers buffer in HPVM table
        1,
        encoded_train_handle, encoded_train_size
    );
    __hetero_wait(enc_train);
    std::cout << "Encoded all training kmers." << std::endl;

    // ---- CPU bucket sum: fold encoded_train into hash_table ----
    size_t  hash_table_size = sizeof(hvtype) * (size_t)N_HVS * Dhv;
    hvtype* hash_table      = new hvtype[N_HVS * Dhv]();

    {
        hvtype* etrain = (hvtype*)encoded_train_handle;
        for (int k = 0; k < train_enc_count; k++) {
            int bucket = k / KMERS_PER_HV;
            if (bucket >= N_HVS) break;
            hvtype* src = &etrain[(size_t)k * Dhv];
            hvtype* dst = &hash_table[(size_t)bucket * Dhv];
            for (int d = 0; d < Dhv; d++)
                dst[d] += src[d];
        }
    }
    std::cout << "Built hash table: " << N_HVS << " buckets." << std::endl;

    // ---- Read test data ----
    char* test_kmers  = new char[(size_t)N_TEST * K]();
    bool* test_labels = new bool[N_TEST]();
    int   actual_test = 0;

    {
        std::ifstream testFile("./dataset/dataTest.csv");
        std::string entry;
        while (testFile >> entry && actual_test < N_TEST) {
            int total_length = (int)entry.length();
            std::string label_str = entry.substr(total_length - 1, 1);
            std::string sequence  = entry.substr(0, total_length - 2);
            if ((int)sequence.length() != K) continue;
            memcpy(&test_kmers[(size_t)actual_test * K], sequence.c_str(), K);
            test_labels[actual_test] = (label_str == "T");
            actual_test++;
        }
    }
    std::cout << "Read " << actual_test << " test entries." << std::endl;

    // ---- Batch encode test kmers on GPU ----
    size_t test_kmers_size    = sizeof(char)   * (size_t)actual_test * K;
    size_t encoded_test_size  = sizeof(hvtype)  * (size_t)N_TEST * Dhv;

    __hypermatrix__<N_TEST, Dhv, hvtype> encoded_test_hm = __hetero_hdc_hypermatrix<N_TEST, Dhv, hvtype>();
    auto   encoded_test_handle = __hetero_hdc_get_handle(encoded_test_hm);

    void* enc_test = __hetero_launch(
        (void*) encode_all_node<N_TEST, K, Dhv>, 6,
        test_kmers,          test_kmers_size,
        A_enc_handle,        sizeof(hvtype) * Dhv,
        C_enc_handle,        sizeof(hvtype) * Dhv,
        G_enc_handle,        sizeof(hvtype) * Dhv,
        T_enc_handle,        sizeof(hvtype) * Dhv,
        encoded_test_handle, encoded_test_size,    // input+output: registers buffer
        1,
        encoded_test_handle, encoded_test_size
    );
    __hetero_wait(enc_test);
    std::cout << "Encoded all test kmers." << std::endl;

    // ---- Batch query: max similarity per test kmer ----
    // hash_table is CPU-computed; pass as both input and output so HPVM registers it.
    size_t max_sims_size = sizeof(hvtype) * actual_test;
    hvtype* max_sims     = new hvtype[actual_test]();

    void* qry = __hetero_launch(
        (void*) query_all_node<N_TEST, N_HVS, Dhv>, 3,
        encoded_test_handle, encoded_test_size,
        hash_table,          hash_table_size,
        max_sims,            max_sims_size,     // also input so HPVM registers it
        1,
        max_sims, max_sims_size
    );
    __hetero_wait(qry);
    std::cout << "Query complete." << std::endl;

    // ---- Evaluate accuracy ----
    hvtype threshold = (hvtype)(Dhv) * THRESHOLD_MULT;

    int true_pos = 0, true_neg = 0, total_pos = 0, total_neg = 0;
    for (int i = 0; i < actual_test; i++) {
        bool prediction    = (max_sims[i] > threshold);
        bool in_hash_table = test_labels[i];

        if (in_hash_table) {
            total_pos++;
            if (prediction) true_pos++;
        } else {
            total_neg++;
            if (!prediction) true_neg++;
        }
    }

    std::cout << "Correctly identified " << true_pos << " / " << total_pos << " kmers in hash table\n";
    std::cout << "Accuracy: " << (100.0 * true_pos / total_pos) << "%\n";
    std::cout << "Correctly identified " << true_neg << " / " << total_neg << " kmers NOT in hash table\n";
    std::cout << "Accuracy: " << (100.0 * true_neg / total_neg) << "%\n";

    float total_accuracy = 100.0f * (true_pos + true_neg) / (total_pos + total_neg);
    std::cout << "Total Accuracy: " << total_accuracy << "%\n";

    auto t_elapsed = std::chrono::high_resolution_clock::now() - start_time;
    long total_mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();

    std::ofstream result_file("results.txt", std::ios_base::app);
    result_file << total_accuracy / 100.0 << "\n";
    result_file << total_mSec << "\n";
    result_file.close();

    delete[] train_kmers;
    delete[] hash_table;
    delete[] test_kmers;
    delete[] max_sims;
    delete[] test_labels;
    __hpvm__cleanup();
    return 0;
}
