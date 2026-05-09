#define HPVM 1
#include <fstream>
#include <iostream>

#ifdef HPVM
#include <heterocc.h>
#include <hpvm_hdc.h>
#include "DFG.hpp"
#include "host.h"
#endif
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <sstream>
#include <chrono>

template <typename T> T zero(size_t loop_index_var) { return 0; }
template <typename T> T one(size_t loop_index_var) { return 1; }

int main(){
    __hpvm__init();

    srand(time(NULL));

    auto start_time = std::chrono::high_resolution_clock::now();

    // --- Encoding scheme ---
    __hypervector__<Dhv, hvtype> A_encoding = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
    __hypervector__<Dhv, hvtype> C_encoding = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
    __hypervector__<Dhv, hvtype> G_encoding = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
    __hypervector__<Dhv, hvtype> T_encoding = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);

    auto A_encoding_handle = __hetero_hdc_get_handle(A_encoding);
    auto C_encoding_handle = __hetero_hdc_get_handle(C_encoding);
    auto G_encoding_handle = __hetero_hdc_get_handle(G_encoding);
    auto T_encoding_handle = __hetero_hdc_get_handle(T_encoding);

    void* init_encoding = __hetero_launch(
        (void*) generate_encoding_scheme<Dhv>, 4,
        A_encoding_handle, sizeof(hvtype) * Dhv,
        C_encoding_handle, sizeof(hvtype) * Dhv,
        G_encoding_handle, sizeof(hvtype) * Dhv,
        T_encoding_handle, sizeof(hvtype) * Dhv,
        4,
        A_encoding_handle, sizeof(hvtype) * Dhv,
        C_encoding_handle, sizeof(hvtype) * Dhv,
        G_encoding_handle, sizeof(hvtype) * Dhv,
        T_encoding_handle, sizeof(hvtype) * Dhv
    );
    __hetero_wait(init_encoding);

    std::cout << "Created encoding scheme!" << std::endl;

    // --- Hash table: N_HVS hypervectors, each accumulating KMERS_PER_HV kmers ---
    __hypermatrix__<N_HVS, Dhv, hvtype> hash_table = __hetero_hdc_hypermatrix<N_HVS, Dhv, hvtype>();
    auto hash_table_handle = __hetero_hdc_get_handle(hash_table);
    size_t hash_table_size = sizeof(hvtype) * Dhv * N_HVS;

    // Scratch buffers for encoding
    __hypervector__<Dhv, hvtype> encoded_hv = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) one<hvtype>);
    auto encoded_hv_handle = __hetero_hdc_get_handle(encoded_hv);
    size_t encoded_size = sizeof(hvtype) * Dhv;

    __hypervector__<Dhv, hvtype> base_hv = __hetero_hdc_hypervector<Dhv, hvtype>();

    // Bucket tracking
    int* bucket_idx = new int(0);
    int* kmers_in_bucket = new int(0);

    // --- Training ---
    std::string hash_file_name = "./dataset/dataTrain.csv";
    std::fstream trainFile(hash_file_name, std::ios_base::in);
    std::string a;

    while (trainFile >> a) {
        if ((int)a.length() != K) continue;
        if (*bucket_idx >= N_HVS) break;

        void* enc = __hetero_launch(
            (void*) encode_kmer<K, Dhv, 1>, 7,
            a.c_str(), sizeof(char) * K,
            encoded_hv_handle, encoded_size,
            __hetero_hdc_get_handle(base_hv), sizeof(hvtype) * Dhv,
            A_encoding_handle, sizeof(hvtype) * Dhv,
            C_encoding_handle, sizeof(hvtype) * Dhv,
            G_encoding_handle, sizeof(hvtype) * Dhv,
            T_encoding_handle, sizeof(hvtype) * Dhv,
            1,
            encoded_hv_handle, encoded_size
        );
        __hetero_wait(enc);

        void* add = __hetero_launch(
            (void*) add_to_bucket_task<N_HVS, Dhv>, 3,
            hash_table_handle, hash_table_size,
            encoded_hv_handle, encoded_size,
            bucket_idx, sizeof(int),
            1,
            hash_table_handle, hash_table_size
        );
        __hetero_wait(add);

        (*kmers_in_bucket)++;
        if (*kmers_in_bucket >= KMERS_PER_HV) {
            (*bucket_idx)++;
            *kmers_in_bucket = 0;
        }
    }
    trainFile.close();
    std::cout << "Built hash table: " << (*bucket_idx) << " full buckets, "
              << (*kmers_in_bucket) << " kmers in last bucket." << std::endl;

    // --- Similarity output buffer ---
    __hypervector__<N_HVS, hvtype> sims = __hetero_hdc_create_hypervector<N_HVS, hvtype>(0, (void*) zero<hvtype>);
    auto sims_handle = __hetero_hdc_get_handle(sims);
    size_t sims_size = sizeof(hvtype) * N_HVS;

    // --- Testing ---
    std::string test_file_name = "./dataset/dataTest.csv";
    std::ifstream testFile(test_file_name);

    int true_pos = 0, true_neg = 0, total_pos = 0, total_neg = 0;
    hvtype threshold = (hvtype)(Dhv) * THRESHOLD_MULT;

    for (int i = 0; i < N_TEST; i++) {
        std::string entry;
        if (!(testFile >> entry)) break;

        int total_length = entry.length();
        std::string label_str = entry.substr(total_length - 1, 1);
        std::string sequence = entry.substr(0, total_length - 2);

        if ((int)sequence.length() != K) continue;

        void* enc2 = __hetero_launch(
            (void*) encode_kmer<K, Dhv, 0>, 7,
            sequence.c_str(), sizeof(char) * K,
            encoded_hv_handle, encoded_size,
            __hetero_hdc_get_handle(base_hv), sizeof(hvtype) * Dhv,
            A_encoding_handle, sizeof(hvtype) * Dhv,
            C_encoding_handle, sizeof(hvtype) * Dhv,
            G_encoding_handle, sizeof(hvtype) * Dhv,
            T_encoding_handle, sizeof(hvtype) * Dhv,
            1,
            encoded_hv_handle, encoded_size
        );
        __hetero_wait(enc2);

        void* qry = __hetero_launch(
            (void*) query_hash_task<N_HVS, Dhv>, 3,
            hash_table_handle, hash_table_size,
            encoded_hv_handle, encoded_size,
            sims_handle, sims_size,
            1,
            sims_handle, sims_size
        );
        __hetero_wait(qry);

        // Check if any bucket HV has similarity above threshold
        hvtype* sims_flat = (hvtype*) sims_handle;
        hvtype max_sim = sims_flat[0];
        for (int j = 1; j < N_HVS; j++)
            if (sims_flat[j] > max_sim) max_sim = sims_flat[j];
        bool prediction = (max_sim > threshold);

        bool in_hash_table = (label_str == "T");

        if (in_hash_table) {
            total_pos++;
            if (prediction) true_pos++;
        } else {
            total_neg++;
            if (!prediction) true_neg++;
        }
    }
    testFile.close();

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

    delete bucket_idx;
    delete kmers_in_bucket;
    __hpvm__cleanup();
    return 0;
}
