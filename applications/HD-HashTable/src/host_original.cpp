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

template <typename T>
T copy(T* data, size_t loop_index_var) {
    return data[loop_index_var];
}

template <int T>
hvtype dot_prod(hvtype* hv1, hvtype* hv2){
    hvtype result = 0;
    for(int i = 0; i < T; i++){
        result += hv1[i] * hv2[i];
    }
    return result;
}

template <typename T>
void parseCSV(const std::string& filename, std::vector<std::vector<T>>& data) {
    std::ifstream file(filename);
    std::string line;
    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<T> row;
        while (std::getline(ss, value, ',')) {
            row.push_back(static_cast<T>(std::stof(value)));
        }
        data.push_back(row);
    }
}

int main(){

	__hpvm__init();

    srand(time(NULL));  // Seed random number generator

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    // Open file to create encoding scheme from file.
    std::string encoding_file_name = "./dataset/encoding.csv";
    // Parse CSV line into vector of floats
    // std::vector<std::vector<hvtype>> encoding_values;
    // parseCSV(encoding_file_name, encoding_values);

    // Dump encoding values
    // std::cout << "Encoding values loaded from file:" << std::endl;
    // std::cout << "Number of encodings: " << encoding_values.size() << std::endl;
    // for (size_t i = 0; i < encoding_values.size(); i++) {
    //     std::cout << "Encoding " << i << ": ";
    //     for (size_t j = 0; j < encoding_values[i].size(); j++) {
    //         std::cout << encoding_values[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

//     __hypervector__<Dhv, hvtype> A_encoding = \
//     __hetero_hdc_create_hypervector<Dhv, hvtype>(1, (void*) copy<hvtype>, (void*) encoding_values[0].data());
// __hypervector__<Dhv, hvtype> C_encoding = \
//     __hetero_hdc_create_hypervector<Dhv, hvtype>(1, (void*) copy<hvtype>, (void*) encoding_values[1].data()); 
// __hypervector__<Dhv, hvtype> G_encoding = \
//     __hetero_hdc_create_hypervector<Dhv, hvtype>(1, (void*) copy<hvtype>, (void*) encoding_values[2].data());
// __hypervector__<Dhv, hvtype> T_encoding = \
//     __hetero_hdc_create_hypervector<Dhv, hvtype>(1, (void*) copy<hvtype>, (void*) encoding_values[3].data());

	__hypervector__<Dhv, hvtype> A_encoding = \
        __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
	__hypervector__<Dhv, hvtype> C_encoding = \
        __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>); 
	__hypervector__<Dhv, hvtype> G_encoding = \
        __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
	__hypervector__<Dhv, hvtype> T_encoding = \
        __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);

        std::cout << "Loaded encoding scheme from file!" << std::endl;
    // Ini
    auto A_encoding_handle = __hetero_hdc_get_handle(A_encoding);
    auto C_encoding_handle = __hetero_hdc_get_handle(C_encoding);
    auto G_encoding_handle = __hetero_hdc_get_handle(G_encoding);
    auto T_encoding_handle = __hetero_hdc_get_handle(T_encoding);

    // Dump hypervector handles
    // std::cout << "A_encoding: \t";
    // for (int i = 0; i < Dhv; i++) {
    //    std::cout << ((hvtype*)A_encoding_handle)[i] << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << "C_encoding: \t";
    // for (int i = 0; i < Dhv; i++) {
    //    std::cout << ((hvtype*)C_encoding_handle)[i] << ", ";
    // }

    void* init_encoding = __hetero_launch(
        (void*)  generate_encoding_scheme<Dhv>, 4,
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

    // Print the dot product of each pairing
    //     print("Print dot product between encodings to ensure they're dissimilar enough")
    // print(f'A and C: {dot(A_encoding, C_encoding)}')
    std::cout << "Print dot product between encodings to ensure they're dissimilar enough" << std::endl;
    std::cout << "A and C: \t" << dot_prod<Dhv>((hvtype*)__hetero_hdc_get_handle(A_encoding), \
        (hvtype*)__hetero_hdc_get_handle(C_encoding)) << std::endl;
    // A and G
    std::cout << "A and G: \t" << dot_prod<Dhv>((hvtype*)__hetero_hdc_get_handle(A_encoding), \
        (hvtype*)__hetero_hdc_get_handle(G_encoding)) << std::endl;
    // A and T
    std::cout << "A and T: \t" << dot_prod<Dhv>((hvtype*)__hetero_hdc_get_handle(A_encoding), \
        (hvtype*)__hetero_hdc_get_handle(T_encoding)) << std::endl;
    // C and G
    std::cout << "C and G: \t" << dot_prod<Dhv>((hvtype*)__hetero_hdc_get_handle(C_encoding), \
        (hvtype*)__hetero_hdc_get_handle(G_encoding)) << std::endl;
    // C and T
    std::cout << "C and T: \t" << dot_prod<Dhv>((hvtype*)__hetero_hdc_get_handle(C_encoding), \
        (hvtype*)__hetero_hdc_get_handle(T_encoding)) << std::endl;
    // G and T
    std::cout << "G and T: \t" << dot_prod<Dhv>((hvtype*)__hetero_hdc_get_handle(G_encoding), \
        (hvtype*)__hetero_hdc_get_handle(T_encoding)) << std::endl;
    // print(f'A and G: {dot(A_encoding, G_encoding)}')
    // print(f'A and T: {dot(A_encoding, T_encoding)}')
    // print(f'C and G: {dot(C_encoding, G_encoding)}')
    // print(f'C and T: {dot(C_encoding, T_encoding)}')
    // print(f'G and T: {dot(G_encoding, T_encoding)}')


    std::cout << "Created encoding scheme!" << std::endl;

    // auto encoding_scheme_handle = __hetero_hdc_get_handle(encoding_scheme);
    // size_t encoding_scheme_size = sizeof(hvtype) * Dhv * 4;

    __hypervector__<Dhv, hvtype> encoded_hv = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) one<hvtype>);	
    auto encoded_hv_handle = __hetero_hdc_get_handle(encoded_hv);

    __hypervector__<Dhv, hvtype> hash_table = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
    auto hash_table_handle = __hetero_hdc_get_handle(hash_table);
    size_t hash_table_size = sizeof(hvtype) * Dhv;


    __hypervector__<Dhv, hvtype> base_hv = __hetero_hdc_hypervector<Dhv, hvtype>();	


    // Populate hash table
    std::string hash_file_name = "./dataset/dataTrain.csv";
    std::fstream trainFile(hash_file_name, std::ios_base::in);

    // Print the length of the first line
    std::string a;
    // int counter = 0;
    while (trainFile >> a){
        // std::cout << a << std::endl;
        if (a.length() != K){
            std::cout << "Error: Kmer length is not correct: " << a.length() << "\n";
            // exit(1);
        }

        // Encode kmer
        void* one = __hetero_launch(
            (void*) encode_kmer<K, Dhv, 1>, 7,
            a.c_str(), sizeof(char) * K,
            encoded_hv_handle, sizeof(hvtype) * Dhv,
            __hetero_hdc_get_handle(base_hv), sizeof(hvtype) * Dhv,
            __hetero_hdc_get_handle(A_encoding), sizeof(hvtype) * Dhv,
            __hetero_hdc_get_handle(C_encoding), sizeof(hvtype) * Dhv,
            __hetero_hdc_get_handle(G_encoding), sizeof(hvtype) * Dhv,
            __hetero_hdc_get_handle(T_encoding), sizeof(hvtype) * Dhv,
            /* Output Buffers: 1*/ 1,
            encoded_hv_handle, sizeof(hvtype) * Dhv
            // "encode_kmer_task"
        );

        __hetero_wait(one);

        // Dump encoded hv handle
        // std::cout << "Encoding kmer: " << a << std::endl;
        // std::cout << "encoded_hv: \t";
        // for (int i = 0; i < Dhv; i++) {
        //    std::cout << ((hvtype*)encoded_hv_handle)[i] << ", ";
        // }
        // std::cout << std::endl;

        // // Add it to hash table 
        void* two = __hetero_launch(
            (void*) sum_in_place_task<Dhv>, 2,
            encoded_hv_handle, sizeof(hvtype) * Dhv,
            hash_table_handle, hash_table_size,
            /* Output Buffers: 1*/ 1,
            hash_table_handle, hash_table_size
            // "sum_task"
        );
        __hetero_wait(two);

        // Dump encoded hv handle
        // std::cout << "hash_table: \t";
        // for (int i = 0; i < Dhv; i++) {
        //    std::cout << ((hvtype*)hash_table_handle)[i] << ", ";
        // }
        // std::cout << std::endl;
    }

    std::string test_file_name = "./dataset/dataTest.csv";
    std::ifstream testFile(test_file_name);
    
    //std::vector<hvtype> kmer(K);
    char kmer[K];

    int true_pos = 0;
    int true_neg = 0;
    int total_pos = 0;
    int total_neg = 0;


    size_t encoded_size = sizeof(hvtype) * Dhv;
    
    std::cout << "Threshold: \t" << (Dhv / 2) << std::endl;


    #pragma clang loop unroll(disable)
    for(int i = 0; i < N_TEST; i++){
        // std::cout << "Iteration "<< i << "\n";

        std::string entry;
        testFile >> entry;

        int total_length = entry.length();
        std::string included_str = entry.substr(total_length-1, 1);
        std::string sequence = entry.substr(0,total_length-2);

        // std::cout << entry << "\t" << sequence << "\t" << included_str << "\n";
        assert(sequence.length() == K);

        void* encode_two = __hetero_launch(
            (void*) encode_kmer<K, Dhv, 0>, 7,
            entry.c_str(), sizeof(char) * K,
            encoded_hv_handle, sizeof(hvtype) * Dhv,
            __hetero_hdc_get_handle(base_hv), sizeof(hvtype) * Dhv,
            __hetero_hdc_get_handle(A_encoding), sizeof(hvtype) * Dhv,
            __hetero_hdc_get_handle(C_encoding), sizeof(hvtype) * Dhv,
            __hetero_hdc_get_handle(G_encoding), sizeof(hvtype) * Dhv,
            __hetero_hdc_get_handle(T_encoding), sizeof(hvtype) * Dhv,
            /* Output Buffers: 1*/ 1,
            encoded_hv_handle, sizeof(hvtype) * Dhv
            // "encode_kmer_task"
        );
        __hetero_wait(encode_two);

        // Add it to hash table 
        // Take  dot product
        // float similarity = __hetero_hdc_cossim<Dhv, hvtype>(encoded_hv, hash_table);
        // take dot product
        float dot_prod = 0.0;
        for (int i = 0; i < (int)(Dhv * QUERY_END); i += STRIDE) {
            hvtype a = ((hvtype*)&encoded_hv)[i];
            hvtype b = ((hvtype*)&hash_table)[i];
            dot_prod += a * b;
        }
        float similarity = dot_prod;

        // std::cout << "similarity:\t" << similarity << std::endl;
        bool in_hash_table = (included_str == "T");

        bool prediction = similarity > (Dhv / 2); // Threshold

        if(in_hash_table){
            total_pos += 1;
            if(prediction){
                true_pos += 1;
            }

        } else {
            total_neg += 1;
            if(!prediction){
                true_neg += 1;
            }
        }

    }

    testFile.close();

    std::cout << "Correctly identified " << true_pos << " / "<< total_pos <<" "<< K<<"-mers that were in the hash table\n";
    std::cout << "Accuracy: "<< ((100.0 *true_pos) / total_pos) << "%\n";

    std::cout << "Correctly identified " << true_neg << " / " << total_neg  <<" "<< K<<"-mers that were not in the hash table\n";
    std::cout << "Accuracy: "<< ((100.0 *true_neg) / total_neg) << "%\n";

    float total_accuracy  = 100.0 * (true_pos + true_neg) / (total_pos + total_neg);
    std::cout <<"Total Accuracy: "<< total_accuracy << "%\n";

    auto t_elapsed = std::chrono::high_resolution_clock::now() - start_time;
    long total_mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();

    std::ofstream result_file("results.txt", std::ios_base::app);
    result_file << total_accuracy / 100.0 << "\n";
    result_file << total_mSec << "\n";
    result_file.close();
	__hpvm__cleanup();

    return 0;
}
