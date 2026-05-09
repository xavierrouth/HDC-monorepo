#define NUM_EDGES       10556

/* HPVM PORT of gpu_opt_cora.py and cora_lp_hd.py */
#include <hpvm_hdc.h>
#include <heterocc.h>
#include "encoding.hpp"
#include "DFG.hpp"

#include "host.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <chrono>


#ifndef D
#define D ENCODING_DIM //4096
#endif

#ifndef Q
#define Q 100 // Quantization levels
#endif 

#ifndef F
#define F 1433 // Num Features
#endif 

#define TOTAL_NODES     2708
#define TRAIN_NODES     140
#define TEST_NODES      1000
#define VAL_NODES       500 // What are val nodes?
#define NUM_CLASSES     7  

// FIXME: Why don't these sum together? 

#define USE_LEVELS_ENCODING // If we use the level hvs to encode or not, or not the ID ones. 

template <typename T>
T zero_hv(size_t loop_index_var) {
	return 0;
}

template <typename T>
T zero_hm(size_t loop_index_var) {
	return 0;
}

template <typename T>
T hv_from_ptr(T* vec, size_t loop_index_var) {
    return vec[loop_index_var];
}

template <typename T>
T hm_from_ptr(T* vec, size_t loop_index_var) {
    return vec[loop_index_var];
}

template <typename T>
T all_neg_one(size_t loop_index_var) {
    return (T)(-1);
}

/* 1, -1 random*/
template <typename T>
T random_hv(size_t loop_index_var) {
    int r = rand() % 2;
    return (r == 1) ? 1.0 : -1.0; 
}

template <typename T>
T random_hm(size_t loop_index_var) {
    int r = rand() % 2;
    return (r == 1) ? 1.0 : -1.0; 
}


std::vector<int> read_mask(std::string filename) {
    std::ifstream filestream(filename, std::ifstream::in);
    std::string line;
    std::getline(filestream, line);
    std::vector<int> mask;
    std::string token;

    std::istringstream ss(line);

    while (std::getline(ss, token, ',')) {
        // std::cout << token << " ";
        mask.push_back(std::stoi(token));
    }
    return mask;
}


std::vector<std::pair<int, int>> read_edge_indicies (std::string filename) {

    std::vector<std::pair<int, int>> out;

    std::ifstream filestream(filename, std::ifstream::in);
    std::string line;

    while (std::getline(filestream, line)) {
        std::string token;
        std::istringstream ss(line);

        int i = 0;

        std::pair<int, int> p;

        while (std::getline(ss, token, ',')) {
            // std::cout << token << " ";
            if (i == 0)
                p.first = std::stoi(token);
            else
                p.second = std::stoi(token);
            i++;
        }

        out.push_back(p);
    }

    return out;
}


// Reads oiut indicies of where features exist. '1' marks that a feature exists, '0' means it doens't. 
std::vector<std::vector<int>> read_features(std::string filename) {

    std::vector<std::vector<int>> out;

    std::ifstream filestream(filename, std::ifstream::in);
    std::string line;

    while (std::getline(filestream, line)) {
        std::string token;
        std::istringstream ss(line);

        int i = 0;

        std::vector<int> line_vec;

        while (std::getline(ss, token, ',')) {
            // std::cout << token << " ";
            line_vec.push_back(std::stoi(token));
        }

        out.push_back(line_vec);
    }
    return out;
}

// FIXME: 
std::vector<int> read_true_labels(std::string filename) {
    std::ifstream filestream(filename, std::ifstream::in);
    std::string line;
    std::getline(filestream, line);
    std::vector<int> mask;
    std::string token;

    std::istringstream ss(line);

    while (std::getline(ss, token, ',')) {
        // std::cout << token << " ";
        mask.push_back(std::stoi(token));
    }
    return mask;
}

int main(int argc, char* argv[]){

    /* Load Dataset */
    int train_iter_num = (argc > 1) ? std::atoi(argv[1]) : 1;

    __hpvm__init();


    // FIXME: Make this command-line args
    // std::vector<int> test_mask = read_mask("/home/xrouth2/dev/HDGraph_JKang/HPVM/dataset/cora.test_mask");
    std::vector<int> test_mask = read_mask("dataset/cora.test_mask");
    std::vector<int> val_mask = read_mask("dataset/cora.val_mask");
    std::vector<int> train_mask = read_mask("dataset/cora.train_mask");

    size_t mask_size = sizeof(int) * test_mask.size(); // FIXME: make this a constant?

    std::vector<std::pair<int, int>> edge_coordinates = read_edge_indicies("dataset/cora.edges"); // FIXME: line 239 of gpu_opt_cora.py

    int n_edge_coordinates;

    size_t edge_coordinates_size = NUM_EDGES * sizeof(std::pair<int, int>);
    std::cout << "edges size: " <<  edge_coordinates.size() << std::endl;

    /*
    for (auto & p : edge_coordinates) {
        std::cout << p.first << " " << p.second << std::endl;
    } */

    std::vector<std::vector<int>> feature_indicies = read_features("dataset/cora.x");

    std::cout << feature_indicies.size() << std::endl;

    std::vector<int> flattened_indicies;

    for (const auto & sub : feature_indicies) {
        // int sum = std::accumulate(sub.begin(), sub.end(), 0);
        // std::cout << " sum: " << sum;
        flattened_indicies.insert(flattened_indicies.end(), sub.begin(), sub.end());
    }

    size_t flattened_indicies_size = TOTAL_NODES * F * sizeof(int);

    
    std::vector<int> arrange;
    for (int i = 0; i < F; i++) {
        arrange.push_back(i);
    }
    /* Normalize node features */
    
    #if 0
    std::vector<int> normalized_node_features = normalize(node_features); //(take l2 norm)

    int max_feature_val = node_features.max();
    int min_feature_val = node_features.min();
    #endif

    std::vector<int> labels_true = read_true_labels("dataset/cora.y");

    size_t labels_size = sizeof(int) * labels_true.size(); // FIXME: Make this a constant

    std::vector<hv_type> feature_hvs_vec = generate_normal_hvs_stdvec<F, D, hv_type>(); // generate_id_hvs_stdvec<F, D, hv_type>();
    printf("debug: generate_normal_hvs_stdvec done\n"); fflush(stdout);

    std::cout << std::endl;

    #if 0
    for (int i = 0; i < 3; i++) {

        std::cout << "[";
        for (int j = 0; j < D; j++) {
            std::cout << feature_hvs_vec[i * D + j] << ", ";
        }
        std::cout << "]\n";
    }
    #endif

    // FIXME: Make sure this reads out correctly.
    __hypermatrix__<F, D, hv_type> feature_hvs = __hetero_hdc_create_hypermatrix<F, D, hv_type>(1, (void*) hm_from_ptr<hv_type>, feature_hvs_vec.data());
    printf("debug: feature_hvs done\n"); fflush(stdout);
    size_t feature_hvs_size = sizeof(hv_type) * F * D;

    // FIXME: Levels are *not* used.
    // __hypermatrix__<D, Q, hv_type> level_hvs = gen_lvs

    // FIXME: make sure these are different.
    __hypervector__<D, hv_type> b0 = __hetero_hdc_create_hypervector<D, hv_type>(0, (void*) random_hv<hv_type>); /* random hv even split of -1, 1*/
    printf("debug: b0 done\n"); fflush(stdout);
    __hypervector__<D, hv_type> b1 = __hetero_hdc_create_hypervector<D, hv_type>(0, (void*) random_hv<hv_type>);
    printf("debug: b1 done\n"); fflush(stdout);
    __hypervector__<D, hv_type> b2 = __hetero_hdc_create_hypervector<D, hv_type>(0, (void*) random_hv<hv_type>);
    printf("debug: b2 done\n"); fflush(stdout);


#ifdef PARALLEL_UPDATE
    __hypervector__<D, hv_type>* temp1 = new __hypervector__<D, hv_type>[TOTAL_NODES];
    __hypervector__<D, hv_type>* temp2 = new __hypervector__<D, hv_type>[TOTAL_NODES];
#else
    __hypervector__<D, hv_type> temp1 = __hetero_hdc_hypervector<D, hv_type>();
    printf("debug: temp1 done\n"); fflush(stdout);
    __hypervector__<D, hv_type> temp2 = __hetero_hdc_hypervector<D, hv_type>();
    printf("debug: temp2 done\n"); fflush(stdout);
#endif

    size_t hv_base_size = sizeof(hv_type) * D;

    // FIXME: Original code transforms to CSR

    __hypermatrix__<TOTAL_NODES, D, hv_type> node_hvs = __hetero_hdc_create_hypermatrix<TOTAL_NODES, D, hv_type>(0, (void*) zero_hm<hv_type>);
    printf("debug: node_hvs done\n"); fflush(stdout);
    size_t node_hvs_size = sizeof(hv_type) * D * TOTAL_NODES;

    __hypermatrix__<TOTAL_NODES, D, hv_type> memory_lv1_hvs = __hetero_hdc_create_hypermatrix<TOTAL_NODES, D, hv_type>(0, (void*) zero_hm<hv_type>);
    printf("debug: memory_lv1_hvs done\n"); fflush(stdout);
    __hypermatrix__<TOTAL_NODES, D, hv_type> memory_lv2_hvs = __hetero_hdc_create_hypermatrix<TOTAL_NODES, D, hv_type>(0, (void*) zero_hm<hv_type>);
    printf("debug: memory_lv2_hvs done\n"); fflush(stdout);

    __hypermatrix__<TOTAL_NODES, D, hv_type> rel_hvs = __hetero_hdc_hypermatrix<TOTAL_NODES, D, hv_type>(); // __hetero_hdc_sum()
    printf("debug: rel_hvs done\n"); fflush(stdout);
    __hypermatrix__<NUM_CLASSES, D, hv_type> class_hvs = __hetero_hdc_create_hypermatrix<NUM_CLASSES, D, hv_type>(0, (void*) zero_hm<hv_type>);
    printf("debug: class_hvs done\n"); fflush(stdout);
    size_t class_hvs_size = NUM_CLASSES * sizeof(hv_type) * D;

    int* labels_predicted = new int[TOTAL_NODES]; // FIXME: Can optimize this.

    // For debug.
    // for (int i = 0; i < TOTAL_NODES; i++) {
    //     labels_predicted[i] = -1;
    // }

    printf("before launch\n");

    // Start timer
    auto t_start = std::chrono::high_resolution_clock::now();


    void* DFG = __hetero_launch((void*) root_node<TOTAL_NODES, NUM_CLASSES, D, F>, 
        17,
        __hetero_hdc_get_handle(class_hvs), class_hvs_size,
        __hetero_hdc_get_handle(memory_lv1_hvs), node_hvs_size,
        __hetero_hdc_get_handle(memory_lv2_hvs), node_hvs_size,
        __hetero_hdc_get_handle(node_hvs), node_hvs_size,
        __hetero_hdc_get_handle(rel_hvs), node_hvs_size,
        __hetero_hdc_get_handle(feature_hvs), feature_hvs_size,
        flattened_indicies.data(), flattened_indicies_size,
        edge_coordinates.data(), edge_coordinates_size,
        train_mask.data(), mask_size,
        test_mask.data(), mask_size, //FIXME: test_mask.data()
        labels_true.data(), labels_size,
        __hetero_hdc_get_handle(b0), hv_base_size,
        __hetero_hdc_get_handle(b1), hv_base_size,
        __hetero_hdc_get_handle(b2), hv_base_size,
#ifdef PARALLEL_UPDATE
        temp1, hv_base_size * TOTAL_NODES,
        temp2, hv_base_size * TOTAL_NODES,
#else
        __hetero_hdc_get_handle(temp1), hv_base_size,
        __hetero_hdc_get_handle(temp2), hv_base_size,
#endif
        labels_predicted, labels_size,
        1,
        labels_predicted, labels_size
    );

    __hetero_wait(DFG);

    /* Corrective retraining loop (iterations 2..train_iter_num) */
    for (int iter = 1; iter < train_iter_num; iter++) {
        void* retrain_DFG = __hetero_launch(
            (void*) retrain_root_node<TOTAL_NODES, NUM_CLASSES, D>,
            4,
            __hetero_hdc_get_handle(class_hvs), class_hvs_size,
            __hetero_hdc_get_handle(rel_hvs),   node_hvs_size,
            labels_true.data(),                 labels_size,
            train_mask.data(),                  mask_size,
            1,
            __hetero_hdc_get_handle(class_hvs), class_hvs_size
        );
        __hetero_wait(retrain_DFG);
    }

    /* Polarize class_hvs: convert to sign(x) = ±1, matching Python */
#if POLARIZE == 1
    if (train_iter_num > 1) {
        void* polarize_DFG = __hetero_launch(
            (void*) polarize_root_node<NUM_CLASSES, D>,
            1,
            __hetero_hdc_get_handle(class_hvs), class_hvs_size,
            1,
            __hetero_hdc_get_handle(class_hvs), class_hvs_size
        );
        __hetero_wait(polarize_DFG);
    }
#endif

    /* Final search after all training iterations */
    if (train_iter_num > 1) {
        void* search_DFG = __hetero_launch(
            (void*) search_root_node<TOTAL_NODES, NUM_CLASSES, D>,
            6,
            __hetero_hdc_get_handle(class_hvs), class_hvs_size,
            test_mask.data(),                   mask_size,
            labels_predicted,                   labels_size,
            __hetero_hdc_get_handle(rel_hvs),   node_hvs_size,
            __hetero_hdc_get_handle(temp1),     hv_base_size,
            __hetero_hdc_get_handle(temp2),     hv_base_size,
            1,
            labels_predicted, labels_size
        );
        __hetero_wait(search_DFG);
    }

    auto t_elapsed = std::chrono::high_resolution_clock::now() - t_start;

    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(t_elapsed).count();

#ifdef PARALLEL_UPDATE
    delete[] temp1;
    delete[] temp2;
#endif

    int correct = 0;

    for (int i = 0; i < TOTAL_NODES; i++) {
        if (test_mask[i] != 1)
            continue;
        // FIXME: Check size of labels_true to make sure it is of NUM_TEST
        // printf("true: %d pred: %d\n", labels_true[i], labels_predicted[i]);

        if (labels_true[i] == labels_predicted[i]) {
            correct++;
        }
    }

    float accuracy = correct / (float) TEST_NODES;

    // FIXME: Make sure this is float. 
    std::cout << "accuracy:" << accuracy << std::endl;

    {
        std::ofstream acc("test_accuracy.txt", std::ios::out | std::ios::trunc);
        if (acc.is_open()) {
            acc << accuracy << std::endl;
        }
		std::ofstream timing("test_time.txt", std::ios::out | std::ios::trunc);
		if (timing.is_open()) {
			timing << microseconds << std::endl;
		}

    }

    __hpvm__cleanup();
    return 0;     
}
