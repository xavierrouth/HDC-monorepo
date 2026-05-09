#include <heterocc.h>
#include <hpvm_hdc.h>
#include "DFG.hpp"
#include "host.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <numeric>
#include <cstring>

using namespace std;

// ---- Binary file reader (4-byte header skip) ----
template <typename T>
vector<T> readBinary(const string& path) {
    ifstream file(path, ios::in | ios::binary);
    assert(file.is_open() && "Couldn't open file!");
    file.seekg(0, ios::end);
    int32_t file_size = (int32_t)file.tellg();
    file.seekg(4, ios::beg);
    int32_t size = (file_size - 4) / sizeof(T);
    vector<T> data(size);
    file.read((char*)data.data(), size * sizeof(T));
    return data;
}

// ---- HV init callbacks ----
template <typename T>
T copy_flat(T* ptr, size_t idx) { return ptr[idx]; }

template <typename T>
T zero_init(size_t) { return (T)0; }


// ============================================================
// Generate level hypervectors (CPU)
//
// Matches TorchHD embeddings.Level(L, D):
//   level[0]   : random ±1
//   level[i]   : level[i-1] with incrementally flipped bits
//   level[L-1] : ~D/2 bits flipped vs level[0]  (orthogonal-ish)
// ============================================================
vector<hvtype> generate_level_vectors(int L, int D, unsigned seed = 42) {
    vector<hvtype> levels(L * D);
    mt19937 rng(seed);
    uniform_int_distribution<int> coin(0, 1);

    // Level 0: random ±1
    for (int j = 0; j < D; j++)
        levels[j] = coin(rng) ? (hvtype)1 : (hvtype)-1;

    if (L == 1) return levels;

    // Pre-shuffle flip order so flips are uniformly distributed across D
    vector<int> flip_order(D);
    iota(flip_order.begin(), flip_order.end(), 0);
    shuffle(flip_order.begin(), flip_order.end(), mt19937(seed + 1));

    // Spread D/2 total flips evenly over L-1 steps
    int total_flips = D / 2;
    for (int i = 1; i < L; i++) {
        // Copy previous level
        for (int j = 0; j < D; j++)
            levels[i * D + j] = levels[(i - 1) * D + j];
        // Flip the slice belonging to step i
        int flip_start = (int)((long long)(i - 1) * total_flips / (L - 1));
        int flip_end   = (int)((long long) i      * total_flips / (L - 1));
        for (int k = flip_start; k < flip_end && k < D; k++)
            levels[i * D + flip_order[k]] *= (hvtype)(-1);
    }
    return levels;
}


// ============================================================
// Generate base hypervectors (CPU)
// Matches TorchHD embeddings.Random(N_FEAT, D): random ±1
// ============================================================
vector<hvtype> generate_base_vectors(int N, int D, unsigned seed = 123) {
    vector<hvtype> base(N * D);
    mt19937 rng(seed);
    uniform_int_distribution<int> coin(0, 1);
    for (int i = 0; i < N * D; i++)
        base[i] = coin(rng) ? (hvtype)1 : (hvtype)-1;
    return base;
}


// ============================================================
// Generate random projection matrix [D × N] of ±1 values
// ============================================================
vector<hvtype> generate_rp_matrix(int D, int N, unsigned seed = 42) {
    vector<hvtype> rp(D * N);
    mt19937 rng(seed);
    uniform_int_distribution<int> coin(0, 1);
    for (auto& v : rp) v = coin(rng) ? (hvtype)1 : (hvtype)-1;
    return rp;
}


// ============================================================
// Precompute level indices for every (sample, feature) pair.
//
// data_stride: floats per sample in the binary file (may be > N_feat due to padding).
// Only the first N_feat elements of each stride-block are used.
// ============================================================
vector<int> compute_level_indices(
    const vector<float>& data, int N_samples, int N_feat, int data_stride, int L,
    const vector<float>& feat_min, const vector<float>& feat_max)
{
    vector<int> indices(N_samples * N_feat);
    for (int i = 0; i < N_samples; i++) {
        for (int j = 0; j < N_feat; j++) {
            float v    = data[(size_t)i * data_stride + j];
            float rng  = feat_max[j] - feat_min[j];
            float norm = (rng > 1e-10f) ? (v - feat_min[j]) / rng : 0.5f;
            norm = max(0.0f, min(1.0f, norm));
            int idx = (int)(norm * (float)(L - 1) + 0.5f);
            idx = max(0, min(L - 1, idx));
            indices[i * N_feat + j] = idx;
        }
    }
    return indices;
}


int main(int argc, char** argv) {
    __hpvm__init();

    int    EPOCH       = argc > 1 ? atoi(argv[1]) : 2;
    string X_train_path = argc > 2 ? argv[2] : "../../datasets/classification/isolet_nq_trainX.bin";
    string y_train_path = argc > 3 ? argv[3] : "../../datasets/classification/isolet_nq_trainY.bin";
    string X_test_path  = argc > 4 ? argv[4] : "../../datasets/classification/isolet_nq_testX.bin";
    string y_test_path  = argc > 5 ? argv[5] : "../../datasets/classification/isolet_nq_testY.bin";

    auto t_total_start = chrono::high_resolution_clock::now();
    auto t_start       = chrono::high_resolution_clock::now();

    // ---- Load data ----
    vector<float> X_train_f = readBinary<float>(X_train_path);
    vector<int>   y_train   = readBinary<int>(y_train_path);
    vector<float> X_test_f  = readBinary<float>(X_test_path);
    vector<int>   y_test    = readBinary<int>(y_test_path);

    assert((int)y_train.size() == N_SAMPLE && "Training label count mismatch");
    assert((int)y_test.size()  == N_TEST   && "Test label count mismatch");

    // Convert from 1-indexed labels to 0-indexed (same as MicroHD dataset handling)
    for (auto& l : y_train) l -= 1;
    for (auto& l : y_test)  l -= 1;

    // Derive actual per-sample stride from file (may include padding beyond N_FEAT)
    int train_stride = (int)(X_train_f.size() / N_SAMPLE);
    int test_stride  = (int)(X_test_f.size()  / N_TEST);
    cout << "Feature stride (train/test): " << train_stride << " / " << test_stride
         << "  (N_FEAT=" << N_FEAT << ")" << endl;

#if ENCODER_TYPE == 0  // IDLev
    // ---- Per-feature min/max from training data (for level quantization) ----
    vector<float> feat_min(N_FEAT,  numeric_limits<float>::max());
    vector<float> feat_max(N_FEAT,  numeric_limits<float>::lowest());
    for (int i = 0; i < N_SAMPLE; i++) {
        for (int j = 0; j < N_FEAT; j++) {
            float v = X_train_f[(size_t)i * train_stride + j];
            feat_min[j] = min(feat_min[j], v);
            feat_max[j] = max(feat_max[j], v);
        }
    }

    // ---- Precompute level indices (CPU, once) ----
    vector<int> train_lev_idx = compute_level_indices(X_train_f, N_SAMPLE, N_FEAT, train_stride, N_LEVELS, feat_min, feat_max);
    vector<int> test_lev_idx  = compute_level_indices(X_test_f,  N_TEST,   N_FEAT, test_stride,  N_LEVELS, feat_min, feat_max);

    size_t train_lev_size = (size_t)N_SAMPLE * N_FEAT * sizeof(int);
    size_t test_lev_size  = (size_t)N_TEST   * N_FEAT * sizeof(int);

    // ---- Generate base & level vectors on CPU, then copy into HPVM ----
    vector<hvtype> base_cpu  = generate_base_vectors(N_FEAT,   Dhv);
    vector<hvtype> level_cpu = generate_level_vectors(N_LEVELS, Dhv);

    auto t_elapsed = chrono::high_resolution_clock::now() - t_start;
    long mSec_load = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
    cout << "Data load + CPU init (IDLev): " << mSec_load << " ms" << endl;

    // ---- Allocate HPVM buffers ----

    // Base & level matrices
    __hypermatrix__<N_FEAT,   Dhv, hvtype> base_matrix  =
        __hetero_hdc_create_hypermatrix<N_FEAT,   Dhv, hvtype>(1, (void*)copy_flat<hvtype>, base_cpu.data());
    __hypermatrix__<N_LEVELS, Dhv, hvtype> level_matrix =
        __hetero_hdc_create_hypermatrix<N_LEVELS, Dhv, hvtype>(1, (void*)copy_flat<hvtype>, level_cpu.data());

    size_t base_matrix_size  = (size_t)N_FEAT   * Dhv * sizeof(hvtype);
    size_t level_matrix_size = (size_t)N_LEVELS  * Dhv * sizeof(hvtype);

    // Scratch hypervector (avoids loop-carried phi node in LLVM IR)
    __hypervector__<Dhv, hvtype> acc_hv =
        __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*)zero_init<hvtype>);
    size_t acc_hv_size = (size_t)Dhv * sizeof(hvtype);

#elif ENCODER_TYPE == 1  // RP
    // ---- Build compact feature matrices (strip stride padding) ----
    vector<hvtype> X_train_compact((size_t)N_SAMPLE * N_FEAT);
    vector<hvtype> X_test_compact((size_t)N_TEST   * N_FEAT);
    for (int i = 0; i < N_SAMPLE; i++)
        for (int j = 0; j < N_FEAT; j++)
            X_train_compact[(size_t)i * N_FEAT + j] = (hvtype)X_train_f[(size_t)i * train_stride + j];
    for (int i = 0; i < N_TEST; i++)
        for (int j = 0; j < N_FEAT; j++)
            X_test_compact[(size_t)i * N_FEAT + j] = (hvtype)X_test_f[(size_t)i * test_stride + j];

    // ---- Generate random projection matrix [Dhv × N_FEAT] ----
    vector<hvtype> rp_cpu = generate_rp_matrix(Dhv, N_FEAT);

    auto t_elapsed = chrono::high_resolution_clock::now() - t_start;
    long mSec_load = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
    cout << "Data load + CPU init (RP): " << mSec_load << " ms" << endl;

    // ---- Allocate HPVM buffers ----
    __hypermatrix__<Dhv, N_FEAT, hvtype> rp_matrix =
        __hetero_hdc_create_hypermatrix<Dhv, N_FEAT, hvtype>(1, (void*)copy_flat<hvtype>, rp_cpu.data());
    size_t rp_matrix_size = (size_t)Dhv * N_FEAT * sizeof(hvtype);

    __hypermatrix__<N_SAMPLE, N_FEAT, hvtype> train_input_hm =
        __hetero_hdc_create_hypermatrix<N_SAMPLE, N_FEAT, hvtype>(1, (void*)copy_flat<hvtype>, X_train_compact.data());
    __hypermatrix__<N_TEST,   N_FEAT, hvtype> test_input_hm =
        __hetero_hdc_create_hypermatrix<N_TEST,   N_FEAT, hvtype>(1, (void*)copy_flat<hvtype>, X_test_compact.data());
    size_t train_input_size = (size_t)N_SAMPLE * N_FEAT * sizeof(hvtype);
    size_t test_input_size  = (size_t)N_TEST   * N_FEAT * sizeof(hvtype);

    // Scratch hypervector (avoids loop-carried phi node in LLVM IR)
    __hypervector__<Dhv, hvtype> acc_hv =
        __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*)zero_init<hvtype>);
    size_t acc_hv_size = (size_t)Dhv * sizeof(hvtype);
#endif

    // Encoded HV buffers
    __hypermatrix__<N_SAMPLE, Dhv, hvtype> encoded_train_hvs =
        __hetero_hdc_create_hypermatrix<N_SAMPLE, Dhv, hvtype>(0, (void*)zero_init<hvtype>);
    __hypermatrix__<N_TEST,   Dhv, hvtype> encoded_test_hvs  =
        __hetero_hdc_create_hypermatrix<N_TEST,   Dhv, hvtype>(0, (void*)zero_init<hvtype>);

    size_t encoded_train_size = (size_t)N_SAMPLE * Dhv * sizeof(hvtype);
    size_t encoded_test_size  = (size_t)N_TEST   * Dhv * sizeof(hvtype);

    auto encoded_train_handle = __hetero_hdc_get_handle(encoded_train_hvs);
    auto encoded_test_handle  = __hetero_hdc_get_handle(encoded_test_hvs);

    // Class vectors (sum of per-class training HVs)
    __hypermatrix__<N_CLASS, Dhv, hvtype> classes =
        __hetero_hdc_create_hypermatrix<N_CLASS, Dhv, hvtype>(0, (void*)zero_init<hvtype>);
    auto classes_handle = __hetero_hdc_get_handle(classes);
    size_t classes_size = (size_t)N_CLASS * Dhv * sizeof(hvtype);

    // Update HV scratch buffer (training)
    __hypervector__<Dhv, hvtype> update_hv =
        __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*)zero_init<hvtype>);
    size_t update_hv_size = (size_t)Dhv * sizeof(hvtype);

    // Inference output
    int inference_labels[N_TEST];
    memset(inference_labels, 0, sizeof(inference_labels));

    // ============================================================
    // TRAINING
    // ============================================================
    t_start = chrono::high_resolution_clock::now();

#if ENCODER_TYPE == 0
    cout << "Encoding training data (IDLev)..." << endl;
    void* train_enc_dag = __hetero_launch(
        (void*) idlev_encoding_node<Dhv, N_FEAT, N_SAMPLE, N_LEVELS>,
        5,
        __hetero_hdc_get_handle(base_matrix),  base_matrix_size,
        __hetero_hdc_get_handle(level_matrix), level_matrix_size,
        train_lev_idx.data(),                  train_lev_size,
        __hetero_hdc_get_handle(acc_hv),       acc_hv_size,
        encoded_train_handle,                  encoded_train_size,
        1,
        encoded_train_handle, encoded_train_size);
    __hetero_wait(train_enc_dag);
#elif ENCODER_TYPE == 1
    cout << "Encoding training data (RP)..." << endl;
    void* train_enc_dag = __hetero_launch(
        (void*) rp_encoding_node<Dhv, N_FEAT, N_SAMPLE>,
        4,
        __hetero_hdc_get_handle(rp_matrix),    rp_matrix_size,
        __hetero_hdc_get_handle(train_input_hm), train_input_size,
        __hetero_hdc_get_handle(acc_hv),       acc_hv_size,
        encoded_train_handle,                  encoded_train_size,
        1,
        encoded_train_handle, encoded_train_size);
    __hetero_wait(train_enc_dag);
#endif

    // CreateClasses: accumulate encoded HVs into class vectors
    cout << "Creating class vectors..." << endl;
    for (int i = 0; i < N_SAMPLE; i++) {
        int label     = y_train[i];
        auto class_hv = __hetero_hdc_get_matrix_row<N_CLASS, Dhv, hvtype>(classes, N_CLASS, Dhv, label);
        auto enc_hv   = __hetero_hdc_get_matrix_row<N_SAMPLE, Dhv, hvtype>(encoded_train_hvs, N_SAMPLE, Dhv, i);
        auto sum_hv   = __hetero_hdc_sum<Dhv, hvtype>(class_hv, enc_hv);
        __hetero_hdc_set_matrix_row<N_CLASS, Dhv, hvtype>(classes, sum_hv, label);
    }

    // Retrain for EPOCH iterations (online update on mispredictions)
    cout << "Retraining for " << EPOCH << " epoch(s)..." << endl;
    for (int ep = 0; ep < EPOCH; ep++) {
        void* train_dag = __hetero_launch(
            (void*) training_node<Dhv, N_CLASS, N_SAMPLE>,
            4,
            encoded_train_handle, encoded_train_size,
            y_train.data(),       (size_t)N_SAMPLE * sizeof(int),
            __hetero_hdc_get_handle(update_hv), update_hv_size,
            classes_handle,       classes_size,
            1,
            classes_handle, classes_size);
        __hetero_wait(train_dag);
    }

    auto t_end = chrono::high_resolution_clock::now();
    long training_time = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count();
    cout << "Training done: " << training_time << " ms" << endl;

    // ============================================================
    // INFERENCE
    // ============================================================
    t_start = chrono::high_resolution_clock::now();

#if ENCODER_TYPE == 0
    cout << "Encoding test data (IDLev)..." << endl;
    void* test_enc_dag = __hetero_launch(
        (void*) idlev_encoding_node<Dhv, N_FEAT, N_TEST, N_LEVELS>,
        5,
        __hetero_hdc_get_handle(base_matrix),  base_matrix_size,
        __hetero_hdc_get_handle(level_matrix), level_matrix_size,
        test_lev_idx.data(),                   test_lev_size,
        __hetero_hdc_get_handle(acc_hv),       acc_hv_size,
        encoded_test_handle,                   encoded_test_size,
        1,
        encoded_test_handle, encoded_test_size);
    __hetero_wait(test_enc_dag);
#elif ENCODER_TYPE == 1
    cout << "Encoding test data (RP)..." << endl;
    void* test_enc_dag = __hetero_launch(
        (void*) rp_encoding_node<Dhv, N_FEAT, N_TEST>,
        4,
        __hetero_hdc_get_handle(rp_matrix),    rp_matrix_size,
        __hetero_hdc_get_handle(test_input_hm), test_input_size,
        __hetero_hdc_get_handle(acc_hv),       acc_hv_size,
        encoded_test_handle,                   encoded_test_size,
        1,
        encoded_test_handle, encoded_test_size);
    __hetero_wait(test_enc_dag);
#endif

    cout << "Running inference..." << endl;
    void* inf_dag = __hetero_launch(
        (void*) inference_node<Dhv, N_CLASS, N_TEST>,
        3,
        encoded_test_handle,  encoded_test_size,
        classes_handle,       classes_size,
        inference_labels,     (size_t)N_TEST * sizeof(int),
        1,
        inference_labels, (size_t)N_TEST * sizeof(int));
    __hetero_wait(inf_dag);

    t_end = chrono::high_resolution_clock::now();
    long inference_time = chrono::duration_cast<chrono::milliseconds>(t_end - t_start).count();
    long total_time     = chrono::duration_cast<chrono::milliseconds>(
                              chrono::high_resolution_clock::now() - t_total_start).count();

    // ============================================================
    // Accuracy + output
    // ============================================================
    int correct = 0;
    ofstream outfile("output/out.txt");
    for (int i = 0; i < N_TEST; i++) {
        correct += (inference_labels[i] == y_test[i]);
        outfile << y_test[i] << " " << inference_labels[i] << "\n";
    }

    float test_accuracy = (float)correct / N_TEST;
    cout << "Test Accuracy: " << test_accuracy << endl;
    cout << "Inference Time: " << inference_time << " ms" << endl;
    cout << "Total Time:     " << total_time     << " ms" << endl;

    {
        ofstream acc("output/test_accuracy.txt", ios::trunc);
        if (acc.is_open()) acc << test_accuracy << "\n";

        ofstream timing("output/test_time.txt", ios::trunc);
        if (timing.is_open()) {
            timing << total_time     << "\n";
            timing << training_time  << "\n";
            timing << inference_time << "\n";
        }
    }

    __hpvm__cleanup();
    return 0;
}
