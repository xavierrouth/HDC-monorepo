#include <hpvm_hdc.h>
#include <heterocc.h>
#include <iostream>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif
typedef DATA_TYPE hvtype;

// ============================================================
// ID-Level encoding DFG node
//
// Implements MicroHD's IDLevEncoder.forward():
//   sample_hv = multiset( bind(base[j], level[lev_idx(x[j])]) )
//             = sum_j( base[j] * level[lev_idx(x[j])] )
//
// base_matrix:    [N_FEATURES x D]        - random ±1 base HVs
// level_matrix:   [N_LEV x D]             - linearly-interpolated level HVs
// lev_indices:    [N_SAMPLES * N_FEATURES] - precomputed level index per feature
// output:         [N_SAMPLES x D]         - encoded HVs
// ============================================================
// acc_ptr: scratch hypervector buffer (avoids loop-carried SSA phi node,
//          same pattern as temp_hv_ptr in HD-Classification InitialEncodingDFG)
template<int D, int N_FEATURES, int N_SAMPLES, int N_LEV>
void idlev_encoding_node(
    __hypermatrix__<N_FEATURES, D, hvtype>* base_matrix_ptr,  size_t base_matrix_size,
    __hypermatrix__<N_LEV, D, hvtype>*      level_matrix_ptr, size_t level_matrix_size,
    int*                                    lev_indices_ptr,   size_t lev_indices_size,
    __hypervector__<D, hvtype>*             acc_ptr,           size_t acc_size,
    __hypermatrix__<N_SAMPLES, D, hvtype>*  output_ptr,        size_t output_size)
{
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
        5,
        base_matrix_ptr,  base_matrix_size,
        level_matrix_ptr, level_matrix_size,
        lev_indices_ptr,  lev_indices_size,
        acc_ptr,          acc_size,
        output_ptr,       output_size,
        1,
        output_ptr, output_size,
        "idlev_encoding_task");

    for (int i = 0; i < N_SAMPLES; i++) {
        // Bootstrap acc with j=0 binding, then accumulate j=1..N_FEATURES-1.
        // Accumulator flows through *acc_ptr (memory), not as an SSA local,
        // so no loop-carried phi node is created in LLVM IR.
        int lev_idx0 = lev_indices_ptr[i * N_FEATURES + 0];
        auto lev_0   = __hetero_hdc_get_matrix_row<N_LEV,      D, hvtype>(*level_matrix_ptr, N_LEV,      D, lev_idx0);
        auto base_0  = __hetero_hdc_get_matrix_row<N_FEATURES, D, hvtype>(*base_matrix_ptr,  N_FEATURES, D, 0);
        *acc_ptr     = __hetero_hdc_mul<D, hvtype>(lev_0, base_0);   // bind, store via pointer

        for (int j = 1; j < N_FEATURES; j++) {
            int  lev_idx = lev_indices_ptr[i * N_FEATURES + j];
            auto lev_j   = __hetero_hdc_get_matrix_row<N_LEV,      D, hvtype>(*level_matrix_ptr, N_LEV,      D, lev_idx);
            auto base_j  = __hetero_hdc_get_matrix_row<N_FEATURES, D, hvtype>(*base_matrix_ptr,  N_FEATURES, D, j);
            auto bound   = __hetero_hdc_mul<D, hvtype>(lev_j, base_j);  // bind
            *acc_ptr     = __hetero_hdc_sum<D, hvtype>(*acc_ptr, bound); // accumulate via pointer
        }

        __hetero_hdc_no_approx(*acc_ptr);
        __hetero_hdc_set_matrix_row<N_SAMPLES, D, hvtype>(*output_ptr, *acc_ptr, i);
    }

    __hetero_task_end(task);
    __hetero_section_end(section);
}


// ============================================================
// Random Projection encoding DFG node
//
// Implements standard RP encoding:
//   encoded_hv = sign(rp_matrix × input_features)
//
// rp_matrix:  [D × N_FEATURES]        - random ±1 projection matrix
// input_data: [N_SAMPLES × N_FEATURES] - raw input features
// temp_ptr:   scratch hypervector (avoids loop-carried phi node)
// output:     [N_SAMPLES × D]          - encoded HVs (binary ±1)
// ============================================================
template<int D, int N_FEATURES, int N_SAMPLES>
void rp_encoding_node(
    __hypermatrix__<D, N_FEATURES, hvtype>*        rp_matrix_ptr, size_t rp_matrix_size,
    __hypermatrix__<N_SAMPLES, N_FEATURES, hvtype>* input_ptr,    size_t input_size,
    __hypervector__<D, hvtype>*                     temp_ptr,     size_t temp_size,
    __hypermatrix__<N_SAMPLES, D, hvtype>*          output_ptr,   size_t output_size)
{
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
        4,
        rp_matrix_ptr, rp_matrix_size,
        input_ptr,     input_size,
        temp_ptr,      temp_size,
        output_ptr,    output_size,
        1,
        output_ptr, output_size,
        "rp_encoding_task");

    for (int i = 0; i < N_SAMPLES; i++) {
        auto row  = __hetero_hdc_get_matrix_row<N_SAMPLES, N_FEATURES, hvtype>(*input_ptr, N_SAMPLES, N_FEATURES, i);
        *temp_ptr = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(row, *rp_matrix_ptr);
        *temp_ptr = __hetero_hdc_sign<D, hvtype>(*temp_ptr);
        __hetero_hdc_no_approx(*temp_ptr);
        __hetero_hdc_set_matrix_row<N_SAMPLES, D, hvtype>(*output_ptr, *temp_ptr, i);
    }

    __hetero_task_end(task);
    __hetero_section_end(section);
}


// ============================================================
// Online retraining DFG node
//
// Matches MicroHD's HDClassifier.add_online():
//   For each mispredicted sample:
//     class[true_label]  += encoded_hv
//     class[pred_label]  -= encoded_hv
//
// (Simplified from MicroHD's weighted version; lr=1, alpha=1.)
// ============================================================
template<int D, int K, int N_VEC>
void training_node(
    __hypermatrix__<N_VEC, D, hvtype>* encoded_hvs_ptr, size_t encoded_hvs_size,
    int*                               labels,           size_t labels_size,
    __hypervector__<D, hvtype>*        update_hv_ptr,    size_t update_hv_size,
    __hypermatrix__<K, D, hvtype>*     classes_ptr,      size_t classes_size)
{
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
        4,
        encoded_hvs_ptr, encoded_hvs_size,
        classes_ptr,     classes_size,
        labels,          labels_size,
        update_hv_ptr,   update_hv_size,
        1,
        classes_ptr, classes_size,
        "training_task");

    for (int i = 0; i < N_VEC; i++) {
        auto enc_hv = __hetero_hdc_get_matrix_row<N_VEC, D, hvtype>(*encoded_hvs_ptr, N_VEC, D, i);

        // Find argmax cosine similarity (MicroHD uses cosine similarity throughout)
        float max_score = -9999999.0f;
        int   max_index = 0;
        for (int c = 0; c < K; c++) {
            auto  class_hv = __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, c);
            float score    = (float)__hetero_hdc_cossim<D, hvtype>(enc_hv, class_hv);
            if (score > max_score) {
                max_score = score;
                max_index = c;
            }
        }

        int label = labels[i];
        if (label != max_index) {
            // class[true_label] += hv
            auto label_row  = __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, label);
            *update_hv_ptr  = __hetero_hdc_sum<D, hvtype>(label_row, enc_hv);
            __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, label);
            // class[pred_label] -= hv
            auto pred_row   = __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, max_index);
            *update_hv_ptr  = __hetero_hdc_sub<D, hvtype>(pred_row, enc_hv);
            __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, max_index);
        }
    }

    __hetero_task_end(task);
    __hetero_section_end(section);
}


// ============================================================
// Inference DFG node
//
// For each test HV: cosine similarity with all class HVs, argmax.
// Matches MicroHD's HDClassifier.forward() + Evaluate().
// ============================================================
template<int D, int K, int N_VEC>
void inference_node(
    __hypermatrix__<N_VEC, D, hvtype>* encoded_hvs_ptr, size_t encoded_hvs_size,
    __hypermatrix__<K, D, hvtype>*     classes_ptr,     size_t classes_size,
    int*                               labels_ptr,      size_t labels_size)
{
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
        3,
        encoded_hvs_ptr, encoded_hvs_size,
        classes_ptr,     classes_size,
        labels_ptr,      labels_size,
        1,
        labels_ptr, labels_size,
        "inference_task");

    __hetero_hint(DEVICE);

    for (int i = 0; i < N_VEC; i++) {
        auto  row       = __hetero_hdc_get_matrix_row<N_VEC, D, hvtype>(*encoded_hvs_ptr, N_VEC, D, i);
        float max_score = -9999999.0f;
        int   max_index = 0;
        for (int c = 0; c < K; c++) {
            auto  class_hv = __hetero_hdc_get_matrix_row<K, D, hvtype>(*classes_ptr, K, D, c);
            float score    = (float)__hetero_hdc_cossim<D, hvtype>(row, class_hv);
            if (score > max_score) {
                max_score = score;
                max_index = c;
            }
        }
        labels_ptr[i] = max_index;
    }

    __hetero_task_end(task);
    __hetero_section_end(section);
}
