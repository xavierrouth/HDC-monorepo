#include <hpvm_hdc.h>
#include "host.h"
#include <heterocc.h>
#include <iostream>
#include <random>

#undef D
#undef N_FEATURES
#undef K

template <int D,  int N_FEATURES>
void gen_hv(
        __hypermatrix__<N_FEATURES, D, hv_type>* gaussian_matrix, size_t  gaussian_matrix_size,
        hv_type mu, hv_type sigma
        ){

    void* root_section = __hetero_section_begin();


    void* root_task = __hetero_task_begin(
            /* Num Inputs */ 3,
            gaussian_matrix,   gaussian_matrix_size,
            mu, sigma,
            /* Num Outputs*/ 1,
            gaussian_matrix,   gaussian_matrix_size,
            "gen_hv_task"
            );

    *gaussian_matrix  = __hetero_hdc_gaussian_hypermatrix<N_FEATURES, D, hv_type>(mu, sigma);

#if BINARIZE == 1

    *gaussian_matrix  = __hetero_hdc_sign<N_FEATURES, D, hv_type>(mu, sigma);

#endif


    __hetero_task_end(root_task); 

    __hetero_section_end(root_section);

}

// Runs over all NODES
// FIXMEL Get rid of the temp pointers (t1_ptr), they are unused lots of places.
template <int TOTAL_NODES, int F, int D>
void encoding_node(
        __hypermatrix__<TOTAL_NODES, D, hv_type>* node_hvs, size_t node_hvs_size,
        __hypermatrix__<F, D, hv_type>* feature_hvs, size_t feature_hvs_size,
        int* feature_indicies, size_t indicies_size
    ){
    void* section = __hetero_section_begin();

            #if 0
    void* task = __hetero_task_begin(
        3, 
        node_hvs, node_hvs_size,
        feature_hvs, feature_hvs_size,
        feature_indicies, indicies_size,
        1,
        feature_hvs, feature_hvs_size
    );
    #endif

    // printf("encoding node\n");
    // FIXME: Reverse lops to make loop over i parallelizable?
    for (int i = 0; i < TOTAL_NODES; i++) {

        __hetero_parallel_loop(1,
            3, 
            node_hvs, node_hvs_size,
            feature_hvs, feature_hvs_size,
            feature_indicies, indicies_size,
            1,
            feature_hvs, feature_hvs_size
        );

        __hetero_hint(DEVICE);
        int* indicies = &feature_indicies[i * F]; // FIXME: Index this correctly.
        // If we dont' transform to CSR, we can just loop over everything
        //__hypervector__<D, hv_type> node_hv = __hetero_hdc_create_hypervector<D, hv_type>(0, (void*) zero_hv<hv_precision>);
        // node_hv
        // *t1_ptr = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*node_hvs, TOTAL_NODES, D, i); 

        for (int j = 0 ; j < F; j++) {
            // printf("ind: %d ", indicies[j]);
            // FIXME: Convert htis to use indicies as expected.
            if (indicies[j] != 1) {
                continue;
            }
            int f = j; //indicies[j];

            auto fhv_slice = __hetero_hdc_get_matrix_row<F,D,hv_type>(*feature_hvs, F, D, f);
            auto acc_slice = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*node_hvs, TOTAL_NODES, D, i);
            auto acc_hv = __hetero_hdc_sum<D, hv_type>(acc_slice, fhv_slice);
            __hetero_hdc_set_matrix_row<TOTAL_NODES, D, hv_type>(*node_hvs, acc_hv, i);
        }
        // printf("\n");
        //__hetero_hdc_set_matrix_row<TOTAL_NODES, D, hv_type>(*node_hvs, *t1_ptr, i);
        // aaaaaaprintf("v: %f ", ((float*) node_hvs)[i * D]); // ope that this works
    }
    // __hetero_task_end(task);



    // Reversed loops are a whole lot less cache efficient, so we change the way feature_indicies is stored also.
    // for (int j = 0; j < )
    __hetero_section_end(section);

    return;
}

template <int TOTAL_NODES, int F, int D>
void relation_hvs_node(
        __hypermatrix__<TOTAL_NODES, D, hv_type>* memory_lv1_hvs, size_t memory_lv1_size,
        __hypermatrix__<TOTAL_NODES, D, hv_type>* memory_lv2_hvs, size_t memory_lv2_size,
        __hypermatrix__<TOTAL_NODES, D, hv_type>* node_hvs, size_t node_hvs_size,
        __hypermatrix__<TOTAL_NODES, D, hv_type>* rel_hvs, size_t rel_hvs_size,
        std::pair<int, int>* edge_coordinates, size_t edge_coordinates_size, // Does this work?
        __hypervector__<D, hv_type>* t1_ptr, size_t t1_size,
        __hypervector__<D, hv_type>* t2_ptr, size_t t2_size,
        __hypervector__<D, hv_type>* b0_ptr, size_t b0_size,
        __hypervector__<D, hv_type>* b1_ptr, size_t b1_size,
        __hypervector__<D, hv_type>* b2_ptr, size_t b2_size
) {
    void* section = __hetero_section_begin();

    void* lv1_task = __hetero_task_begin(
        3, 
        memory_lv1_hvs, memory_lv1_size,
        node_hvs, node_hvs_size,
        edge_coordinates, edge_coordinates_size,
        1,
        memory_lv1_hvs, memory_lv1_size
    );

    /* Memory LV1 HV Generation */
    for (int i = 0; i < NUM_EDGES; i++) {
        std::pair<int, int> p = edge_coordinates[i];
        int h = p.first;
        int t = p.second;

        // printf("edge: %d, %d\n", h, t);
        // Can we do this in-place and avoid copies?
        // lv1_hv = memory_lv1_hvs[t];
        // __hypervector__<D, hv_type> lv1_hv 
        auto lv1_hv = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*memory_lv1_hvs, TOTAL_NODES, D, t);
        // node_hv = node_hvs[h];
        // __hypervector__<D, hv_type> node_hv 
        auto node_hv = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*node_hvs, TOTAL_NODES, D, h);
        //__hypervector__<D, hv_type> result_hv 
        auto result_hv = __hetero_hdc_sum<D, hv_type>(lv1_hv, node_hv);
        // memory_lv1_hvs[t] = result_hv;
        __hetero_hdc_set_matrix_row<TOTAL_NODES, D, hv_type>(*memory_lv1_hvs, result_hv, t);
    }

    __hetero_task_end(lv1_task);

    /* Memory LV2 HV Generation */
    void* lv2_task = __hetero_task_begin(
        3, 
        memory_lv1_hvs, memory_lv1_size,
        memory_lv2_hvs, memory_lv2_size,
        edge_coordinates, edge_coordinates_size,
        1,
        memory_lv1_hvs, memory_lv1_size
    );

    for (int i = 0; i < NUM_EDGES; i++) {
        std::pair<int, int> p = edge_coordinates[i];
        int h = p.first;
        int t = p.second;
        // This loading and then storing of rows is horribly inefficient, hopefully we optimize this out. 

        // lv2_hv = memory_lv2_hvs[t];
        auto lv2_hv = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*memory_lv2_hvs, TOTAL_NODES, D, t);
        // lv1_hv = memory_lv1_hvs[h];
        auto lv1_hv = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*memory_lv1_hvs, TOTAL_NODES, D, h);

        auto result_hv = __hetero_hdc_sum<D, hv_type>(lv2_hv, lv1_hv);
        // memory_lv2_hvs[t] = result_hv;

        __hetero_hdc_set_matrix_row<TOTAL_NODES, D, hv_type>(*memory_lv2_hvs, result_hv, t);
    }

    __hetero_task_end(lv2_task);

#if RHO_SHIFT == 1
    /* Circular shift each row of memory_lv2_hvs by -1 (rho), matching gpu_opt_cora.py */
    void* rho_task = __hetero_task_begin(
        1,
        memory_lv2_hvs, memory_lv2_size,
        1,
        memory_lv2_hvs, memory_lv2_size
    );
    {
        hv_type* flat = reinterpret_cast<hv_type*>(memory_lv2_hvs);
        for (int i = 0; i < TOTAL_NODES; i++) {
            hv_type first = flat[i * D];
            for (int j = 0; j < D - 1; j++) {
                flat[i * D + j] = flat[i * D + j + 1];
            }
            flat[i * D + D - 1] = first;
        }
    }
    __hetero_task_end(rho_task);
#endif

#ifndef PARALLEL_UPDATE 
    void* rel_task = __hetero_task_begin(
        9, 
        memory_lv1_hvs, memory_lv1_size,
        memory_lv2_hvs, memory_lv2_size,
        node_hvs, node_hvs_size,
        t1_ptr, t1_size,
        t2_ptr, t2_size,
        rel_hvs, rel_hvs_size,
        b0_ptr, b0_size,
        b1_ptr, b1_size,
        b2_ptr, b2_size,
        1,
        rel_hvs, rel_hvs_size,
        "relational_update"
    );

       __hetero_hint(DEVICE);
#endif
    /* Build Relation HVs */
    for (int i = 0; i < TOTAL_NODES; i++) {
#ifdef PARALLEL_UPDATE

    __hetero_parallel_loop(1,
        9, 
        memory_lv1_hvs, memory_lv1_size,
        memory_lv2_hvs, memory_lv2_size,
        node_hvs, node_hvs_size,
        t1_ptr, t1_size,
        t2_ptr, t2_size,
        rel_hvs, rel_hvs_size,
        b0_ptr, b0_size,
        b1_ptr, b1_size,
        b2_ptr, b2_size,
        1,
        rel_hvs, rel_hvs_size,
        "parallel_relational_update"
    );

        // __hetero_hint(DEVICE);
        __hypervector__<D, hv_type>* t1_i = t1_ptr + i;
        __hypervector__<D, hv_type>* t2_i = t2_ptr + i;
#else
        __hypervector__<D, hv_type>* t1_i = t1_ptr ;
        __hypervector__<D, hv_type>* t2_i = t2_ptr ;
#endif

        *t1_i = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*memory_lv2_hvs, TOTAL_NODES, D, i);
        *t1_i = __hetero_hdc_mul<D, hv_type>(*t1_i, *b2_ptr);


        *t2_i = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*memory_lv1_hvs, TOTAL_NODES, D, i);
        *t2_i  = __hetero_hdc_mul<D, hv_type>(*t2_i , *b1_ptr);


        *t1_i = __hetero_hdc_sum<D, hv_type>(*t2_i, *t1_i);

        *t2_i = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*node_hvs, TOTAL_NODES, D, i);
        *t2_i = __hetero_hdc_mul<D, hv_type>(*t2_i, *b0_ptr);


        *t1_i = __hetero_hdc_sum<D, hv_type>(*t1_i, *t2_i);
        __hetero_hdc_set_matrix_row<TOTAL_NODES, D, hv_type>(*rel_hvs, *t1_i, i);

    }

#ifndef PARALLEL_UPDATE
    __hetero_task_end(rel_task);
#endif
    __hetero_section_end(section);
    return;
}

template <int TOTAL_NODES, int NUM_CLASSES, int D>
void training_node (
    __hypermatrix__<NUM_CLASSES, D, hv_type>* class_hvs, size_t class_hvs_size,
    int* labels, size_t labels_size,
    int* train_mask, size_t train_mask_size,
    __hypermatrix__<TOTAL_NODES, D, hv_type>* rel_hvs, size_t rel_hvs_size
) {
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
        4,
        class_hvs, class_hvs_size,
        labels, labels_size,
        train_mask, train_mask_size,
        rel_hvs, rel_hvs_size,
        1,
        class_hvs, class_hvs_size
    );


    int train_num = 0;
    // We can parallelize this?
    for (int i = 0; i < TOTAL_NODES; i++) {
        // __hetero_hint(DEVICE);

        if (train_mask[i] == 0)
            continue;

        auto train_hv = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*rel_hvs, TOTAL_NODES, D, i); // Index as labels or train_num?

        int label = labels[i]; // Is this correct?, i.e labels[i] or labels[train_num];

        // printf("%d label %d\n", train_num, label);

        // FIXME: Is get_matrix_row optimized to return a reference?
        auto class_hv  = __hetero_hdc_get_matrix_row<NUM_CLASSES, D, hv_type>(*class_hvs, NUM_CLASSES, D, label);
        auto sum_hv = __hetero_hdc_sum<D, hv_type>(train_hv, class_hv);
        __hetero_hdc_set_matrix_row<NUM_CLASSES, D>(*class_hvs, sum_hv, label);

        train_num++;
    }

    __hetero_task_end(task);

    __hetero_section_end(section);
}

template <int TOTAL_NODES, int NUM_CLASSES, int D>
void search_node (
    __hypermatrix__<NUM_CLASSES, D, hv_type>* class_hvs, size_t class_hvs_size,
    int* test_mask, size_t test_mask_size,
    int* labels_predicted, size_t labels_predicted_size,
    __hypermatrix__<TOTAL_NODES, D, hv_type>* rel_hvs, size_t rel_hvs_size,
    __hypervector__<D, hv_type>* t1_ptr, size_t t1_size,
    __hypervector__<D, hv_type>* t2_ptr, size_t t2_size
) {
    
    void* section = __hetero_section_begin();
    #if 0
    void* task = __hetero_task_begin(
        6,
        class_hvs, class_hvs_size,
        test_mask, test_mask_size,
        labels_predicted, labels_predicted_size,
        rel_hvs, rel_hvs_size,
        t1_ptr, t1_size,
        t2_ptr, t2_size,
        1,
        class_hvs, class_hvs_size
    );
    #endif

    int test_num = 0;

    // We can parallelize this?
    for (int i = 0; i < TOTAL_NODES; i++) {
        __hetero_parallel_loop(1,
            6,
            class_hvs, class_hvs_size,
            test_mask, test_mask_size,
            labels_predicted, labels_predicted_size,
            rel_hvs, rel_hvs_size,
            t1_ptr, t1_size,
            t2_ptr, t2_size,
            1,
            class_hvs, class_hvs_size
        );

        __hetero_hint(DEVICE);

        if (test_mask[i] == 0)
            continue;
        

        int top_label = -1;
        hv_type top_score = -1.;

        auto node_hv = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*rel_hvs, TOTAL_NODES, D, i); // Index as labels or test_num

        #pragma clang loop unroll(disable)
        for (int j = 0; j < NUM_CLASSES; j++) {
            auto class_hv = __hetero_hdc_get_matrix_row<NUM_CLASSES, D, hv_type>(*class_hvs, NUM_CLASSES, D, j);

            #if HAMMING_DIST == 1
            // FIXME: We never do hdc sign, so the hamming distnace is never high at all
            // We don't want to do 'sign' in_place, so we have to assign to the temp pointers.
            *t1_ptr = __hetero_hdc_sign<D, hv_type>(node_hv);
            *t2_ptr = __hetero_hdc_sign<D, hv_type>(class_hv);
            hv_type score = D - __hetero_hdc_hamming_distance<D, hv_type>(*t1_ptr, *t2_ptr);
            #else
            hv_type score = __hetero_hdc_cossim<D, hv_type>(node_hv, class_hv);
            #endif

            // printf("score: %d ", score);
            if (score > top_score) {
                top_label = j;
                top_score = score;
            }
        }
        labels_predicted[i] = top_label;
        test_num++;
    }

    // __hetero_task_end(task);
    __hetero_section_end(section);
}


/* Corrective retraining: for each misclassified training node, add rel_hv to
   the correct class and subtract from the predicted class.  Matches the
   iterative refinement loop in gpu_opt_cora.py.
   Uses flat-array pointer access (same pattern as the rho shift) to avoid
   HDC intrinsics in a CPU task, which crash the HPVM GPU codegen pass. */
template <int TOTAL_NODES, int NUM_CLASSES, int D>
void retrain_root_node(
    __hypermatrix__<NUM_CLASSES, D, hv_type>* class_hvs, size_t class_hvs_size,
    __hypermatrix__<TOTAL_NODES, D, hv_type>* rel_hvs,   size_t rel_hvs_size,
    int* labels,     size_t labels_size,
    int* train_mask, size_t train_mask_size
) {
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
        4,
        class_hvs,  class_hvs_size,
        rel_hvs,    rel_hvs_size,
        labels,     labels_size,
        train_mask, train_mask_size,
        1,
        class_hvs,  class_hvs_size
    );

    {
        hv_type*       cls_flat = reinterpret_cast<hv_type*>(class_hvs);
        const hv_type* rel_flat = reinterpret_cast<const hv_type*>(rel_hvs);

        for (int i = 0; i < TOTAL_NODES; i++) {
            if (train_mask[i] == 0) continue;
            int true_l = labels[i];

            /* Find predicted label — normalize by class_hv norm only,
               matching Python: sim_vec / linalg.norm(class_hvs, dim=1) */
            int   pred_l   = 0;
            hv_type best   = -2.0f;
            for (int j = 0; j < NUM_CLASSES; j++) {
                hv_type dot = 0, nc = 0;
                for (int d = 0; d < D; d++) {
                    hv_type c = cls_flat[j * D + d];
                    hv_type r = rel_flat[i * D + d];
                    dot += c * r;  nc += c * c;
                }
                hv_type sim = dot / (std::sqrt(nc) + 1e-8f);
                if (sim > best) { best = sim; pred_l = j; }
            }

            if (pred_l != true_l) {
                for (int d = 0; d < D; d++) {
                    cls_flat[true_l * D + d] += rel_flat[i * D + d];
                    cls_flat[pred_l * D + d] -= rel_flat[i * D + d];
                }
            }
        }
    }

    __hetero_task_end(task);
    __hetero_section_end(section);
}


/* Polarize class_hvs: convert each element to sign(x) = +1 or -1.
   Python applies this once after all retrain iterations. */
template <int NUM_CLASSES, int D>
void polarize_root_node(
    __hypermatrix__<NUM_CLASSES, D, hv_type>* class_hvs, size_t class_hvs_size
) {
    void* section = __hetero_section_begin();
    void* task = __hetero_task_begin(
        1, class_hvs, class_hvs_size,
        1, class_hvs, class_hvs_size
    );
    {
        hv_type* flat = reinterpret_cast<hv_type*>(class_hvs);
        for (int j = 0; j < NUM_CLASSES; j++) {
            for (int d = 0; d < D; d++) {
                hv_type v = flat[j * D + d];
                flat[j * D + d] = (v >= 0.0f) ? 1.0f : -1.0f;
            }
        }
    }
    __hetero_task_end(task);
    __hetero_section_end(section);
}

/* Final inference after all training iterations.
   Inlines search_node's GPU parallel-loop logic directly (no nested DFG
   function call) to avoid the ClearDFG segfault that occurs when
   calling search_node (which has its own DFG sections) from a task. */
template <int TOTAL_NODES, int NUM_CLASSES, int D>
void search_root_node(
    __hypermatrix__<NUM_CLASSES, D, hv_type>* class_hvs,  size_t class_hvs_size,
    int* test_mask,                                         size_t test_mask_size,
    int* labels_predicted,                                  size_t labels_predicted_size,
    __hypermatrix__<TOTAL_NODES, D, hv_type>* rel_hvs,     size_t rel_hvs_size,
    __hypervector__<D, hv_type>* t1_ptr,                   size_t t1_size,
    __hypervector__<D, hv_type>* t2_ptr,                   size_t t2_size
) {
    void* section = __hetero_section_begin();

    for (int i = 0; i < TOTAL_NODES; i++) {
        __hetero_parallel_loop(1,
            6,
            class_hvs,        class_hvs_size,
            test_mask,        test_mask_size,
            labels_predicted, labels_predicted_size,
            rel_hvs,          rel_hvs_size,
            t1_ptr,           t1_size,
            t2_ptr,           t2_size,
            1,
            labels_predicted, labels_predicted_size
        );
        __hetero_hint(DEVICE);

        if (test_mask[i] == 0) continue;

        int top_label = -1;
        hv_type top_score = -1.;

        auto node_hv = __hetero_hdc_get_matrix_row<TOTAL_NODES, D, hv_type>(*rel_hvs, TOTAL_NODES, D, i);

        #pragma clang loop unroll(disable)
        for (int j = 0; j < NUM_CLASSES; j++) {
            auto class_hv = __hetero_hdc_get_matrix_row<NUM_CLASSES, D, hv_type>(*class_hvs, NUM_CLASSES, D, j);

            #if HAMMING_DIST == 1
            *t1_ptr = __hetero_hdc_sign<D, hv_type>(node_hv);
            *t2_ptr = __hetero_hdc_sign<D, hv_type>(class_hv);
            hv_type score = D - __hetero_hdc_hamming_distance<D, hv_type>(*t1_ptr, *t2_ptr);
            #else
            hv_type score = __hetero_hdc_cossim<D, hv_type>(node_hv, class_hv);
            #endif

            if (score > top_score) {
                top_label = j;
                top_score = score;
            }
        }
        labels_predicted[i] = top_label;
    }

    __hetero_section_end(section);
}


template <int TOTAL_NODES, int NUM_CLASSES, int D, int F>
void root_node(
    __hypermatrix__<NUM_CLASSES, D, hv_type>* class_hvs, size_t class_hvs_size,
    __hypermatrix__<TOTAL_NODES, D, hv_type>* memory_lv1_hvs, size_t memory_lv1_size,
    __hypermatrix__<TOTAL_NODES, D, hv_type>* memory_lv2_hvs, size_t memory_lv2_size,
    __hypermatrix__<TOTAL_NODES, D, hv_type>* node_hvs, size_t node_hvs_size,
     __hypermatrix__<TOTAL_NODES, D, hv_type>* rel_hvs, size_t rel_hvs_size,
    __hypermatrix__<F, D, hv_type>* feature_hvs, size_t feature_hvs_size,
    int* feature_indicies, size_t indicies_size,
    std::pair<int, int>* edge_coordinates, size_t edge_coordinates_size, // Does this work?
    int* train_mask, size_t train_mask_size,
    int* test_mask, size_t test_mask_size,
    int* labels, size_t labels_size,

    __hypervector__<D, hv_type>* b0_ptr, size_t b0_size,
    __hypervector__<D, hv_type>* b1_ptr, size_t b1_size,
    __hypervector__<D, hv_type>* b2_ptr, size_t b2_size,
    __hypervector__<D, hv_type>* t1_ptr, size_t t1_size,
    __hypervector__<D, hv_type>* t2_ptr, size_t t2_size,
    int* labels_predicted, size_t labels_predicted_size
) {
    void* section = __hetero_section_begin();

    void* encoding_task = __hetero_task_begin(
        3, 
        node_hvs, node_hvs_size,
        feature_hvs, feature_hvs_size,
        feature_indicies, indicies_size,
        1,
        feature_hvs, feature_hvs_size
    );

    encoding_node<TOTAL_NODES, F, D>(
        node_hvs, node_hvs_size, 
        feature_hvs, feature_hvs_size, 
        feature_indicies, indicies_size
    );

    __hetero_task_end(encoding_task);


    void* relation_outer_task = __hetero_task_begin(
        10, 
        memory_lv1_hvs, memory_lv1_size,
        memory_lv2_hvs, memory_lv2_size,
        node_hvs, node_hvs_size,
        edge_coordinates, edge_coordinates_size,
        rel_hvs, rel_hvs_size,
        t1_ptr, t1_size,
        t2_ptr, t2_size,
        b0_ptr, b0_size,
        b1_ptr, b1_size,
        b2_ptr, b2_size,
        1,
        rel_hvs, rel_hvs_size
    );

    relation_hvs_node<TOTAL_NODES, F, D>(
        memory_lv1_hvs, memory_lv1_size,
        memory_lv2_hvs, memory_lv2_size,
        node_hvs, node_hvs_size,
        rel_hvs, rel_hvs_size,
        edge_coordinates, edge_coordinates_size,
        t1_ptr, t1_size,
        t2_ptr, t2_size,
        b0_ptr, b0_size,
        b1_ptr, b1_size,
        b2_ptr, b2_size
    );

    __hetero_task_end(relation_outer_task);


     void* training_task = __hetero_task_begin(
        4,
        class_hvs, class_hvs_size,
        labels, labels_size,
        train_mask, train_mask_size,
        rel_hvs, rel_hvs_size,
        1,
        class_hvs, class_hvs_size
    );
    
    training_node<TOTAL_NODES, NUM_CLASSES, D>(
        class_hvs, class_hvs_size,
        labels, labels_size,
        train_mask, train_mask_size,
        rel_hvs, rel_hvs_size
    );  

    __hetero_task_end(training_task);

    void* search_task = __hetero_task_begin(
        6,
        class_hvs, class_hvs_size,
        test_mask, test_mask_size,
        labels_predicted, labels_predicted_size,
        rel_hvs, rel_hvs_size,
        t1_ptr, t1_size,
        t2_ptr, t2_size,
        1,
        class_hvs, class_hvs_size
    );

    search_node<TOTAL_NODES, NUM_CLASSES, D>(
        class_hvs, class_hvs_size,
        test_mask, test_mask_size,
        labels_predicted, labels_predicted_size,
        rel_hvs, rel_hvs_size,
        t1_ptr, t1_size,
        t2_ptr, t2_size
    );

    __hetero_task_end(search_task);


    __hetero_section_end(section);


    return;
}
