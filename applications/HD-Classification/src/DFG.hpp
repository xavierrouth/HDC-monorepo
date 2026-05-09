#include <hpvm_hdc.h>
#include <heterocc.h>
#include <iostream>

#ifndef HAMMING_DIST
#define HAMMING_DIST 1
#endif

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#ifndef BINARIZE_ENCODING
#define BINARIZE_ENCODING 1
#endif
typedef DATA_TYPE hvtype;


// RANDOM PROJECTION ENCODING!!
// Matrix-vector mul
// Encodes a single vector using a random projection matrix
//
// RP encoding reduces N_features -> D 

// Need multiple copies because of HPVM limitations, so add unused template parameter.
// template<int D, int N_FEATURES, int unused>
// void rp_encoding_node(/* Input Buffers: 2*/ __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, __hypervector__<N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size, /* Output Buffers: 1*/ __hypervector__<D, hvtype>* output_hv_ptr, size_t output_hv_size) {

// #ifndef NODFG
//     void* section = __hetero_section_begin();

//     void* task = __hetero_task_begin(/* Input Buffers: 2*/ 3, rp_matrix_ptr, rp_matrix_size, 
//     input_datapoint_ptr, input_datapoint_size, 
//     output_hv_ptr, output_hv_size, /* Parameters: 0*/ 
//     /* Output Buffers: 1*/ 1, output_hv_ptr, output_hv_size, "inner_rp_encoding_task");

//     //__hetero_hint(DEVICE);
// #endif

//     // This zero initialization should no longer be needed as matmul self-initializes with 0
//     //__hypervector__<D, hvtype> encoded_hv = __hetero_hdc_create_hypervector<D, hvtype>(0, (void*) zero_hv<hvtype>);
//     // *output_hv_ptr = encoded_hv;
//     *output_hv_ptr = encoded_hv;

//     __hetero_task_end(task); 

//     __hetero_section_end(section);
//     return;
// }

template<int D, int N_FEATURES, int N_SAMPLES>
void encoding_dfg_train(/* Input Buffers: 2*/
    __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, 
    __hypermatrix__<N_SAMPLES, N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size, 
    __hypervector__<D, hvtype>* temp_hv_ptr, size_t temp_hv_size,
    __hypervector__<N_FEATURES, hvtype>* temp_row_ptr, size_t temp_row_size,
    /* Output Buffers: 1*/ 
    __hypermatrix__<N_SAMPLES, D, hvtype>* output_hm_ptr, size_t output_hv_size) {
    
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(/* Input Buffers: 2*/ 5, 
        rp_matrix_ptr, rp_matrix_size, 
        input_datapoint_ptr, input_datapoint_size,
        temp_hv_ptr, temp_hv_size
        , temp_row_ptr, temp_row_size,
        output_hm_ptr, output_hv_size, /* Output Buffers: 1*/ 1, 
        output_hm_ptr, output_hv_size, "initial_encoding_wrapper");

    for (int i = 0; i < N_SAMPLES; i++) {
        auto& a = *reinterpret_cast<__hypervector__<N_FEATURES, hvtype>*>(
            reinterpret_cast<hvtype*>(input_datapoint_ptr) + i * N_FEATURES);
        *temp_hv_ptr = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(a, *rp_matrix_ptr);
#if NO_APPROX_HINT == 1
        // Place no_approx before binarize so CollectApproxKnobs can trace:
        // matmul -> store -> load -> no_approx (no intervening stores on temp_hv_ptr).
        __hetero_hdc_no_approx(*temp_hv_ptr);
#endif
#if BINARIZE_ENCODING
        *temp_hv_ptr = __hetero_hdc_sign<D, hvtype>(*temp_hv_ptr);
#endif
        __hetero_hdc_set_matrix_row<N_SAMPLES, D, hvtype>(*output_hm_ptr, *temp_hv_ptr, i);
    }
    __hetero_task_end(task);

    __hetero_section_end(section);
    return;
}

template<int D, int N_FEATURES, int N_SAMPLES>
void encoding_dfg_infer(/* Input Buffers: 2*/
    __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size,
    __hypermatrix__<N_SAMPLES, N_FEATURES, hvtype>* input_datapoint_ptr, size_t input_datapoint_size,
    __hypervector__<D, hvtype>* temp_hv_ptr, size_t temp_hv_size,
    __hypervector__<N_FEATURES, hvtype>* temp_row_ptr, size_t temp_row_size,
    /* Output Buffers: 1*/
    __hypermatrix__<N_SAMPLES, D, hvtype>* output_hm_ptr, size_t output_hv_size) {

    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(/* Input Buffers: 2*/ 5,
        rp_matrix_ptr, rp_matrix_size,
        input_datapoint_ptr, input_datapoint_size,
        temp_hv_ptr, temp_hv_size
        , temp_row_ptr, temp_row_size,
        output_hm_ptr, output_hv_size, /* Output Buffers: 1*/ 1,
        output_hm_ptr, output_hv_size, "initial_encoding_wrapper");

    for (int i = 0; i < N_SAMPLES; i++) {
        auto& a = *reinterpret_cast<__hypervector__<N_FEATURES, hvtype>*>(
            reinterpret_cast<hvtype*>(input_datapoint_ptr) + i * N_FEATURES);
        *temp_hv_ptr = __hetero_hdc_matmul<D, N_FEATURES, hvtype>(a, *rp_matrix_ptr);
#if NO_APPROX_HINT == 1
        __hetero_hdc_no_approx(*temp_hv_ptr);
#endif
#if BINARIZE_ENCODING
        *temp_hv_ptr = __hetero_hdc_sign<D, hvtype>(*temp_hv_ptr);
#endif
        __hetero_hdc_set_matrix_row<N_SAMPLES, D, hvtype>(*output_hm_ptr, *temp_hv_ptr, i);
    }
    __hetero_task_end(task);

    __hetero_section_end(section);
    return;
}

// RP Matrix Node Generation. Performs element wise rotation on ID matrix rows and stores into destination buffer.
template <int D,  int N_FEATURES>
void gen_rp_matrix(/* Input Buffers*/ __hypervector__<D, hvtype>* rp_seed_vector, size_t  rp_seed_vector_size, __hypermatrix__<N_FEATURES, D, hvtype>* shifted_matrix, size_t  shifted_matrix_size, __hypermatrix__<D, N_FEATURES, hvtype>* transposed_matrix, size_t  transposed_matrix_size){

    void* root_section = __hetero_section_begin();

    void* root_task = __hetero_task_begin(/* Num Inputs */ 3, rp_seed_vector,   rp_seed_vector_size, shifted_matrix,   shifted_matrix_size, transposed_matrix,   transposed_matrix_size, /* Num Outputs*/ 1, transposed_matrix,   transposed_matrix_size, "gen_root_task");

    void* wrapper_section = __hetero_section_begin();

    {
    void* gen_shifted_task = __hetero_task_begin(/* Num Inputs */ 2, rp_seed_vector,   rp_seed_vector_size, shifted_matrix,   shifted_matrix_size, /* Num Outputs */ 1, shifted_matrix,   shifted_matrix_size, "gen_shifted_matrix_task");

    //__hetero_hint(DEVICE);


    for (int i = 0; i < N_FEATURES; i++) {
    	__hypervector__<D, hvtype>  row = __hetero_hdc_wrap_shift<D, hvtype>(*rp_seed_vector, i);
        __hetero_hdc_set_matrix_row<N_FEATURES, D, hvtype>(*shifted_matrix, row, i);
    } 
   __hetero_task_end(gen_shifted_task); 
   }

   {
    void* transpose_task = __hetero_task_begin(/* Num Inputs */ 2, shifted_matrix,   shifted_matrix_size, transposed_matrix,   transposed_matrix_size, /* Num Outputs */ 1, transposed_matrix,   transposed_matrix_size, "gen_tranpose_task");
    //__hetero_hint(DEVICE);
    *transposed_matrix = __hetero_hdc_matrix_transpose<N_FEATURES, D, hvtype>(*shifted_matrix, N_FEATURES, D);
    __hetero_task_end(transpose_task); 
   }


   __hetero_section_end(wrapper_section);

   __hetero_task_end(root_task); 

   __hetero_section_end(root_section);
}

// /* Just make guesses based on cossim  */
// template<int D, int K, int N_VEC> // ONLY RUNS ONCE
// void __attribute__ ((noinline)) classification_node_inference( __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, __hypervector__<K, hvtype>* scores_ptr, size_t scores_size, __hypervector__<K, hvtype>* norms_ptr, size_t norms_size, int* label_ptr, size_t label_size ) {   
//     // Read classes hvs from host.

//      void* section = __hetero_section_begin();

//     void* task1 = __hetero_task_begin( /* Input Buffers: */ 4, encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size, norms_ptr, norms_size,  /* Output Buffers: */ 1, scores_ptr, scores_size, "inference_calculate_score_task" );

//     __hetero_hint(DEVICE);
    
//     // Class HVs are created via 'clustering' on +1, -1 encoded hypervectors. (loop 269).
//     __hypervector__<D, hvtype> encoded_hv = *encoded_hv_ptr;
//     __hypermatrix__<K, D, hvtype> classes = *classes_ptr;

//     __hypervector__<K, hvtype> scores = *scores_ptr; // Precision of these scores might need to be increased.

//     #if HAMMING_DIST == 1
//     __hypervector__<K, hvtype> score =  __hetero_hdc_hamming_distance<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
//     *scores_ptr = score;
//     #else
//     // *scores_ptr = __hetero_hdc_matmul<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr); 
//     // *scores_ptr = __hetero_hdc_div<K, hvtype>(*scores_ptr, *norms_ptr);
//     // *scores_ptr = __hetero_hdc_absolute_value<K, hvtype>(*scores_ptr);
//     *scores_ptr  =  __hetero_hdc_cossim<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
//     #endif

//     __hetero_task_end(task1);

//     void* task2 = __hetero_task_begin( /* Input Buffers: 1*/ 3, scores_ptr, scores_size, label_ptr, label_size, encoded_hv_ptr, encoded_hv_size, /* Output Buffers: 1*/ 1,  label_ptr, label_size, "inference_find_max_task" );  

//     __hetero_hint(DEVICE);

//     {
//     __hypervector__<K, hvtype> scores = *scores_ptr;
//     int max_idx = 0;
    

//     #if HAMMING_DIST == 1
//     // For hamming distance take the smallest value
//     max_idx = __hetero_hdc_arg_min<K, hvtype>(*scores_ptr);

// #else

//     max_idx = __hetero_hdc_arg_max<K, hvtype>(*scores_ptr);

// #endif
    
//     // Set the label to our guess 
//     *label_ptr = max_idx; 
//     }
// #ifndef NODFG
//     __hetero_task_end(task2);

//     __hetero_section_end(section);
// #endif
//     return;
// }


// Dimensionality, Clusters, data point vectors, features per.
// DO NOT LAZY ENCODE, 
template <int D, int K, int N_VEC, int N_FEATURES>
void training_root_node( /* Input buffers: 2*/  
    __hypermatrix__<N_VEC, D, hvtype>* encoded_hvs_ptr, size_t encoded_hvs_size,
    int* labels, size_t labels_size,
    /* Parameters: 0 */ 
    /* Local Vars: 4 */ 
    __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, 
    __hypervector__<K, hvtype>* scores_ptr, size_t scores_size, 
    __hypervector__<K, hvtype>* norms_ptr, size_t norms_size, 
    __hypervector__<D, hvtype>* update_hv_ptr, size_t update_hv_size,  
    // Classes
     /* Output Buffers: 1 (Classes)*/ 
     __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size

    ){
    void* root_section = __hetero_section_begin();

    void* root_task = __hetero_task_begin( /* Input Buffers: 3 */ 
        7,  
        encoded_hvs_ptr, encoded_hvs_size,
        classes_ptr, classes_size, 
        labels, labels_size,
        encoded_hv_ptr, encoded_hv_size,  
        scores_ptr, scores_size, 
        norms_ptr, norms_size, 
        update_hv_ptr, update_hv_size, 
        
        /* Output Buffers: 1 */ 1,  
        classes_ptr, classes_size, "root_task"   );
    
    for (int i = 0; i < N_VEC; i++) {
        *encoded_hv_ptr = *reinterpret_cast<__hypervector__<D, hvtype>*>(
            reinterpret_cast<hvtype*>(encoded_hvs_ptr) + i * D);

        // #if HAMMING_DIST == 1
        // *scores_ptr = __hetero_hdc_hamming_distance<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
        // #else
        // // *scores_ptr = __hetero_hdc_matmul<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr); 
        // // *scores_ptr = __hetero_hdc_div<K, hvtype>(*scores_ptr, *norms_ptr);
        // // *scores_ptr = __hetero_hdc_absolute_value<K, hvtype>(*scores_ptr);
        // *scores_ptr  =  __hetero_hdc_cossim<K, D, hvtype>(*encoded_hv_ptr, *classes_ptr);
        // //*scores_ptr = score;
        // #endif

        // auto row = __hetero_hdc_get_matrix_row<N_VEC, D, hvtype>(*encoded_hv_ptr, N_VEC, D, i);
        #if HAMMING_DIST != 1
        float max_score = -9999999.0;
        int max_index = -1;
        for (int i = 0; i < K; i++) {
            auto& class_hv = *reinterpret_cast<__hypervector__<D, hvtype>*>(
                reinterpret_cast<hvtype*>(classes_ptr) + i * D);
            float score = __hetero_hdc_cossim<D, hvtype>(*encoded_hv_ptr, class_hv);
            if (score > max_score) {
                max_score = score;
                max_index = i;
            }
        }
        #else
        int max_index = 0;
        int min_score = 9999999;
        for (int i = 0; i < K; i++) {
            auto& class_hv = *reinterpret_cast<__hypervector__<D, hvtype>*>(
                reinterpret_cast<hvtype*>(classes_ptr) + i * D);
            int score = __hetero_hdc_hamming_distance<D, hvtype>(*encoded_hv_ptr, class_hv);
            if (score < min_score) {
                min_score = score;
                max_index = i;
            }
        }
        #endif
        

        // #if HAMMING_DIST == 1
        //     // For hamming distance take the smallest value
        //     max_idx = __hetero_hdc_arg_min<K, hvtype>(*scores_ptr);
        //     // printf("argmax: %d\n", *argmax);
        // #else
        //     max_idx = __hetero_hdc_arg_max<K, hvtype>(*scores_ptr);
        // #endif

        int label = labels[i]; // Labels are 0-indexed in the dataset files
        // Update the correct and mispredicted class
        if (label != max_index) { // Incorrect prediction

            auto& label_row = *reinterpret_cast<__hypervector__<D, hvtype>*>(
                reinterpret_cast<hvtype*>(classes_ptr) + label * D);
            *update_hv_ptr = __hetero_hdc_sum<D, hvtype>(label_row, *encoded_hv_ptr); // May need an instrinsic for this.
            __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, label); // How do we normalize?

            auto& max_idx_row = *reinterpret_cast<__hypervector__<D, hvtype>*>(
                reinterpret_cast<hvtype*>(classes_ptr) + max_index * D);
            *update_hv_ptr = __hetero_hdc_sub<D, hvtype>(max_idx_row, *encoded_hv_ptr); // May need an instrinsic for this.
            __hetero_hdc_set_matrix_row<K, D, hvtype>(*classes_ptr, *update_hv_ptr, max_index); // How do we normalize?

        }
    }
    __hetero_task_end(root_task);

    __hetero_section_end(root_section);
    return;
}

/* Just make guesses based on cossim  */
template<int D, int K, int N_VEC>
void inference_node(
    __hypermatrix__<N_VEC, D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, 
    __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, 
    __hypervector__<K, hvtype>* scores_ptr, size_t scores_size, 
    __hypervector__<K, hvtype>* norms_ptr, size_t norms_size, 
    int* label_ptr, size_t label_size ) {   
    // Read classes hvs from host.

     void* section = __hetero_section_begin();

    void* task1 = __hetero_task_begin( /* Input Buffers: */ 5, encoded_hv_ptr, encoded_hv_size, 
        classes_ptr, classes_size, 
        scores_ptr, scores_size, 
        norms_ptr, norms_size, 
        label_ptr, label_size,  
        /* Output Buffers: */ 1, 
        label_ptr, label_size, "inference_task" );
    __hetero_hint(DEVICE);
    
    for (int i = 0; i < N_VEC; i++) {
        auto& row = *reinterpret_cast<__hypervector__<D, hvtype>*>(
            reinterpret_cast<hvtype*>(encoded_hv_ptr) + i * D);
        #if HAMMING_DIST != 1
        float max_score = -9999999.0;
        int max_index = -1;
        for (int i = 0; i < K; i++) {
            auto& class_hv = *reinterpret_cast<__hypervector__<D, hvtype>*>(
                reinterpret_cast<hvtype*>(classes_ptr) + i * D);
            float score = __hetero_hdc_cossim<D, hvtype>(row, class_hv);
            if (score > max_score) {
                max_score = score;
                max_index = i;
            }
        }
        #else
        int max_index = 0;
        int min_score = 9999999;
        for (int i = 0; i < K; i++) {
            auto& class_hv = *reinterpret_cast<__hypervector__<D, hvtype>*>(
                reinterpret_cast<hvtype*>(classes_ptr) + i * D);
            int score = __hetero_hdc_hamming_distance<D, hvtype>(row, class_hv);
            if (score < min_score) {
                min_score = score;
                max_index = i;
            }
        }
        #endif
        label_ptr[i] = max_index;
        // #if HAMMING_DIST == 1
        // *scores_ptr=  __hetero_hdc_hamming_distance<K, D, hvtype>(row, *classes_ptr);
        // #else
        // *scores_ptr = __hetero_hdc_matmul<K, D, hvtype>(row, *classes_ptr); 
        // // *scores_ptr = __hetero_hdc_div<K, hvtype>(*scores_ptr, *norms_ptr);
        // // *scores_ptr = __hetero_hdc_absolute_value<K, hvtype>(*scores_ptr);
        // // *scores_ptr  =  __hetero_hdc_cossim<K, D, hvtype>(row, *classes_ptr);


        // #endif

        // int max_idx = 0;
    
        // #if HAMMING_DIST == 1
        // // For hamming distance take the smallest value
        // max_idx = __hetero_hdc_arg_min<K, hvtype>(*scores_ptr);
        // #else
        // max_idx = __hetero_hdc_arg_max<K, hvtype>(*scores_ptr);
        // #endif
        
        // // Set the label to our guess 
        // printf("max_idx: %d\n", max_idx);
        // label_ptr[i] = max_idx; 
    }
    __hetero_task_end(task1);

    __hetero_section_end(section);
    return;
}


// // Dimensionality, Clusters, data point vectors, features per.
// template <int D, int K, int N_VEC, int N_FEATURES>
// void encoding_and_inference_node( /* Input buffers: 3*/ 
//     __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size, 
//     __hypervector__<N_FEATURES, hvtype>* datapoint_vec_ptr, size_t datapoint_vec_size, 
//     __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, 
//     int* label_ptr, size_t label_size, 
//     /* Local Vars: 2*/ __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, 
//     __hypervector__<K, hvtype>* scores_ptr, size_t scores_size, 
//     __hypervector__<K, hvtype>* norms_ptr, size_t norms_size /* Output Buffers: 1*/ ){


//     void* inference_task = __hetero_task_begin( /* Input Buffers: 5 */  5,  encoded_hv_ptr, encoded_hv_size,  classes_ptr, classes_size,  label_ptr, label_size, scores_ptr, scores_size, norms_ptr, norms_size, /* Output Buffers: 1 */ 1, label_ptr, label_size, "inference_task"   );

//     classification_node_inference<D, K, N_VEC>(encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size, scores_ptr, scores_size,norms_ptr, norms_size , label_ptr, label_size); 

//     __hetero_task_end(inference_task);

//     __hetero_section_end(root_section);
//     return;
// }

// // Dimensionality, Clusters, data point vectors, features per.
// template <int D, int K, int N_VEC, int N_FEATURES>
// void inference_root_node( 
//     /* Input buffers: 3*/  
//     __hypermatrix__<D, N_FEATURES, hvtype>* rp_matrix_ptr, size_t rp_matrix_size,
//     __hypervector__<N_FEATURES, hvtype>* datapoint_vec_ptr, size_t datapoint_vec_size, 
//     __hypermatrix__<K, D, hvtype>* classes_ptr, size_t classes_size, 
//     /* Output Buffers: 1*/ int* label_ptr, size_t label_size, 
//     /* Local Vars: 2*/ __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, 
//     __hypervector__<K, hvtype>* scores_ptr, size_t scores_size, 
//     __hypervector__<K, hvtype>* norms_ptr, size_t norms_size){

//     void* root_section = __hetero_section_begin();

//     void* inference_task = __hetero_task_begin( /* Input Buffers: 3 */ 7,  rp_matrix_ptr, rp_matrix_size,  datapoint_vec_ptr, datapoint_vec_size,  encoded_hv_ptr, encoded_hv_size, classes_ptr, classes_size,  label_ptr, label_size, scores_ptr, scores_size, norms_ptr, norms_size, /* Output Buffers: 1 */ 1,encoded_hv_ptr, encoded_hv_size, "inference_task" );

//     encoding_and_inference_node<D, K, N_VEC, N_FEATURES>(
//         rp_matrix_ptr, rp_matrix_size, 
//         datapoint_vec_ptr, datapoint_vec_size, 
//         classes_ptr, classes_size, 
//         label_ptr, label_size, 
//         encoded_hv_ptr, encoded_hv_size, 
//         scores_ptr, scores_size, 
//         norms_ptr, norms_size);


//     __hetero_task_end(inference_task);

//     __hetero_section_end(root_section);
// }


template<int NC, int NS , int D>
void InitialUpdateDFG(__hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, __hypermatrix__<NC ,D, hvtype>* class_hm_ptr, size_t class_hm_size, int label, int idx) {
    
#ifndef NODFG
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
            4, encoded_hv_ptr, encoded_hv_size, class_hm_ptr, class_hm_size, label, idx, 
            1, class_hm_ptr, class_hm_size,
            "InitNode"
            );
#endif

		auto& class_hv = *reinterpret_cast<__hypervector__<D, hvtype>*>(
            reinterpret_cast<hvtype*>(class_hm_ptr) + label * D);
		//auto encoded_hv = __hetero_hdc_get_matrix_row<NS, D, hvtype>(*encoded_hv_ptr, NS, D, idx);
		auto sum_hv = __hetero_hdc_sum<D, hvtype>(class_hv, *encoded_hv_ptr); 
		__hetero_hdc_set_matrix_row<NC, D, hvtype>(*class_hm_ptr, sum_hv, label); 


#ifndef NODFG
    __hetero_task_end(task); 

    __hetero_section_end(section);
#endif
    return;
}


// template<int N_CLASS, int Dhv> 
// void L2NormDFG(__hypervector__<N_CLASS, hvtype> *norms_buffer, size_t norms_size,  __hypermatrix__<N_CLASS, Dhv, hvtype> *classes, size_t classes_size) {

    
//     void* section = __hetero_section_begin();

//     void* task = __hetero_task_begin(
//             2, norms_buffer, norms_size, classes, classes_size,
//             1, norms_buffer, norms_size,
//             "norm_node"
//             );
    
// #if HAMMING_DIST != 1
// 	*norms_buffer = __hetero_hdc_l2norm<N_CLASS, Dhv, hvtype>(*classes);
// #endif

//     __hetero_task_end(task); 
//     __hetero_section_end(section);
//     return;

// }
