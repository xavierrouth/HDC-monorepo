
#include "host.h"
#include <assert.h>
#undef D
#undef K

template<typename T>
void shuffle(T *arr, int n) {
    
    for (int i = n - 1; i > 0; i--) {
        // Generate random index from 0 to i (inclusive)
        int j = rand() % (i + 1);
        
        // Swap arr[i] and arr[j]
        T temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

template<int D>
void sum_in_place_task(
        __hypervector__<D, hvtype>* hv1_ptr, size_t hv1_size,
        __hypervector__<D, hvtype>* output_ptr, size_t output_size
        ){

    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
         2, hv1_ptr, hv1_size,
         output_ptr, output_size,
        /* Output Buffers: 1*/ 1, output_ptr, output_size,
        "sum_task_inner"
    );

    *output_ptr = __hetero_hdc_sum<D, hvtype>(*hv1_ptr, *output_ptr);

    __hetero_task_end(task); 

    __hetero_section_end(section);

}
template<int D>
void generate_encoding_scheme(
        __hypervector__<D, hvtype>* A_encoding_ptr, size_t encoded_scheme_size_a,
        __hypervector__<D, hvtype>* C_encoding_ptr, size_t encoded_scheme_size_c,
        __hypervector__<D, hvtype>* G_encoding_ptr, size_t encoded_scheme_size_g,
        __hypervector__<D, hvtype>* T_encoding_ptr, size_t encoded_scheme_size_t
        ){
    // Encoding scheme: each encoding will have half its values be -1 and the other half be 1


    void* section = __hetero_section_begin();

    // One task per encoding
    void* task_A = __hetero_task_begin(
         1, A_encoding_ptr, encoded_scheme_size_a,
        /* Output Buffers: 1*/ 1, A_encoding_ptr, encoded_scheme_size_a,
        "generate_encoding_a");
    
        for (int i = 0; i < D; i++){
            if(i < D/2)
                ((hvtype*)A_encoding_ptr)[i] = (hvtype)1;
            else
                ((hvtype*)A_encoding_ptr)[i] = (hvtype)-1;
        }
        // Shuffle the encoding
        shuffle((hvtype*)A_encoding_ptr, D);
    __hetero_task_end(task_A);

    void* task_C = __hetero_task_begin(
         1, C_encoding_ptr, encoded_scheme_size_c,
        /* Output Buffers: 1*/ 1, C_encoding_ptr, encoded_scheme_size_c,
        "generate_encoding_c");
    
        for (int i = 0; i < D; i++){
            if(i < D/2)
                ((hvtype*)C_encoding_ptr)[i] = (hvtype)1;
            else
                ((hvtype*)C_encoding_ptr)[i] = (hvtype)-1;
        }
        // Shuffle the encoding
        shuffle((hvtype*)C_encoding_ptr, D);
    __hetero_task_end(task_C);

    void* task_G = __hetero_task_begin(
         1, G_encoding_ptr, encoded_scheme_size_g,
        /* Output Buffers: 1*/ 1, G_encoding_ptr, encoded_scheme_size_g,
        "generate_encoding_g");

        for (int i = 0; i < D; i++){
            if(i < D/2)
                ((hvtype*)G_encoding_ptr)[i] = (hvtype)1;
            else
                ((hvtype*)G_encoding_ptr)[i] = (hvtype)-1;
        }
        // Shuffle the encoding
        shuffle((hvtype*)G_encoding_ptr, D);
    __hetero_task_end(task_G);

    void* task_T = __hetero_task_begin(
         1, T_encoding_ptr, encoded_scheme_size_a,
        /* Output Buffers: 1*/ 1, T_encoding_ptr, encoded_scheme_size_a,
        "generate_encoding_t");
        for (int i = 0; i < D; i++){
            if(i < D/2)
                ((hvtype*)T_encoding_ptr)[i] = (hvtype)1;
            else
                ((hvtype*)T_encoding_ptr)[i] = (hvtype)-1;
        }
        // Shuffle the encoding
        shuffle((hvtype*)T_encoding_ptr, D);
    __hetero_task_end(task_T);

    __hetero_section_end(section);
    return;
}

template<int D>
void encode_kmer_partial(
    char inputr_kmer_ptr, size_t input_kmer_size, 
    __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_size, 
    __hypervector__<D, hvtype>* base_ptr, size_t base_size,
    __hypervector__<D, hvtype>* encoding_ptr, size_t encoded
)
{
    // To be implemented if needed

}

template<int D>
void roll_into(hvtype* __restrict__ src, int shift_amount, hvtype* __restrict__ dst) {
    // np.roll
    shift_amount = ((shift_amount % D) + D) % D;
    for (int i = 0; i < D; i++) {
        dst[(i + shift_amount) % D] = src[i];
    }
}
template<int K, int D, int misc>
void encode_kmer(
        char* __restrict__ input_kmer_ptr, size_t input_kmer_size, 
        hvtype* __restrict__ encoded_hv_ptr, size_t encoded_size, 
        hvtype* __restrict__ base_ptr, size_t base_size,
        hvtype* __restrict__ A_encoding_ptr, size_t encoded_scheme_size_a,
        hvtype* __restrict__ C_encoding_ptr, size_t encoded_scheme_size_c,
        hvtype* __restrict__ G_encoding_ptr, size_t encoded_scheme_size_g,
        hvtype* __restrict__ T_encoding_ptr, size_t encoded_scheme_size_t
        ){

    void* section = __hetero_section_begin();
    
    void* task = __hetero_task_begin(
         6 + 1, 
         A_encoding_ptr, encoded_scheme_size_a, 
         C_encoding_ptr, encoded_scheme_size_c, 
         G_encoding_ptr, encoded_scheme_size_g, 
         T_encoding_ptr, encoded_scheme_size_t, 
         input_kmer_ptr, input_kmer_size,
         base_ptr, base_size,
         // output
         encoded_hv_ptr, encoded_size,
        /* Output Buffers: 1*/ 1, encoded_hv_ptr, encoded_size,
        "encode_kmer_inner_task"
    );
    __hetero_hint(DEVICE);
    // Set enc_hv_ptr to all 1s
    for (int i = 0; i < D; i++){
        *((hvtype*)encoded_hv_ptr + i) = (hvtype)1;
    }
    for (int i = 0; i < K; i += KMER_STRIDE) {
        char token = input_kmer_ptr[i];

        switch(token){
            case 'A':
                roll_into<D>((hvtype*)A_encoding_ptr, i, (hvtype*)base_ptr);
                for (int i = 0; i < D; i++){
                    // std::cout << "temp["<< i <<"]:\t" << ((hvtype*)&temp)[i] << "\n";
                    *((hvtype*)encoded_hv_ptr + i) *= *((hvtype*)base_ptr + i);
                }   
                break;
            case 'C':
                roll_into<D>((hvtype*)C_encoding_ptr, i, (hvtype*)base_ptr);
                for (int i = 0; i < D; i++){
                    // std::cout << "temp["<< i <<"]:\t" << ((hvtype*)&temp)[i] << "\n";
                    *((hvtype*)encoded_hv_ptr + i) *= *((hvtype*)base_ptr + i);
                }
                break;
            case 'G':
                roll_into<D>((hvtype*)G_encoding_ptr, i, (hvtype*)base_ptr);
                for (int i = 0; i < D; i++){
                    // std::cout << "temp["<< i <<"]:\t" << ((hvtype*)&temp)[i] << "\n";
                    *((hvtype*)encoded_hv_ptr + i) *= *((hvtype*)base_ptr + i);
                }   
                break;
            case 'T':
                roll_into<D>((hvtype*)T_encoding_ptr, i, (hvtype*)base_ptr);
                for (int i = 0; i < D; i++){
                    // std::cout << "temp["<< i <<"]:\t" << ((hvtype*)&temp)[i] << "\n";
                    *((hvtype*)encoded_hv_ptr + i) *= *((hvtype*)base_ptr + i);
                }   
                break;
            default:
                // assert(false && "Invalid character in kmer");
                break;
        };
        
        
        
    }

    // Reset 

    __hetero_task_end(task);

    __hetero_section_end(section);
}





template<int K, int D, int N>
void produce_dot_prod(
        __hypermatrix__<N,D, hvtype>* hash_table, size_t hash_table_size,
        __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_size,
        __hypervector__<N, hvtype>* output_ptr, size_t output_size,
        int* argmax, size_t argmax_size
        ){

    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(
         3, hash_table, hash_table_size, encoded_hv_ptr, encoded_size, output_ptr, output_size,
        /* Output Buffers: 1*/ 1, output_ptr, output_size,
        "produce_dot_prod_inner_task"
    );


    *output_ptr = __hetero_hdc_matmul<N, D, hvtype>(*encoded_hv_ptr, *hash_table); 

    __hetero_task_end(task); 

    void* task2 = __hetero_task_begin(
         2, output_ptr, output_size, argmax, argmax_size,
         2, output_ptr, output_size, argmax, argmax_size,
        "produce_dot_prod_argmax_task"
    );


    *argmax = __hetero_hdc_arg_max<N, hvtype>(*output_ptr);


    __hetero_task_end(task2); 
    __hetero_section_end(section);

}




template<int K, int D, int N>
bool __attribute__ ((noinline))  query(int* kmer, size_t kmer_size,
        __hypermatrix__<4,D, hvtype>* encoding_scheme_ptr, size_t encoded_scheme_size,
        __hypermatrix__<N,D, hvtype>* hash_table, size_t hash_table_size,
        __hypervector__<D, hvtype>* base_ptr, size_t base_size, 
        __hypervector__<D, hvtype>* shifted_hv_ptr, size_t shifted_size,
        __hypervector__<D, hvtype>* encoded_hv_handle, size_t encoded_size ,
        __hypervector__<N, hvtype>* output_handle, size_t output_size 
        ){

    encode_kmer_wrapper<K,D>(kmer, kmer_size,encoded_hv_handle, encoded_size, base_ptr, base_size, shifted_hv_ptr, shifted_size, encoding_scheme_ptr, encoded_scheme_size);

    //return false;
    int arg_max[1];
    size_t arg_max_size = sizeof(int);

        void* ProduceDAG = __hetero_launch(
        (void*) produce_dot_prod<K,D, N>, 4, 
        hash_table, hash_table_size,
        encoded_hv_handle, encoded_size,
        output_handle, output_size,
        arg_max, arg_max_size, 
        2, 
        output_handle, output_size,
        arg_max, arg_max_size);

        __hetero_wait(ProduceDAG);
    //std::cout <<"arg max: "<< arg_max <<"\n";

    hvtype* output_base = (hvtype*) output_handle;
    return output_base[arg_max[0]] > 0.5 * D;//output_base[arg_max] > (0.8 * D);

}
