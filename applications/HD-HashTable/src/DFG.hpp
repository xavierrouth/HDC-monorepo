
#include "host.h"
#include <assert.h>
#undef D
#undef K

template<typename T>
void shuffle(T *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        T temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

template<int D>
void generate_encoding_scheme(
        __hypervector__<D, hvtype>* A_encoding_ptr, size_t encoded_scheme_size_a,
        __hypervector__<D, hvtype>* C_encoding_ptr, size_t encoded_scheme_size_c,
        __hypervector__<D, hvtype>* G_encoding_ptr, size_t encoded_scheme_size_g,
        __hypervector__<D, hvtype>* T_encoding_ptr, size_t encoded_scheme_size_t
        ){
    void* section = __hetero_section_begin();

    void* task_A = __hetero_task_begin(
         1, A_encoding_ptr, encoded_scheme_size_a,
        1, A_encoding_ptr, encoded_scheme_size_a,
        "generate_encoding_a");
        for (int i = 0; i < D; i++)
            ((hvtype*)A_encoding_ptr)[i] = (i < D/2) ? (hvtype)1 : (hvtype)-1;
        shuffle((hvtype*)A_encoding_ptr, D);
    __hetero_task_end(task_A);

    void* task_C = __hetero_task_begin(
         1, C_encoding_ptr, encoded_scheme_size_c,
        1, C_encoding_ptr, encoded_scheme_size_c,
        "generate_encoding_c");
        for (int i = 0; i < D; i++)
            ((hvtype*)C_encoding_ptr)[i] = (i < D/2) ? (hvtype)1 : (hvtype)-1;
        shuffle((hvtype*)C_encoding_ptr, D);
    __hetero_task_end(task_C);

    void* task_G = __hetero_task_begin(
         1, G_encoding_ptr, encoded_scheme_size_g,
        1, G_encoding_ptr, encoded_scheme_size_g,
        "generate_encoding_g");
        for (int i = 0; i < D; i++)
            ((hvtype*)G_encoding_ptr)[i] = (i < D/2) ? (hvtype)1 : (hvtype)-1;
        shuffle((hvtype*)G_encoding_ptr, D);
    __hetero_task_end(task_G);

    void* task_T = __hetero_task_begin(
         1, T_encoding_ptr, encoded_scheme_size_a,
        1, T_encoding_ptr, encoded_scheme_size_a,
        "generate_encoding_t");
        for (int i = 0; i < D; i++)
            ((hvtype*)T_encoding_ptr)[i] = (i < D/2) ? (hvtype)1 : (hvtype)-1;
        shuffle((hvtype*)T_encoding_ptr, D);
    __hetero_task_end(task_T);

    __hetero_section_end(section);
}

template<int D>
void roll_into(hvtype* __restrict__ src, int shift_amount, hvtype* __restrict__ dst) {
    shift_amount = ((shift_amount % D) + D) % D;
    for (int i = 0; i < D; i++)
        dst[(i + shift_amount) % D] = src[i];
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
         encoded_hv_ptr, encoded_size,
        1, encoded_hv_ptr, encoded_size,
        "encode_kmer_inner_task"
    );
    __hetero_hint(DEVICE);
    for (int i = 0; i < D; i++)
        *((hvtype*)encoded_hv_ptr + i) = (hvtype)1;

    for (int i = 0; i < K; i += KMER_STRIDE) {
        char token = input_kmer_ptr[i];
        switch(token){
            case 'A':
                roll_into<D>((hvtype*)A_encoding_ptr, i, (hvtype*)base_ptr);
                for (int j = 0; j < D; j++)
                    *((hvtype*)encoded_hv_ptr + j) *= *((hvtype*)base_ptr + j);
                break;
            case 'C':
                roll_into<D>((hvtype*)C_encoding_ptr, i, (hvtype*)base_ptr);
                for (int j = 0; j < D; j++)
                    *((hvtype*)encoded_hv_ptr + j) *= *((hvtype*)base_ptr + j);
                break;
            case 'G':
                roll_into<D>((hvtype*)G_encoding_ptr, i, (hvtype*)base_ptr);
                for (int j = 0; j < D; j++)
                    *((hvtype*)encoded_hv_ptr + j) *= *((hvtype*)base_ptr + j);
                break;
            case 'T':
                roll_into<D>((hvtype*)T_encoding_ptr, i, (hvtype*)base_ptr);
                for (int j = 0; j < D; j++)
                    *((hvtype*)encoded_hv_ptr + j) *= *((hvtype*)base_ptr + j);
                break;
            default:
                break;
        }
    }

    __hetero_task_end(task);
    __hetero_section_end(section);
}

// Add encoded_hv into bucket bucket_idx of hash_table matrix
template<int N, int D>
void add_to_bucket_task(
        __hypermatrix__<N, D, hvtype>* hash_table_ptr, size_t hash_table_size,
        hvtype* encoded_hv_ptr, size_t encoded_size,
        int* bucket_idx_ptr, size_t bucket_idx_size
        ){
    void* section = __hetero_section_begin();
    void* task = __hetero_task_begin(
         3, hash_table_ptr, hash_table_size,
            encoded_hv_ptr, encoded_size,
            bucket_idx_ptr, bucket_idx_size,
        1, hash_table_ptr, hash_table_size,
        "add_to_bucket_inner"
    );
    {
        int bucket = *bucket_idx_ptr;
        hvtype* table_flat = reinterpret_cast<hvtype*>(hash_table_ptr);
        for (int i = 0; i < D; i++)
            table_flat[bucket * D + i] += encoded_hv_ptr[i];
    }
    __hetero_task_end(task);
    __hetero_section_end(section);
}

// Compute dot products of encoded_hv against all N rows of hash_table
template<int N, int D>
void query_hash_task(
        __hypermatrix__<N, D, hvtype>* hash_table_ptr, size_t hash_table_size,
        __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_size,
        __hypervector__<N, hvtype>* sims_ptr, size_t sims_size
        ){
    void* section = __hetero_section_begin();
    void* task = __hetero_task_begin(
         2, hash_table_ptr, hash_table_size,
            encoded_hv_ptr, encoded_size,
        1, sims_ptr, sims_size,
        "query_hash_inner"
    );
    *sims_ptr = __hetero_hdc_matmul<N, D, hvtype>(*encoded_hv_ptr, *hash_table_ptr);
    __hetero_task_end(task);
    __hetero_section_end(section);
}
