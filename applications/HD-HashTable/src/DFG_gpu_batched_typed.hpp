#include "host_typed.h"
#include <assert.h>

template<typename T>
static void shuffle_typed(T *arr, int n) {
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
        shuffle_typed((hvtype*)A_encoding_ptr, D);
    __hetero_task_end(task_A);

    void* task_C = __hetero_task_begin(
         1, C_encoding_ptr, encoded_scheme_size_c,
        1, C_encoding_ptr, encoded_scheme_size_c,
        "generate_encoding_c");
        for (int i = 0; i < D; i++)
            ((hvtype*)C_encoding_ptr)[i] = (i < D/2) ? (hvtype)1 : (hvtype)-1;
        shuffle_typed((hvtype*)C_encoding_ptr, D);
    __hetero_task_end(task_C);

    void* task_G = __hetero_task_begin(
         1, G_encoding_ptr, encoded_scheme_size_g,
        1, G_encoding_ptr, encoded_scheme_size_g,
        "generate_encoding_g");
        for (int i = 0; i < D; i++)
            ((hvtype*)G_encoding_ptr)[i] = (i < D/2) ? (hvtype)1 : (hvtype)-1;
        shuffle_typed((hvtype*)G_encoding_ptr, D);
    __hetero_task_end(task_G);

    void* task_T = __hetero_task_begin(
         1, T_encoding_ptr, encoded_scheme_size_a,
        1, T_encoding_ptr, encoded_scheme_size_a,
        "generate_encoding_t");
        for (int i = 0; i < D; i++)
            ((hvtype*)T_encoding_ptr)[i] = (i < D/2) ? (hvtype)1 : (hvtype)-1;
        shuffle_typed((hvtype*)T_encoding_ptr, D);
    __hetero_task_end(task_T);

    __hetero_section_end(section);
}

// encode_all_node: encode N kmers in parallel, one GPU thread per kmer.
template<int N, int K_LEN, int D>
void encode_all_node(
    char* kmers_flat,      size_t kmers_flat_size,
    hvtype* A_enc,         size_t a_size,
    hvtype* C_enc,         size_t c_size,
    hvtype* G_enc,         size_t g_size,
    hvtype* T_enc,         size_t t_size,
    hvtype* encoded_out,   size_t encoded_out_size
) {
    void* section = __hetero_section_begin();

    for (int i = 0; i < N; i++) {
        __hetero_parallel_loop(1,
            6,
            kmers_flat,  kmers_flat_size,
            A_enc,       a_size,
            C_enc,       c_size,
            G_enc,       g_size,
            T_enc,       t_size,
            encoded_out, encoded_out_size,
            1,
            encoded_out, encoded_out_size
        );
        __hetero_hint(DEVICE);

        char*   kmer = &kmers_flat[i * K_LEN];
        hvtype* out  = &encoded_out[i * D];

        for (int d = 0; d < D; d++) out[d] = (hvtype)1;

        for (int pos = 0; pos < K_LEN; pos += KMER_STRIDE) {
            char token = kmer[pos];
            hvtype* src;
            switch (token) {
                case 'A': src = A_enc; break;
                case 'C': src = C_enc; break;
                case 'G': src = G_enc; break;
                case 'T': src = T_enc; break;
                default:  continue;
            }
            int shift = pos % D;
            for (int d = 0; d < D; d++) {
                int idx = d - shift;
                if (idx < 0) idx += D;
                out[d] *= src[idx];
            }
        }
    }

    __hetero_section_end(section);
}

// query_all_node: for each of N_Q test encoded HVs, compute dot product against
// all N_HVS_Q hash-table buckets and record the maximum similarity.
template<int N_Q, int N_HVS_Q, int D>
void query_all_node(
    hvtype* encoded_test,  size_t encoded_test_size,
    hvtype* hash_table,    size_t hash_table_size,
    hvtype* max_sims,      size_t max_sims_size
) {
    void* section = __hetero_section_begin();

    for (int i = 0; i < N_Q; i++) {
        __hetero_parallel_loop(1,
            3,
            encoded_test, encoded_test_size,
            hash_table,   hash_table_size,
            max_sims,     max_sims_size,
            1,
            max_sims, max_sims_size
        );
        __hetero_hint(DEVICE);

        hvtype* enc     = &encoded_test[i * D];
        hvtype  max_sim = (hvtype)(-2000000000);

        int d_end = (int)(D * QUERY_END);
        if (d_end < 1)  d_end = 1;
        if (d_end > D)  d_end = D;

        for (int b = 0; b < N_HVS_Q; b++) {
            hvtype*  bucket = &hash_table[b * D];
            hvtype   dot    = 0;
            for (int d = 0; d < d_end; d += STRIDE)
                dot += enc[d] * bucket[d];
            if (dot > max_sim) max_sim = dot;
        }
        max_sims[i] = max_sim;
    }

    __hetero_section_end(section);
}
