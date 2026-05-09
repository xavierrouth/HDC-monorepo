
#include "DFG.hpp"   // for generate_encoding_scheme; also does #undef D, #undef K

// encode_all_node: encode N kmers in parallel, one GPU thread per kmer.
// Uses inline roll (no temp buffer) to avoid per-thread scratch conflicts.
// encoded_out[i*D .. i*D+D-1] = product over kmer positions of rolled encoding HV.
template<int N, int K_LEN, int D>
void encode_all_node(
    char* kmers_flat,      size_t kmers_flat_size,    // N * K_LEN chars
    hvtype* A_enc,         size_t a_size,              // D floats
    hvtype* C_enc,         size_t c_size,
    hvtype* G_enc,         size_t g_size,
    hvtype* T_enc,         size_t t_size,
    hvtype* encoded_out,   size_t encoded_out_size     // N * D floats (output)
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
            encoded_out, encoded_out_size,  // listed as input so HPVM registers it
            1,
            encoded_out, encoded_out_size
        );
        __hetero_hint(DEVICE);

        char*   kmer = &kmers_flat[i * K_LEN];
        hvtype* out  = &encoded_out[i * D];

        // Initialize to all-ones
        for (int d = 0; d < D; d++) out[d] = (hvtype)1;

        // Accumulate binding product: out *= roll(enc_ACGT, pos)
        // Inline roll: rolled[d] = src[(d - shift + D) % D]
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
            int shift = pos % D;   // pos < K_LEN=500 < D for all typical configs
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
// max_sims[i] = max over b of dot(encoded_test[i], hash_table[b])
// max_sims must also be passed as input to ensure HPVM registers the buffer.
template<int N_Q, int N_HVS_Q, int D>
void query_all_node(
    hvtype* encoded_test,  size_t encoded_test_size,   // N_Q * D floats
    hvtype* hash_table,    size_t hash_table_size,     // N_HVS_Q * D floats
    hvtype* max_sims,      size_t max_sims_size        // N_Q floats (input+output)
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
        hvtype  max_sim = -1e30f;

        // Reduction perforation: only compute first QUERY_END fraction of D,
        // stepping by STRIDE (matches the redperf knobs from autotune.py).
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
