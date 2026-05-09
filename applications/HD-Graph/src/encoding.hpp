#include <vector>
#include <random>

/* Generates a F x D hypermatrix representing the id hypervectors for encoding. */
template<int F, int D, typename elemTy>
std::vector<elemTy> generate_id_hvs_stdvec() {
    // TODO: We may eventually be able to rewrite this using hypervector intrinsics, for now, we can't index them, so it makes more sense to generates 
    // these in a 2d std::vector and create a hypermatrix from that. 
    

    int flip_factor = 2;   // TODO: Parameterize
    float mu = 0.0;
    float sigma = 1.0;
    int nFlip = int(D / flip_factor);
    std::vector<elemTy> bases(D);
    for (int i = 0; i < D; i++) {
        float tmp = float(rand() / float(RAND_MAX) * 2.0 * sigma + mu);
        bases[i] = (tmp > 0 ? 1. : -1.);
    }

    std::vector<float> id_hvs; // this is F * D
    id_hvs.insert(id_hvs.end(), bases.begin(), bases.end());

    for (int ii = 0; ii < F - 1; ii++) {
        std::vector<int> idx_to_flip(nFlip);
        for (int i = 0; i < nFlip; i++)
            idx_to_flip[i] = rand() % D;
        for (int i = 0; i < nFlip; i++)
            bases[idx_to_flip[i]] *= -1;

        id_hvs.insert(id_hvs.end(), bases.begin(), bases.end());
    }

    // FIXME: make sure this is reading out in the correct memory order (i.e row major vs col major.)
    std::vector<elemTy> out_vec(id_hvs.begin(), id_hvs.end());

    /*
    for (int i = 0; i < 10; i++) {
        std::cout << "[";
        for (int j = 0; j < 32; j++) {
            std::cout << " " << out_vec[(i * 32) + j] << ", ";
        }
        std::cout << "]\n";
    } */

    return out_vec;
}

/* Generates a F x D hypermatrix representing the id hypervectors for encoding. */
template<int F, int D, typename elemTy>
std::vector<elemTy> generate_normal_hvs_stdvec() {

    // FIXME: 
    int mu = 0;
    int sigma = 1;

    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<float> d{0.0, 1.0};


    std::vector<elemTy> out_vec (F * D);

    // FIXME: Round down.
    for (int i = 0; i < F * D; i++) {
        out_vec[i] = (d(gen) > 0.0) ? -1.0 : 1.0;
    }


    // FIXME: make sure this is reading out in the correct memory order (i.e row major vs col major.)
    // std::vector<elemTy> out_vec(id_hvs.begin(), id_hvs.end());



    return out_vec;
}