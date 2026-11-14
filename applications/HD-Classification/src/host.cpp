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
#include <cstring>
#include "mnist_reader.hpp"


#define POSTFIX ".cpu.txt"

using namespace std;

// #define MNIST
template <typename T>
void datasetRead(std::vector<T> &data, std::string path){
	std::ifstream file_(path, std::ios::in | std::ios::binary);
	assert(file_.is_open() && "Couldn't open file!");
	T size;
	file_.read((char*)&size, sizeof(size));
	T temp;
	for(int i = 0; i < size; i++){
		file_.read((char*)&temp, sizeof(temp));
		data.push_back(temp);
	}
	file_.close();
}

template<typename T>
T parse(const std::string& str) {
    std::istringstream iss(str);
    T value;
    iss >> value;
    return value;
}


// covnert to T, Read as S
template <typename T, typename S>
std::vector<std::vector<T>> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<T>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::vector<T> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            row.push_back((T) parse<S>(value));
        }
        data.push_back(row);
    }

    return data;
}

template <typename T>
std::vector<T> flatten2D(std::vector<std::vector<T>> vec2D) {
    std::vector<T> flattened;
    for (const auto& row : vec2D) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

// Read a binary file: 4-byte header (skipped), followed by T-typed elements.
// Element count is derived from file size to avoid header type ambiguity
// (X files write the count as float32; Y files write it as int32).
template <typename T>
std::vector<T> readBinary(const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    assert(file.is_open() && "Couldn't open file!");
    file.seekg(0, std::ios::end);
    int32_t file_size = (int32_t)file.tellg();
    file.seekg(4, std::ios::beg);  // skip 4-byte header
    int32_t size = (file_size - 4) / sizeof(T);
    std::vector<T> data(size);
    file.read((char*)data.data(), size * sizeof(T));
    return data;
}

// Scale float data to the range of T. For float, copies as-is.
template <typename T>
std::vector<T> quantize(const std::vector<float>& data) {
    std::vector<T> out;
    out.reserve(data.size());
    if (std::is_floating_point<T>::value) {
        for (float v : data) out.push_back(static_cast<T>(v));
    } else {
        float max_abs = 0.0f;
        for (float v : data) max_abs = std::max(max_abs, std::abs(v));
        if (max_abs == 0.0f) max_abs = 1.0f;
        // Give the feature magnitude half of T's value bits and leave the rest
        // as headroom for the encoding matmul, which sums N_FEAT products of
        // (+/-1 x feature).
        //
        // This is an empirical setting, not a derived bound. The naive bound --
        // budget ceil(log2(N_FEAT)) bits for the accumulation and give the rest
        // to the feature -- was measured and is WRONG: it puts int32 features at
        // 2^20, whose worst-case sum (617 * 2^20 = 6.5e8) sits comfortably under
        // INT32_MAX, yet accuracy collapses to chance. int32 empirically needs
        // ~100x headroom rather than the ~3x that bound predicts, so something
        // in the lowered pipeline accumulates more aggressively than a single
        // N_FEAT-term dot product. Until that is pinned down, use the half-bits
        // split, which is measured good:
        //
        //   int32 (features 2^15): 0.871    int64 (features 2^31): 0.874
        //   float:                 0.874    double:                0.874
        //
        // int8 and int16 return chance accuracy at every scale tried and are
        // not usable for this application; 7 and 15 value bits cannot absorb a
        // 617-term accumulation with any useful feature precision.
        //
        // The previous formula was max/max_abs/N_FEAT, which places the
        // worst-case sum on exactly INT32_MAX with zero headroom and drove every
        // integer configuration to chance (0.042 vs 0.874 for float).
        //
        // The previous formula was max/max_abs/N_FEAT, which places the
        // worst-case sum on exactly INT32_MAX with zero headroom. It overflowed
        // in practice and drove every DATA_TYPE=int configuration to chance
        // accuracy (0.042, vs 0.874 for float): gen_rp_matrix builds projection
        // rows by wrap-shifting one seed vector, so the rows are correlated and
        // partial sums excurse much further than an independent-random model
        // predicts. Measured: this formula restores int to 0.871.
        const int value_bits = std::numeric_limits<T>::digits;  // excludes sign
        const float feature_max = (float)(1ull << (value_bits / 2));
        float scale = feature_max / max_abs;
        std::cout << "[DEBUG quantize] max_abs=" << max_abs
                  << " T_max=" << (int64_t)std::numeric_limits<T>::max()
                  << " scale=" << scale << std::endl;
        for (float v : data) {
            float s = v * scale;
            s = std::max(s, (float)std::numeric_limits<T>::min());
            s = std::min(s, (float)std::numeric_limits<T>::max());
            out.push_back(static_cast<T>(s));
        }
        std::cout << "[DEBUG quantize] first 10 quantized values: ";
        for (int i = 0; i < 10 && i < (int)out.size(); i++)
            std::cout << (int64_t)out[i] << " ";
        std::cout << std::endl;
    }
    return out;
}

// void __attribute__ ((noinline)) l2norm(__hypervector__<N_CLASS, hvtype> *norms_buffer, __hypermatrix__<N_CLASS, Dhv, hvtype> *classes) {
// 	*norms_buffer = __hetero_hdc_l2norm<N_CLASS, Dhv, hvtype>(*classes);

//         void* InitDAG = __hetero_launch(
//                 (void*) L2NormDFG<N_CLASS, Dhv>, 
//                 2,
//                 norms_buffer,
//                 sizeof(hvtype) * N_CLASS,
//                 classes,
//                 sizeof(hvtype) * N_CLASS * Dhv,
//                 1,
//                 norms_buffer,
//                 sizeof(hvtype) * N_CLASS
//                 );

//         __hetero_wait(InitDAG);

// }


void __attribute__ ((noinline)) InitializeClasses(__hypervector__<Dhv, hvtype> *encoded_train_hvs, __hypermatrix__<N_CLASS, Dhv, hvtype> *classes, int label ,int index) {

        void* InitDAG = __hetero_launch(
                (void*) InitialUpdateDFG< N_CLASS, N_SAMPLE, Dhv>, 4, /* Input Buffers: 4*/ 
                encoded_train_hvs, sizeof(hvtype) * Dhv, 
                classes, sizeof(hvtype) * (N_CLASS * Dhv), 
                label, index,
                1, /* Output Buffers */ 
                classes, sizeof(hvtype) * (N_CLASS * Dhv)
                );

        __hetero_wait(InitDAG);

}

// This doesn't work on hypermatricies for some reason.
template <typename T>
T initialize_rp_seed(size_t loop_index_var) {
	int i = loop_index_var / 32;
	int j = loop_index_var % 32;

	//std::cout << i << " " << j << "\n";
    
	long double temp = log2(i+2.5) * pow(2, 31);
	long long int temp2 = (long long int)(temp);
	temp2 = temp2 % 2147483648;

	int ele = temp2 & (0x01 << j); //temp2 && (0x01 << j);
	T out = ele ? (T) 1 : (T) -1;
	// std::cout << "idx, out: " << loop_index_var << ", " << out  << std::endl;

	return out;
}

template <typename T>
T copy(T* vec, size_t loop_index_var) {
	return vec[loop_index_var];
}

template <typename T>
T copy_no_padding(T* vec, size_t loop_index_var) {
	// The given data vectors are strided by N_FEAT_PAD elements, even though the vectors are each N_FEAT elements in length.
    return vec[loop_index_var + (N_FEAT_PAD - N_FEAT) * (loop_index_var / N_FEAT)];
}

template <typename T>
T zero(size_t loop_index_var) {
	return 0;
}


int main(int argc, char** argv) {
	__hpvm__init();

	auto t_start = std::chrono::high_resolution_clock::now();
	auto t_total_start = std::chrono::high_resolution_clock::now();

	srand(time(NULL));

	int EPOCH = std::atoi(argv[1]);
	std::string X_train_path = argc > 2 ? argv[2] : "../../datasets/classification/isolet_nq_trainX.bin";
	std::string y_train_path = argc > 3 ? argv[3] : "../../datasets/classification/isolet_nq_trainY.bin";
	std::string X_test_path  = argc > 4 ? argv[4] : "../../datasets/classification/isolet_nq_testX.bin";
	std::string y_test_path  = argc > 5 ? argv[5] : "../../datasets/classification/isolet_nq_testY.bin";

	std::vector<hvtype> X_train = quantize<hvtype>(readBinary<float>(X_train_path));
	std::vector<int>    y_train = readBinary<int>(y_train_path);

	std::vector<hvtype> X_test  = quantize<hvtype>(readBinary<float>(X_test_path));
	std::vector<int>    y_test  = readBinary<int>(y_test_path);

	size_t X_train_samples = X_train.size() / N_FEAT_PAD;

	std::cout << "N_FEAT: " << N_FEAT << ", N_FEAT_PAD: " << N_FEAT_PAD << std::endl;
	std::cout << "X Train, y Train: / N_FEAT_PAD " << X_train_samples << ", " << y_train.size() << std::endl;
	assert(X_train_samples == y_train.size() && "Incorrect number of training labels");
	assert(N_SAMPLE == y_train.size());

	std::cout << "X Test, y Test: / N_FEAT_PAD " << X_test.size() / N_FEAT_PAD << ", " << y_test.size() << std::endl;
	assert(N_TEST == y_test.size());
	
	hvtype* training_input_vectors_cpu = X_train.data();

	// N_FEAT is number of entries per vector
	size_t input_vector_size = N_FEAT * sizeof(hvtype); // Size of a single vector

	int* training_labels = y_train.data(); // Get your training labels.
	// N_SAMPLE is number of input vectors
	size_t training_labels_size = N_SAMPLE * sizeof(int);

	// INFERENCE DATA / TEST DATA
	//int inference_test_labels[N_TEST];
	size_t inference_labels_size = N_TEST * sizeof(int);
	int* inference_test_labels = y_test.data();

	hvtype* inference_input_vectors_cpu = X_test.data();
	assert((X_test.size() / N_FEAT_PAD) == N_TEST && "Incorrect number of tests");

	auto t_elapsed = std::chrono::high_resolution_clock::now() - t_start;
	long mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();
	long mSec1 = mSec;

	t_start = std::chrono::high_resolution_clock::now();

	__hypermatrix__<N_SAMPLE, N_FEAT, hvtype> training_input_vectors = __hetero_hdc_create_hypermatrix<N_SAMPLE, N_FEAT, hvtype>(1, (void*) copy_no_padding<hvtype>, training_input_vectors_cpu);
	__hypermatrix__<N_TEST, N_FEAT, hvtype> inference_input_vectors = __hetero_hdc_create_hypermatrix<N_TEST, N_FEAT, hvtype>(1, (void*) copy_no_padding<hvtype>, inference_input_vectors_cpu);

	size_t rp_matrix_size = N_FEAT * Dhv * sizeof(hvtype);
	// __hypervector__<Dhv, hvtype> rp_seed = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) initialize_rp_seed<hvtype>);	
	// Dhv needs to be greater than N_FEAT for the orthognality to hold.
	__hypermatrix__<Dhv, N_FEAT, hvtype> rp_matrix_buffer = __hetero_hdc_create_hypermatrix<Dhv, N_FEAT, hvtype>(0, (void *) initialize_rp_seed<hvtype>);
	// __hypermatrix__<N_FEAT, Dhv, hvtype> shifted_buffer = __hetero_hdc_create_hypermatrix<N_FEAT, Dhv, hvtype>(0, (void *) zero<hvtype>);

	// #ifndef NODFG
	//     void* GenRPMatDAG = __hetero_launch((void*) gen_rp_matrix<Dhv,  N_FEAT>, 3, /* Input Buffers: 3*/ __hetero_hdc_get_handle(rp_seed), sizeof(hvtype) * Dhv, __hetero_hdc_get_handle(shifted_buffer), sizeof(hvtype) * (N_FEAT * Dhv), __hetero_hdc_get_handle(rp_matrix_buffer), sizeof(hvtype) * (N_FEAT * Dhv), /* Output Buffers: 1*/ 1, __hetero_hdc_get_handle(rp_matrix_buffer), sizeof(hvtype) * (N_FEAT * Dhv));

	//     __hetero_wait(GenRPMatDAG);
	// #else
	//     gen_rp_matrix<Dhv, N_FEAT>(__hetero_hdc_get_handle(rp_seed), sizeof(hvtype) * Dhv, __hetero_hdc_get_handle(shifted_buffer), sizeof(hvtype) * (N_FEAT * Dhv), __hetero_hdc_get_handle(rp_matrix_buffer), sizeof(hvtype) * (N_FEAT * Dhv));
	// #endif

	// Read encoded hvs into a new flattened vec from a csv file.
	// std::vector<std::vector<hvtype>> encoded_train_hvs_vec2D = readCSV<hvtype, hvtype>("/home/xrouth2/hdc/HD-Classification/CPU/train_enc_hvs.csv");
	// std::vector<hvtype> encoded_train_hvs_flattened = flatten2D<hvtype>(encoded_train_hvs_vec2D);

	// // Read encoded test hvs into a new flattened vec from a csv file.
	// std::vector<std::vector<hvtype>> encoded_test_hvs_vec2D = readCSV<hvtype, hvtype>("/home/xrouth2/hdc/HD-Classification/CPU/test_enc_hvs.csv");
	// std::vector<hvtype> encoded_test_hvs_flattened = flatten2D<hvtype>(encoded_test_hvs_vec2D);

	// // read class hvs from file
	// std::vector<std::vector<hvtype>> class_hvs_vec2D = readCSV<hvtype, hvtype>("/home/xrouth2/hdc/HD-Classification/CPU/class_hvs.csv");
	// std::vector<hvtype> class_hvs_flattened = flatten2D<hvtype>(class_hvs_vec2D);

	size_t class_size = Dhv * sizeof(hvtype);
	size_t classes_size = N_CLASS * Dhv * sizeof(hvtype);
	size_t encoded_hv_size = Dhv * sizeof(hvtype);
	size_t update_hv_size = Dhv * sizeof(hvtype);
	size_t scores_size = N_CLASS * sizeof(hvtype);
	size_t norms_size = N_CLASS * sizeof(hvtype);

	__hypervector__<Dhv, hvtype> update_hv = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);

	__hypermatrix__<N_CLASS, Dhv, hvtype> classes = __hetero_hdc_create_hypermatrix<N_CLASS, Dhv, hvtype> \ 
		(0, (void*) zero<hvtype>);
		// (1, (void*) copy<hvtype>, class_hvs_flattened.data());
		
	auto classes_handle =__hetero_hdc_get_handle(classes);

	__hypermatrix__<N_SAMPLE, Dhv, hvtype> encoded_train_hvs = __hetero_hdc_create_hypermatrix<N_SAMPLE, Dhv, hvtype>\
		(0, (void*) zero<hvtype>); 
		// (1, (void*) copy<hvtype>, encoded_train_hvs_flattened.data());
	auto encoded_train_hvs_handle = __hetero_hdc_get_handle(encoded_train_hvs);
	size_t encoded_train_hvs_size = N_SAMPLE * Dhv * sizeof(hvtype);

	__hypermatrix__<N_TEST, Dhv, hvtype> encoded_test_hvs = __hetero_hdc_create_hypermatrix<N_TEST, Dhv, hvtype>\
		(0, (void*) zero<hvtype>);
		// (1, (void*) copy<hvtype>, encoded_test_hvs_flattened.data());

	auto encoded_test_hvs_handle = __hetero_hdc_get_handle(encoded_test_hvs);
	size_t encoded_test_hvs_size = N_TEST * Dhv * sizeof(hvtype);

	__hypervector__<Dhv, hvtype> encoded_hv_buffer = __hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);
	__hypervector__<N_CLASS, hvtype> scores_buffer = __hetero_hdc_create_hypervector<N_CLASS, hvtype>(0, (void*) zero<hvtype>);
	__hypervector__<N_CLASS, hvtype> norms_buffer = __hetero_hdc_create_hypervector<N_CLASS, hvtype>(0, (void*) zero<hvtype>);

	// Scratch buffers used to hold binarized (+1/-1) copies of the query and
	// class hypervectors when HAMMING_DIST == 1. __hetero_hdc_sign cannot be
	// applied in place, so its results need somewhere to live. These live in
	// the inference region, so they carry the inference element type.
	size_t sign_hv_size = Dhv * sizeof(hvtype_infer);
	__hypervector__<Dhv, hvtype_infer> sign_q_buffer = __hetero_hdc_create_hypervector<Dhv, hvtype_infer>(0, (void*) zero<hvtype_infer>);
	__hypervector__<Dhv, hvtype_infer> sign_c_buffer = __hetero_hdc_create_hypervector<Dhv, hvtype_infer>(0, (void*) zero<hvtype_infer>);

	// ---- Inference representation region -------------------------------
	// When INFER_DATA_TYPE == DATA_TYPE these alias the training-region
	// buffers and no conversion is emitted. Otherwise they are distinct
	// storage, populated once by cast_region_node at the region boundary.
	size_t scores_infer_size = N_CLASS * sizeof(hvtype_infer);
	size_t norms_infer_size  = N_CLASS * sizeof(hvtype_infer);
	__hypervector__<N_CLASS, hvtype_infer> scores_infer_buffer = __hetero_hdc_create_hypervector<N_CLASS, hvtype_infer>(0, (void*) zero<hvtype_infer>);
	__hypervector__<N_CLASS, hvtype_infer> norms_infer_buffer  = __hetero_hdc_create_hypervector<N_CLASS, hvtype_infer>(0, (void*) zero<hvtype_infer>);

#if INFER_NEEDS_CAST
	size_t classes_infer_size          = N_CLASS * Dhv * sizeof(hvtype_infer);
	size_t encoded_test_hvs_infer_size = N_TEST  * Dhv * sizeof(hvtype_infer);
	size_t cast_row_size               = Dhv * sizeof(hvtype);

	__hypermatrix__<N_CLASS, Dhv, hvtype_infer> classes_infer =
		__hetero_hdc_create_hypermatrix<N_CLASS, Dhv, hvtype_infer>(0, (void*) zero<hvtype_infer>);
	__hypermatrix__<N_TEST, Dhv, hvtype_infer> encoded_test_hvs_infer =
		__hetero_hdc_create_hypermatrix<N_TEST, Dhv, hvtype_infer>(0, (void*) zero<hvtype_infer>);
	__hypervector__<Dhv, hvtype> cast_row_buffer =
		__hetero_hdc_create_hypervector<Dhv, hvtype>(0, (void*) zero<hvtype>);

	auto classes_infer_handle          = __hetero_hdc_get_handle(classes_infer);
	auto encoded_test_hvs_infer_handle = __hetero_hdc_get_handle(encoded_test_hvs_infer);
#endif

	__hypervector__<N_FEAT, hvtype> row_buffer = __hetero_hdc_create_hypervector<N_FEAT, hvtype>(0, (void*) zero<hvtype>);
	size_t row_buffer_size = N_SAMPLE * sizeof(hvtype);
	int inference_labels[N_TEST];

	// ============ Training ===============

	std::cout << "before encoding" << std::endl;
	t_start = std::chrono::high_resolution_clock::now();

	// Initialize class hvs, encode every HV.
	// for (int i = 0; i < N_SAMPLE; i++) {
	void* encoding_dag = __hetero_launch(
		(void*) encoding_dfg_train<Dhv, N_FEAT, N_SAMPLE>,
		/* input buffers*/
		4 + 1,
		__hetero_hdc_get_handle(rp_matrix_buffer), 
		rp_matrix_size, 
		__hetero_hdc_get_handle(training_input_vectors), 
		input_vector_size * N_SAMPLE, 
		__hetero_hdc_get_handle(encoded_hv_buffer),
		encoded_hv_size,
		__hetero_hdc_get_handle(row_buffer),
		row_buffer_size,
		/* output buffers*/
		encoded_train_hvs_handle, 
		encoded_train_hvs_size,
		1,
		encoded_train_hvs_handle, 
		encoded_train_hvs_size);

	__hetero_wait(encoding_dag);

	std::cout << "initailizing all classes" << std::endl;
	// Set classes as sum of all encoded HVs that share the same label.
	// NOTE: Using plain pointer ops here because __hetero_hdc_get_matrix_row
	// segfaults in the HPVM CPU runtime (TODO: investigate root cause).
	// Labels are 0-indexed in the dataset files (0..N_CLASS-1).
	{
		hvtype* raw_encoded = (hvtype*)encoded_train_hvs_handle;
		hvtype* raw_classes = (hvtype*)classes_handle;
		for (int i = 0; i < N_SAMPLE; i++) {
			int label = training_labels[i];
			for (int d = 0; d < Dhv; d++) {
				raw_classes[label * Dhv + d] += raw_encoded[i * Dhv + d];
			}
		}
	}

	// // Something is slightly wrong here... 
	// // TODO: Check the buffer that  is being read from here, and check the size, and make sure the math i * N_CLASS + j works out, 
	// // And also check that we are printing them in the correct order w/r to memory layout.
	// std::cout << "Dumping class hypervectors after initialization:" << std::endl;
	// for (int i = 0; i < N_CLASS; i++) {
	// 	auto class_hv =  __hetero_hdc_get_matrix_row<N_CLASS, Dhv, hvtype>(classes, N_CLASS, Dhv, i);
	// 	for (int j = 0; j < Dhv; j++) {
	// 		hvtype val = ((hvtype*) __hetero_hdc_get_handle(class_hv))[j];
	// 		std::cout << val << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	int argmax[1];

	// ======= Retraining Epochs ======= 

	// #if HAMMING_DIST != 1
	// #endif
	std::cout << "Beginning Training." << std::endl;

	// EPOCH = 5;
	// Loops training over epochs AND? N_SAMPLEs,
	// reencodes each time.
	// Increments label and training_input_vectors each time. 
	if (true) {
	for (int i = 0; i < EPOCH; i++) {
		// l2norm(__hetero_hdc_get_handle(norms_buffer), __hetero_hdc_get_handle(classes));

		void* training_dag = __hetero_launch(
			(void*) training_root_node<Dhv, N_CLASS, N_SAMPLE, N_FEAT>,
			/* input buffers*/
			9,
			encoded_train_hvs_handle,
			encoded_train_hvs_size,
			training_labels,
			training_labels_size,
			__hetero_hdc_get_handle(encoded_hv_buffer),
			encoded_hv_size,
			__hetero_hdc_get_handle(scores_buffer),
			scores_size,
			__hetero_hdc_get_handle(norms_buffer),
			norms_size,
			__hetero_hdc_get_handle(update_hv),
			update_hv_size,
			__hetero_hdc_get_handle(sign_q_buffer),
			sign_hv_size,
			__hetero_hdc_get_handle(sign_c_buffer),
			sign_hv_size,
			classes_handle,
			classes_size,
			/* output buffers*/
			1,
			classes_handle,
			classes_size
		);
		__hetero_wait(training_dag);
	}

	// Dump classes after training.
	// std::cout << "Dumping class hypervectors after training:" << std::endl;
	// for (int i = 0; i < N_CLASS; i++) {
	// 	auto class_hv =  __hetero_hdc_get_matrix_row<N_CLASS, Dhv, hvtype>(classes, N_CLASS, Dhv, i);
	// 	for (int j = 0; j < Dhv; j++) {
	// 		hvtype val = ((hvtype*) __hetero_hdc_get_handle(class_hv))[j];
	// 		std::cout << val << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	std::cout << "Finished Training." << std::endl;
	}
	auto t_end = std::chrono::high_resolution_clock::now();
	auto training_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
	

	// ============ Test / Evaluation / Inference =============== //

	t_start = std::chrono::high_resolution_clock::now();
	// Encode vectors for test
	std::cout << "before inference encoding" << std::endl;
	// Initialize class hvs, encode every HV.
	if (true) {
	void* inference_encoding_dag = __hetero_launch(
		(void*) encoding_dfg_infer<Dhv, N_FEAT, N_TEST>,
		/* input buffers*/
		4 + 1,
		__hetero_hdc_get_handle(rp_matrix_buffer), 
		rp_matrix_size, 
		__hetero_hdc_get_handle(inference_input_vectors), 
		input_vector_size * N_TEST, 
		__hetero_hdc_get_handle(encoded_hv_buffer),
		encoded_hv_size,
		__hetero_hdc_get_handle(row_buffer),
		row_buffer_size,
		/* output buffers*/
		encoded_test_hvs_handle, 
		encoded_test_hvs_size,
		1,
		encoded_test_hvs_handle, 
		encoded_test_hvs_size);

	__hetero_wait(inference_encoding_dag);
}
	std::cout << "after inference encoding" << std::endl;

	
	// #if HAMMING_DIST != 1
	// l2norm(__hetero_hdc_get_handle(norms_buffer), __hetero_hdc_get_handle(classes));
	// #endif
#if INFER_NEEDS_CAST
	// Representation-region boundary: convert the class prototypes and the
	// encoded test hypervectors from the training element type to the
	// inference element type. Executed once, at a program point fixed at
	// compile time; its cost is included in this configuration's measurement.
	{
		void* cast_classes_dag = __hetero_launch(
			(void*) cast_region_node<N_CLASS, Dhv>,
			3,
			classes_handle, classes_size,
			__hetero_hdc_get_handle(cast_row_buffer), cast_row_size,
			classes_infer_handle, classes_infer_size,
			1,
			classes_infer_handle, classes_infer_size);
		__hetero_wait(cast_classes_dag);

		void* cast_test_dag = __hetero_launch(
			(void*) cast_region_node<N_TEST, Dhv>,
			3,
			encoded_test_hvs_handle, encoded_test_hvs_size,
			__hetero_hdc_get_handle(cast_row_buffer), cast_row_size,
			encoded_test_hvs_infer_handle, encoded_test_hvs_infer_size,
			1,
			encoded_test_hvs_infer_handle, encoded_test_hvs_infer_size);
		__hetero_wait(cast_test_dag);
	}
	std::cout << "Converted to inference representation." << std::endl;
#endif

	std::cout << "Beginning Inference." << std::endl;

	// FIXME, this currently launches N_TEST different tasks.
	// instead, we want to batch this into a single task that processes all test samples.
	// std::cout << "Classifying test sample " << i << std::endl;
	void* infernece_node_dag_handle = __hetero_launch(
		(void*) inference_node<Dhv, N_CLASS, N_TEST>,
		/* input buffers*/
		7,
#if INFER_NEEDS_CAST
		encoded_test_hvs_infer_handle,
		encoded_test_hvs_infer_size,
		classes_infer_handle,
		classes_infer_size,
#else
		encoded_test_hvs_handle,
		encoded_hv_size * N_TEST,
		classes_handle,
		classes_size,
#endif
		__hetero_hdc_get_handle(scores_infer_buffer),
		scores_infer_size,
		__hetero_hdc_get_handle(norms_infer_buffer),
		norms_infer_size,
		__hetero_hdc_get_handle(sign_q_buffer),
		sign_hv_size,
		__hetero_hdc_get_handle(sign_c_buffer),
		sign_hv_size,
		inference_labels,
		sizeof(int) * N_TEST,
		/* output buffers*/
		1,
		inference_labels,
		sizeof(int) * N_TEST
	);

	__hetero_wait(infernece_node_dag_handle);


	std::cout << "Finished Inference." << std::endl;

	t_end = std::chrono::high_resolution_clock::now();
	auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
	auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t_total_start).count();
	std::cout << "Inference Time: " << inference_time << " ms" << std::endl;

	std::ofstream myfile("out");
	int correct = 0;
	for(int i = 0; i < N_TEST; i++) {
		correct += inference_labels[i] == y_test[i]; // Labels are 0-indexed in the dataset files
		myfile << y_test[i] << " " << inference_labels[i] << std::endl;
	}

	float test_accuracy=float(correct)/N_TEST;
    

	// Dump test accuracy to a file
	


	std::cout << "Test Accuracy: " << test_accuracy << std::endl;

	{
        std::ofstream acc("output/test_accuracy.txt", std::ios::out | std::ios::trunc);
        if (acc.is_open()) {
            acc << test_accuracy << std::endl;
        }
		std::ofstream timing("output/test_time.txt", std::ios::out | std::ios::trunc);
		if (timing.is_open()) {
			timing << total_time << std::endl;
			timing << training_time << std::endl;
			timing << inference_time << std::endl;
		}

    }

	__hpvm__cleanup();
	return 0;
}


