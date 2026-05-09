#include <heterocc.h>
#include <hpvm_hdc.h>
#include <fstream>
#include <iostream>

// Tuneable Parameters
#ifndef DHV
#define DHV ENCODING_DIM
#endif

#define H 10

template <typename T>
T zero(size_t loop_index_var) {
	return 0;
}

template <typename T>
T init_hm(size_t loop_index_var) {
    if ((loop_index_var / DHV) == 1) {
        return loop_index_var / DHV;
    } else {
        if (loop_index_var % 2 == 0) {
            return -1.0 * loop_index_var;
        } else {
            return 1.0 * loop_index_var;
        }
    }
	return 0;
}

template <typename T>
T init_hv(size_t loop_index_var) {
	return 1;
}


typedef float hvtype;


void node(__hypervector__<DHV, hvtype>* hv, size_t hv_size,  __hypermatrix__<H, DHV, hvtype>* hm, size_t hm_size, 
    __hypervector__<H, hvtype>* sim, size_t sim_size,
    int* max, size_t max_size) {
    void* section = __hetero_section_begin();

    void* task = __hetero_task_begin(/* Input Buffers: 2*/ 4, hv, hv_size, 
        hm, hm_size,
        sim, sim_size, 
        max, max_size, /* Parameters: 0*/ 
    /* Output Buffers: 1*/ 2, sim, sim_size, max, max_size, "similarity_task");
    
    *sim = __hetero_hdc_cossim<H, DHV, hvtype>(*hv, *hm);

    *max = __hetero_hdc_arg_max<H, hvtype>(*sim);
    

    __hetero_task_end(task);
    __hetero_section_end(section);
}

int main(int argc, char** argv) {

	__hypermatrix__<H, DHV, hvtype> hm = 
        __hetero_hdc_create_hypermatrix<H, DHV, hvtype>(0, (void*) init_hm<hvtype>);

    
    __hypervector__<DHV, hvtype> hv = __hetero_hdc_create_hypervector<DHV, hvtype>(0, (void*) init_hv<hvtype>);	
    __hypervector__<H, hvtype> sim_hv = __hetero_hdc_hypervector<H, hvtype>();	


    auto hv_handle = (hvtype*)&hv;
    auto hm_handle = (hvtype*)&hm;

    // std::cout << "HM: \n";
    // for (int i = 0; i < H; i++) {
    //     std::cout << "[ ";
    //     for (int j = 0; j < DHV; j++){
    //         std::cout << hm_handle[i * DHV + j] << " ";
    //     }
    //     std::cout << " ] \n";
    // }

    // std::cout << "HV: \n";
    // std::cout << "[ ";
    // for (int j = 0; j < DHV; j++){
    //     std::cout << hv_handle[j] << " ";
    // }
    // std::cout << " ] \n";


    int max = 0;

    // auto sim_handle =  __hetero_hdc_get_handle(sim_hv);
    auto sim_handle = (hvtype*)&sim_hv;


    void* dag = __hetero_launch((void*) node, 4, 
        /* Input Buffers: 3*/ hv_handle, sizeof(hvtype) * DHV, hm_handle, sizeof(hvtype) * (H * DHV),
        sim_handle, sizeof(hvtype) * H,
        &max, sizeof(int), 
    /* Output Buffers: 1*/ 2, sim_handle, sizeof(hvtype) * H, &max, sizeof(int));

    std::cout << "sim: \n";
    std::cout << "[ ";
    for (int j = 0; j < H; j++){
        std::cout << sim_handle[j] << " ";
    }

    std::cout << " ] \n";

    __hetero_wait(dag);


    float accuracy = 0.0;
    if (max == 1) {
        accuracy = 1.0;
    }

    std::ofstream file("out.txt");

    if (file.is_open()) {
        file << accuracy;
        file.close();
    }

    std::cout << "Max: " << max << std::endl;
    std::cout << "Accuracy: " << accuracy << std::endl;

}