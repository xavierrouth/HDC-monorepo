#include "hpvm_interface.hpp"
#include "indicators.hpp"
#include "mnist_reader.hpp"

using namespace std;
using namespace indicators;

int main() {
  // Create configs
  config cfg;
  // cfg.num_classes = 26;
  // cfg.num_features = 617;
  // cfg.hypervector_dim = 2048;
  cfg.num_classes = 10;
  cfg.num_features = int(28 * 28);
  // Params
  cfg.hypervector_dim = HV_DIM;//original hv dimensions, don't count dimension/mlc packing
  cfg.mlc_level=MLC;//1 for reference mode,do not use mlc=0
  cfg.adc_resolution=ADC_RES;
  cfg.adc_ceil=ADC_CEIL;//adc input ceil
  cfg.adc_floor=ADC_FLOOR;//adc input floor
  cfg.adc_step=ADC_STEP;//quantization step,should be: (ceil-floor)/step<=2^res
  int test_verify_cycles=WRITE_VERIFY_CYCLE;//0 reference mode,maxium 7

  // Load MNIST Dataset
  auto dataset =
      mnist::read_dataset<std::vector, std::vector, int16_t, uint8_t>("./data");

  // ProgressBar bar{option::BarWidth{50},
  //                 option::Start{"["},
  //                 option::Fill{"="},
  //                 option::Lead{">"},
  //                 option::Remainder{" "},
  //                 option::End{"]"},
  //                 option::PrefixText{"HDC"},
  //                 option::ForegroundColor{Color::unspecified},
  //                 option::ShowElapsedTime{true},
  //                 option::ShowRemainingTime{true},
  //                 option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

  /////////////////////////////////////////////
  // Train
  int8_t *BHv_Ptr = nullptr;
  int16_t *FHv_Ptr = nullptr;
  size_t FHv_Bytes = cfg.num_features * sizeof(int16_t);

  size_t CHv_size= cfg.num_classes*cfg.hypervector_dim;
  size_t CHv_size_packed=CHv_size/cfg.mlc_level;

  int16_t *CHv_Ptr = new int16_t[CHv_size_packed];
  std::fill_n(CHv_Ptr, CHv_size_packed, 0);

  size_t CHv_Bytes = CHv_size * sizeof(int16_t);
  size_t CHv_Bytes_packed=CHv_size_packed*sizeof(int16_t);

  // printf("%d", HV_DIM);
  // return 0;
  int n_samples = dataset.training_images.size();
  for (int i = 0; i < n_samples; i++) {
    // bar.set_progress(int(i * 100.0 / n_samples));

    // __hetero_hdc_training
    {
      initialize_encoder(cfg);

      allocate_base_mem((void *)BHv_Ptr, 0);
      FHv_Ptr = &dataset.training_images[i][0];
      allocate_feature_mem((void *)FHv_Ptr, FHv_Bytes);
      allocate_class_mem((void *)CHv_Ptr, CHv_Bytes_packed);

      execute_train(dataset.training_labels[i],test_verify_cycles);
      read_class_mem((void *)CHv_Ptr, CHv_Bytes_packed);
    }
  }



  n_samples = dataset.test_images.size();
  // for (int i = 0; i < n_samples; i++) {
  //   // bar.set_progress(int(i * 100.0 / n_samples));

  //   // __hetero_hdc_inference
  //   {
  //     initialize_encoder(cfg);

  //     allocate_base_mem((void *)BHv_Ptr, 0);
  //     FHv_Ptr = &dataset.test_images[i][0];
  //     allocate_feature_mem((void *)FHv_Ptr, FHv_Bytes);
  //     allocate_class_mem((void *)CHv_Ptr, CHv_Bytes_packed);

  //     int pred = execute_inference();

  //     correct += pred == dataset.test_labels[i];
  //     // read_score_mem(SHv_Ptr, SHv_Bytes);
  //   }
  // }
  // printf("\n Testing accuracy = %f\n", correct * 1.0 / n_samples);

  /////////////////////////////////////////////
  int epochs = EPOCHS;
  for (int e = 0; e < epochs; e++) {
    // bar.set_progress(int(e * 100.0 / epochs));


    n_samples = dataset.training_images.size();
    for (int i = 0; i < n_samples; i++) {
      // __hetero_hdc_retraining
      {
        initialize_encoder(cfg);

        allocate_base_mem((void *)BHv_Ptr, 0);
        FHv_Ptr = &dataset.training_images[i][0];
        allocate_feature_mem((void *)FHv_Ptr, FHv_Bytes);
        allocate_class_mem((void *)CHv_Ptr, CHv_Bytes_packed);

        execute_retrain(dataset.training_labels[i],test_verify_cycles);

        read_class_mem((void *)CHv_Ptr, CHv_Bytes_packed);
      }
    }


    // printf("\n Testing accuracy @ Epoch %d/%d = %f\n", e, epochs,
    //        correct * 1.0 / n_samples);
  }
  int correct = 0;

  n_samples = dataset.test_images.size();
  for (int i = 0; i < n_samples; i++) {
    // __hetero_hdc_inference
    {
      initialize_encoder(cfg);

      allocate_base_mem((void *)BHv_Ptr, 0);
      FHv_Ptr = &dataset.test_images[i][0];
      allocate_feature_mem((void *)FHv_Ptr, FHv_Bytes);
      allocate_class_mem((void *)CHv_Ptr, CHv_Bytes_packed);

      int pred = execute_inference();

      correct += pred == dataset.test_labels[i];
      // read_score_mem(SHv_Ptr, SHv_Bytes);
    }
  }


  int time_ = get_latency();
  double energy_ = get_energy();
  /////////////////////////////////////////////
  FILE *file = fopen("results.txt", "w");
  if (file == NULL) {
    printf("Error opening file!\n");
    return 1;
  }

  fprintf(file, "%d\n", time_);
  fprintf(file, "%f\n", correct * 1.0 / n_samples);
  fprintf(file, "%f", energy_);
  // std::ofstream myfile("/home/xrouth2/hdc/HD-Sim-Tune/accuracy.txt");
	// myfile << (correct * 1.0 / n_samples) << std::endl;
	fclose(file);

  delete[] CHv_Ptr;

  if (ins_hdnn_reram)
    delete ins_hdnn_reram;



  return 0;
}