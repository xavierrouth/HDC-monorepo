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
  cfg.hypervector_dim = 1024;//original hv dimensions, don't count dimension/mlc packing
  cfg.mlc_level=2;//1 for reference mode

  int test_verify_cycles=0;//0 reference mode

  initialize_encoder(cfg);


   ins_hdnn_reram->program_reram_bit_mlc(
   5, 5, 10,test_verify_cycles);

   int16_t test_output=ins_hdnn_reram->readout_reram_bit_mlc(
   5, 10);

  printf("tested output %d \n",test_output);

  int16_t test_hv[cfg.hypervector_dim]={-1,-1,-1,1,1,-1,1,1};
  int16_t result_hv[cfg.hypervector_dim/cfg.mlc_level];

  dimension_packing(test_hv,result_hv);

  for (int i = 0; i < 4; i++)
  {
    printf("result hv %d \n",result_hv[i]);
  }
  

}