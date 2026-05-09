# $ python3.11 check.py --test-limit 5
from hdc_approximation_tuner.Harness import OpenTunerHarness, process_opentuner_config
from hdc_approximation_tuner.ParseHDCOpDesc import parse_hdc_op
from hdc_approximation_tuner.ApproxTypes import ApproxEncodingDimension
import json
import opentuner


test_json = {
    "arg_types":["<1024 x float>","<10 x 1024 x float>"],
    "intrinsic":"llvm.hpvm.hdc.cossim",
    "number":0,
    "op_type": "hmhv",
    "parent_func":"Z4nodePu11matrix_typeIL",
    "ret_type":"<10 x float>"
}

operations = [test_json]


apx_descs = [parse_hdc_op(dict_) for dict_ in operations]


# Global Configurations

# Encoding Dimension
DHv = 1024

apx_descs.append(
    ApproxEncodingDimension(1, DHv, True) # True: Use only powers of 2, False otherwise

)



def eval_config(cfg):
    print("Configuration:", cfg)
    pass_config_json = process_opentuner_config(cfg)
    print(json.dumps(pass_config_json, indent = 4))
    return {'time': 1, "accuracy": 0.5}


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    args = argparser.parse_args()
    # Register approximation description and eval config to arg parse object
    args.approx_descs = apx_descs
    args.evaluate_config_fn = eval_config
    args.accuracy_target = 0.75
    args.num_knobs = len(apx_descs)

    tuner_harness = OpenTunerHarness(args)
    tuner_harness.main(args)
