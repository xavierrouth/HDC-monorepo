# $ python3.11 check.py --test-limit 5
from hdc_approximation_tuner.Harness import OpenTunerHarness, process_opentuner_config
from hdc_approximation_tuner.ParseHDCOpDesc import parse_hdc_op
from hdc_approximation_tuner.ApproxTypes import ApproxEncodingDimension, ApproxAutoBinarize, ApproxCustomEnum, ApproxCustomInteger
import json
import opentuner
import time
import subprocess
import os
import shlex
import ast
import sys
import time

from gpu_power import *

root_dir_str = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Classification"
config_out_path = root_dir_str + "/config/config_out.json"

# All element-type representations the backend can be asked for. Types that
# miscompile or score poorly are deliberately left IN the space: the tuner
# evaluates them, a failed build returns the penalty result and a low-quality
# one is rejected by the QoS constraint, so "this type is a poor choice here"
# is a search outcome rather than a hand-coded exclusion.
#
# Measured on HD-Classification/ISOLET: int8_t and int16_t return chance
# accuracy (too few bits to absorb the encoding accumulation); int32_t, int64_t,
# float and double all work. On HD-HashTable int16_t and double fail to build.
DATA_TYPES = ["int8_t", "int16_t", "int32_t", "int64_t", "float", "double"]
FLOAT_TYPES = {"float", "double"}
config_in_path =   root_dir_str + "/config/config_in.json"

DATASETS = {
    "ISOLET": {
        "X_train": "../../datasets/classification/isolet_nq_trainX.bin",
        "y_train": "../../datasets/classification/isolet_nq_trainY.bin",
        "X_test":  "../../datasets/classification/isolet_nq_testX.bin",
        "y_test":  "../../datasets/classification/isolet_nq_testY.bin",
        "n_feat":     617,
        "n_sample":   6238,
        "n_test":     1559,
        "n_class":    26,
        "n_feat_pad": 624,
    },
    "UCIHAR": {
        "X_train": "../../datasets/classification/ucihar_trainX.bin",
        "y_train": "../../datasets/classification/ucihar_trainY.bin",
        "X_test":  "../../datasets/classification/ucihar_testX.bin",
        "y_test":  "../../datasets/classification/ucihar_testY.bin",
        "n_feat":     568,
        "n_sample":   6213,
        "n_test":     1554,
        "n_class":    12,
        "n_feat_pad": 568,
    },
}

# Global variables set by command-line arguments
dataset_x_train = None
dataset_y_train = None
dataset_x_test  = None
dataset_y_test  = None
dataset_n_feat     = None
dataset_n_sample   = None
dataset_n_test     = None
dataset_n_class    = None
dataset_n_feat_pad = None
device_type = None
no_approx_hint = False

my_env = os.environ.copy()
my_env["CONFIG_IN_PATH"] = config_in_path
my_env["CONFIG_OUT_PATH"] = config_out_path

def get_apx_descs(enable_redperf=True, enable_algchoice=True, redperf_end_only=False):
    with open(config_out_path, "r") as file:
        test_json = json.load(file)

    operations = test_json

    print("OPERATIONS: ", operations)

    apx_descs = [parse_hdc_op(dict_, enable_redperf=enable_redperf, enable_algchoice=enable_algchoice, redperf_end_only=redperf_end_only) for dict_ in operations]

    # Global Configurations

    # # Encoding Dimension
    DHv = 2048

    apx_descs.append(
        ApproxEncodingDimension(1, DHv, True) # True: Use only powers of 2, False otherwise
    )

    # Integerfor custom types which have numerical properties
    apx_descs.append(
        ApproxCustomInteger("TRAIN_ITERATIONS", 20) # 10 possible values.
    )

    apx_descs.append(
        ApproxCustomInteger("HAMMING_DIST", 2)
    )

    apx_descs.append(
        ApproxCustomInteger("DATA_TYPE", len(DATA_TYPES) - 1)
    )

    return apx_descs
def eval_config(cfg):
    global dataset_x_train, dataset_y_train, dataset_x_test, dataset_y_test, device_type, no_approx_hint
    global dataset_n_feat, dataset_n_sample, dataset_n_test, dataset_n_class, dataset_n_feat_pad
    print("HDC Configuration:", cfg)
    print("\n")
    pass_config_json = process_opentuner_config(cfg)
    print("Json:", pass_config_json)
    print("\n")

    # Extract Enc Dimension
    dhv_entry = next((item for item in pass_config_json if "EncodingDim" in item), None)
    dhv = dhv_entry["EncodingDim"]
    train_entry = next((item for item in pass_config_json if "TRAIN_ITERATIONS" in item), None)
    train_iterations = train_entry["TRAIN_ITERATIONS"]

    data_type_entry = next((item for item in pass_config_json if "DATA_TYPE" in item), None)
    data_type_raw = data_type_entry["DATA_TYPE"] if data_type_entry else DATA_TYPES.index("float")
    data_type = DATA_TYPES[min(data_type_raw, len(DATA_TYPES) - 1)]

    hamming_dist_entry = next((item for item in pass_config_json if "HAMMING_DIST" in item), None)
    hamming_dist = hamming_dist_entry["HAMMING_DIST"] if hamming_dist_entry else 1

    # Cosine similarity is defined only for non-integer element types, so an
    # integer representation is legal only on the Hamming path.
    if data_type not in FLOAT_TYPES and hamming_dist != 1:
        print(f"Skipping invalid combination: DATA_TYPE={data_type} requires HAMMING_DIST=1")
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

    binarize_encoding_entry = next((item for item in pass_config_json if "BINARIZE_ENCODING" in item), None)
    binarize_encoding = binarize_encoding_entry["BINARIZE_ENCODING"] if binarize_encoding_entry else 1

    pass_config_json = [item for item in pass_config_json if "intrinsic" in item]

    with open(config_in_path, "w+") as file:
        json.dump(pass_config_json, file)

    subprocess.run(["make", f"--directory={root_dir_str}",
        "clean"], env=my_env)

    compile_start = time.time()
    make_target = "host-gpu-approx" if device_type == "gpu" else "host-cpu-approx"
    make_cmd = ["make", f"--directory={root_dir_str}",
        make_target, "-B",
        "HPVM_BUILD=release",
        "APPROX_CONFIG=1",  # this config is intentional; see util/check_approx_config.py
        f"ENCODING_DIM={dhv}",
        f"DATA_TYPE={data_type}",
        f"HAMMING_DIST={hamming_dist}",
        f"BINARIZE_ENCODING={binarize_encoding}",
        f"N_FEAT={dataset_n_feat}",
        f"N_SAMPLE={dataset_n_sample}",
        f"N_TEST={dataset_n_test}",
        f"N_CLASS={dataset_n_class}",
        f"N_FEAT_PAD={dataset_n_feat_pad}"]
    if no_approx_hint:
        make_cmd.append("NO_APPROX_HINT=1")
    compile_result = subprocess.run(make_cmd, env=my_env)
    compile_end = time.time()

    compile_time = compile_end - compile_start
    if compile_result.returncode != 0:
        print("Compilation failed — skipping this configuration.")
        stats_dump = root_dir_str + "/output/stats.txt"
        with open(stats_dump, "a") as f:
            f.write(f"inf, 0.0, {compile_time}, 0.0, 0.0\n")
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0,
                "compile_time": compile_time, "training_time": 0.0, "inference_time": 0.0}

    print("Running Application:")

    monitor = GPUPowerMonitor(gpu_id=0, sampling_interval=0.1)
    time.sleep(5)

    print("Starting GPU power monitoring...")
    monitor.start()
    start = time.time()

    executable = "./host-gpu-approx" if device_type == "gpu" else "./host-cpu-approx"
    run_result = subprocess.run([executable, f"{train_iterations}",
                    dataset_x_train, dataset_y_train, dataset_x_test, dataset_y_test],
                   cwd=f"{root_dir_str}", env=my_env)
    end = time.time()

    print("Stopping monitoring...")
    stats = monitor.stop()

    print(stats)
    energy_ = stats['total_energy_joules'] if stats is not None else 0.0

    score_file = root_dir_str + "/output/test_accuracy.txt"
    time_file = root_dir_str + "/output/test_time.txt"

    try:
        with open(time_file, "r") as f:
            time_ = float(f.readline().strip())
            training_time = float(f.readline().strip())
            inference_time = float(f.readline().strip())
        with open(time_file, "w") as f:
            pass

        with open(score_file, "r") as f:
            accuracy = float(f.readline().strip())
        with open(score_file, "w") as f:
            pass
    except (ValueError, FileNotFoundError):
        print("Failed to read output files (binary likely crashed) — skipping this configuration.")
        stats_dump = root_dir_str + "/output/stats.txt"
        with open(stats_dump, "a") as f:
            f.write(f"inf, 0.0, {compile_time}, 0.0, 0.0\n")
        return {'time': float('inf'), "accuracy": 0.0, "energy": energy_,
                "compile_time": compile_time, "training_time": 0.0, "inference_time": 0.0}

    stats_dump = root_dir_str + "/output/stats.txt"
    with open(stats_dump, "a") as f:
        f.write(f"{time_}, {accuracy}, {compile_time}, {training_time}, {inference_time}\n")

    time_ = training_time + inference_time
    print("time:\t", time_, "\taccuracy: ", accuracy)
    return {'time': time_, "accuracy": accuracy, "energy": energy_, "compile_time": compile_time, "training_time": training_time, "inference_time": inference_time}

if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    _results_dbs = os.path.join(root_dir_str, "../../results/dbs")
    os.makedirs(_results_dbs, exist_ok=True)
    argparser.set_defaults(database=os.path.join(_results_dbs, "hd_classification_opentuner.db"))
    argparser.add_argument('--dataset', type=str,
                          default='ISOLET',
                          choices=list(DATASETS.keys()),
                          help='Dataset to use')
    argparser.add_argument('--device', type=str, choices=['cpu', 'gpu'],
                          default='cpu',
                          help='Device to use (cpu or gpu)')
    argparser.add_argument('--accuracy-target', type=float,
                          default=0.85,
                          help='Target accuracy threshold')
    argparser.add_argument('--direct-eval', type=str, metavar='CONFIG_FILE',
                          default=None,
                          help='Directly evaluate a configuration file without tuning')
    argparser.add_argument('--collect-approx', action='store_true',
                          default=False,
                          help='Collect approximation opportunities and exit')
    argparser.add_argument('--no-redperf', action='store_true',
                          default=False,
                          help='Disable tuning over reduction perforation')
    argparser.add_argument('--redperf-end-only', action='store_true',
                          default=False,
                          help='Tune only the end (extent) of reduction perforation, keeping base and stride fixed')
    argparser.add_argument('--no-algchoice', action='store_true',
                          default=False,
                          help='Disable tuning over algorithmic choice')
    argparser.add_argument('--enable-no-approx-hint', action='store_true',
                          default=False,
                          help='Pass NO_APPROX_HINT=1 macro to compilation')

    args = argparser.parse_args()

    # Set global variables from args
    paths = DATASETS[args.dataset]
    dataset_x_train    = paths["X_train"]
    dataset_y_train    = paths["y_train"]
    dataset_x_test     = paths["X_test"]
    dataset_y_test     = paths["y_test"]
    dataset_n_feat     = paths["n_feat"]
    dataset_n_sample   = paths["n_sample"]
    dataset_n_test     = paths["n_test"]
    dataset_n_class    = paths["n_class"]
    dataset_n_feat_pad = paths["n_feat_pad"]
    device_type = args.device
    no_approx_hint = args.enable_no_approx_hint

    if args.direct_eval:
        config_file = args.direct_eval
        with open(config_file, "r") as f:
            cfg = f.read()
            cfg = ast.literal_eval(cfg)
            result = eval_config(cfg)
            print("Result:", result)
            sys.exit()
        
    # Always regenerate config_out.json so the knob space matches the current
    # hint mode. Without this, a stale config_out left over from an earlier
    # run (potentially in the opposite hint mode) silently changes the
    # search-space size reported by --print-search-space-size and the knobs
    # passed to OpenTuner.
    collect_cmd = ["make", f"--directory={root_dir_str}",
        "collect-approx", "-B",
        "HPVM_BUILD=release",
        "ENCODING_DIM=2048",
        "DATA_TYPE=float",
        "HAMMING_DIST=0",
        "BINARIZE_ENCODING=1",
        "N_FEAT=617",
        "N_SAMPLE=6238",
        "N_TEST=1559",
        "N_CLASS=26",
        "N_FEAT_PAD=624"]
    if args.enable_no_approx_hint:
        collect_cmd.append("NO_APPROX_HINT=1")
    subprocess.run(collect_cmd, env=my_env, check=True)
    if args.collect_approx:
        sys.exit()

    # Create opentuner argparser and add custom arguments

    # Register approximation description and eval config to arg parse object
    enable_redperf = not args.no_redperf
    enable_algchoice = not args.no_algchoice
    redperf_end_only = args.redperf_end_only
    apx_descs = get_apx_descs(enable_redperf=enable_redperf, enable_algchoice=enable_algchoice, redperf_end_only=redperf_end_only)

    # breakpoint()

    args.approx_descs = apx_descs
    args.evaluate_config_fn = eval_config
    args.num_knobs = len(apx_descs)
    # Provide path to baseline/seed configuration
    args.seed_config_path = root_dir_str + "/config/baseline_config.json"

    
    print(apx_descs)
    # breakpoint()

    tuner_harness = OpenTunerHarness(args)
    tuner_harness.main(args)
