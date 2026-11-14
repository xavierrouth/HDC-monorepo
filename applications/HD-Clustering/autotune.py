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

root_dir_str = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Clustering"
VENV_PYTHON  = os.environ.get('HDC_VENV_PYTHON', "/home/xrouth2/hdc/HDC-monorepo/approx-tuner/venv/bin/python3.11")
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
        "X": "../../datasets/clustering/isolet_train_trainX.bin",
        "Y": "../../datasets/clustering/isolet_train_trainY.bin",
        "n_feat":    624,
        "n_sample":  6238,
        "n_center":  26,
    },
    "MNIST": {
        "X": "../../datasets/clustering/mnist_testX.bin",
        "Y": "../../datasets/clustering/mnist_testY.bin",
        "n_feat":    784,
        "n_sample":  10000,
        "n_center":  10,
    },
    "UCIHAR": {
        "X": "../../datasets/clustering/ucihar_trainX.bin",
        "Y": "../../datasets/clustering/ucihar_trainY.bin",
        "n_feat":    568,
        "n_sample":  6213,
        "n_center":  12,
    },
}

# Global variables set by command-line arguments
dataset_x = None
dataset_y = None
dataset_n_feat   = None
dataset_n_sample = None
dataset_n_center = None
device_type = None
num_runs = 1
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
    DHv = 1024

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
    global dataset_x, dataset_y, dataset_n_feat, dataset_n_sample, dataset_n_center, device_type, no_approx_hint, num_runs
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

    entry = next((item for item in pass_config_json if "DATA_TYPE" in item), None)
    data_type = DATA_TYPES[min(entry["DATA_TYPE"], len(DATA_TYPES) - 1)]

    entry = next((item for item in pass_config_json if "HAMMING_DIST" in item), None)
    hamming_dist = entry["HAMMING_DIST"]

    if hamming_dist == 1:
        hamming_dist = "1"
    else:
        hamming_dist = "0"

    # Integer element types require HAMMING_DIST=1: cosine similarity is
    # float-only, and the combination fails to compile rather than merely
    # scoring poorly. Skip it here instead of paying for a doomed build.
    if data_type not in FLOAT_TYPES and hamming_dist == "0":
        print(f"Skipping invalid combination: DATA_TYPE={data_type} requires HAMMING_DIST=1")
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

    pass_config_json = [item for item in pass_config_json if "intrinsic" in item]

    with open(config_in_path, "w+") as file:
        json.dump(pass_config_json, file)

    subprocess.run(["make", f"--directory={root_dir_str}", 
        "clean"], env=my_env)

    compile_start = time.time()
    make_target = "host-gpu-approx" if device_type == "gpu" else "host-cpu-approx"
    make_cmd = ["make", f"--directory={root_dir_str}",
        make_target, "-B",
        "APPROX_CONFIG=1",  # this config is intentional; see util/check_approx_config.py
        f"ENCODING_DIM={dhv}",
        f"HAMMING_DIST={hamming_dist}",
        f"DATA_TYPE={data_type}",
        f"N_FEAT={dataset_n_feat}",
        f"N_SAMPLE={dataset_n_sample}",
        f"N_CENTER={dataset_n_center}"]
    if no_approx_hint:
        make_cmd.append("NO_APPROX_HINT=1")
    compile_result = subprocess.run(make_cmd, env=my_env)
    compile_end = time.time()

    if compile_result.returncode != 0:
        print("Compilation failed — skipping this configuration.")
        compile_time = compile_end - compile_start
        stats_dump = root_dir_str + "/stats.txt"
        with open(stats_dump, "a") as f:
            f.write(f"inf, 0.0, {compile_time}\n")
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

    print("Running Application:")

    executable = "./host-gpu-approx" if device_type == "gpu" else "./host-cpu-approx"
    score_script = root_dir_str + "/mutual_info.py"
    predictions_file = root_dir_str + "/output/out.txt"
    score_file = root_dir_str + "/score.txt"

    times_ = []
    accuracies = []
    energies = []

    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        monitor = GPUPowerMonitor(gpu_id=0, sampling_interval=0.1)
        time.sleep(5)
        monitor.start()
        start = time.time()
        try:
            subprocess.run([executable, f"{train_iterations}", dataset_x, dataset_y],
                           cwd=f"{root_dir_str}", env=my_env)
        except FileNotFoundError:
            print(f"Executable {executable} not found — skipping this configuration.")
            monitor.stop()
            return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}
        end = time.time()
        stats = monitor.stop()

        energies.append(stats['total_energy_joules'] if stats is not None else 0.0)
        times_.append(end - start)

        with open(f"{score_file}", "w") as f:
            subprocess.run([VENV_PYTHON, score_script, predictions_file],
                           stdout=f, stderr=subprocess.STDOUT, env=my_env)
        try:
            with open(score_file, "r") as f:
                accuracies.append(float(f.readline().strip()))
        except (ValueError, FileNotFoundError):
            print("Failed to read accuracy from score.txt — skipping this configuration.")
            return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}
        with open(score_file, "w"):
            pass

    avg_time     = sum(times_) / len(times_) if times_ else 0.0
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    avg_energy   = sum(energies) / len(energies) if energies else 0.0

    compile_time = compile_end - compile_start
    stats_dump = root_dir_str + "/stats.txt"
    with open(stats_dump, "a") as f:
        f.write(f"{avg_time}, {avg_accuracy}, {compile_time}\n")

    print(f"Averages ({num_runs} runs) -> time: {avg_time}, accuracy: {avg_accuracy}, energy/run: {avg_energy} J")
    return {'time': avg_time, "accuracy": avg_accuracy, "energy": avg_energy}

if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    _results_dbs = os.path.join(root_dir_str, "../../results/dbs")
    os.makedirs(_results_dbs, exist_ok=True)
    argparser.set_defaults(database=os.path.join(_results_dbs, "hd_clustering_opentuner.db"))
    argparser.add_argument('--dataset', type=str,
                          default='ISOLET',
                          choices=list(DATASETS.keys()),
                          help='Dataset to use')
    argparser.add_argument('--device', type=str, choices=['cpu', 'gpu'],
                          default='cpu',
                          help='Device to use (cpu or gpu)')
    argparser.add_argument('--accuracy-target', type=float,
                          default=0.62,
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
    argparser.add_argument('--num-runs', type=int, default=1,
                          help='Number of repeated runs per config for averaging (default: 1)')

    args = argparser.parse_args()

    # Set global variables from args
    paths = DATASETS[args.dataset]
    dataset_x       = paths["X"]
    dataset_y       = paths["Y"]
    dataset_n_feat   = paths["n_feat"]
    dataset_n_sample = paths["n_sample"]
    dataset_n_center = paths["n_center"]
    device_type = args.device
    no_approx_hint = args.enable_no_approx_hint
    num_runs = args.num_runs

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
        f"ENCODING_DIM=1024",
        f"HAMMING_DIST=0",
        f"DATA_TYPE=float",
        f"N_FEAT={dataset_n_feat}",
        f"N_SAMPLE={dataset_n_sample}",
        f"N_CENTER={dataset_n_center}"]
    if args.enable_no_approx_hint:
        collect_cmd.append("NO_APPROX_HINT=1")
    subprocess.run(collect_cmd, env=my_env, check=True)
    if args.collect_approx:
        sys.exit()

    # Register approximation description and eval config to arg parse object
    enable_redperf = not args.no_redperf
    enable_algchoice = not args.no_algchoice
    redperf_end_only = args.redperf_end_only
    apx_descs = get_apx_descs(enable_redperf=enable_redperf, enable_algchoice=enable_algchoice, redperf_end_only=redperf_end_only)

    args.approx_descs = apx_descs
    args.evaluate_config_fn = eval_config
    args.num_knobs = len(apx_descs)
    # Provide path to baseline/seed configuration
    args.seed_config_path = root_dir_str + "/config/baseline_config.json"

    
    print(apx_descs)
    # breakpoint()

    tuner_harness = OpenTunerHarness(args)
    tuner_harness.main(args)
