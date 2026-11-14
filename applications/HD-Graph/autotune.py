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

from gpu_power import *

root_dir_str = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Graph"
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

# Global variables set by command-line arguments
device_type = None
no_approx_hint = False
num_runs = 5

my_env = os.environ.copy()
my_env["CONFIG_IN_PATH"] = config_in_path
my_env["CONFIG_OUT_PATH"] = config_out_path

def get_apx_descs(enable_redperf=True, enable_algchoice=True, redperf_end_only=False):
    try:
        with open(config_out_path, "r") as file:
            try:
                test_json = json.load(file)
            except json.JSONDecodeError:
                print("Warning: config_out.json is empty/invalid. Proceeding with no per-op approximations.")
                test_json = []
    except FileNotFoundError:
        print("Warning: config_out.json not found. Proceeding with no per-op approximations.")
        test_json = []

    operations = test_json if isinstance(test_json, list) else []
    if not operations:
        print("No operations found in config_out.json; using only global knobs.")


    print("OPERATIONS: ", operations)

    apx_descs = [parse_hdc_op(dict_, enable_redperf=enable_redperf, enable_algchoice=enable_algchoice, redperf_end_only=redperf_end_only) for dict_ in operations]

    # Global Configurations

    # # Encoding Dimension
    DHv = 4096

    apx_descs.append(
        ApproxEncodingDimension(8, DHv, True) # True: Use only powers of 2, False otherwise; min=8 to avoid HPVM codegen bug at D=1 (| instead of + in loop increment → infinite GPU loop)
    )

    apx_descs.append(ApproxCustomInteger("TRAIN_ITERATIONS", 20))

    apx_descs.append(
        ApproxCustomInteger("HAMMING_DIST", 2)
    )

    apx_descs.append(
        ApproxCustomInteger("BINARIZE", 2)
    )

    apx_descs.append(
        ApproxCustomInteger("RHO_SHIFT", 2)
    )

    apx_descs.append(
        ApproxCustomInteger("POLARIZE", 2)
    )

    apx_descs.append(
        ApproxCustomInteger("DATA_TYPE", len(DATA_TYPES) - 1)
    )

    return apx_descs
def eval_config(cfg):
    global device_type, no_approx_hint, num_runs
    print("HDC Configuration:", cfg)
    print("\n")
    pass_config_json = process_opentuner_config(cfg)
    print("Json:", pass_config_json)
    print("\n")

    # Extract Enc Dimension
    dhv_entry = next((item for item in pass_config_json if "EncodingDim" in item), None)
    dhv = dhv_entry["EncodingDim"]

    train_entry = next((item for item in pass_config_json if "TRAIN_ITERATIONS" in item), None)
    train_iterations = train_entry["TRAIN_ITERATIONS"] if train_entry else 1

    data_type_entry = next((item for item in pass_config_json if "DATA_TYPE" in item), None)
    data_type = DATA_TYPES[min(data_type_entry["DATA_TYPE"], len(DATA_TYPES) - 1)] \
        if data_type_entry else "float"

    hamming_dist_entry = next((item for item in pass_config_json if "HAMMING_DIST" in item), None)
    hamming_dist = hamming_dist_entry["HAMMING_DIST"] if hamming_dist_entry else 1

    binarize_entry = next((item for item in pass_config_json if "BINARIZE" in item), None)
    binarize = binarize_entry["BINARIZE"] if binarize_entry else 0

    rho_shift_entry = next((item for item in pass_config_json if "RHO_SHIFT" in item), None)
    rho_shift = rho_shift_entry["RHO_SHIFT"] if rho_shift_entry else 0

    polarize_entry = next((item for item in pass_config_json if "POLARIZE" in item), None)
    polarize = polarize_entry["POLARIZE"] if polarize_entry else 0

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
        f"HAMMING_DIST={hamming_dist}",
        f"DATA_TYPE={data_type}",
        f"BINARIZE={binarize}",
        f"RHO_SHIFT={rho_shift}",
        f"POLARIZE={polarize}"]
    if no_approx_hint:
        make_cmd.append("NO_APPROX_HINT=1")
    run_res = subprocess.run(make_cmd, env=my_env)
    compile_end = time.time()

    if run_res.returncode != 0:
        print(f"Application exited with non-zero code: {run_res.returncode}")
        print("Stopping monitoring...")
        # stats = monitor.stop()

        compile_time = compile_end - compile_start
        stats_dump = root_dir_str + "/stats.txt"
        with open(stats_dump, "a") as f:
            f.write(f"inf, 0.0, {compile_time}\n")

        print("Compilation failed — skipping this configuration.")
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

    print("Running Application:")

    energies = []
    times_ = []
    accuracies = []

    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        monitor = GPUPowerMonitor(gpu_id=0, sampling_interval=0.1)
        time.sleep(5)
        monitor.start()

        executable = "./host-gpu-approx" if device_type == "gpu" else "./host-cpu-approx"
        # Timeout: 2× the baseline wall time to catch GPU hangs at extreme configs
        RUN_TIMEOUT_S = 4000
        try:
            subprocess.run([executable, f"{train_iterations}"], cwd=f"{root_dir_str}",
                           env=my_env, timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"WARNING: binary timed out after {RUN_TIMEOUT_S}s — treating as penalty")
            subprocess.run(["pkill", "-f", executable], capture_output=True)
            return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

        print("Stopping monitoring...")
        stats = monitor.stop()

        # Energy (Joules)
        energies.append(stats.get('total_energy_joules', 0.0) if stats is not None else 0.0)

        # Time
        time_file = root_dir_str + "/test_time.txt"
        with open(time_file, "r") as f:
            line = f.readline().strip()
            times_.append(float(line) if line else 0.0)
        with open(time_file, "w"):
            pass

        # Accuracy
        score_file = root_dir_str + "/test_accuracy.txt"
        with open(score_file, "r") as f:
            line = f.readline().strip()
            accuracies.append(float(line) if line else 0.0)
        with open(score_file, "w"):
            pass

    avg_time = sum(times_) / len(times_) if times_ else 0.0
    avg_energy = sum(energies) / len(energies) if energies else 0.0
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    total_energy = sum(energies)
    total_time = sum(times_)
    avg_power_watts = (total_energy / total_time) if total_time > 0 else 0.0

    compile_time = compile_end - compile_start
    stats_dump = root_dir_str + "/stats.txt"
    with open(stats_dump, "a") as f:
        f.write(f"{avg_time}, {avg_accuracy}, {compile_time},\n")

    print(f"Averages ({num_runs} runs) -> time: {avg_time}, accuracy: {avg_accuracy}, energy/run: {avg_energy}, avg_power: {avg_power_watts} W")
    return {'time': avg_time, "accuracy": avg_accuracy, "energy": avg_energy}

if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    _results_dbs = os.path.join(root_dir_str, "../../results/dbs")
    os.makedirs(_results_dbs, exist_ok=True)
    argparser.set_defaults(database=os.path.join(_results_dbs, "hd_graph_opentuner.db"))
    argparser.add_argument('--device', type=str, choices=['cpu', 'gpu'],
                          default='gpu',
                          help='Device to use (cpu or gpu)')
    argparser.add_argument('--accuracy-target', type=float,
                          default=0.70,
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
    argparser.add_argument('--num-runs', type=int, default=5,
                          help='Number of repeated runs per config for averaging (default: 5, use 1 to skip repeats)')

    args = argparser.parse_args()

    # Set global variables from args
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

    if args.collect_approx:
        subprocess.run(["make", f"--directory={root_dir_str}",
        "collect-approx", "-B",
        f"ENCODING_DIM=1024"], env=my_env)
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

    tuner_harness = OpenTunerHarness(args)
    tuner_harness.main(args)
