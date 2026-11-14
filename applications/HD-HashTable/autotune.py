from hdc_approximation_tuner.Harness import OpenTunerHarness, process_opentuner_config
from hdc_approximation_tuner.ParseHDCOpDesc import parse_hdc_op
from hdc_approximation_tuner.ApproxTypes import ApproxEncodingDimension, ApproxCustomInteger, ApproxCustomFloat
import json
import opentuner
import time
import subprocess
import os
import ast
import sys

from gpu_power import *

root_dir_str = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-HashTable"
config_out_path = root_dir_str + "/config/config_out.json"
config_in_path  = root_dir_str + "/config/config_in.json"

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


# Global variables set by command-line arguments
device_type = None
no_approx_hint = False
num_runs = 3

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

    DHv = 32768

    apx_descs.append(
        ApproxEncodingDimension(8, DHv, True)
    )

    # Reduction perforation knobs: STRIDE (step over D dimensions) and END
    # (fraction of D to compute) in the query dot product.
    # Controlled by --no-redperf / --redperf-end-only, same as other apps.
    if enable_redperf:
        if not redperf_end_only:
            apx_descs.append(
                ApproxCustomInteger("STRIDE", DHv - 1)
            )
        apx_descs.append(
            ApproxCustomFloat("END", 0.0, 1.0)
        )

    # KMER_STRIDE defines the problem layout; not tuned. Fixed at 1.
    apx_descs.append(
        ApproxCustomInteger("KMERS_PER_HV", 2000)
    )

    apx_descs.append(
        ApproxCustomFloat("THRESHOLD_MULT", 0.0, 1.0)
    )

    apx_descs.append(
        ApproxCustomInteger("DATA_TYPE", len(DATA_TYPES) - 1)
    )

    apx_descs.append(
        ApproxCustomInteger("BINARIZE_BUCKETS", 1)  # 0=off, 1=on
    )

    return apx_descs


def eval_config(cfg):
    global device_type, no_approx_hint, num_runs
    print("HDC Configuration:", cfg)
    print("\n")
    pass_config_json = process_opentuner_config(cfg)
    print("Json:", pass_config_json)
    print("\n")

    dhv_entry = next((item for item in pass_config_json if "EncodingDim" in item), None)
    dhv = dhv_entry["EncodingDim"]

    stride_entry = next((item for item in pass_config_json if "STRIDE" in item), None)
    stride = stride_entry["STRIDE"] + 1 if stride_entry else 1

    end_entry = next((item for item in pass_config_json if "END" in item), None)
    query_end = end_entry["END"] if end_entry else 1.0

    kmer_stride = 1

    kmers_entry = next((item for item in pass_config_json if "KMERS_PER_HV" in item), None)
    kmers_per_hv = kmers_entry["KMERS_PER_HV"] if kmers_entry else 400

    thresh_entry = next((item for item in pass_config_json if "THRESHOLD_MULT" in item), None)
    threshold_mult = thresh_entry["THRESHOLD_MULT"] if thresh_entry else 1.0

    data_type_entry = next((item for item in pass_config_json if "DATA_TYPE" in item), None)
    data_type_raw = data_type_entry["DATA_TYPE"] if data_type_entry else 1
    data_type = DATA_TYPES[min(data_type_raw, len(DATA_TYPES) - 1)]

    binarize_entry = next((item for item in pass_config_json if "BINARIZE_BUCKETS" in item), None)
    binarize_buckets = binarize_entry["BINARIZE_BUCKETS"] if binarize_entry else 0

    # if stride > dhv:
    #     print(f"Skipping invalid combination: STRIDE={stride} > ENCODING_DIM={dhv}")
    #     return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

    pass_config_json = [item for item in pass_config_json if "intrinsic" in item]

    with open(config_in_path, "w+") as file:
        json.dump(pass_config_json, file)

    # Reject degenerate KMERS_PER_HV before compile: 0 causes division-by-zero
    # in the N_HVS macro, and the compile silently fails leaving a stale binary.
    if kmers_per_hv < 1:
        print(f"Skipping invalid KMERS_PER_HV={kmers_per_hv} (must be >= 1)")
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

    subprocess.run(["make", f"--directory={root_dir_str}",
        "clean"], env=my_env)

    compile_start = time.time()
    make_target = "host-gpu-batched-typed" if device_type == "gpu" else "host-cpu-batched-typed"

    # Remove the target binary explicitly so a failed compile can't run a stale one.
    binary_path = os.path.join(root_dir_str, make_target)
    if os.path.exists(binary_path):
        os.remove(binary_path)

    make_cmd = ["make", f"--directory={root_dir_str}",
        make_target, "-B",
        "HPVM_BUILD=release",
        "APPROX_CONFIG=1",  # this config is intentional; see util/check_approx_config.py
        f"ENCODING_DIM={dhv}",
        f"STRIDE={stride}",
        f"QUERY_END={query_end}",
        f"KMER_STRIDE={kmer_stride}",
        f"KMERS_PER_HV={kmers_per_hv}",
        f"THRESHOLD_MULT={threshold_mult}",
        f"DATA_TYPE={data_type}",
        f"BINARIZE_BUCKETS={binarize_buckets}"]
    run_res = subprocess.run(make_cmd, env=my_env)
    compile_end = time.time()

    if run_res.returncode != 0:
        print(f"Compilation exited with non-zero code: {run_res.returncode}")
        compile_time = compile_end - compile_start
        stats_dump = root_dir_str + "/stats.txt"
        with open(stats_dump, "a") as f:
            f.write(f"inf, 0.0, {compile_time}\n")
        print("Compilation failed — skipping this configuration.")
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

    # Verify the binary was actually produced (guards against partial builds).
    if not os.path.exists(binary_path):
        print(f"Binary {make_target} not found after successful make — skipping.")
        compile_time = compile_end - compile_start
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

    print("Running Application:")

    times_ = []
    accuracies = []

    # Generous timeout: unapproximated CPU at D=8192 takes well above the
    # 9s used for GPU-only tuning. 180s still protects against degenerate
    # infinite-loop configs but lets the baseline and near-baseline CPU
    # configs finish honestly.
    run_timeout_s = 180

    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        executable = "./host-gpu-batched-typed" if device_type == "gpu" else "./host-cpu-batched-typed"
        try:
            subprocess.run([executable], cwd=f"{root_dir_str}", env=my_env,
                           timeout=run_timeout_s)
        except subprocess.TimeoutExpired:
            print(f"Run {i+1} exceeded {run_timeout_s}s timeout — killing and skipping config.")
            subprocess.run(["pkill", "-9", "-f", f"{executable}"])
            compile_time = compile_end - compile_start
            stats_dump = root_dir_str + "/stats.txt"
            with open(stats_dump, "a") as f:
                f.write(f"inf, 0.0, {compile_time}, timeout\n")
            return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

        results_file = root_dir_str + "/results.txt"
        try:
            with open(results_file, "r") as f:
                accuracy = float(f.readline().strip())
                time_ms  = float(f.readline().strip())
        except (ValueError, FileNotFoundError):
            print(f"Run {i+1} produced no results (likely crashed) — skipping config.")
            compile_time = compile_end - compile_start
            return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}
        with open(results_file, "w") as f:
            pass

        accuracies.append(accuracy)
        times_.append(time_ms)

    avg_time = sum(times_) / len(times_) if times_ else 0.0
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    compile_time = compile_end - compile_start

    stats_dump = root_dir_str + "/stats.txt"
    with open(stats_dump, "a") as f:
        f.write(f"{avg_time}, {avg_accuracy}, {compile_time},\n")

    print(f"Averages ({num_runs} runs) -> time: {avg_time}, accuracy: {avg_accuracy}")
    return {'time': avg_time, "accuracy": avg_accuracy, "energy": 0.0}


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    _results_dbs = os.path.join(root_dir_str, "../../results/dbs")
    os.makedirs(_results_dbs, exist_ok=True)
    argparser.set_defaults(database=os.path.join(_results_dbs, "hd_hashtable_multihv_opentuner.db"))
    argparser.add_argument('--device', type=str, choices=['cpu', 'gpu'],
                          default='cpu',
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
                          help='Tune only the end (extent) of reduction perforation')
    argparser.add_argument('--no-algchoice', action='store_true',
                          default=False,
                          help='Disable tuning over algorithmic choice')
    argparser.add_argument('--enable-no-approx-hint', action='store_true',
                          default=False,
                          help='Pass NO_APPROX_HINT=1 macro to compilation')
    argparser.add_argument('--num-runs', type=int, default=5,
                          help='Number of repeated runs per config for averaging (default: 5)')

    args = argparser.parse_args()

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
    # hint mode. CLEAR_CONFIG_OUT=1 drops the stale HPVM-level matmul op so
    # that only the custom knobs (STRIDE, QUERY_END, KMER_STRIDE, etc.) remain.
    collect_cmd = ["make", f"--directory={root_dir_str}",
        "collect-approx", "-B",
        "ENCODING_DIM=4096",
        "CLEAR_CONFIG_OUT=1"]
    if args.enable_no_approx_hint:
        collect_cmd.append("NO_APPROX_HINT=1")
    subprocess.run(collect_cmd, env=my_env, check=True)
    if args.collect_approx:
        sys.exit()

    enable_redperf   = not args.no_redperf
    enable_algchoice = not args.no_algchoice
    redperf_end_only = args.redperf_end_only
    apx_descs = get_apx_descs(enable_redperf=enable_redperf, enable_algchoice=enable_algchoice, redperf_end_only=redperf_end_only)

    args.approx_descs        = apx_descs
    args.evaluate_config_fn  = eval_config
    args.num_knobs           = len(apx_descs)
    args.seed_config_path    = root_dir_str + "/config/baseline_config.json"

    print(apx_descs)

    tuner_harness = OpenTunerHarness(args)
    tuner_harness.main(args)
