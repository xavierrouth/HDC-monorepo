from hdc_approximation_tuner.Harness import OpenTunerHarness, process_opentuner_config
from hdc_approximation_tuner.ParseHDCOpDesc import parse_hdc_op
from hdc_approximation_tuner.ApproxTypes import ApproxEncodingDimension, ApproxCustomInteger
import json
import opentuner
import time
import subprocess
import os
import ast
import sys

root_dir_str = "/home/xrouth2/hdc/HDC-monorepo/applications/Micro-HD-Classification"
config_out_path = root_dir_str + "/config/config_out.json"
config_in_path  = root_dir_str + "/config/config_in.json"

# Global variables set by command-line arguments
device_type = None
tuning_start_time = None

my_env = os.environ.copy()
my_env["CONFIG_IN_PATH"] = config_in_path
my_env["CONFIG_OUT_PATH"] = config_out_path

# Dataset: ISOLET only
N_FEAT   = 617
N_CLASS  = 26
N_SAMPLE = 6238
N_TEST   = 1559
X_TRAIN  = "../../datasets/classification/isolet_nq_trainX.bin"
Y_TRAIN  = "../../datasets/classification/isolet_nq_trainY.bin"
X_TEST   = "../../datasets/classification/isolet_nq_testX.bin"
Y_TEST   = "../../datasets/classification/isolet_nq_testY.bin"


def get_apx_descs(enable_redperf=True, enable_algchoice=True, redperf_end_only=False):
    try:
        with open(config_out_path, "r") as f:
            try:
                test_json = json.load(f)
            except json.JSONDecodeError:
                print("Warning: config_out.json empty/invalid; no per-op approximations.")
                test_json = []
    except FileNotFoundError:
        print("Warning: config_out.json not found; no per-op approximations.")
        test_json = []

    operations = test_json if isinstance(test_json, list) else []
    print("OPERATIONS:", operations)

    apx_descs = [
        parse_hdc_op(d, enable_redperf=enable_redperf,
                     enable_algchoice=enable_algchoice,
                     redperf_end_only=redperf_end_only)
        for d in operations
    ]

    # ENCODING_DIM: powers of 2 from 1 to 16384 (covers MicroHD's 10000 baseline)
    apx_descs.append(ApproxEncodingDimension(1, 16384, True))

    # N_LEVELS: integer 1..1024 (MicroHD admissible: powers of 2 up to 1024)
    apx_descs.append(ApproxCustomInteger("N_LEVELS", 1024))

    # ENCODER_TYPE: 0 = IDLev, 1 = RP. N_LEVELS is ignored when ENCODER_TYPE=1.
    apx_descs.append(ApproxCustomInteger("ENCODER_TYPE", 1))

    return apx_descs


def eval_config(cfg):
    global device_type, tuning_start_time
    print("HDC Configuration:", cfg)
    print()
    pass_config_json = process_opentuner_config(cfg)
    print("Json:", pass_config_json)
    print()

    dhv_entry = next((item for item in pass_config_json if "EncodingDim" in item), None)
    dhv = dhv_entry["EncodingDim"]

    nlevels_entry = next((item for item in pass_config_json if "N_LEVELS" in item), None)
    n_levels = max(2, nlevels_entry["N_LEVELS"]) if nlevels_entry else 128

    enc_entry = next((item for item in pass_config_json if "ENCODER_TYPE" in item), None)
    encoder_type = enc_entry["ENCODER_TYPE"] if enc_entry else 0

    # Keep only HPVM intrinsic ops for config_in.json
    pass_config_json = [item for item in pass_config_json if "intrinsic" in item]
    with open(config_in_path, "w+") as f:
        json.dump(pass_config_json, f)

    subprocess.run(["make", f"--directory={root_dir_str}", "clean"], env=my_env)

    compile_start = time.time()
    make_target = "host-gpu-approx" if device_type == "gpu" else "host-cpu-approx"
    make_cmd = [
        "make", f"--directory={root_dir_str}", make_target, "-B",
        f"ENCODING_DIM={dhv}",
        f"N_LEVELS={n_levels}",
        f"ENCODER_TYPE={encoder_type}",
        f"N_FEAT={N_FEAT}",
        f"N_CLASS={N_CLASS}",
        f"N_SAMPLE={N_SAMPLE}",
        f"N_TEST={N_TEST}",
    ]
    run_res = subprocess.run(make_cmd, env=my_env)
    compile_end = time.time()
    compile_time = compile_end - compile_start

    if run_res.returncode != 0:
        print(f"Compilation failed (exit {run_res.returncode}); skipping.")
        _log_result(float('inf'), 0.0, compile_time, float('inf'), float('inf'))
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

    executable = "./host-gpu-approx" if device_type == "gpu" else "./host-cpu-approx"
    subprocess.run(
        [executable, "2", X_TRAIN, Y_TRAIN, X_TEST, Y_TEST],
        cwd=root_dir_str, env=my_env
    )

    time_file     = root_dir_str + "/output/test_time.txt"
    accuracy_file = root_dir_str + "/output/test_accuracy.txt"

    with open(time_file, "r") as f:
        _total_ms  = float(f.readline().strip())
        train_time = float(f.readline().strip())
        infer_time = float(f.readline().strip())
    with open(time_file, "w"):
        pass

    with open(accuracy_file, "r") as f:
        accuracy = float(f.readline().strip())
    with open(accuracy_file, "w"):
        pass

    total_time = train_time + infer_time
    _log_result(total_time, accuracy, compile_time, train_time, infer_time)
    print(f"time: {total_time:.1f}ms  accuracy: {accuracy:.4f}")
    return {'time': total_time, "accuracy": accuracy, "energy": 0.0}


def _log_result(total_time_ms, accuracy, compile_time_s, train_time_ms, infer_time_ms):
    """Append a row to output/tuning_log.csv with a wall-clock timestamp."""
    global tuning_start_time
    wall_clock_s = time.time() - tuning_start_time if tuning_start_time else float('nan')
    log_path = root_dir_str + "/output/tuning_log.csv"
    write_header = not os.path.exists(log_path)
    with open(log_path, "a") as f:
        if write_header:
            f.write("wall_clock_s,accuracy,exec_time_ms,compile_time_s,train_time_ms,infer_time_ms\n")
        f.write(f"{wall_clock_s:.2f},{accuracy:.6f},{total_time_ms:.2f},"
                f"{compile_time_s:.2f},{train_time_ms:.2f},{infer_time_ms:.2f}\n")


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    _results_dbs = os.path.join(root_dir_str, "../../results/dbs")
    os.makedirs(_results_dbs, exist_ok=True)
    argparser.set_defaults(
        database=os.path.join(_results_dbs, "micro_hd_classification_opentuner.db")
    )
    argparser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu')
    argparser.add_argument('--accuracy-target', type=float, default=0.82)
    argparser.add_argument('--no-redperf', action='store_true', default=False,
                           help='Disable reduction perforation knobs')
    argparser.add_argument('--redperf-end-only', action='store_true', default=False,
                           help='Tune only the end (extent) of reduction perforation')
    argparser.add_argument('--no-algchoice', action='store_true', default=False,
                           help='Disable algorithmic choice knobs')
    argparser.add_argument('--collect-approx', action='store_true', default=False,
                           help='Run collect-approx and exit')
    argparser.add_argument('--create-baseline', action='store_true', default=False,
                           help='Write config/baseline_config.json and exit')
    argparser.add_argument('--direct-eval', type=str, metavar='CONFIG_FILE', default=None,
                           help='Directly evaluate a config file without tuning')

    args = argparser.parse_args()
    device_type = args.device

    if args.collect_approx:
        subprocess.run(["make", f"--directory={root_dir_str}", "collect-approx", "-B",
                        "ENCODING_DIM=1024"], env=my_env)
        sys.exit()

    if args.create_baseline:
        # Enumerate every manipulator param and fill it with its
        # non-approximating value, so the seed actually evaluates the
        # baseline (D=8192, L=128, extent=1.0, stride=1, base=0, idlev).
        apx_descs = get_apx_descs(enable_redperf=True,
                                  enable_algchoice=False,
                                  redperf_end_only=False)
        baseline = {}
        for desc in apx_descs:
            for p in desc.get_opentuner_parameters():
                name = p.name
                if name.endswith("+RedPerfBase"):
                    baseline[name] = 0.0
                elif name.endswith("+RedPerfStride"):
                    baseline[name] = 1
                elif name.endswith("+RedPerfExtent"):
                    baseline[name] = 1.0
                elif name == "GLOBAL_EncodingDim":
                    baseline[name] = 8192
                elif name == "CUSTOM+N_LEVELS":
                    baseline[name] = 128
                elif name == "CUSTOM+ENCODER_TYPE":
                    baseline[name] = 0
                else:
                    # Unknown knob — fall back to the param's min. Warn so
                    # the user notices and adds a case above.
                    print(f"Warning: no baseline default for {name}; using min")
                    baseline[name] = getattr(p, 'min_value', 0)
        os.makedirs(root_dir_str + "/config", exist_ok=True)
        with open(root_dir_str + "/config/baseline_config.json", "w") as f:
            json.dump(baseline, f, indent=4)
        print("Wrote config/baseline_config.json with "
              f"{len(baseline)} knobs")
        sys.exit()

    if args.direct_eval:
        tuning_start_time = time.time()
        with open(args.direct_eval, "r") as f:
            cfg = ast.literal_eval(f.read())
        result = eval_config(cfg)
        print("Result:", result)
        sys.exit()

    tuning_start_time = time.time()

    enable_redperf   = not args.no_redperf
    enable_algchoice = not args.no_algchoice
    redperf_end_only = args.redperf_end_only
    apx_descs = get_apx_descs(
        enable_redperf=enable_redperf,
        enable_algchoice=enable_algchoice,
        redperf_end_only=redperf_end_only,
    )

    args.approx_descs       = apx_descs
    args.evaluate_config_fn = eval_config
    args.num_knobs          = len(apx_descs)
    args.seed_config_path   = root_dir_str + "/config/baseline_config.json"

    print(apx_descs)
    tuner_harness = OpenTunerHarness(args)
    tuner_harness.main(args)
