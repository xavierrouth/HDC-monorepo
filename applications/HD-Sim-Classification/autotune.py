from hdc_approximation_tuner.Harness import OpenTunerHarness, process_opentuner_config
from hdc_approximation_tuner.ApproxTypes import ApproxPCM, ApproxEncodingDimension, ApproxCustomInteger
from hdc_approximation_tuner.PCMDesc import PCMDesc
import json
import opentuner
import time
import subprocess
import os
import ast
import sys

root_dir_str = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Sim-Classification"

# Global variables set by command-line arguments
device_type = None
num_runs = 5

my_env = os.environ.copy()


def get_apx_descs():
    DHv = 8192

    apx_descs = [
        ApproxEncodingDimension(8, DHv, True),
        ApproxCustomInteger("TRAIN_ITERATIONS", 10),
    ]

    pcm_desc = PCMDesc()
    pcm_desc.mlc_level_min = 1  # mlc_level=0 is invalid per the C++ simulator
    pcm_desc.mlc_level_max = 4
    pcm_desc.pcm_adc_floor_min = -127
    pcm_desc.pcm_adc_ceil_max = 128
    apx_descs.append(ApproxPCM(pcm_desc))

    return apx_descs


def eval_config(cfg):
    global device_type, num_runs
    print("HDC Configuration:", cfg)
    print("\n")
    pass_config_json = process_opentuner_config(cfg)
    print("Json:", pass_config_json)
    print("\n")

    dhv_entry = next((item for item in pass_config_json if "EncodingDim" in item), None)
    dhv = dhv_entry["EncodingDim"]

    train_entry = next((item for item in pass_config_json if "TRAIN_ITERATIONS" in item), None)
    train_iterations = train_entry["TRAIN_ITERATIONS"] if train_entry else 1

    mlc      = next((item for item in pass_config_json if "MLC"      in item), {}).get("MLC", 1)
    adc_res  = next((item for item in pass_config_json if "ADC_RES"  in item), {}).get("ADC_RES", 8)
    adc_step = next((item for item in pass_config_json if "ADC_STEP" in item), {}).get("ADC_STEP", 1)
    adc_floor= next((item for item in pass_config_json if "ADC_FLOOR" in item), {}).get("ADC_FLOOR", -255)
    adc_ceil = next((item for item in pass_config_json if "ADC_CEIL" in item), {}).get("ADC_CEIL", 256)
    wvc      = next((item for item in pass_config_json if "WRITE_VERIFY_CYCLE" in item), {}).get("WRITE_VERIFY_CYCLE", 0)

    if adc_floor >= adc_ceil:
        print(f"Invalid ADC range: floor {adc_floor} >= ceil {adc_ceil}. Skipping config.")
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}
    # ADC quantization constraint: (ceil - floor) / step must fit within 2^res
    # levels, otherwise the quantizer is under-resolved for the requested range.
    adc_levels = (adc_ceil - adc_floor) / max(adc_step, 1)
    if adc_levels > (1 << adc_res):
        print(f"Invalid ADC quantization: (ceil-floor)/step = {adc_levels} > 2^res = {1 << adc_res} "
              f"(floor={adc_floor}, ceil={adc_ceil}, step={adc_step}, res={adc_res}). Skipping config.")
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}
    # HV_DIM must be a multiple of 128 (the PCM array_size) and divisible
    # by mlc_level. Round up to the nearest 128-multiple divisible by mlc.
    unit = 128 * max(mlc, 1)
    dhv = ((dhv + unit - 1) // unit) * unit
    if dhv < unit:
        dhv = unit

    subprocess.run(["make", f"--directory={root_dir_str}", "clean"], env=my_env)

    compile_start = time.time()
    make_cmd = ["make", f"--directory={root_dir_str}",
        "all", "-B",
        "HPVM_BUILD=release",
        f"HV_DIM={dhv}",
        f"EPOCHS={train_iterations}",
        f"MLC={mlc}",
        f"ADC_RES={adc_res}",
        f"ADC_STEP={adc_step}",
        f"ADC_FLOOR={adc_floor}",
        f"ADC_CEIL={adc_ceil}",
        f"WRITE_VERIFY_CYCLE={wvc}"]
    run_res = subprocess.run(make_cmd, env=my_env)
    compile_end = time.time()

    if run_res.returncode != 0:
        print(f"Application exited with non-zero code: {run_res.returncode}")
        compile_time = compile_end - compile_start
        stats_dump = root_dir_str + "/stats.txt"
        with open(stats_dump, "a") as f:
            f.write(f"inf, 0.0, {compile_time}\n")
        print("Compilation failed — skipping this configuration.")
        return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

    print("Running Application:")

    times_ = []
    accuracies = []
    energies = []

    # Simulator is slow and deterministic-ish; a single run is usually enough,
    # but average `num_runs` to smooth system noise. Timeout guards against
    # degenerate knob combinations (e.g. huge HV_DIM with wvc=7).
    run_timeout_s = 300

    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        try:
            subprocess.run(["./test"], cwd=root_dir_str, env=my_env,
                           timeout=run_timeout_s)
        except subprocess.TimeoutExpired:
            print(f"Run {i+1} exceeded {run_timeout_s}s timeout — killing and skipping config.")
            subprocess.run(["pkill", "-9", "-f", root_dir_str + "/test"])
            compile_time = compile_end - compile_start
            stats_dump = root_dir_str + "/stats.txt"
            with open(stats_dump, "a") as f:
                f.write(f"inf, 0.0, {compile_time}, timeout\n")
            return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}

        results_file = root_dir_str + "/results.txt"
        try:
            with open(results_file, "r") as f:
                # test.cpp writes: line1=cycles, line2=accuracy, line3=energy(pJ)
                time_cycles = float(f.readline().strip())
                accuracy    = float(f.readline().strip())
                energy_pj   = float(f.readline().strip())
        except (ValueError, FileNotFoundError):
            print(f"Run {i+1} produced no results (likely crashed) — skipping config.")
            return {'time': float('inf'), "accuracy": 0.0, "energy": 0.0}
        with open(results_file, "w") as f:
            pass

        accuracies.append(accuracy)
        times_.append(time_cycles)
        energies.append(energy_pj)

    avg_time = sum(times_) / len(times_) if times_ else 0.0
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    avg_energy = sum(energies) / len(energies) if energies else 0.0
    compile_time = compile_end - compile_start

    stats_dump = root_dir_str + "/stats.txt"
    with open(stats_dump, "a") as f:
        f.write(f"{avg_time}, {avg_accuracy}, {avg_energy}, {compile_time},\n")

    print(f"Averages ({num_runs} runs) -> time: {avg_time}, accuracy: {avg_accuracy}, energy(pJ): {avg_energy}")
    return {'time': avg_time, "accuracy": avg_accuracy, "energy": avg_energy}


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    _results_dbs = os.path.join(root_dir_str, "../../results/dbs")
    os.makedirs(_results_dbs, exist_ok=True)
    argparser.set_defaults(database=os.path.join(_results_dbs, "hd_sim_classification_opentuner.db"))
    argparser.add_argument('--device', type=str, choices=['cpu'],
                          default='cpu',
                          help='Device to use (simulator is CPU-only)')
    argparser.add_argument('--accuracy-target', type=float,
                          default=0.60,
                          help='Target accuracy threshold')
    argparser.add_argument('--direct-eval', type=str, metavar='CONFIG_FILE',
                          default=None,
                          help='Directly evaluate a configuration file without tuning')
    argparser.add_argument('--num-runs', type=int, default=1,
                          help='Number of repeated runs per config for averaging (default: 1)')

    args = argparser.parse_args()

    device_type = args.device
    num_runs = args.num_runs

    if args.direct_eval:
        config_file = args.direct_eval
        with open(config_file, "r") as f:
            cfg = f.read()
            cfg = ast.literal_eval(cfg)
            result = eval_config(cfg)
            print("Result:", result)
            sys.exit()

    apx_descs = get_apx_descs()

    args.approx_descs        = apx_descs
    args.evaluate_config_fn  = eval_config
    args.num_knobs           = len(apx_descs)
    args.seed_config_path    = root_dir_str + "/config/baseline_config.json"

    print(apx_descs)

    tuner_harness = OpenTunerHarness(args)
    tuner_harness.main(args)
