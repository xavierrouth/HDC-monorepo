#!/usr/bin/env python3
"""
Compare ApproxHDC vs MicroHD on Micro-HD-Classification (ISOLET, IDLev encoder).

Metric: max speedup at accuracy >= threshold vs cumulative tuning time.
Speedup is always measured against the baseline HPVM binary (D=1024, L=128, no approx).

Usage (from repo root, inside approx-tuner venv):
    python3 results/compare_microhd.py --budget 3600 --device gpu --accuracy-target 0.82

Prerequisites:
    source approx-tuner/venv/bin/activate
    cd applications/Micro-HD-Classification
    make collect-approx -B ENCODING_DIM=1024    # generates config/config_out.json
    python3 autotune.py --create-baseline       # writes config/baseline_config.json
    cd ../..

Note: MicroHD requires torchhd. If not in the active venv, pass a separate interpreter:
    --microhd-python /path/to/torchhd-venv/bin/python3
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

MONOREPO_ROOT = "/home/xrouth2/hdc/HDC-monorepo"
APP_ROOT      = os.path.join(MONOREPO_ROOT, "applications/Micro-HD-Classification")
MICROHD_CODE  = os.path.join(MONOREPO_ROOT, "MicroHD/code")
RESULTS_DIR   = os.path.join(MONOREPO_ROOT, "results")
ATTACH_DIR    = os.path.join(RESULTS_DIR, "attachments")
os.makedirs(ATTACH_DIR, exist_ok=True)

# ISOLET dataset paths (relative to APP_ROOT, as the binary expects)
X_TRAIN = "../../datasets/classification/isolet_nq_trainX.bin"
Y_TRAIN = "../../datasets/classification/isolet_nq_trainY.bin"
X_TEST  = "../../datasets/classification/isolet_nq_testX.bin"
Y_TEST  = "../../datasets/classification/isolet_nq_testY.bin"

N_FEAT   = 617
N_CLASS  = 26
N_SAMPLE = 6238
N_TEST   = 1559


# ---------------------------------------------------------------------------
# HPVM compile + run helpers
# ---------------------------------------------------------------------------

def _make_clean():
    subprocess.run(["make", f"--directory={APP_ROOT}", "clean"],
                   capture_output=True)


def make_hpvm(dhv, n_levels, device, config_in=None, encoder_type=0):
    """
    Compile HPVM binary for given (D, L, encoder_type).
    encoder_type: 0 = IDLev, 1 = RP. N_LEVELS is ignored when encoder_type=1.
    config_in: list of per-op dicts for config_in.json; defaults to [] (no approx).
    Returns (returncode, compile_time_s).
    """
    cfg = config_in if config_in is not None else []
    with open(os.path.join(APP_ROOT, "config/config_in.json"), "w") as f:
        json.dump(cfg, f)

    _make_clean()

    target = "host-gpu-approx" if device == "gpu" else "host-cpu-approx"
    cmd = [
        "make", f"--directory={APP_ROOT}", target, "-B",
        f"ENCODING_DIM={dhv}",
        f"N_LEVELS={n_levels}",
        f"ENCODER_TYPE={encoder_type}",
        f"N_FEAT={N_FEAT}",
        f"N_CLASS={N_CLASS}",
        f"N_SAMPLE={N_SAMPLE}",
        f"N_TEST={N_TEST}",
    ]
    t0 = time.time()
    res = subprocess.run(cmd)
    return res.returncode, time.time() - t0


def run_hpvm(device, n_runs=3):
    """
    Run the already-compiled binary n_runs times.
    Returns (avg_exec_time_ms, accuracy) where exec_time = train + infer.
    """
    exe = "./host-gpu-approx" if device == "gpu" else "./host-cpu-approx"
    times_ms = []
    accuracy  = 0.0

    for _ in range(n_runs):
        subprocess.run(
            [exe, "2", X_TRAIN, Y_TRAIN, X_TEST, Y_TEST],
            cwd=APP_ROOT,
        )
        acc_file  = os.path.join(APP_ROOT, "output/test_accuracy.txt")
        time_file = os.path.join(APP_ROOT, "output/test_time.txt")

        with open(acc_file) as f:
            accuracy = float(f.readline())
        with open(acc_file, "w"):
            pass

        with open(time_file) as f:
            f.readline()              # total (unused)
            train = float(f.readline())
            infer = float(f.readline())
        with open(time_file, "w"):
            pass

        times_ms.append(train + infer)

    return sum(times_ms) / len(times_ms), accuracy


# ---------------------------------------------------------------------------
# Step 1: Baseline
# ---------------------------------------------------------------------------

BASELINE_D = 8192
BASELINE_L = 128

def measure_baseline(device, n_runs=3, encoder_type=0):
    """Compile and run baseline (D=8192, L=128, no approx). Returns (time_ms, accuracy)."""
    print(f"=== Measuring baseline (D={BASELINE_D}, L={BASELINE_L}, enc={encoder_type}) ===")
    ret, ct = make_hpvm(BASELINE_D, BASELINE_L, device, encoder_type=encoder_type)
    if ret != 0:
        sys.exit(f"ERROR: Baseline compilation failed (exit {ret})")
    avg_ms, acc = run_hpvm(device, n_runs=n_runs)
    print(f"Baseline: {avg_ms:.1f} ms, accuracy={acc:.4f}, compile={ct:.0f}s")
    return avg_ms, acc


# ---------------------------------------------------------------------------
# Step 2: MicroHD binary search
# ---------------------------------------------------------------------------

def run_microhd(python_bin="python3"):
    """
    Run MicroHD-idlev-binary-search.py on ISOLET, capturing stdout with timestamps.

    Returns:
        microhd_wall_s (float): total wall-clock time for the Python search
        configs (list[dict]): accepted configs per drop level:
            {'drop': float, 'D': int, 'L': int, 'Q': int, 'accuracy': float}
    """
    script = os.path.join(MICROHD_CODE, "MicroHD-idlev-binary-search.py")
    print("=== Running MicroHD binary search (ISOLET, IDLev) ===")

    t0 = time.time()
    proc = subprocess.Popen(
        [python_bin, script, "--dataset", "isolet"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=MICROHD_CODE,
    )

    captured = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        ts = time.time() - t0
        print(f"[{ts:6.1f}s] {line}", end="", flush=True)
        captured.append(line)

    proc.wait()
    microhd_wall_s = time.time() - t0
    print(f"\nMicroHD finished in {microhd_wall_s:.1f}s")

    full_text = "".join(captured)

    # Parse "Optimization Results" blocks
    blocks = re.findall(
        r"Optimization Results - Drop = ([\d.]+).*?"
        r"Dimensionality = (\d+).*?"
        r"Levels\s+=\s+(\d+).*?"
        r"Quantization\s+=\s+(\d+).*?"
        r"New Accuracy\s+=\s+([\d.]+)",
        full_text, re.DOTALL
    )
    configs = []
    for drop_s, D_s, L_s, Q_s, acc_s in blocks:
        configs.append({
            "drop": float(drop_s),
            "D":    int(D_s),
            "L":    int(L_s),
            "Q":    int(Q_s),
            "accuracy": float(acc_s),
        })

    print(f"MicroHD accepted configs: {configs}")
    return microhd_wall_s, configs


# ---------------------------------------------------------------------------
# MicroHD PyTorch timing (for Plot 2 — relative speedup in own execution model)
# ---------------------------------------------------------------------------

def _run_pytorch_time_config(D, L, Q, python_bin, n_runs=3, encoder="idlev"):
    """
    Run MicroHD/code/time_config.py n_runs times for a given (D, L, Q) on ISOLET.
    Returns (avg_total_ms, accuracy) where total = train + inference.
    """
    script = os.path.join(MICROHD_CODE, "time_config.py")
    times_ms = []
    accuracy  = 0.0

    for _ in range(n_runs):
        result = subprocess.run(
            [python_bin, script,
             "--D", str(D), "--L", str(L), "--Q", str(Q),
             "--encoder", encoder,
             "--dataset", "isolet"],
            capture_output=True, text=True, cwd=MICROHD_CODE,
        )
        train_ms = infer_ms = 0.0
        for line in result.stdout.splitlines():
            if line.startswith("TRAIN_MS:"):
                train_ms = float(line.split(":", 1)[1])
            elif line.startswith("INFER_MS:"):
                infer_ms = float(line.split(":", 1)[1])
            elif line.startswith("ACCURACY:"):
                accuracy = float(line.split(":", 1)[1])
        times_ms.append(train_ms + infer_ms)

    return sum(times_ms) / len(times_ms), accuracy


def measure_microhd_pytorch_baseline(python_bin, n_runs=3, encoder="idlev"):
    """
    Run the MicroHD PyTorch model at D=8192, L=128, Q=16 (the shared baseline config).
    Returns (avg_total_ms, accuracy).
    """
    print(f"=== Measuring MicroHD PyTorch baseline (D={BASELINE_D}, L={BASELINE_L}, Q=16, {encoder}) ===")
    avg_ms, acc = _run_pytorch_time_config(BASELINE_D, BASELINE_L, 16, python_bin, n_runs=n_runs, encoder=encoder)
    print(f"MicroHD PyTorch baseline: {avg_ms:.1f} ms, accuracy={acc:.4f}")
    return avg_ms, acc


def eval_microhd_pytorch(configs, microhd_wall_s, mhd_pytorch_baseline_ms,
                          accuracy_target, python_bin, n_runs=3):
    """
    For each MicroHD-accepted (D, L, Q) config, run the PyTorch model and measure
    speedup relative to the PyTorch baseline at D=1024, L=128, Q=16.

    Cumulative time = microhd_wall_s (binary search) + timing measurement overhead.
    Returns list of dicts: {cumulative_time_s, speedup, accuracy, D, L, Q, drop}.
    Only includes configs with accuracy >= accuracy_target.
    """
    results = []
    cumulative_s = microhd_wall_s  # binary search already done

    for cfg in configs:
        D, L, Q = cfg["D"], cfg["L"], cfg["Q"]
        print(f"\n  PyTorch timing for MicroHD config D={D}, L={L}, Q={Q}...")
        avg_ms, acc = _run_pytorch_time_config(D, L, Q, python_bin, n_runs=n_runs)
        # Add measurement overhead (n_runs × avg time per run)
        cumulative_s += (avg_ms / 1000.0) * n_runs

        speedup = mhd_pytorch_baseline_ms / avg_ms if avg_ms > 0 else 0.0
        print(f"  D={D} L={L} Q={Q}: {avg_ms:.1f}ms  acc={acc:.4f}  speedup={speedup:.2f}x  "
              f"cumulative={cumulative_s:.0f}s")

        if acc >= accuracy_target:
            results.append({
                "cumulative_time_s": cumulative_s,
                "speedup":  speedup,
                "accuracy": acc,
                "D": D, "L": L, "Q": Q,
                "drop": cfg["drop"],
            })

    return results


# ---------------------------------------------------------------------------
# Step 3: Evaluate MicroHD configs on HPVM
# ---------------------------------------------------------------------------

def eval_microhd_on_hpvm(configs, microhd_wall_s, baseline_ms, accuracy_target, device):
    """
    For each MicroHD-accepted (D, L) config, compile the HPVM binary and measure
    actual hardware speedup. Compile time is added to the cumulative tuning time.

    Returns list of dicts: {cumulative_time_s, speedup, accuracy, D, L, drop}
    Only configs with accuracy >= accuracy_target are included.
    """
    results = []
    cumulative_s = microhd_wall_s  # start after MicroHD Python search

    for cfg in configs:
        D, L = cfg["D"], cfg["L"]

        print(f"\n  Compiling HPVM for MicroHD config: D={D}, L={L}")
        ret, ct = make_hpvm(D, L, device)
        cumulative_s += ct

        if ret != 0:
            print(f"  Compilation failed for D={D}, L={L}")
            continue

        avg_ms, hpvm_acc = run_hpvm(device, n_runs=3)
        # Add approximate run time (3 runs × avg_ms)
        cumulative_s += (avg_ms / 1000.0) * 3

        speedup = baseline_ms / avg_ms if avg_ms > 0 else 0.0
        print(f"  D={D} L={L}: {avg_ms:.1f}ms  acc={hpvm_acc:.4f}  "
              f"speedup={speedup:.2f}x  cumulative={cumulative_s:.0f}s")

        if hpvm_acc >= accuracy_target:
            results.append({
                "cumulative_time_s": cumulative_s,
                "speedup":    speedup,
                "accuracy":   hpvm_acc,
                "D":          D,
                "L":          L,
                "drop":       cfg["drop"],
            })

    return results


# ---------------------------------------------------------------------------
# Step 4: Run ApproxHDC
# ---------------------------------------------------------------------------

def run_approxhdc(budget_s, device, accuracy_target, db_suffix=""):
    """
    Launch applications/Micro-HD-Classification/autotune.py for budget_s seconds.
    Uses --no-algchoice --redperf-end-only (best condition from prior studies).
    Returns path to tuning_log.csv.
    """
    autotune_script = os.path.join(APP_ROOT, "autotune.py")
    db_name = f"micro_hd_classification_comparison{db_suffix}.db"
    db_path = os.path.join(MONOREPO_ROOT, "results/dbs", db_name)

    # Clear old tuning log so we get a fresh curve
    log_path = os.path.join(APP_ROOT, "output/tuning_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)

    cmd = [
        sys.executable, autotune_script,
        "--device", device,
        "--accuracy-target", str(accuracy_target),
        "--no-algchoice",
        "--redperf-end-only",
        "--stop-after", str(int(budget_s)),
        "--database", db_path,
    ]
    print(f"\n=== Running ApproxHDC for {budget_s}s ===")
    print(" ".join(cmd))
    subprocess.run(cmd)
    return log_path


# ---------------------------------------------------------------------------
# Step 5: Read ApproxHDC curve from tuning_log.csv
# ---------------------------------------------------------------------------

def read_approxhdc_curve(log_path, baseline_ms, accuracy_target):
    """
    Parse output/tuning_log.csv and return a running-best speedup curve.
    Returns list of (cumulative_time_s, best_speedup_so_far).
    """
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found")
        return []

    rows = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc   = float(row["accuracy"])
            t_s   = float(row["wall_clock_s"])
            t_ms  = float(row["exec_time_ms"])
            if acc >= accuracy_target and t_ms > 0 and t_ms < 1e9:
                speedup = baseline_ms / t_ms
                rows.append((t_s, speedup))

    if not rows:
        print(f"No valid results at accuracy >= {accuracy_target} in {log_path}")
        return []

    # Running maximum
    curve = []
    best = 0.0
    for t_s, sp in sorted(rows, key=lambda x: x[0]):
        if sp > best:
            best = sp
        curve.append((t_s, best))
    return curve


# ---------------------------------------------------------------------------
# Step 6: Plot
# ---------------------------------------------------------------------------

def plot_comparison(microhd_pts, approxhdc_curve, accuracy_target, out_path):
    def fmt_hm(x, _):
        h, m = int(x), int((x % 1) * 60)
        return f"{h}h{m:02d}m"

    fig, ax = plt.subplots(figsize=(10, 5))

    if approxhdc_curve:
        hours = [p[0] / 3600 for p in approxhdc_curve]
        ys    = [p[1] for p in approxhdc_curve]
        lbl = f"ApproxHDC (OpenTuner + HPVM, max={max(ys):.2f}×)"
        ax.step(hours, ys, where="post", color="steelblue", linewidth=2, label=lbl)
        ax.scatter(hours, ys, color="steelblue", s=30, zorder=5)

    if microhd_pts:
        best_mhd = max(pt["speedup"] for pt in microhd_pts)
        ax.axhline(best_mhd, color="coral", linewidth=2,
                   label=f"MicroHD best ({best_mhd:.2f}×)")

    ax.axhline(1.0, color="red", linewidth=1.5, label="baseline (1×)")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_hm))
    ax.set_xlabel("Time since start", fontsize=16)
    ax.set_ylabel("Max speedup achieved", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def plot_relative_comparison(microhd_pytorch_pts, approxhdc_curve, accuracy_target, out_path):
    """
    Plot 2: relative speedup from the same D=1024, L=128 baseline, each system in
    its own execution model.
      MicroHD:   speedup in PyTorch vs PyTorch baseline at D=1024, L=128, Q=16
      ApproxHDC: speedup in HPVM vs HPVM baseline at D=1024, L=128
    Same style as plot_comparison: step curve + new-best dots + horizontal
    MicroHD-best line + red 1× baseline.
    """
    def fmt_hm(x, _):
        h, m = int(x), int((x % 1) * 60)
        return f"{h}h{m:02d}m"

    fig, ax = plt.subplots(figsize=(10, 5))

    if approxhdc_curve:
        hours = [p[0] / 3600 for p in approxhdc_curve]
        ys    = [p[1] for p in approxhdc_curve]
        lbl = f"ApproxHDC — HPVM (max={max(ys):.2f}×)"
        ax.step(hours, ys, where="post", color="steelblue", linewidth=2, label=lbl)
        ax.scatter(hours, ys, color="steelblue", s=30, zorder=5)

    if microhd_pytorch_pts:
        best_mhd = max(pt["speedup"] for pt in microhd_pytorch_pts)
        ax.axhline(best_mhd, color="coral", linewidth=2,
                   label=f"MicroHD best — PyTorch ({best_mhd:.2f}×)")

    ax.axhline(1.0, color="red", linewidth=1.5, label="baseline (1×)")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_hm))
    ax.set_xlabel("Time since start", fontsize=16)
    ax.set_ylabel("Max speedup achieved", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved Plot 2 (relative speedup) → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare ApproxHDC vs MicroHD on Micro-HD-Classification (ISOLET).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--budget", type=int, default=3600,
                        help="ApproxHDC budget in seconds")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--accuracy-target", type=float, default=0.82,
                        help="Min accuracy to count a config as valid")
    parser.add_argument("--baseline-runs", type=int, default=3,
                        help="Number of runs to average for baseline measurement")
    parser.add_argument("--microhd-python", type=str, default="python3",
                        help="Python interpreter with torchhd installed")
    parser.add_argument("--skip-microhd", action="store_true",
                        help="Skip MicroHD run (load from --microhd-results if given)")
    parser.add_argument("--skip-approxhdc", action="store_true",
                        help="Skip ApproxHDC run; use existing tuning_log.csv")
    parser.add_argument("--skip-relative", action="store_true",
                        help="Skip Plot 2 (relative speedup / PyTorch timing)")
    parser.add_argument("--microhd-results", type=str, default=None,
                        help="Path to a previously saved microhd_configs_*.json")
    parser.add_argument("--tuning-log", type=str, default=None,
                        help="Path to an existing tuning_log.csv to plot from")
    parser.add_argument("--pytorch-only", action="store_true",
                        help="Run MicroHD binary search and measure speedup in PyTorch only "
                             "(no HPVM compilation). Baseline is PyTorch at D=BASELINE_D, "
                             "L=BASELINE_L. Accuracy target is treated as absolute drop from "
                             "PyTorch baseline accuracy.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip all measurement; re-plot from an existing ApproxHDC "
                             "OpenTuner DB and an existing microhd_sweep *_hpvm.csv")
    parser.add_argument("--approxhdc-db", type=str,
                        default=os.path.join(MONOREPO_ROOT, "results/dbs",
                                              "micro_hd_classification_opentuner.db"),
                        help="[--plot-only] OpenTuner DB with ApproxHDC results")
    parser.add_argument("--microhd-hpvm-csv", type=str,
                        default=os.path.join(ATTACH_DIR,
                                              "microhd_sweep_20260414_092845_hpvm.csv"),
                        help="[--plot-only] microhd_sweep *_hpvm.csv")
    parser.add_argument("--baseline-ms", type=float, default=None,
                        help="[--plot-only] HPVM baseline in ms; if unset, read from "
                             "<microhd-hpvm-csv sibling>_meta.json")
    parser.add_argument("--include-deprecated-hpvm-plot", action="store_true",
                        help="[--plot-only] also emit the deprecated HPVM-vs-HPVM plot "
                             "(forces MicroHD configs through HPVM; not MicroHD's target backend)")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # --plot-only mode: replay from existing DB + microhd csv, no HPVM runs
    # ------------------------------------------------------------------
    if args.plot_only:
        import sqlite3
        import pandas as pd

        baseline_ms = args.baseline_ms
        if baseline_ms is None:
            meta_path = args.microhd_hpvm_csv.replace("_hpvm.csv", "_meta.json")
            with open(meta_path) as f:
                baseline_ms = json.load(f)["hpvm_baseline_ms"]
        print(f"HPVM baseline: {baseline_ms:.1f} ms")

        conn = sqlite3.connect(args.approxhdc_db)
        df = pd.read_sql_query(
            "SELECT collection_date, time, accuracy FROM result "
            "WHERE accuracy >= ? AND time < 1e30 ORDER BY collection_date",
            conn, params=(args.accuracy_target,))
        conn.close()
        df["collection_date"] = pd.to_datetime(df["collection_date"])
        if df.empty:
            print(f"No ApproxHDC configs meet accuracy ≥ {args.accuracy_target}")
            approxhdc_curve = []
        else:
            t0 = df["collection_date"].iloc[0]
            secs = (df["collection_date"] - t0).dt.total_seconds().values
            sp   = (baseline_ms / df["time"]).values
            running = []
            best = 0.0
            for t_s, s in sorted(zip(secs, sp), key=lambda x: x[0]):
                if s > best:
                    best = s
                    running.append((float(t_s), float(best)))
            approxhdc_curve = running
            print(f"ApproxHDC: {len(approxhdc_curve)} new-best points, "
                  f"max={approxhdc_curve[-1][1]:.2f}×")

        if args.include_deprecated_hpvm_plot:
            microhd_pts = []
            with open(args.microhd_hpvm_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    acc      = float(row["accuracy"])
                    sp       = float(row["speedup"])
                    accepted = bool(int(row["microhd_accepted"]))
                    if accepted and acc >= args.accuracy_target and sp > 0:
                        microhd_pts.append({
                            "cumulative_time_s": float(row["cumulative_s"]),
                            "speedup": sp,
                            "D": int(row["D"]), "L": int(row["L"]),
                            "drop": 0.05,
                        })
            if microhd_pts:
                best = max(microhd_pts, key=lambda p: p["speedup"])
                print(f"MicroHD best (HPVM): {best['speedup']:.2f}×  "
                      f"(D={best['D']}, L={best['L']})")
            out_path = os.path.join(ATTACH_DIR,
                                     f"microhd_vs_approxhdc_hpvm_{timestamp}.pdf")
            plot_comparison(microhd_pts, approxhdc_curve, args.accuracy_target, out_path)

        # Canonical plot: relative (PyTorch) — requires sibling _pytorch.csv
        pyt_csv = args.microhd_hpvm_csv.replace("_hpvm.csv", "_pytorch.csv")
        if not os.path.exists(pyt_csv):
            print(f"[skip Plot 2] no PyTorch CSV at {pyt_csv}")
            return
        microhd_pyt_pts = []
        with open(pyt_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                acc      = float(row["accuracy"])
                sp       = float(row["speedup"])
                accepted = bool(int(row["microhd_accepted"]))
                # Filter to configs MicroHD's binary search actually accepted
                # (microhd_accepted=1) AND that re-time above our cutoff.
                if accepted and acc >= args.accuracy_target and sp > 0:
                    microhd_pyt_pts.append({
                        "cumulative_time_s": float(row["cumulative_s"]),
                        "speedup": sp,
                        "D": int(row["D"]), "L": int(row["L"]), "Q": int(row["Q"]),
                        "drop": 0.05,
                    })
        if microhd_pyt_pts:
            best = max(microhd_pyt_pts, key=lambda p: p["speedup"])
            print(f"MicroHD best (PyTorch, accepted only): {best['speedup']:.2f}×  "
                  f"(D={best['D']}, L={best['L']}, Q={best['Q']})")
            first_beat = next(((t, s) for t, s in approxhdc_curve
                               if s > best['speedup']), None)
            if first_beat:
                t_s, sp = first_beat
                h, m, sec = int(t_s // 3600), int((t_s % 3600) // 60), int(t_s % 60)
                print(f"ApproxHDC first beats MicroHD best ({best['speedup']:.2f}×) "
                      f"at t={h}h{m:02d}m{sec:02d}s ({t_s:.0f}s) with {sp:.2f}×")
            else:
                print("ApproxHDC never exceeds MicroHD best at this cutoff")
        out2 = os.path.join(ATTACH_DIR, f"microhd_vs_approxhdc_relative_{timestamp}.pdf")
        plot_relative_comparison(microhd_pyt_pts, approxhdc_curve,
                                  args.accuracy_target, out2)
        return

    # ------------------------------------------------------------------
    # --pytorch-only mode: MicroHD binary search + PyTorch timing only
    # ------------------------------------------------------------------
    if args.pytorch_only:
        print("=== PyTorch-only mode: MicroHD binary search + PyTorch timing ===")
        pytorch_baseline_ms, pytorch_baseline_acc = measure_microhd_pytorch_baseline(
            args.microhd_python, n_runs=args.baseline_runs
        )
        acc_threshold = pytorch_baseline_acc - args.accuracy_target
        print(f"Accuracy threshold: {acc_threshold:.4f} "
              f"(baseline {pytorch_baseline_acc:.4f} - drop {args.accuracy_target})")

        microhd_wall_s, microhd_configs = run_microhd(python_bin=args.microhd_python)
        pytorch_pts = eval_microhd_pytorch(
            microhd_configs, microhd_wall_s, pytorch_baseline_ms,
            acc_threshold, args.microhd_python, n_runs=args.baseline_runs,
        )

        print("\n=== Summary (PyTorch-only) ===")
        print(f"PyTorch baseline (D={BASELINE_D}, L={BASELINE_L}): "
              f"{pytorch_baseline_ms:.1f} ms, acc={pytorch_baseline_acc:.4f}")
        if pytorch_pts:
            best = max(pytorch_pts, key=lambda p: p["speedup"])
            print(f"MicroHD best (PyTorch): {best['speedup']:.2f}x at "
                  f"{best['cumulative_time_s']:.0f}s  "
                  f"(D={best['D']}, L={best['L']}, Q={best['Q']}, acc={best['accuracy']:.4f})")
        else:
            print("No valid configs found above accuracy threshold.")
        return

    # ------------------------------------------------------------------
    # Step 1: HPVM baseline (D=1024, L=128) — shared reference for both plots
    # ------------------------------------------------------------------
    baseline_ms, baseline_acc = measure_baseline(args.device, n_runs=args.baseline_runs)

    # ------------------------------------------------------------------
    # Steps 2+3: MicroHD binary search + HPVM evaluation (Plot 1)
    # ------------------------------------------------------------------
    microhd_pts   = []
    microhd_wall_s = 0.0
    microhd_configs = []
    if not args.skip_microhd:
        if args.microhd_results:
            print(f"Loading MicroHD results from {args.microhd_results}")
            with open(args.microhd_results) as f:
                saved = json.load(f)
            baseline_ms     = saved.get("baseline_ms", baseline_ms)
            microhd_wall_s  = saved.get("microhd_wall_s", 0.0)
            microhd_configs = saved.get("raw_configs", [])
            microhd_pts     = saved.get("hpvm_configs", [])
        else:
            microhd_wall_s, microhd_configs = run_microhd(python_bin=args.microhd_python)
            microhd_pts = eval_microhd_on_hpvm(
                microhd_configs, microhd_wall_s, baseline_ms,
                args.accuracy_target, args.device,
            )

            mhd_out = os.path.join(RESULTS_DIR, f"microhd_configs_{timestamp}.json")
            with open(mhd_out, "w") as f:
                json.dump({
                    "baseline_ms":     baseline_ms,
                    "baseline_acc":    baseline_acc,
                    "microhd_wall_s":  microhd_wall_s,
                    "accuracy_target": args.accuracy_target,
                    "raw_configs":     microhd_configs,
                    "hpvm_configs":    microhd_pts,
                }, f, indent=2)
            print(f"Saved MicroHD results to {mhd_out}")

    # ------------------------------------------------------------------
    # Step 4: ApproxHDC
    # ------------------------------------------------------------------
    log_path = args.tuning_log or os.path.join(APP_ROOT, "output/tuning_log.csv")
    if not args.skip_approxhdc:
        log_path = run_approxhdc(
            args.budget, args.device, args.accuracy_target,
            db_suffix=f"_{timestamp}",
        )

    # ------------------------------------------------------------------
    # Step 5: Read ApproxHDC curve (same for both plots)
    # ------------------------------------------------------------------
    approxhdc_curve = read_approxhdc_curve(log_path, baseline_ms, args.accuracy_target)
    print(f"ApproxHDC: {len(approxhdc_curve)} improving points found")

    # ------------------------------------------------------------------
    # Step 6a: Plot 1 — both systems vs shared HPVM baseline
    # ------------------------------------------------------------------
    out_plot1 = os.path.join(ATTACH_DIR, f"microhd_vs_approxhdc_hpvm_{timestamp}.pdf")
    plot_comparison(microhd_pts, approxhdc_curve, args.accuracy_target, out_plot1)

    # ------------------------------------------------------------------
    # Step 6b: Plot 2 — relative speedup, each in own execution model
    # ------------------------------------------------------------------
    if not args.skip_relative and microhd_configs:
        mhd_pytorch_baseline_ms, _ = measure_microhd_pytorch_baseline(
            args.microhd_python, n_runs=args.baseline_runs
        )
        microhd_pytorch_pts = eval_microhd_pytorch(
            microhd_configs, microhd_wall_s, mhd_pytorch_baseline_ms,
            args.accuracy_target, args.microhd_python,
        )
        out_plot2 = os.path.join(ATTACH_DIR,
                                  f"microhd_vs_approxhdc_relative_{timestamp}.pdf")
        plot_relative_comparison(microhd_pytorch_pts, approxhdc_curve,
                                  args.accuracy_target, out_plot2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"HPVM baseline (D=1024, L=128): {baseline_ms:.1f} ms, acc={baseline_acc:.4f}")
    if microhd_pts:
        best = max(microhd_pts, key=lambda p: p["speedup"])
        print(f"MicroHD best (HPVM): {best['speedup']:.2f}x at "
              f"{best['cumulative_time_s']:.0f}s  (D={best['D']}, L={best['L']})")
    if approxhdc_curve:
        print(f"ApproxHDC best:      {approxhdc_curve[-1][1]:.2f}x at "
              f"{approxhdc_curve[-1][0]:.0f}s tuning time")


if __name__ == "__main__":
    main()
