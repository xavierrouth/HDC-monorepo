#!/usr/bin/env python3
"""
MicroHD sweep — run the IDLev binary search at drop=0.05, time every visited
candidate on both HPVM and MicroHD/PyTorch, and plot max-speedup-over-time
curves analogous to eval_redperf.py.

Two plots are written to results/attachments/:
    microhd_sweep_<ts>.pdf              — MicroHD HPVM + PyTorch curves only
    microhd_sweep_vs_approxhdc_<ts>.pdf — adds overlay of existing ApproxHDC run

Per-candidate JSONL logs, HPVM CSV, PyTorch CSV also written for reuse.

Usage:
    source approx-tuner/venv/bin/activate
    python3 results/scripts/microhd_sweep.py \
        --microhd-python /home/xrouth2/miniconda3/bin/python3 \
        --device gpu

    # Replay without re-running the search:
    python3 results/scripts/microhd_sweep.py --replay results/attachments/microhd_sweep_<ts>
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_microhd as cmp

MONOREPO_ROOT = cmp.MONOREPO_ROOT
APP_ROOT      = cmp.APP_ROOT
MICROHD_CODE  = cmp.MICROHD_CODE
ATTACH_DIR    = cmp.ATTACH_DIR
BASELINE_D    = cmp.BASELINE_D
BASELINE_L    = cmp.BASELINE_L

APPROXHDC_TUNING_LOG = os.path.join(APP_ROOT, "output/tuning_log.csv")


# ---------------------------------------------------------------------------
# Step 1: run MicroHD search with per-candidate logging
# ---------------------------------------------------------------------------

def run_logged_search(python_bin, log_path, drops="0.05", encoder="idlev"):
    """
    Launch the MicroHD binary search with MICROHD_LOG_PATH set so every
    visited candidate is appended to `log_path`. Returns (wall_s, records).
    """
    script_name = ("MicroHD-idlev-binary-search.py" if encoder == "idlev"
                   else "MicroHD-proj-binary-search.py")
    script = os.path.join(MICROHD_CODE, script_name)
    env = os.environ.copy()
    env["MICROHD_LOG_PATH"] = log_path
    env["MICROHD_DROPS_ONLY"] = drops

    print(f"=== Running MicroHD binary search ({encoder}, drops={drops}) ===")
    t0 = time.time()
    proc = subprocess.Popen(
        [python_bin, script, "--dataset", "isolet"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=MICROHD_CODE, env=env,
    )
    for line in iter(proc.stdout.readline, ""):
        ts = time.time() - t0
        print(f"[{ts:6.1f}s] {line}", end="", flush=True)
    proc.wait()
    wall_s = time.time() - t0
    print(f"MicroHD search finished in {wall_s:.1f}s\n")

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Parsed {len(records)} candidates from {log_path}")
    return wall_s, records


# ---------------------------------------------------------------------------
# Step 2: time every visited candidate on HPVM + PyTorch
# ---------------------------------------------------------------------------

def time_candidates_on_hpvm(records, device, baseline_ms, csv_path, encoder_type=0):
    """
    For each (D, L) in records, compile and time the HPVM binary. The per-
    candidate cost (compile_s + 3*run_s) accumulates into `cumulative_s`,
    which the returned rows use as X-axis for the running-best curve.
    Deduplicates by (D, L) — the first visit pays the full cost; repeats are
    treated as free (cached evaluation).
    """
    rows = []
    cumulative_s = 0.0
    cache = {}  # (D, L) -> (exec_ms, acc)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "wall_s", "cumulative_s", "D", "L", "Q",
                    "compile_s", "exec_ms", "accuracy", "speedup",
                    "microhd_accepted", "microhd_accuracy"])

        for i, rec in enumerate(records):
            D, L = rec["D"], rec["L"]
            key = (D, L)
            if key in cache:
                exec_ms, acc = cache[key]
                compile_s = 0.0
            else:
                print(f"[HPVM {i+1}/{len(records)}] compile D={D} L={L} enc={encoder_type}")
                ret, compile_s = cmp.make_hpvm(D, L, device, encoder_type=encoder_type)
                if ret != 0:
                    print(f"  compile failed (ret={ret}) — skipping")
                    cumulative_s += compile_s
                    cache[key] = (float("inf"), 0.0)
                    w.writerow([i, rec["wall_s"], cumulative_s, D, L, rec["Q"],
                                compile_s, float("inf"), 0.0, 0.0,
                                int(rec["successful"]), rec["accuracy"]])
                    continue
                exec_ms, acc = cmp.run_hpvm(device, n_runs=3)
                run_cost = (exec_ms / 1000.0) * 3
                cumulative_s += compile_s + run_cost
                cache[key] = (exec_ms, acc)
                print(f"  exec={exec_ms:.1f}ms acc={acc:.4f} "
                      f"compile={compile_s:.0f}s cumulative={cumulative_s:.0f}s")

            speedup = (baseline_ms / exec_ms) if exec_ms > 0 and exec_ms != float("inf") else 0.0
            rows.append({
                "idx": i, "wall_s": rec["wall_s"], "cumulative_s": cumulative_s,
                "D": D, "L": L, "Q": rec["Q"],
                "compile_s": compile_s, "exec_ms": exec_ms,
                "accuracy": acc, "speedup": speedup,
                "microhd_accepted": rec["successful"],
                "microhd_accuracy": rec["accuracy"],
            })
            w.writerow([i, rec["wall_s"], cumulative_s, D, L, rec["Q"],
                        compile_s, exec_ms, acc, speedup,
                        int(rec["successful"]), rec["accuracy"]])
    return rows


def time_candidates_on_pytorch(records, python_bin, baseline_ms, csv_path, encoder="idlev"):
    """
    For each (D, L, Q) in records, run time_config.py and accumulate the
    cost (3 * avg_ms) into cumulative_s. Deduplicates by (D, L, Q).
    """
    rows = []
    cumulative_s = 0.0
    cache = {}

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "wall_s", "cumulative_s", "D", "L", "Q",
                    "exec_ms", "accuracy", "speedup",
                    "microhd_accepted", "microhd_accuracy"])

        for i, rec in enumerate(records):
            D, L, Q = rec["D"], rec["L"], rec["Q"]
            key = (D, L, Q)
            if key in cache:
                exec_ms, acc = cache[key]
            else:
                print(f"[PyTorch {i+1}/{len(records)}] D={D} L={L} Q={Q} enc={encoder}")
                exec_ms, acc = cmp._run_pytorch_time_config(D, L, Q, python_bin, n_runs=3, encoder=encoder)
                cumulative_s += (exec_ms / 1000.0) * 3
                cache[key] = (exec_ms, acc)
                print(f"  exec={exec_ms:.1f}ms acc={acc:.4f} cumulative={cumulative_s:.0f}s")

            speedup = (baseline_ms / exec_ms) if exec_ms > 0 else 0.0
            rows.append({
                "idx": i, "wall_s": rec["wall_s"], "cumulative_s": cumulative_s,
                "D": D, "L": L, "Q": Q, "exec_ms": exec_ms,
                "accuracy": acc, "speedup": speedup,
                "microhd_accepted": rec["successful"],
                "microhd_accuracy": rec["accuracy"],
            })
            w.writerow([i, rec["wall_s"], cumulative_s, D, L, Q,
                        exec_ms, acc, speedup,
                        int(rec["successful"]), rec["accuracy"]])
    return rows


# ---------------------------------------------------------------------------
# Step 3: running-best curves
# ---------------------------------------------------------------------------

def running_best_curve(rows, search_wall_s, accuracy_target):
    """
    Build a step curve (wall_s, best_speedup_so_far). X-axis is the MicroHD
    search wall-time only — HPVM compile+run and PyTorch time_config.py are
    post-hoc evaluation and do NOT contribute to MicroHD's tuning budget.
    """
    points = []
    for r in rows:
        if r["accuracy"] >= accuracy_target and r["speedup"] > 0:
            points.append((r["wall_s"], r["speedup"]))
    points.sort(key=lambda x: x[0])
    curve = []
    best = 0.0
    for t, sp in points:
        if sp > best:
            best = sp
            curve.append((t, best))
    return curve


# ---------------------------------------------------------------------------
# Step 4: plots
# ---------------------------------------------------------------------------

def plot_microhd_only(hpvm_curve, pyt_curve, hpvm_rows, pyt_rows,
                       accuracy_target, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    if hpvm_curve:
        xs = [p[0] for p in hpvm_curve]
        ys = [p[1] for p in hpvm_curve]
        ax.step(xs, ys, where="post", color="steelblue", linewidth=2,
                label="MicroHD → HPVM (running best)")
    if pyt_curve:
        xs = [p[0] for p in pyt_curve]
        ys = [p[1] for p in pyt_curve]
        ax.step(xs, ys, where="post", color="coral", linewidth=2,
                label="MicroHD → PyTorch (running best)")

    _scatter_valid(ax, hpvm_rows, accuracy_target, "steelblue", marker="o")
    _scatter_valid(ax, pyt_rows,  accuracy_target, "coral",     marker="^")

    ax.set_xlabel("Cumulative tuning time (s)", fontsize=16)
    ax.set_ylabel(f"Best speedup vs D={BASELINE_D}, L={BASELINE_L} "
                  f"(accuracy ≥ {accuracy_target})", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved MicroHD-only plot → {out_path}")


def plot_with_approxhdc(hpvm_curve, pyt_curve, approxhdc_curve,
                         hpvm_rows, pyt_rows, accuracy_target, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    if approxhdc_curve:
        xs = [p[0] for p in approxhdc_curve]
        ys = [p[1] for p in approxhdc_curve]
        ax.step(xs, ys, where="post", color="forestgreen", linewidth=2,
                label="ApproxHDC → HPVM (OpenTuner)")
    if hpvm_curve:
        xs = [p[0] for p in hpvm_curve]
        ys = [p[1] for p in hpvm_curve]
        ax.step(xs, ys, where="post", color="steelblue", linewidth=2,
                label="MicroHD → HPVM (running best)")
    if pyt_curve:
        xs = [p[0] for p in pyt_curve]
        ys = [p[1] for p in pyt_curve]
        ax.step(xs, ys, where="post", color="coral", linewidth=2,
                label="MicroHD → PyTorch (running best)")

    _scatter_valid(ax, hpvm_rows, accuracy_target, "steelblue", marker="o")
    _scatter_valid(ax, pyt_rows,  accuracy_target, "coral",     marker="^")

    ax.set_xlabel("Cumulative tuning time (s)", fontsize=16)
    ax.set_ylabel(f"Best speedup vs D={BASELINE_D}, L={BASELINE_L} "
                  f"(accuracy ≥ {accuracy_target})", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved overlay plot → {out_path}")


def _scatter_valid(ax, rows, accuracy_target, color, marker):
    xs, ys = [], []
    for r in rows:
        if r["accuracy"] >= accuracy_target and r["speedup"] > 0:
            xs.append(r["wall_s"])
            ys.append(r["speedup"])
    if xs:
        ax.scatter(xs, ys, color=color, marker=marker, s=35,
                   alpha=0.6, zorder=4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--encoder", choices=["idlev", "rp"], default="idlev",
                   help="MicroHD encoder: idlev (IDLev) or rp (random projection)")
    p.add_argument("--microhd-python", default="/home/xrouth2/miniconda3/bin/python3",
                   help="Python interpreter with torchhd installed")
    p.add_argument("--accuracy-target", type=float, default=0.82)
    p.add_argument("--drops", default="0.05",
                   help="Comma-separated drop values to run in the binary search")
    p.add_argument("--approxhdc-log", default=APPROXHDC_TUNING_LOG,
                   help="Path to existing ApproxHDC tuning_log.csv to overlay")
    p.add_argument("--replay", default=None,
                   help="Prefix of a previous run (<prefix>_candidates.jsonl etc.) "
                        "to skip the search+timing and just re-plot")
    args = p.parse_args()

    encoder_type = 0 if args.encoder == "idlev" else 1

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.replay or os.path.join(ATTACH_DIR, f"microhd_sweep_{args.encoder}_{ts}")
    log_path = prefix + "_candidates.jsonl"
    hpvm_csv = prefix + "_hpvm.csv"
    pyt_csv  = prefix + "_pytorch.csv"
    meta_path = prefix + "_meta.json"

    if args.replay:
        with open(meta_path) as f:
            meta = json.load(f)
        hpvm_baseline_ms = meta["hpvm_baseline_ms"]
        pyt_baseline_ms  = meta["pyt_baseline_ms"]
        hpvm_rows = _load_csv_rows(hpvm_csv)
        pyt_rows  = _load_csv_rows(pyt_csv)
    else:
        # Baselines
        hpvm_baseline_ms, hpvm_baseline_acc = cmp.measure_baseline(
            args.device, n_runs=3, encoder_type=encoder_type)
        pyt_baseline_ms,  pyt_baseline_acc  = cmp.measure_microhd_pytorch_baseline(
            args.microhd_python, n_runs=3, encoder=args.encoder)

        # Search
        search_wall_s, records = run_logged_search(
            args.microhd_python, log_path, drops=args.drops, encoder=args.encoder)

        # Time each candidate
        hpvm_rows = time_candidates_on_hpvm(records, args.device,
                                             hpvm_baseline_ms, hpvm_csv,
                                             encoder_type=encoder_type)
        pyt_rows  = time_candidates_on_pytorch(records, args.microhd_python,
                                                pyt_baseline_ms, pyt_csv,
                                                encoder=args.encoder)

        with open(meta_path, "w") as f:
            json.dump({
                "timestamp": ts,
                "device": args.device,
                "encoder": args.encoder,
                "drops": args.drops,
                "hpvm_baseline_ms":  hpvm_baseline_ms,
                "hpvm_baseline_acc": hpvm_baseline_acc,
                "pyt_baseline_ms":   pyt_baseline_ms,
                "pyt_baseline_acc":  pyt_baseline_acc,
                "search_wall_s":     search_wall_s,
                "n_candidates":      len(records),
            }, f, indent=2)

    hpvm_curve = running_best_curve(hpvm_rows, 0, args.accuracy_target)
    pyt_curve  = running_best_curve(pyt_rows,  0, args.accuracy_target)

    plot_microhd_only(hpvm_curve, pyt_curve, hpvm_rows, pyt_rows,
                      args.accuracy_target, prefix + ".pdf")

    approxhdc_curve = cmp.read_approxhdc_curve(
        args.approxhdc_log, hpvm_baseline_ms if not args.replay
        else meta["hpvm_baseline_ms"], args.accuracy_target)
    plot_with_approxhdc(hpvm_curve, pyt_curve, approxhdc_curve,
                         hpvm_rows, pyt_rows, args.accuracy_target,
                         prefix + "_vs_approxhdc.pdf")


def _load_csv_rows(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "idx": int(row["idx"]),
                "wall_s": float(row["wall_s"]),
                "cumulative_s": float(row["cumulative_s"]),
                "D": int(row["D"]), "L": int(row["L"]), "Q": int(row["Q"]),
                "exec_ms": float(row["exec_ms"]),
                "accuracy": float(row["accuracy"]),
                "speedup":  float(row["speedup"]),
                "microhd_accepted": bool(int(row["microhd_accepted"])),
                "microhd_accuracy": float(row["microhd_accuracy"]),
            })
    return rows


if __name__ == "__main__":
    main()
