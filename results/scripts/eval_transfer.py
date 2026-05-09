#!/usr/bin/env python3
"""
Dataset transfer experiment for ApproxHDC.

For HD-Clustering (ISOLET/MNIST/UCIHAR) and HD-Classification (ISOLET/UCIHAR),
shows how approximation configs found by tuning on one dataset transfer to others.

Two phases:
  Phase 1 -- Tuning: run end-only condition for each (app, dataset) pair (5h GPU).
             ISOLET results already exist; new runs for MNIST/UCIHAR (clustering)
             and UCIHAR (classification).
  Phase 2 -- Transfer eval: for each (source, target) pair, extract the best valid
             config from the source DB, recompile with target dataset constants,
             and measure speedup and accuracy on the target dataset.

Usage:
    # Run new Phase 1 tuning runs (skips existing ISOLET runs):
    python3 eval_transfer.py --app clustering --device gpu --stop-after 18000 --run-tuning

    # Run Phase 2 transfer evaluation from existing DBs:
    python3 eval_transfer.py --app clustering --device gpu --eval-transfer

    # Both phases + plots:
    python3 eval_transfer.py --app clustering --device gpu --stop-after 18000

    # Plot only from saved results:
    python3 eval_transfer.py --app clustering --device gpu --plot-only
"""

import argparse
import ast
import csv
import datetime
import os
import re
import subprocess
import sys
import tempfile

from db_utils import decode_config, seed_time, db_glob, find_best_transfer_db, extract_top_configs

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MONOREPO_DIR = "/home/xrouth2/hdc/HDC-monorepo"
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.dirname(SCRIPT_DIR)
DB_DIR       = os.path.join(RESULTS_DIR, "dbs")
OUT_DIR      = os.path.join(RESULTS_DIR, "attachments")
VENV_PYTHON  = os.path.join(MONOREPO_DIR, "approx-tuner/venv/bin/python3.11")

os.makedirs(DB_DIR,  exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# App / dataset configuration
# ---------------------------------------------------------------------------

CLUSTERING_DATASETS = {
    "ISOLET": {
        "n_feat": 624, "n_sample": 6238, "n_center": 26,
        "accuracy_target": 0.62,
        "baseline_nmi":    0.7284,
    },
    "MNIST": {
        "n_feat": 784, "n_sample": 10000, "n_center": 10,
        "accuracy_target": 0.32,   # 80% of baseline 0.399
        "baseline_nmi":    0.3994,
    },
    "UCIHAR": {
        "n_feat": 568, "n_sample": 6213, "n_center": 12,
        "accuracy_target": 0.46,   # 80% of baseline 0.574
        "baseline_nmi":    0.5735,
    },
}

CLASSIFICATION_DATASETS = {
    "ISOLET": {
        "n_feat":     617,
        "n_sample":   6238,
        "n_test":     1559,
        "n_class":    26,
        "n_feat_pad": 624,
        "accuracy_target": 0.82,
        "baseline_acc":    0.8743,
    },
    "UCIHAR": {
        "n_feat":     568,
        "n_sample":   6213,
        "n_test":     1554,
        "n_class":    12,
        "n_feat_pad": 568,
        "accuracy_target": 0.63,   # 80% of baseline 0.788
        "baseline_acc":    0.7883,
    },
}

APPS = {
    "clustering": {
        "dir":      os.path.join(MONOREPO_DIR, "applications/HD-Clustering"),
        "datasets": CLUSTERING_DATASETS,
        "db_prefix": "hd_clustering",
        "label":    "HD-Clustering",
        "baseline_config": {
            "GLOBAL_EncodingDim":     1024,
            "CUSTOM+TRAIN_ITERATIONS": 20,
            "CUSTOM+HAMMING_DIST":     0,   # cosine
            "CUSTOM+DATA_TYPE":        1,   # float
        },
    },
    "classification": {
        "dir":      os.path.join(MONOREPO_DIR, "applications/HD-Classification"),
        "datasets": CLASSIFICATION_DATASETS,
        "db_prefix": "hd_classification",
        "label":    "HD-Classification",
        "baseline_config": {
            "GLOBAL_EncodingDim":     2048,
            "CUSTOM+TRAIN_ITERATIONS": 10,
            "CUSTOM+HAMMING_DIST":     0,   # cosine
        },
    },
}

# Flags for the end-only + enc-protected tuning condition (best from redperf study)
TUNING_FLAGS = ["--no-algchoice", "--redperf-end-only", "--enable-no-approx-hint"]


# ---------------------------------------------------------------------------
# Direct-eval helpers
# ---------------------------------------------------------------------------

def _write_config_tempfile(cfg: dict) -> str:
    """Write config dict to a temp file as a Python literal; return path."""
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="hdc_transfer_")
    with os.fdopen(fd, "w") as f:
        f.write(repr(cfg))
    return path


def _run_direct_eval(app_dir: str, autotune_script: str,
                     cfg: dict, dataset: str, device: str,
                     enable_no_approx_hint: bool = True) -> dict | None:
    """Run autotune.py --direct-eval and return the result dict."""
    cfg_path = _write_config_tempfile(cfg)
    try:
        cmd = [
            VENV_PYTHON, autotune_script,
            "--direct-eval", cfg_path,
            "--dataset",     dataset,
            "--device",      device,
        ]
        if enable_no_approx_hint:
            cmd.append("--enable-no-approx-hint")

        result = subprocess.run(
            cmd, cwd=app_dir, capture_output=True, text=True,
        )
        # Parse "Result: {...}" from stdout
        for line in result.stdout.splitlines():
            if line.startswith("Result:"):
                try:
                    return ast.literal_eval(line[len("Result:"):].strip())
                except Exception:
                    pass
        print(f"  [WARN] Could not parse result from direct-eval stdout:")
        print(result.stdout[-2000:])
        return None
    finally:
        os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# Phase 1 — Tuning runs
# ---------------------------------------------------------------------------

def run_tuning(app_name: str, dataset: str, device: str,
               stop_after: int, timestamp: str) -> str | None:
    """Run a single end-only tuning run for (app, dataset). Returns DB path."""
    cfg            = APPS[app_name]
    app_dir        = cfg["dir"]
    autotune_script = os.path.join(app_dir, "autotune.py")
    db_prefix      = cfg["db_prefix"]
    acc_target     = cfg["datasets"][dataset]["accuracy_target"]

    db_name = f"{db_prefix}_{dataset.lower()}_redperf_endonly_{timestamp}.db"
    db_path = os.path.join(DB_DIR, db_name)

    print(f"\n{'='*70}")
    print(f"  {cfg['label']} — {dataset} — end-only tuning")
    print(f"  DB: {db_name}")
    print(f"{'='*70}\n")

    cmd = [
        VENV_PYTHON, autotune_script,
        "--dataset",         dataset,
        "--device",          device,
        "--accuracy-target", str(acc_target),
        "--stop-after",      str(stop_after),
        "--no-dups",
        "--database",        db_path,
    ] + TUNING_FLAGS

    print(f"Command: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, cwd=app_dir)
    except KeyboardInterrupt:
        print("  Interrupted.")
    return db_path


def run_all_tuning(app_name: str, device: str, stop_after: int,
                   skip_existing: bool = True) -> None:
    """Phase 1: run end-only tuning for all datasets that don't have a DB yet."""
    cfg       = APPS[app_name]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset in cfg["datasets"]:
        if skip_existing:
            # strict=True: only skip if a dataset-specific DB exists (don't match ISOLET fallback)
            existing_candidates = db_glob(cfg["db_prefix"], dataset, "redperf_endonly", strict=True, db_dir=DB_DIR)
            existing = next(
                (p for p in existing_candidates
                 if (not device or not os.path.exists(p + ".device") or
                     open(p + ".device").read().strip().lower() == device.lower())),
                None,
            )
            if existing:
                print(f"  Skipping {app_name}/{dataset}: existing DB found at {existing}")
                continue
        run_tuning(app_name, dataset, device, stop_after, timestamp)


# ---------------------------------------------------------------------------
# Phase 2 — Transfer evaluation
# ---------------------------------------------------------------------------

def eval_transfer_pair(app_name: str, source_dataset: str, target_dataset: str,
                       device: str, n_runs: int = 3, top_k: int = 1) -> dict | None:
    """Evaluate the top-k configs from source_dataset on target_dataset.

    Each of the k source configs is evaluated n_runs times on the target; the
    best result (highest speedup with accuracy >= target) is returned.

    Returns dict with keys: source, target, config, time_ms, accuracy, speedup, valid.
    Returns None on failure.
    """
    cfg            = APPS[app_name]
    app_dir        = cfg["dir"]
    autotune_script = os.path.join(app_dir, "autotune.py")
    ds_cfg         = cfg["datasets"]
    acc_target_src = ds_cfg[source_dataset]["accuracy_target"]
    acc_target_tgt = ds_cfg[target_dataset]["accuracy_target"]

    # Find source DB
    src_db = find_best_transfer_db(cfg["db_prefix"], source_dataset, device=device, db_dir=DB_DIR)
    if src_db is None:
        print(f"  [SKIP] No DB found for {app_name}/{source_dataset}")
        return None
    print(f"  Source DB: {os.path.basename(src_db)}")

    # Extract top-k configs from source DB
    top_cfgs = extract_top_configs(src_db, acc_target_src, k=top_k)
    if not top_cfgs:
        print(f"  [SKIP] No valid configs in {os.path.basename(src_db)} (acc >= {acc_target_src})")
        return None
    print(f"  Evaluating top-{len(top_cfgs)} source configs on {target_dataset}...")

    # Get target baseline time (seed time from target DB, or run baseline)
    tgt_db = find_best_transfer_db(cfg["db_prefix"], target_dataset, device=device, db_dir=DB_DIR)
    baseline_time = seed_time(tgt_db) if tgt_db else None
    if baseline_time is None:
        print(f"  No seed time for target {target_dataset}; running baseline eval...")
        baseline_result = _run_direct_eval(
            app_dir, autotune_script,
            cfg["baseline_config"], target_dataset, device,
            enable_no_approx_hint=False,
        )
        if baseline_result is None:
            print(f"  [FAIL] Baseline eval failed for {target_dataset}")
            return None
        baseline_time = baseline_result["time"]
    print(f"  Target baseline time: {baseline_time*1000:.1f} ms")

    # Evaluate each of the top-k source configs on target; keep best by speedup
    best_result = None
    for cfg_i, src_cfg in enumerate(top_cfgs):
        print(f"  Config {cfg_i+1}/{len(top_cfgs)} ({source_dataset} → {target_dataset})...")
        times, accs = [], []
        for run_i in range(n_runs):
            res = _run_direct_eval(
                app_dir, autotune_script,
                src_cfg, target_dataset, device,
                enable_no_approx_hint=True,
            )
            if res is None:
                continue
            times.append(res["time"])
            accs.append(res["accuracy"])

        if not times:
            print(f"    [FAIL] All runs failed for config {cfg_i+1}.")
            continue

        avg_time = float(np.mean(times))
        avg_acc  = float(np.mean(accs))
        speedup  = baseline_time / avg_time if avg_time > 0 else 0.0
        valid    = avg_acc >= acc_target_tgt
        print(f"    speedup={speedup:.2f}x, acc={avg_acc:.4f}, valid={valid}")

        if best_result is None or speedup > best_result["speedup"]:
            best_result = {
                "source":      source_dataset,
                "target":      target_dataset,
                "config_rank": cfg_i + 1,
                "baseline_ms": baseline_time * 1000,
                "transfer_ms": avg_time * 1000,
                "accuracy":    avg_acc,
                "speedup":     speedup,
                "valid":       valid,
                "acc_target":  acc_target_tgt,
            }

    if best_result is None:
        print(f"  [FAIL] All configs failed for {source_dataset}→{target_dataset}.")
        return None

    print(f"  Best transfer {source_dataset}→{target_dataset}: "
          f"speedup={best_result['speedup']:.2f}x (config rank {best_result['config_rank']}), "
          f"acc={best_result['accuracy']:.4f}, valid={best_result['valid']}")
    return best_result


def eval_all_transfers(app_name: str, device: str,
                       n_runs: int = 3, top_k: int = 1) -> list[dict]:
    """Phase 2: run all (source, target) transfer pairs for an app."""
    cfg      = APPS[app_name]
    datasets = list(cfg["datasets"].keys())
    results  = []

    for src in datasets:
        for tgt in datasets:
            print(f"\n--- {app_name}: {src} → {tgt} ---")
            pair_result = eval_transfer_pair(
                app_name, src, tgt, device, n_runs=n_runs, top_k=top_k
            )
            if pair_result is None:
                continue
            results.append(pair_result)

    # Fill in direct_speedup (diagonal) and speedup_retention for each result
    direct_speedups = {
        r["speedup"]: r["speedup"]
        for r in results if r["source"] == r["target"]
    }
    # Rebuild as source==target → speedup mapping
    direct_speedups = {
        r["source"]: r["speedup"]
        for r in results if r["source"] == r["target"]
    }
    for r in results:
        dir_sp = direct_speedups.get(r["target"])
        r["direct_speedup"] = dir_sp
        r["speedup_retention"] = (r["speedup"] / dir_sp) if (dir_sp and dir_sp > 0) else None

    return results


# ---------------------------------------------------------------------------
# Saving / loading results
# ---------------------------------------------------------------------------

def save_results(results: list[dict], app_name: str, device: str) -> str:
    """Save transfer results to a CSV file in OUT_DIR."""
    fname = os.path.join(OUT_DIR, f"{APPS[app_name]['db_prefix']}_{device}_transfer_results.csv")
    if not results:
        return fname
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source", "target", "baseline_ms", "transfer_ms", "accuracy",
            "speedup", "valid", "acc_target", "direct_speedup", "speedup_retention",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})
    print(f"\nResults saved: {fname}")
    return fname


def load_results(app_name: str, device: str) -> list[dict]:
    """Load transfer results from CSV."""
    fname = os.path.join(OUT_DIR, f"{APPS[app_name]['db_prefix']}_{device}_transfer_results.csv")
    if not os.path.exists(fname):
        return []
    results = []
    with open(fname, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ["baseline_ms", "transfer_ms", "accuracy", "speedup",
                        "acc_target", "direct_speedup", "speedup_retention"]:
                if row.get(key) not in (None, "", "None"):
                    row[key] = float(row[key])
                else:
                    row[key] = None
            row["valid"] = row.get("valid", "").lower() in ("true", "1", "yes")
            results.append(row)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_transfer_matrix(results: list[dict], app_name: str, device: str) -> None:
    """Plot speedup heatmap and speedup-retention bar chart."""
    cfg      = APPS[app_name]
    datasets = list(cfg["datasets"].keys())
    n        = len(datasets)
    idx      = {d: i for i, d in enumerate(datasets)}

    # Build speedup matrix
    speedup_mat    = np.full((n, n), np.nan)
    retention_mat  = np.full((n, n), np.nan)
    valid_mat      = np.zeros((n, n), dtype=bool)

    for r in results:
        i = idx.get(r["source"])
        j = idx.get(r["target"])
        if i is None or j is None:
            continue
        speedup_mat[i, j]   = r["speedup"] or 0.0
        retention_mat[i, j] = r["speedup_retention"] or np.nan
        valid_mat[i, j]     = bool(r["valid"])

    # --- Heatmap: speedup ---
    fig, ax = plt.subplots(figsize=(max(5, n + 2), max(4, n + 1)))
    im = ax.imshow(speedup_mat, cmap="YlGn", vmin=1.0,
                   vmax=max(2.0, float(np.nanmax(speedup_mat)) * 1.05))

    for i in range(n):
        for j in range(n):
            v = speedup_mat[i, j]
            if not np.isnan(v):
                mark = "" if valid_mat[i, j] else "*"
                ax.text(j, i, f"{v:.1f}×{mark}", ha="center", va="center",
                        fontsize=14, color="black" if v < 5 else "white")

    cb = plt.colorbar(im, ax=ax, label="Speedup (×)")
    cb.ax.tick_params(labelsize=13)
    cb.set_label("Speedup (×)", fontsize=15)
    ax.set_xticks(range(n)); ax.set_xticklabels(datasets, fontsize=14)
    ax.set_yticks(range(n)); ax.set_yticklabels(datasets, fontsize=14)
    ax.set_xlabel("Target dataset", fontsize=16)
    ax.set_ylabel("Source (tuned on)", fontsize=16)
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, f"{cfg['db_prefix']}_{device}_transfer_matrix.pdf")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close()

    # --- Heatmap: speedup retention ---
    fig, ax = plt.subplots(figsize=(max(5, n + 2), max(4, n + 1)))
    im2 = ax.imshow(retention_mat, cmap="RdYlGn", vmin=0.0, vmax=1.2)
    for i in range(n):
        for j in range(n):
            v = retention_mat[i, j]
            if not np.isnan(v):
                mark = "" if valid_mat[i, j] else "*"
                ax.text(j, i, f"{v:.2f}{mark}", ha="center", va="center",
                        fontsize=14, color="black")

    cb2 = plt.colorbar(im2, ax=ax, label="Speedup retention (transfer/direct)")
    cb2.ax.tick_params(labelsize=13)
    cb2.set_label("Speedup retention (transfer/direct)", fontsize=15)
    ax.set_xticks(range(n)); ax.set_xticklabels(datasets, fontsize=14)
    ax.set_yticks(range(n)); ax.set_yticklabels(datasets, fontsize=14)
    ax.set_xlabel("Target dataset", fontsize=16)
    ax.set_ylabel("Source (tuned on)", fontsize=16)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, f"{cfg['db_prefix']}_{device}_transfer_retention.pdf")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()


def print_transfer_table(results: list[dict], app_name: str) -> None:
    """Print a formatted transfer results table to stdout."""
    if not results:
        print("  No results to display.")
        return
    hdr = f"{'Source':>12}  {'Target':>12}  {'Speedup':>9}  {'Retention':>10}  {'Accuracy':>9}  {'Valid':>6}"
    print(f"\n{APPS[app_name]['label']} — Transfer Results")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: (x["source"], x["target"])):
        ret = f"{r['speedup_retention']:.2f}" if r.get("speedup_retention") else "—"
        sp  = f"{r['speedup']:.2f}×"   if r.get("speedup")           else "—"
        acc = f"{r['accuracy']:.4f}"    if r.get("accuracy") is not None else "—"
        print(f"{r['source']:>12}  {r['target']:>12}  {sp:>9}  {ret:>10}  {acc:>9}  {str(r['valid']):>6}")
    print("-" * len(hdr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--app", choices=["clustering", "classification", "all"],
                        default="clustering",
                        help="Application to evaluate (default: clustering)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                        help="Device (default: gpu)")
    parser.add_argument("--stop-after", type=int, default=5 * 3600, metavar="SECS",
                        help="Tuning budget per run in seconds (default: 18000)")
    parser.add_argument("--n-runs", type=int, default=3, metavar="N",
                        help="Number of repeated evals per config per transfer pair (default: 3)")
    parser.add_argument("--top-k", type=int, default=1, metavar="K",
                        help="Evaluate top-K fastest valid source configs; report best transfer (default: 1)")
    parser.add_argument("--run-tuning", action="store_true",
                        help="Phase 1: run new tuning runs for datasets without an existing DB")
    parser.add_argument("--eval-transfer", action="store_true",
                        help="Phase 2: evaluate transfer across all dataset pairs")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip evaluation; plot from saved CSV results")
    parser.add_argument("--venv-python", type=str, default=None, metavar="PATH",
                        help="Path to python3.11 binary to use")
    args = parser.parse_args()

    global VENV_PYTHON
    if args.venv_python:
        VENV_PYTHON = args.venv_python

    # Default: run both phases if neither flag is set
    run_both = not args.run_tuning and not args.eval_transfer and not args.plot_only

    apps_to_run = ["clustering", "classification"] if args.app == "all" else [args.app]

    for app_name in apps_to_run:
        print(f"\n{'#'*70}")
        print(f"  {APPS[app_name]['label']}")
        print(f"{'#'*70}")

        if args.plot_only:
            results = load_results(app_name, args.device)
            if not results:
                print(f"  No saved results found for {app_name}/{args.device}")
                continue
        else:
            if args.run_tuning or run_both:
                run_all_tuning(app_name, args.device, args.stop_after, skip_existing=True)

            if args.eval_transfer or run_both:
                results = eval_all_transfers(app_name, args.device,
                                             n_runs=args.n_runs, top_k=args.top_k)
                save_results(results, app_name, args.device)
            else:
                results = load_results(app_name, args.device)

        print_transfer_table(results, app_name)
        if results:
            plot_transfer_matrix(results, app_name, args.device)

    return 0


if __name__ == "__main__":
    sys.exit(main())
