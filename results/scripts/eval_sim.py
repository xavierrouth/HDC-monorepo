#!/usr/bin/env python3
"""
Launch a tuning run for HD-Sim-Classification, and plot results.

HD-Sim-Classification is self-contained (its own autotune.py, no HPVM passes,
no redperf conditions). This wrapper forwards to that autotune.py with sensible
defaults, writes the DB into results/dbs/ alongside the other apps, and can
generate two plots from a finished DB:

  1) max-speedup-over-time (subject to accuracy cutoff)
  2) Pareto: energy reduction vs. speedup

Baseline is the seed config written at launch (measured manually with
`autotune.py --direct-eval` against the same seed).
"""
import argparse
import glob
import json
import os
import sqlite3
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(REPO_ROOT, "applications", "HD-Sim-Classification")
DB_DIR = os.path.join(REPO_ROOT, "results", "dbs")
OUT_DIR = os.path.join(REPO_ROOT, "results", "attachments")

# Seed config for the tuning run. Written to config/baseline_config.json before
# launching autotune.py (which hardcodes that seed path). Measured baseline on
# MNIST, HV_DIM=1024, 10 epochs, ADC range [-127, 128]: time 12.2M cycles,
# accuracy 0.8424, energy 429.44 µJ (MVM-only; see hpvm_interface.hpp get_energy).
SEED_CONFIG = {
    "GLOBAL_EncodingDim": 1024,
    "CUSTOM+TRAIN_ITERATIONS": 10,
    "GLOBAL_PCM_MLC": 1,
    "GLOBAL_PCM_ADC_RES": 8,
    "GLOBAL_PCM_ADC_STEP": 1,
    "GLOBAL_PCM_ADC_FLOOR": -127,
    "GLOBAL_PCM_ADC_CEIL": 128,
    "GLOBAL_PCM_WRITE_VERIFY_CYCLE": 0,
}
BASELINE_TIME = 12_200_000.0          # cycles
BASELINE_ACCURACY = 0.8424
BASELINE_ENERGY = 429_440_000.0       # pJ
DEFAULT_ACCURACY_TARGET = round(BASELINE_ACCURACY - 0.05, 4)  # 0.7924


def write_seed():
    seed_path = os.path.join(APP_DIR, "config", "baseline_config.json")
    with open(seed_path, "w") as f:
        json.dump(SEED_CONFIG, f, indent=4)
    print(f"[eval_sim] seed written to {seed_path}: {SEED_CONFIG}")
    return seed_path


def run_tuning(args):
    os.makedirs(DB_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    db_path = os.path.join(DB_DIR, f"hd_sim_classification_{ts}.db")

    write_seed()

    cmd = [
        sys.executable, "autotune.py",
        "--accuracy-target", str(args.accuracy_target),
        "--stop-after", str(args.stop_after),
        "--no-dups",
        "--num-runs", str(args.num_runs),
        "--database", f"sqlite:///{db_path}",
    ] + list(args.extra or [])

    print(f"[eval_sim] DB: {db_path}")
    print(f"[eval_sim] CMD: {' '.join(cmd)}")
    sys.stdout.flush()
    rc = subprocess.run(cmd, cwd=APP_DIR).returncode
    if rc != 0:
        sys.exit(rc)
    return db_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _latest_db() -> str:
    pattern = os.path.join(DB_DIR, "hd_sim_classification_*.db")
    matches = [p for p in glob.glob(pattern) if not p.endswith(".device") and not p.endswith(".searchspace")]
    if not matches:
        raise FileNotFoundError(f"no HD-Sim DBs in {DB_DIR}")
    return max(matches, key=os.path.getmtime)


def _load_df(db_path: str, accuracy_cutoff: float):
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT collection_date, time, accuracy, energy FROM result "
        "WHERE state='OK' AND accuracy >= ? AND time < 1e30 "
        "ORDER BY collection_date",
        conn, params=(accuracy_cutoff,),
    )
    conn.close()
    df["collection_date"] = pd.to_datetime(df["collection_date"])
    return df


def plot_speedup_over_time(db_path: str, accuracy_cutoff: float, out_path: str):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    df = _load_df(db_path, accuracy_cutoff)
    print(f"[plot] {len(df)} valid configs at cutoff {accuracy_cutoff}")
    if df.empty:
        print("  no valid configs — skipping speedup plot")
        return

    t0 = df["collection_date"].iloc[0]
    hours = (df["collection_date"] - t0).dt.total_seconds().values / 3600.0
    speedup = BASELINE_TIME / df["time"].values
    max_speedup = []
    running = 0.0
    for s in speedup:
        running = max(running, s)
        max_speedup.append(running)

    def fmt_hm(x, _):
        h, m = int(x), int((x % 1) * 60)
        return f"{h}h{m:02d}m"

    fig, ax = plt.subplots(figsize=(10, 5))
    lbl = f"HD-Sim-Classification (max={max_speedup[-1]:.2f}×)"
    ax.step(hours, max_speedup, where="post", color="tab:blue", linewidth=2, label=lbl)
    ax.scatter(hours, max_speedup, color="tab:blue", s=20, zorder=5)
    ax.axhline(1.0, color="red", linewidth=1.5, label="baseline (1×)")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_hm))
    ax.set_xlabel("Time since start", fontsize=16)
    ax.set_ylabel("Max speedup achieved (×)", fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_pareto_energy_vs_speedup(db_path: str, accuracy_cutoff: float, out_path: str):
    import matplotlib.pyplot as plt

    df = _load_df(db_path, accuracy_cutoff)
    if df.empty:
        print("  no valid configs — skipping pareto plot")
        return

    speedup = BASELINE_TIME / df["time"].values
    energy_reduction = BASELINE_ENERGY / df["energy"].values

    # Pareto frontier: maximize both speedup and energy_reduction.
    # Sort by speedup descending, then sweep keeping only points that improve
    # on the best energy_reduction seen so far.
    order = sorted(range(len(speedup)), key=lambda i: (-speedup[i], -energy_reduction[i]))
    frontier = []
    best_er = -1.0
    for i in order:
        if energy_reduction[i] > best_er:
            frontier.append(i)
            best_er = energy_reduction[i]
    fs = [speedup[i] for i in frontier]
    fe = [energy_reduction[i] for i in frontier]
    # Draw the frontier left-to-right (ascending speedup)
    fs_sorted, fe_sorted = zip(*sorted(zip(fs, fe)))

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(speedup, energy_reduction, c=df["accuracy"].values,
                    cmap="viridis", s=28, alpha=0.75, edgecolors="none",
                    label="configs")
    ax.plot(fs_sorted, fe_sorted, color="red", linewidth=2,
            marker="o", markersize=6, label="Pareto frontier")
    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":")
    ax.axvline(1.0, color="gray", linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Speedup (×, log)", fontsize=16)
    ax.set_ylabel("Energy reduction (×, log)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Accuracy", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    ax.legend(fontsize=14, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_all(db_path: str, accuracy_cutoff: float):
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(db_path))[0]
    sp = os.path.join(OUT_DIR, f"{stem}_speedup.pdf")
    pr = os.path.join(OUT_DIR, f"{stem}_pareto_energy.pdf")
    plot_speedup_over_time(db_path, accuracy_cutoff, sp)
    plot_pareto_energy_vs_speedup(db_path, accuracy_cutoff, pr)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="HD-Sim-Classification tuning + plots")
    ap.add_argument("--stop-after", type=int, default=10800,
                    help="Tuning budget in seconds (default 10800 = 3h)")
    ap.add_argument("--accuracy-target", type=float, default=DEFAULT_ACCURACY_TARGET,
                    help=f"Accuracy target (default {DEFAULT_ACCURACY_TARGET} = baseline {BASELINE_ACCURACY} - 0.05)")
    ap.add_argument("--num-runs", type=int, default=1,
                    help="Repeated runs per config (default 1)")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip tuning; just generate plots from --db (or latest DB)")
    ap.add_argument("--db", type=str, default=None,
                    help="DB path for --plot-only (default: latest hd_sim_classification_*.db)")
    ap.add_argument("--plot-cutoff", type=float, default=None,
                    help="Accuracy cutoff for plots (default: --accuracy-target)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra args passed through to autotune.py")
    args = ap.parse_args()

    cutoff = args.plot_cutoff if args.plot_cutoff is not None else args.accuracy_target

    if args.plot_only:
        db = args.db or _latest_db()
        print(f"[eval_sim] plot-only on DB: {db}")
        plot_all(db, cutoff)
        return

    db = run_tuning(args)
    plot_all(db, cutoff)


if __name__ == "__main__":
    main()
