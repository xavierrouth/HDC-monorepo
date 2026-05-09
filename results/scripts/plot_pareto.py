#!/usr/bin/env python3
"""
Pareto-frontier plots of speedup vs. energy reduction for ApproxHDC GPU results.

For each application, loads all valid configurations (accuracy >= target) that
have energy measurements, computes speedup and energy reduction relative to the
seed (baseline) configuration, and plots:
  - Scatter of all valid configs coloured by condition
  - Pareto-optimal front per condition (no other config dominates on both axes)
  - Pareto front across all conditions combined

Applications with energy data:
  HD-Clustering    GPU 20260331  (all 4 conditions)
  HD-Classification GPU 20260401 (all 4 conditions)
  HD-Graph          GPU 20260403 (all 4 conditions)

Usage:
    source approx-tuner/venv/bin/activate
    python3 results/plot_pareto.py [--app {clustering,classification,graph,all}]
                                   [--save-only]
"""

import argparse
import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
MONOREPO_DIR = "/home/xrouth2/hdc/HDC-monorepo"
DB_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dbs")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attachments")
os.makedirs(OUT_DIR, exist_ok=True)

APPS = {
    "clustering": {
        "label":       "HD-Clustering (ISOLET, GPU)",
        "acc_target":  0.62,
        "out_prefix":  "hd_clustering_gpu_pareto",
        "series": [
            ("Full redperf",          "steelblue",      "-",  "hd_clustering_redperf_on_20260331_115714.db"),
            ("End-only",              "mediumseagreen", "-.", "hd_clustering_redperf_endonly_20260331_115714.db"),
            ("No redperf",            "coral",          "--", "hd_clustering_redperf_off_20260331_115714.db"),
            ("Full + enc. protected", "mediumpurple",   ":",  "hd_clustering_redperf_on_hint_20260331_115714.db"),
        ],
    },
    "classification": {
        "label":       "HD-Classification (ISOLET, GPU)",
        "acc_target":  0.82,
        "out_prefix":  "hd_classification_gpu_pareto",
        "series": [
            ("Full redperf",          "steelblue",      "-",  "hd_classification_redperf_on_20260401_150114.db"),
            ("End-only",              "mediumseagreen", "-.", "hd_classification_redperf_endonly_20260401_150114.db"),
            ("No redperf",            "coral",          "--", "hd_classification_redperf_off_20260401_150114.db"),
            ("Full + enc. protected", "mediumpurple",   ":",  "hd_classification_redperf_on_hint_20260401_150114.db"),
        ],
    },
    "graph": {
        "label":       "RelHD (Cora, GPU)",
        "acc_target":  0.69,
        "out_prefix":  "hd_graph_gpu_pareto",
        "series": [
            ("Full redperf",          "steelblue",      "-",  "hd_graph_redperf_on_20260403_102951.db"),
            ("End-only",              "mediumseagreen", "-.", "hd_graph_redperf_endonly_20260403_102951.db"),
            ("No redperf",            "coral",          "--", "hd_graph_redperf_off_20260403_102951.db"),
            ("Full + enc. protected", "mediumpurple",   ":",  "hd_graph_redperf_on_hint_20260403_102951.db"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _seed_row(db_path: str) -> tuple[float, float]:
    """Return (seed_time, seed_energy) — first chronological valid result."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT time, energy FROM result "
        "WHERE time < 1e30 AND energy > 0 "
        "ORDER BY collection_date LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"No seed with energy > 0 in {db_path}")
    return float(row[0]), float(row[1])


def load_valid(db_path: str, acc_target: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, energy, accuracy) arrays for all valid configs with energy."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT time, energy, accuracy FROM result "
        "WHERE accuracy >= ? AND time < 1e30 AND energy > 0",
        (acc_target,),
    ).fetchall()
    conn.close()
    if not rows:
        return np.array([]), np.array([]), np.array([])
    t, e, a = zip(*rows)
    return np.array(t, float), np.array(e, float), np.array(a, float)


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (maximise both x and y)."""
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if x[j] >= x[i] and y[j] >= y[i] and (x[j] > x[i] or y[j] > y[i]):
                dominated[i] = True
                break
    return ~dominated


def pareto_staircase(x: np.ndarray, y: np.ndarray):
    """Return (xs, ys) for a step-function Pareto front, sorted by x."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    # Build staircase: keep running max of y
    stair_x, stair_y = [xs[0]], [ys[0]]
    cur_max_y = ys[0]
    for xi, yi in zip(xs[1:], ys[1:]):
        if yi >= cur_max_y:
            stair_x.append(xi)
            stair_y.append(yi)
            cur_max_y = yi
    return np.array(stair_x), np.array(stair_y)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pareto(app_name: str, save_only: bool = False) -> None:
    cfg        = APPS[app_name]
    acc_target = cfg["acc_target"]
    title      = cfg["label"]
    out_prefix = cfg["out_prefix"]

    # Shared baseline: seed from the first series' DB
    first_db = os.path.join(DB_DIR, cfg["series"][0][3])
    try:
        bl_time, bl_energy = _seed_row(first_db)
    except ValueError as e:
        print(f"  [SKIP] {app_name}: {e}")
        return

    print(f"\n{title}")
    print(f"  Baseline: time={bl_time:.2f}, energy={bl_energy:.2f} J")

    fig, ax = plt.subplots(figsize=(9, 6))

    all_sp, all_er = [], []   # collect across all series for combined Pareto

    for label, color, ls, db_file in cfg["series"]:
        db_path = os.path.join(DB_DIR, db_file)
        if not os.path.exists(db_path):
            print(f"  [MISSING] {db_file}")
            continue

        t, e, a = load_valid(db_path, acc_target)
        if len(t) == 0:
            print(f"  [{label}] no valid configs with energy — skipped")
            continue

        speedup  = bl_time   / t
        en_reduc = bl_energy / e

        print(f"  [{label}] n={len(t)}, "
              f"max_speedup={speedup.max():.2f}x, "
              f"max_energy_reduc={en_reduc.max():.2f}x")

        # Scatter all valid configs
        ax.scatter(speedup, en_reduc, color=color, alpha=0.25, s=18, zorder=2)

        # Per-condition Pareto front
        mask = pareto_front(speedup, en_reduc)
        if mask.sum() >= 2:
            px, py = pareto_staircase(speedup[mask], en_reduc[mask])
            ax.step(px, py, where="post", color=color, linewidth=2,
                    linestyle=ls, label=f"{label} (n={len(t)})", zorder=4)
            ax.scatter(speedup[mask], en_reduc[mask],
                       color=color, s=50, zorder=5, edgecolors="white", linewidths=0.5)

        all_sp.extend(speedup.tolist())
        all_er.extend(en_reduc.tolist())

    # Combined Pareto front across all conditions
    if all_sp:
        all_sp = np.array(all_sp)
        all_er = np.array(all_er)
        comb_mask = pareto_front(all_sp, all_er)
        if comb_mask.sum() >= 2:
            cx, cy = pareto_staircase(all_sp[comb_mask], all_er[comb_mask])
            ax.step(cx, cy, where="post", color="black", linewidth=2.5,
                    linestyle="-", label="Combined Pareto front", zorder=6, alpha=0.7)

    ax.axvline(1.0, color="gray", linewidth=1, linestyle=":", alpha=0.6)
    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":", alpha=0.6)
    ax.set_xlabel("Speedup (×)", fontsize=16)
    ax.set_ylabel("Energy reduction (×)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14, loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{out_prefix}.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    if not save_only:
        plt.show()
    plt.close()

    # --- Second plot: accuracy vs speedup, coloured by energy reduction ---
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for label, color, ls, db_file in cfg["series"]:
        db_path = os.path.join(DB_DIR, db_file)
        if not os.path.exists(db_path):
            continue
        t, e, a = load_valid(db_path, acc_target)
        if len(t) == 0:
            continue
        speedup  = bl_time   / t
        en_reduc = bl_energy / e
        sc = ax2.scatter(speedup, a, c=en_reduc, cmap="YlOrRd",
                         alpha=0.6, s=25, zorder=3,
                         vmin=1.0, vmax=max(2.0, float(en_reduc.max()) * 0.9))
    cb2 = plt.colorbar(sc, ax=ax2, label="Energy reduction (×)")
    cb2.ax.tick_params(labelsize=13)
    cb2.set_label("Energy reduction (×)", fontsize=15)
    ax2.axhline(acc_target, color="gray", linewidth=1, linestyle="--",
                label=f"Accuracy target ({acc_target})")
    ax2.set_xlabel("Speedup (×)", fontsize=16)
    ax2.set_ylabel("Accuracy", fontsize=16)
    ax2.tick_params(axis="both", labelsize=14)
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, f"{out_prefix}_accuracy.pdf")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out2}")
    if not save_only:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", choices=["clustering", "classification", "graph", "all"],
                        default="all")
    parser.add_argument("--save-only", action="store_true",
                        help="Save plots to files without displaying them")
    args = parser.parse_args()

    apps = list(APPS.keys()) if args.app == "all" else [args.app]
    for app in apps:
        plot_pareto(app, save_only=args.save_only)


if __name__ == "__main__":
    main()
