#!/usr/bin/env python3
"""
Plot CDFs of speedup for HD-Clustering tuning runs:
  - redperf ENABLED
  - redperf DISABLED

Speedup is relative to the slowest valid configuration across both runs.
Only configurations meeting the accuracy cutoff are included.

Usage:
    python3 plot_clustering_redperf_cdf.py \
        --on  results/dbs/hd_clustering_redperf_on_TIMESTAMP.db \
        --off results/dbs/hd_clustering_redperf_off_TIMESTAMP.db \
        [--accuracy-cutoff 0.62] \
        [--output results/attachments/clustering_redperf_cdf.pdf]

If --on/--off are omitted, the most recent matching pair in results/dbs/ is used.
"""

import argparse
import glob
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.dirname(SCRIPT_DIR)
DB_DIR       = os.path.join(RESULTS_DIR, "dbs")
OUT_DIR      = os.path.join(RESULTS_DIR, "attachments")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def latest_pair() -> tuple[str, str]:
    """Return the most recent (redperf-on, redperf-off) pair that both have >0 rows."""
    on_dbs  = sorted(glob.glob(os.path.join(DB_DIR, "hd_clustering_redperf_on_*.db")),  reverse=True)
    off_dbs = sorted(glob.glob(os.path.join(DB_DIR, "hd_clustering_redperf_off_*.db")), reverse=True)

    for on_db in on_dbs:
        timestamp = os.path.basename(on_db).replace("hd_clustering_redperf_on_", "").replace(".db", "")
        off_db = os.path.join(DB_DIR, f"hd_clustering_redperf_off_{timestamp}.db")
        if not os.path.exists(off_db):
            continue
        if _row_count(on_db) > 0 and _row_count(off_db) > 0:
            return on_db, off_db

    raise FileNotFoundError("No complete redperf ON/OFF pair found in " + DB_DIR)


def _row_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM result").fetchone()[0]
    conn.close()
    return n


def _search_space_size(db_path: str) -> str:
    sidecar = db_path + ".searchspace"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            return f.read().strip()
    return None


def load_results(db_path: str, accuracy_cutoff: float) -> np.ndarray:
    """Return array of (time,) for all results meeting the accuracy cutoff."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT time, accuracy FROM result WHERE accuracy >= ? AND time < 1e30",
        (accuracy_cutoff,)
    ).fetchall()
    conn.close()
    return np.array([r[0] for r in rows], dtype=float)


def plot_cdf(times_on: np.ndarray, times_off: np.ndarray,
             ss_on: str, ss_off: str,
             accuracy_cutoff: float, output_path: str) -> None:

    if len(times_on) == 0 and len(times_off) == 0:
        print("No results meet the accuracy cutoff — nothing to plot.")
        return

    # Baseline = slowest valid config across both runs
    all_times = np.concatenate([times_on, times_off])
    baseline  = all_times.max()
    print(f"Baseline time (slowest valid config): {baseline:.3f}s")

    def speedup_cdf(times):
        if len(times) == 0:
            return np.array([]), np.array([])
        speedups = baseline / times
        speedups_sorted = np.sort(speedups)
        cdf = np.arange(1, len(speedups_sorted) + 1) / len(speedups_sorted)
        return speedups_sorted, cdf

    sp_on,  cdf_on  = speedup_cdf(times_on)
    sp_off, cdf_off = speedup_cdf(times_off)

    fig, ax = plt.subplots(figsize=(9, 5))

    lbl_on  = f"redperf ON  (n={len(sp_on)}" + (f", space={ss_on}" if ss_on else "") + ")"
    lbl_off = f"redperf OFF (n={len(sp_off)}" + (f", space={ss_off}" if ss_off else "") + ")"

    if len(sp_on) > 0:
        ax.plot(sp_on,  cdf_on,  color="steelblue", linewidth=2, label=lbl_on)
    if len(sp_off) > 0:
        ax.plot(sp_off, cdf_off, color="coral", linewidth=2, linestyle="--", label=lbl_off)

    ax.axvline(1.0, color="gray", linewidth=1, linestyle=":", label="baseline (1×)")
    ax.set_xlabel("Speedup (×)", fontsize=12)
    ax.set_ylabel("Fraction of configurations", fontsize=12)
    ax.set_title(
        f"HD-Clustering — CDF of Speedup\n(accuracy ≥ {accuracy_cutoff})",
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

    # Summary stats
    for label, sp in [("redperf ON", sp_on), ("redperf OFF", sp_off)]:
        if len(sp) == 0:
            print(f"{label}: no valid configs")
        else:
            print(f"{label}: n={len(sp)}  max={sp.max():.2f}×  median={np.median(sp):.2f}×  "
                  f"mean={sp.mean():.2f}×")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--on",  metavar="DB", default=None,
                        help="redperf-ON database (default: latest pair)")
    parser.add_argument("--off", metavar="DB", default=None,
                        help="redperf-OFF database (default: latest pair)")
    parser.add_argument("--accuracy-cutoff", type=float, default=0.62,
                        help="Minimum accuracy to include (default: 0.62)")
    parser.add_argument("--output", metavar="PNG", default=None,
                        help="Output image path (default: attachments/clustering_redperf_cdf.pdf)")
    args = parser.parse_args()

    if args.on and args.off:
        db_on, db_off = args.on, args.off
    else:
        db_on, db_off = latest_pair()
        print(f"Using:\n  ON  {db_on}\n  OFF {db_off}\n")

    output_path = args.output or os.path.join(OUT_DIR, "clustering_redperf_cdf.pdf")

    times_on  = load_results(db_on,  args.accuracy_cutoff)
    times_off = load_results(db_off, args.accuracy_cutoff)

    print(f"Valid configs — redperf ON: {len(times_on)}, redperf OFF: {len(times_off)}")

    ss_on  = _search_space_size(db_on)
    ss_off = _search_space_size(db_off)

    plot_cdf(times_on, times_off, ss_on, ss_off, args.accuracy_cutoff, output_path)


if __name__ == "__main__":
    main()
