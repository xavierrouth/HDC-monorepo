#!/usr/bin/env python3
"""
Plot max speedup over time for HD-Classification redperf ON vs OFF tuning runs.
Uses the most recent matching pair in results/dbs/ by default.

Usage:
    python3 plot_classification_redperf_speedup.py \
        [--on  results/dbs/hd_classification_redperf_on_TIMESTAMP.db] \
        [--off results/dbs/hd_classification_redperf_off_TIMESTAMP.db] \
        [--accuracy-cutoff 0.85] \
        [--output results/attachments/classification_redperf_speedup.pdf]
"""

import argparse
import glob
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
DB_DIR      = os.path.join(RESULTS_DIR, "dbs")
OUT_DIR     = os.path.join(RESULTS_DIR, "attachments")
os.makedirs(OUT_DIR, exist_ok=True)


def latest_pair() -> tuple[str, str]:
    on_dbs = sorted(glob.glob(os.path.join(DB_DIR, "hd_classification_redperf_on_*.db")), reverse=True)
    for on_db in on_dbs:
        ts = os.path.basename(on_db).replace("hd_classification_redperf_on_", "").replace(".db", "")
        off_db = os.path.join(DB_DIR, f"hd_classification_redperf_off_{ts}.db")
        if os.path.exists(off_db):
            conn = sqlite3.connect(on_db)
            n_on = conn.execute("SELECT COUNT(*) FROM result").fetchone()[0]
            conn.close()
            conn = sqlite3.connect(off_db)
            n_off = conn.execute("SELECT COUNT(*) FROM result").fetchone()[0]
            conn.close()
            if n_on > 0 and n_off > 0:
                return on_db, off_db
    raise FileNotFoundError("No complete redperf ON/OFF pair found in " + DB_DIR)


def load_df(db_path: str, accuracy_cutoff: float) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT collection_date, time, accuracy FROM result "
        "WHERE accuracy >= ? AND time < 1e30 ORDER BY collection_date",
        conn, params=(accuracy_cutoff,)
    )
    conn.close()
    df["collection_date"] = pd.to_datetime(df["collection_date"])
    return df


def max_speedup_trace(df: pd.DataFrame, baseline: float):
    if df.empty:
        return np.array([]), np.array([])
    df = df.copy().sort_values("collection_date")
    t0 = df["collection_date"].iloc[0]
    df["hours"] = (df["collection_date"] - t0).dt.total_seconds() / 3600
    df["speedup"] = baseline / df["time"]
    df["max_speedup"] = df["speedup"].cummax()
    return df["hours"].values, df["max_speedup"].values


def plot(db_on: str, db_off: str, accuracy_cutoff: float, output_path: str) -> None:
    df_on  = load_df(db_on,  accuracy_cutoff)
    df_off = load_df(db_off, accuracy_cutoff)

    print(f"Valid configs — redperf ON: {len(df_on)}, redperf OFF: {len(df_off)}")

    if df_on.empty and df_off.empty:
        print("No results meet the accuracy cutoff.")
        return

    all_times = pd.concat([df_on["time"], df_off["time"]])
    baseline  = all_times.max()
    print(f"Baseline time (slowest valid config): {baseline:.3f}s")

    hours_on,  ms_on  = max_speedup_trace(df_on,  baseline)
    hours_off, ms_off = max_speedup_trace(df_off, baseline)

    fig, ax = plt.subplots(figsize=(10, 5))

    if len(hours_on) > 0:
        ax.step(hours_on,  ms_on,  where="post", color="steelblue", linewidth=2,
                label=f"redperf ON  (n={len(df_on)}, max={ms_on.max():.2f}×)")
        ax.scatter(hours_on,  ms_on,  color="steelblue", s=30, zorder=5)

    if len(hours_off) > 0:
        ax.step(hours_off, ms_off, where="post", color="coral", linewidth=2,
                linestyle="--", label=f"redperf OFF (n={len(df_off)}, max={ms_off.max():.2f}×)")
        ax.scatter(hours_off, ms_off, color="coral", s=30, zorder=5)

    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":", label="baseline (1×)")

    def fmt_hm(x, _):
        h, m = int(x), int((x % 1) * 60)
        return f"{h}h{m:02d}m"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_hm))
    ax.set_xlabel("Time since start", fontsize=12)
    ax.set_ylabel("Max speedup achieved (×)", fontsize=12)
    ax.set_title(
        f"HD-Classification — Max Speedup Over Time\n(accuracy ≥ {accuracy_cutoff})",
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()

    for label, ms in [("redperf ON", ms_on), ("redperf OFF", ms_off)]:
        if len(ms):
            print(f"{label}: max speedup = {ms.max():.2f}×")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--on",  metavar="DB", default=None)
    parser.add_argument("--off", metavar="DB", default=None)
    parser.add_argument("--accuracy-cutoff", type=float, default=0.85)
    parser.add_argument("--output", metavar="PNG", default=None)
    args = parser.parse_args()

    db_on, db_off = (args.on, args.off) if (args.on and args.off) else latest_pair()
    print(f"Using:\n  ON  {db_on}\n  OFF {db_off}\n")

    output_path = args.output or os.path.join(OUT_DIR, "classification_redperf_speedup.pdf")
    plot(db_on, db_off, args.accuracy_cutoff, output_path)


if __name__ == "__main__":
    main()
