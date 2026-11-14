#!/usr/bin/env python3
"""
HDC autotuning evaluation: reduction perforation ON vs end-only vs OFF.

Runs three back-to-back OpenTuner searches per application:
  1. Reduction perforation ENABLED  (full: start + stride + end)
  2. Reduction perforation END-ONLY (--redperf-end-only)
  3. Reduction perforation DISABLED  (--no-redperf)

Results land in separate SQLite databases. Search space sizes are written to
<db>.searchspace sidecar files. Plots are saved to results/attachments/.

Usage:
    # Run experiments + generate plots
    python3 eval_redperf.py [--app {clustering,classification,all}]
                            [--device {cpu,gpu}]
                            [--stop-after SECONDS]

    # Generate plots only (from most recent DBs)
    python3 eval_redperf.py --plot-only [--app {clustering,classification,all}]
                            [--accuracy-cutoff FLOAT]
"""

import argparse
import pickle
import re
import sqlite3
import subprocess
import os
import sys
import datetime
import zlib

import shutil
from db_utils import db_device, search_space_size, best_db, seed_time, load_times, decode_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Paths / per-app config
# ---------------------------------------------------------------------------
MONOREPO_DIR = "/home/xrouth2/hdc/HDC-monorepo"
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.dirname(SCRIPT_DIR)
DB_DIR       = os.path.join(RESULTS_DIR, "dbs")       # override with --db-dir
DB_COPY_DIR  = None                                    # override with --db-copy-dir
USE_BASELINE_OVERRIDE = False                          # enable with --baseline-override
OUT_DIR      = os.path.join(RESULTS_DIR, "attachments")
VENV_PYTHON  = os.path.join(MONOREPO_DIR, "approx-tuner/venv/bin/python3.11")  # override with --venv-python

os.makedirs(DB_DIR,  exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

APPS = {
    "clustering": {
        "dir":             os.path.join(MONOREPO_DIR, "applications/HD-Clustering"),
        "accuracy_target": 0.62,
        "plot_cutoff":     0.67,
        "db_prefix":       "hd_clustering",
        "label_prefix":    "HD-Clustering",
        "base_flags":      ["--dataset", "ISOLET", "--num-runs", "3"],
        # Per-device baseline override: time of the slowest `redperf_off` config
        # with NMI just above 0.67 (no redperf; algorithmic knobs only). Using
        # this as the speedup denominator measures gain past a "minimum-viable
        # quality" reference instead of the full unapproximated run.
        "baseline_override": {
            "gpu": 51.2,  # D=1024, TRAIN_ITER=4, DATA_TYPE=2, HAMMING=1, NMI 0.6746
            "cpu":  9.3,  # D=512,  TRAIN_ITER=4, DATA_TYPE=1, HAMMING=0, NMI 0.6721
        },
        # Parameters forwarded to `make collect-approx` to regenerate config_out.json
        "collect_approx_make_vars": [
            "ENCODING_DIM=1024", "HAMMING_DIST=0", "DATA_TYPE=float",
            "N_FEAT=624", "N_SAMPLE=6238", "N_CENTER=26",
        ],
    },
    "classification": {
        "dir":             os.path.join(MONOREPO_DIR, "applications/HD-Classification"),
        "accuracy_target": 0.84,
        "plot_cutoff":     0.82,
        "db_prefix":       "hd_classification",
        "label_prefix":    "HD-Classification",
        "base_flags":      [],
        # Regenerate config_out.json per condition so hint=OFF gets 4 ops
        # (matmul + cossim) and hint=ON gets 2 ops (cossim only).
        "collect_approx_make_vars": [
            "HPVM_BUILD=release",
            "ENCODING_DIM=2048", "DATA_TYPE=float", "HAMMING_DIST=0",
            "BINARIZE_ENCODING=1", "N_FEAT=617", "N_SAMPLE=6238",
            "N_TEST=1559", "N_CLASS=26", "N_FEAT_PAD=624",
        ],
    },
    "graph": {
        "dir":             os.path.join(MONOREPO_DIR, "applications/HD-Graph"),
        "accuracy_target": 0.69,
        "db_prefix":       "hd_graph",
        "label_prefix":    "RelHD",
        "base_flags":      ["--num-runs", "3"],
        # Pin every macro that affects which HDC ops exist, so config_out.json is
        # generated against the canonical build rather than the Makefile's
        # defaults. Leaving this empty previously meant "don't regenerate" and,
        # once that was fixed, would have meant "regenerate against defaults" --
        # which enables BINARIZE/RHO_SHIFT/POLARIZE and shifts the op indices
        # from 0 to 3, silently misaligning every knob in the config.
        "collect_approx_make_vars": [
            "HPVM_BUILD=release",
            "ENCODING_DIM=2048", "HAMMING_DIST=0",
            "BINARIZE=0", "RHO_SHIFT=0", "POLARIZE=0",
        ],
        # HD-Graph DFG.hpp has no __hetero_hdc_no_approx calls, so the +hint
        # condition is a no-op. Skip it; only 3 conditions are meaningful.
        "series": [
            (["--no-algchoice"],                         False, "redperf_on",      "redperf ON",       "steelblue",      "-"),
            (["--no-algchoice", "--redperf-end-only"],   False, "redperf_endonly", "redperf end-only", "mediumseagreen", "-."),
            (["--no-algchoice", "--no-redperf"],         False, "redperf_off",     "redperf OFF",      "coral",          "--"),
        ],
    },
    "hashtable": {
        "dir":             os.path.join(MONOREPO_DIR, "applications/HD-HashTable"),
        "accuracy_target": 0.80,
        "plot_cutoff":     0.7795,  # fresh GPU baseline 0.8295 - 0.05
        "db_prefix":       "hd_hashtable_multihv",
        "label_prefix":    "HD-HashTable",
        "base_flags":      ["--num-runs", "3"],
        # Fresh GPU baseline: D=8192, THRESHOLD_MULT=0.9, remeasured 2026-04-15.
        # always_override: the GPU DB seed is from a stale D=4096 config, so
        # seed_time is wrong. This override must be active regardless of the
        # --baseline-override flag (which is opt-in for HD-Clustering's
        # conceptual override).
        "baseline_override": {
            "gpu": 3499.0,
        },
        "always_override_baseline": True,
        # Reject configs where KMERS_PER_HV=0: causes division-by-zero in N_HVS
        # macro, so the compile silently fails and a stale binary is evaluated.
        "config_reject": lambda cfg: cfg.get("CUSTOM+KMERS_PER_HV", 400) == 0,
        # Clears the stale HPVM-level matmul op from config_out.json so that
        # only the custom knobs (STRIDE, QUERY_END, KMER_STRIDE, etc.) are tuned.
        "collect_approx_make_vars": ["CLEAR_CONFIG_OUT=1"],
        # 3-condition series: nohint is irrelevant for the GPU-batched pipeline.
        # redperf here = STRIDE + QUERY_END applied to the query dot product.
        "series": [
            (["--no-algchoice"],                         False, "redperf_on",      "redperf ON",       "steelblue",      "-"),
            (["--no-algchoice", "--redperf-end-only"],   False, "redperf_endonly", "redperf end-only", "mediumseagreen", "-."),
            (["--no-algchoice", "--no-redperf"],         False, "redperf_off",     "redperf OFF",      "coral",          "--"),
        ],
    },
}

STOP_AFTER_DEFAULT = 5 * 3600
DEVICE_DEFAULT     = "cpu"
PLOT_XLIM_HOURS    = 3.0  # clamp x-axis on all speedup-over-time plots

# Series descriptors: (flags, no_approx_hint, db_suffix, label, color, linestyle)
SERIES = [
    (["--no-algchoice"],                                                          False, "redperf_on",       "redperf ON",              "steelblue",      "-"),
    (["--no-algchoice", "--redperf-end-only", "--enable-no-approx-hint"],        True,  "redperf_endonly",  "redperf end-only", "mediumseagreen", "-."),
    (["--no-algchoice", "--no-redperf"],                                          False, "redperf_off",      "redperf OFF",             "coral",          "--"),
    (["--no-algchoice", "--enable-no-approx-hint"],                              True,  "redperf_on_hint",  "redperf ON +hint",        "mediumpurple",   ":"),
]


# ===========================================================================
# Tuning helpers
# ===========================================================================

def regenerate_config_out(app_dir: str, make_vars: list[str], no_approx_hint: bool,
                          skip: bool = False) -> None:
    """Re-run `make collect-approx` so config_out.json matches the series' hint setting.

    Regeneration is the default and an app opts out explicitly via
    `skip_config_regen: True`. It used to be skipped whenever
    `collect_approx_make_vars` was falsy, which silently conflated "this app
    needs no extra make variables" with "do not regenerate at all". HD-Graph
    meant the former and got the latter, so every one of its tuning runs used a
    stale checked-in config_out.json that listed 1 of its 2 perforable
    operators -- understating its search space by ~23 orders of magnitude.
    An empty list now means what it reads as: regenerate, passing no extra vars.
    """
    if skip:
        print("  Skipping config_out.json regeneration (skip_config_regen=True); "
              "using the checked-in file.")
        return
    hint_var = ["NO_APPROX_HINT=1"] if no_approx_hint else ["NO_APPROX_HINT=0"]
    cmd = ["make", f"--directory={app_dir}", "collect-approx", "-B"] + make_vars + hint_var
    print(f"  Regenerating config_out.json (NO_APPROX_HINT={'1' if no_approx_hint else '0'})...")
    subprocess.run(cmd, capture_output=True)


def capturesearch_space_size(app_dir: str, autotune_script: str,
                               base_flags: list[str],
                               extra_flags: list[str]) -> str:
    """Run autotune --print-search-space-size and return the '10^{X.XX}' string."""
    result = subprocess.run(
        [VENV_PYTHON, autotune_script, "--print-search-space-size"]
        + base_flags + extra_flags,
        cwd=app_dir, capture_output=True, text=True
    )
    m = re.search(r'10\^\{[\d.]+\}', result.stdout)
    return m.group(0) if m else "unknown"


def run_autotune(app_dir: str, autotune_script: str,
                 label: str, db_path: str,
                 base_flags: list[str], extra_flags: list[str],
                 device: str, accuracy_target: float, stop_after: int) -> int:
    """Launch a single OpenTuner session. Returns the subprocess return code."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Started: {datetime.datetime.now()}")
    print(f"{'='*80}\n")

    cmd = [
        VENV_PYTHON, autotune_script,
        "--device",          device,
        "--accuracy-target", str(accuracy_target),
        "--stop-after",      str(stop_after),
        "--no-dups",
        "--database",        db_path,
        "--label",           label,
    ] + base_flags + extra_flags

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, cwd=app_dir)
        status = "SUCCESS" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"\n  Finished: {datetime.datetime.now()}  —  {status}")
        return result.returncode
    except KeyboardInterrupt:
        print(f"\n  Interrupted by user.")
        return 1


def run_app(app_name: str, device: str, stop_after: int, timestamp: str,
            accuracy_target_override: float = None,
            series_filter: list[str] | None = None) -> dict:
    """Run all redperf conditions for one application. Returns result dict."""
    cfg             = APPS[app_name]
    app_dir         = cfg["dir"]
    autotune_script = os.path.join(app_dir, "autotune.py")
    base_flags      = cfg["base_flags"]
    prefix          = cfg["label_prefix"]
    db_prefix       = cfg["db_prefix"]
    acc_target      = accuracy_target_override if accuracy_target_override is not None else cfg["accuracy_target"]
    make_vars       = cfg.get("collect_approx_make_vars", [])
    skip_regen      = cfg.get("skip_config_regen", False)

    print(f"\n{'#'*80}")
    print(f"  Application: {prefix}")
    print(f"{'#'*80}")

    series = cfg.get("series", SERIES)
    if series_filter:
        series = [s for s in series if s[2] in series_filter]
        if not series:
            print(f"  No series matched filter {series_filter}; skipping.")
            return {}

    results = {}
    for extra_flags, no_approx_hint, db_suffix, label, _, _ in series:
        regenerate_config_out(app_dir, make_vars, no_approx_hint, skip=skip_regen)
        db_path = os.path.join(DB_DIR, f"{db_prefix}_{db_suffix}_{timestamp}.db")
        ss = capturesearch_space_size(app_dir, autotune_script, base_flags, extra_flags)
        print(f"Search space ({label}): {ss}")
        with open(db_path + ".searchspace", "w") as f:
            f.write(ss)
        with open(db_path + ".device", "w") as f:
            f.write(device)
        rc = run_autotune(app_dir, autotune_script,
                          label=f"{prefix} — {label}",
                          db_path=db_path, base_flags=base_flags,
                          extra_flags=extra_flags,
                          device=device, accuracy_target=acc_target,
                          stop_after=stop_after)
        if DB_COPY_DIR:
            os.makedirs(DB_COPY_DIR, exist_ok=True)
            for f in [db_path, db_path + ".searchspace"]:
                if os.path.exists(f):
                    shutil.copy2(f, DB_COPY_DIR)
                    print(f"  Copied {os.path.basename(f)} → {DB_COPY_DIR}")
        results[db_suffix] = {"rc": rc, "db": db_path}

    # Restore config_out.json to the default (no-hint) state after all series
    regenerate_config_out(app_dir, make_vars, no_approx_hint=False)
    return results


# ===========================================================================
# Plotting helpers
# ===========================================================================

def latest_dbs(db_prefix: str, device: str = "gpu", series=None) -> list[tuple[str, str, str, str]]:
    """Return list of (db_path, label, color, linestyle) using the most recent valid
    DB for each series independently (allows mixing timestamps)."""
    if series is None:
        series = SERIES
    series_out = []
    for _, _hint, db_suffix, label, color, ls in series:
        db_path = best_db(db_prefix, db_suffix, device=device, db_dir=DB_DIR)
        if db_path:
            series_out.append((db_path, label, color, ls))
    if not series_out:
        raise FileNotFoundError(f"No redperf DBs found for prefix '{db_prefix}' device='{device}' in {DB_DIR}")
    return series_out


# --- CDF ---

def plot_cdf(app_name: str, series: list, accuracy_cutoff: float, output_path: str,
             device: str = "gpu") -> None:
    label_prefix = APPS[app_name]["label_prefix"]

    cfg_reject = APPS[app_name].get("config_reject")
    loaded = []
    for db_path, label, color, ls in series:
        if cfg_reject is not None:
            df = _load_df(db_path, accuracy_cutoff, config_reject=cfg_reject)
            times = df["time"].values if not df.empty else np.array([])
        else:
            times = load_times(db_path, accuracy_cutoff)
        ss    = search_space_size(db_path)
        loaded.append((times, label, color, ls, ss))
        print(f"[{label_prefix} CDF] {label}: {len(times)} valid configs")

    all_times = np.concatenate([t for t, *_ in loaded if len(t) > 0])
    if len(all_times) == 0:
        print("  No results meet the accuracy cutoff — skipping CDF plot.")
        return
    baseline = resolve_baseline(app_name, device, series) or all_times.max()

    fig, ax = plt.subplots(figsize=(9, 5))
    for times, label, color, ls, ss in loaded:
        if len(times) == 0:
            continue
        sp  = np.sort(baseline / times)
        cdf = np.arange(1, len(sp) + 1) / len(sp)
        lbl = f"{label} (n={len(sp)}" + (f", space={ss}" if ss else "") + ")"
        ax.plot(sp, cdf, color=color, linewidth=2, linestyle=ls, label=lbl)

    ax.axvline(1.0, color="gray", linewidth=1, linestyle=":", label="baseline (1×)")
    ax.set_xlabel("Speedup (×)", fontsize=16)
    ax.set_ylabel("Fraction of configurations", fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.show()


# --- Config decoding ---

def _load_df_with_configs(db_path: str, accuracy_cutoff: float) -> pd.DataFrame:
    """Load valid results joined with decoded configuration dicts."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT r.collection_date, r.time, r.accuracy, c.data
           FROM result r
           JOIN desired_result dr ON r.id = dr.result_id
           JOIN configuration c  ON dr.configuration_id = c.id
           WHERE r.accuracy >= ? AND r.time < 1e30
           ORDER BY r.collection_date""",
        conn, params=(accuracy_cutoff,)
    )
    conn.close()
    df["collection_date"] = pd.to_datetime(df["collection_date"])
    df["config"] = df["data"].apply(decode_config)
    df = df.drop(columns=["data"])
    return df


# --- Speedup over time ---

def _load_df(db_path: str, accuracy_cutoff: float,
             config_reject=None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    if config_reject is None:
        # Fast path: no config decoding needed.
        df = pd.read_sql_query(
            "SELECT collection_date, time, accuracy FROM result "
            "WHERE accuracy >= ? AND time < 1e30 ORDER BY collection_date",
            conn, params=(accuracy_cutoff,)
        )
    else:
        # Slow path: join with configuration, decode blobs, apply filter.
        rows = conn.execute(
            "SELECT r.collection_date, r.time, r.accuracy, c.data "
            "FROM result r JOIN configuration c ON r.configuration_id=c.id "
            "WHERE r.accuracy >= ? AND r.time < 1e30 "
            "ORDER BY r.collection_date",
            (accuracy_cutoff,)
        ).fetchall()
        kept = []
        for ts, t, a, data in rows:
            cfg = pickle.loads(zlib.decompress(data))
            if config_reject(cfg):
                continue
            kept.append({"collection_date": ts, "time": t, "accuracy": a})
        df = pd.DataFrame(kept) if kept else pd.DataFrame(
            columns=["collection_date", "time", "accuracy"])
    conn.close()
    df["collection_date"] = pd.to_datetime(df["collection_date"])
    return df


def pick_baseline(series: list) -> float | None:
    """Shared baseline rule for plots and summaries: the seed_time of the
    first series entry that has one. Using a single source guarantees the
    DB-queried speedup numbers and the plot speedup numbers agree."""
    return next((seed_time(db) for db, *_ in series if seed_time(db)), None)


def resolve_baseline(app_name: str, device: str, series: list) -> float | None:
    """Per-app/device baseline: `baseline_override[device]` if defined on
    the app entry AND either `USE_BASELINE_OVERRIDE` is enabled (opt-in via
    `--baseline-override`) or the app sets `always_override_baseline: True`
    (for apps whose DB seed is known-stale). Else `pick_baseline(series)`."""
    app = APPS.get(app_name, {})
    override = app.get("baseline_override", {}).get(device)
    if override is not None:
        if app.get("always_override_baseline", False) or USE_BASELINE_OVERRIDE:
            return override
    return pick_baseline(series)


def _max_speedup_trace(df: pd.DataFrame, baseline: float):
    if df.empty:
        return np.array([]), np.array([])
    df = df.copy().sort_values("collection_date")
    t0 = df["collection_date"].iloc[0]
    df["hours"]       = (df["collection_date"] - t0).dt.total_seconds() / 3600
    df["speedup"]     = baseline / df["time"]
    df["max_speedup"] = df["speedup"].cummax()
    return df["hours"].values, df["max_speedup"].values


def _max_speedup_trace_full(df: pd.DataFrame, baseline: float) -> pd.DataFrame:
    """Like _max_speedup_trace but returns a full DataFrame with per-row metadata."""
    if df.empty:
        return df
    df = df.copy().sort_values("collection_date")
    t0 = df["collection_date"].iloc[0]
    df["hours"]       = (df["collection_date"] - t0).dt.total_seconds() / 3600
    df["speedup"]     = baseline / df["time"]
    df["max_speedup"] = df["speedup"].cummax()
    # Only keep rows that improved the running best
    df["is_new_best"] = df["speedup"] == df["max_speedup"]
    return df


def dump_configs(app_name: str, series: list, accuracy_cutoff: float,
                 output_path: str, baseline: float) -> None:
    """Write a text file of new-best configs per series, numbered to match graph annotations."""
    label_prefix = APPS[app_name]["label_prefix"]
    total = 0

    with open(output_path, "w") as f:
        f.write(f"{label_prefix} — New-best configurations (accuracy ≥ {accuracy_cutoff})\n")
        f.write(f"Baseline time: {baseline:.2f}\n")
        f.write("Each entry number matches the annotation on the speedup plot for that series.\n")
        f.write("=" * 80 + "\n\n")

        for db_path, label, _color, _ls in series:
            df = _load_df_with_configs(db_path, accuracy_cutoff)
            if df.empty:
                f.write(f"── {label}  (no valid configs) ──\n\n")
                continue

            df = df.sort_values("collection_date").reset_index(drop=True)
            t0 = df["collection_date"].iloc[0]
            df["hours"]       = (df["collection_date"] - t0).dt.total_seconds() / 3600
            df["speedup"]     = baseline / df["time"]
            df["max_speedup"] = df["speedup"].cummax()
            df["is_new_best"] = df["speedup"] == df["max_speedup"]

            new_bests = df[df["is_new_best"]].reset_index(drop=True)
            f.write(f"── {label}  ({len(new_bests)} new-best configs, {len(df)} valid total) ──\n\n")

            for cfg_num, (_, row) in enumerate(new_bests.iterrows(), 1):
                f.write(f"[{cfg_num:04d}]  speedup={row['speedup']:.3f}×  "
                        f"time={row['time']:.1f}  acc={row['accuracy']:.4f}  "
                        f"wall_clock={row['hours']:.3f}h\n")
                if row["config"]:
                    for k, v in sorted(row["config"].items()):
                        f.write(f"        {k} = {v}\n")
                f.write("\n")
            total += len(new_bests)

    print(f"  Configs dumped: {output_path}  ({total} new-best entries)")


def plot_speedup(app_name: str, series: list, accuracy_cutoff: float, output_path: str,
                 annotate: bool = False, dump_path: str | None = None,
                 device: str = "gpu") -> None:
    label_prefix = APPS[app_name]["label_prefix"]

    cfg_reject = APPS[app_name].get("config_reject")
    loaded = []
    for db_path, label, color, ls in series:
        df = _load_df_with_configs(db_path, accuracy_cutoff) if (annotate or dump_path) \
             else _load_df(db_path, accuracy_cutoff, config_reject=cfg_reject)
        ss = search_space_size(db_path)
        loaded.append((db_path, df, label, color, ls, ss))
        print(f"[{label_prefix} Speedup] {label}: {len(df)} valid configs")

    all_times = pd.concat([df["time"] for _, df, *_ in loaded if not df.empty])
    if all_times.empty:
        print("  No results meet the accuracy cutoff — skipping speedup plot.")
        return
    baseline = resolve_baseline(app_name, device, series) or all_times.max()

    if dump_path:
        dump_configs(app_name, series, accuracy_cutoff, dump_path, baseline)

    def fmt_hm(x, _):
        h, m = int(x), int((x % 1) * 60)
        return f"{h}h{m:02d}m"

    fig, ax = plt.subplots(figsize=(10, 5))
    for db_path, df, label, color, ls, ss in loaded:
        full = _max_speedup_trace_full(df, baseline)
        if full.empty:
            continue
        hours = full["hours"].values
        ms    = full["max_speedup"].values
        lbl = f"{label} (n={len(df)}, max={ms.max():.2f}×" + (f", space={ss}" if ss else "") + ")"
        ax.step(hours, ms, where="post", color=color, linewidth=2, linestyle=ls, label=lbl)
        ax.scatter(hours, ms, color=color, s=30, zorder=5)

        if annotate:
            new_bests = full[full["is_new_best"]].copy()
            for cfg_num, (_, row) in enumerate(new_bests.iterrows(), 1):
                sp = baseline / row["time"]
                ax.annotate(
                    str(cfg_num),
                    xy=(row["hours"], sp),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, color=color, zorder=6,
                )

    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":", label="baseline (1×)")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_hm))
    ax.set_xlim(left=0, right=PLOT_XLIM_HOURS)
    ax.set_xlabel("Time since start", fontsize=16)
    ax.set_ylabel("Max speedup achieved (×)", fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.show()


def plot_app(app_name: str, series: list, accuracy_cutoff: float, device: str = "gpu",
             annotate: bool = False) -> None:
    prefix     = APPS[app_name]["db_prefix"]
    cdf_path   = os.path.join(OUT_DIR, f"{prefix}_{device}_redperf_cdf.pdf")
    speed_path = os.path.join(OUT_DIR, f"{prefix}_{device}_redperf_speedup.pdf")
    dump_path  = os.path.join(OUT_DIR, f"{prefix}_{device}_redperf_configs.txt") if annotate else None
    plot_cdf(app_name, series, accuracy_cutoff, cdf_path)
    plot_speedup(app_name, series, accuracy_cutoff, speed_path,
                 annotate=annotate, dump_path=dump_path)


# ===========================================================================
# Matrix plot, search-space table, knobs list
# ===========================================================================

def plot_matrix(apps: list[str], devices: list[str],
                accuracy_cutoffs: dict, output_path: str,
                transpose: bool = False) -> None:
    """Grid of max-speedup-over-time curves with a shared legend.

    Default (transpose=False): rows = devices, cols = apps  (2 rows × 4 cols)
    Transposed  (transpose=True):  rows = apps,    cols = devices  (4 rows × 2 cols)
    """
    if transpose:
        row_keys, col_keys = apps, devices
        figsize = (5 * len(col_keys), 4 * len(row_keys))
    else:
        row_keys, col_keys = devices, apps
        figsize = (5 * len(col_keys), 4 * len(row_keys))

    nrows, ncols = len(row_keys), len(col_keys)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    legend_handles: dict[str, tuple] = {}
    baseline_line = None

    for row_i, rkey in enumerate(row_keys):
        for col_i, ckey in enumerate(col_keys):
            ax = axes[row_i][col_i]
            device   = ckey if transpose else rkey
            app_name = rkey if transpose else ckey

            app_cfg    = APPS[app_name]
            app_series = app_cfg.get("series", SERIES)
            cutoff     = accuracy_cutoffs.get(app_name, app_cfg["accuracy_target"])
            prefix     = app_cfg["db_prefix"]

            try:
                series = latest_dbs(prefix, device=device, series=app_series)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=14)
                ax.set_xticks([]); ax.set_yticks([])
                if row_i == 0:
                    ax.set_title(device.upper() if transpose else app_cfg["label_prefix"],
                                 fontsize=16, fontweight="bold")
                if col_i == 0:
                    ax.set_ylabel(app_cfg["label_prefix"] if transpose else device.upper(),
                                  fontsize=16, fontweight="bold")
                continue

            baseline = resolve_baseline(app_name, device, series)
            if baseline is None:
                ax.text(0.5, 0.5, "No seed", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                continue

            cfg_reject = app_cfg.get("config_reject")
            for db_path, label, color, ls in series:
                df = _load_df(db_path, cutoff, config_reject=cfg_reject)
                hours, ms = _max_speedup_trace(df, baseline)
                if len(hours) == 0:
                    continue
                ax.step(hours, ms, where="post", color=color, linewidth=1.5, linestyle=ls)
                ax.scatter(hours, ms, color=color, s=15, zorder=5)
                if label not in legend_handles:
                    legend_handles[label] = (color, ls)

            bl = ax.axhline(1.0, color="gray", linewidth=1, linestyle=":", alpha=0.5)
            if baseline_line is None:
                baseline_line = bl

            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}h"))
            ax.set_xlim(left=0, right=PLOT_XLIM_HOURS)
            ax.tick_params(axis="both", labelsize=13)
            ax.grid(True, alpha=0.2)

            # Column headers (top row)
            if row_i == 0:
                title = device.upper() if transpose else app_cfg["label_prefix"]
                ax.set_title(title, fontsize=16, fontweight="bold")
            # Row labels (left column)
            if col_i == 0:
                if transpose:
                    # App name horizontal, above the rotated y-axis label
                    ax.set_ylabel("Max speedup (×)", fontsize=14)
                    ax.text(-0.28, 0.5, app_cfg["label_prefix"],
                            transform=ax.transAxes, rotation=0,
                            ha="right", va="center",
                            fontsize=16, fontweight="bold")
                else:
                    ax.set_ylabel(f"{device.upper()}\nMax speedup (×)", fontsize=14)
            if row_i == nrows - 1:
                ax.set_xlabel("Hours", fontsize=14)

    # Shared legend below the grid
    handles = [
        plt.Line2D([0], [0], color=c, linewidth=2, linestyle=ls, label=lbl)
        for lbl, (c, ls) in legend_handles.items()
    ]
    handles.append(plt.Line2D([0], [0], color="gray", linewidth=1,
                               linestyle=":", label="baseline (1×)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
               fontsize=16, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved matrix plot: {output_path}")
    plt.close()


def dump_search_space_table(apps: list[str], devices: list[str],
                             output_path: str) -> None:
    """Write a text table of search-space sizes for all (app, device, condition) combos."""
    # Build rows: one per condition; columns: one per (app, device) pair
    cols = [(a, d) for a in apps for d in devices]
    col_headers = [f"{APPS[a]['label_prefix']}\n{d.upper()}" for a, d in cols]

    rows_data = {}
    for _, _hint, db_suffix, label, _, _ in SERIES:
        rows_data[label] = {}
        for app_name, device in cols:
            app_cfg = APPS[app_name]
            prefix = app_cfg["db_prefix"]
            app_series = app_cfg.get("series", SERIES)
            # find the matching suffix in this app's series
            this_suffix = next(
                (s for _, _, s, lbl, _, _ in app_series if lbl == label),
                db_suffix
            )
            db_path = best_db(prefix, this_suffix, device=device, db_dir=DB_DIR)
            ss = (search_space_size(db_path) or "—") if db_path else "—"
            rows_data[label][(app_name, device)] = ss

    with open(output_path, "w") as f:
        f.write("ApproxHDC — Approximation Search Space Sizes\n")
        f.write("=" * 80 + "\n\n")
        # Header
        cw = 18
        f.write(f"{'Condition':<28}")
        for a, d in cols:
            hdr = f"{APPS[a]['label_prefix']} {d.upper()}"
            f.write(f"  {hdr:<{cw}}")
        f.write("\n")
        f.write("-" * (28 + (cw + 2) * len(cols)) + "\n")
        for _, _hint, _, label, _, _ in SERIES:
            f.write(f"{label:<28}")
            for key in cols:
                val = rows_data[label].get(key, "—")
                f.write(f"  {val:<{cw}}")
            f.write("\n")

    print(f"  Search space table: {output_path}")


def dump_knobs_list(apps: list[str], output_path: str) -> None:
    """For each app, decode a sample config from the redperf_on DB and list all knob names."""
    with open(output_path, "w") as f:
        f.write("ApproxHDC — Approximation Knobs per Application\n")
        f.write("(Knob names from redperf ON condition, no no-approx hint)\n")
        f.write("=" * 80 + "\n\n")

        for app_name in apps:
            app_cfg    = APPS[app_name]
            prefix     = app_cfg["db_prefix"]
            app_series = app_cfg.get("series", SERIES)

            # Prefer GPU redperf_on, fall back to CPU or any device
            db_path = best_db(prefix, "redperf_on", device="gpu", db_dir=DB_DIR) or \
                      best_db(prefix, "redperf_on", device="cpu", db_dir=DB_DIR)

            f.write(f"{app_cfg['label_prefix']}\n")
            f.write("-" * 40 + "\n")

            if not db_path:
                f.write("  (no DB found)\n\n")
                continue

            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT data FROM configuration LIMIT 20").fetchall()
            conn.close()

            param_info: dict[str, set] = {}
            for (data,) in rows:
                cfg_dict = decode_config(data)
                if not cfg_dict:
                    continue
                for k, v in cfg_dict.items():
                    param_info.setdefault(k, set()).add(type(v).__name__)

            if not param_info:
                f.write("  (could not decode configs)\n\n")
                continue

            for name in sorted(param_info):
                vtype = "/".join(sorted(param_info[name]))
                f.write(f"  {name}  [{vtype}]\n")
            f.write("\n")

    print(f"  Knobs list: {output_path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", choices=["clustering", "classification", "graph", "hashtable", "all"],
                        default="all",
                        help="Which application(s) to run (default: all)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Generate plots from the most recent DBs without re-running experiments")
    parser.add_argument("--accuracy-cutoff", type=float, default=None, metavar="FLOAT",
                        help="Accuracy cutoff for plots (default: per-app target)")
    parser.add_argument("--accuracy-target", type=float, default=None, metavar="FLOAT",
                        help="Override accuracy target for tuning (default: per-app default)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default=DEVICE_DEFAULT,
                        help=f"Device to use (default: {DEVICE_DEFAULT})")
    parser.add_argument("--stop-after", type=int, default=STOP_AFTER_DEFAULT, metavar="SECONDS",
                        help=f"Tuning budget per run in seconds (default: {STOP_AFTER_DEFAULT})")
    parser.add_argument("--venv-python", type=str, default=None, metavar="PATH",
                        help="Path to python3.11 binary to use (default: venv/bin/python3.11 in monorepo)")
    parser.add_argument("--db-dir", type=str, default=None, metavar="PATH",
                        help="Directory for SQLite DBs (use a local path to avoid SQLite-on-NFS issues)")
    parser.add_argument("--db-copy-dir", type=str, default=None, metavar="PATH",
                        help="After each condition, copy DBs here (e.g. shared results/dbs/)")
    parser.add_argument("--annotate-configs", action="store_true",
                        help="Annotate speedup plot with config sequence numbers at each new best; "
                             "also dump decoded configs to a .txt file in results/attachments/")
    parser.add_argument("--plot-matrix", action="store_true",
                        help="Generate BOTH matrix orientations (horizontal rows=GPU/CPU cols=apps, "
                             "and vertical rows=apps cols=GPU/CPU) with shared legend; "
                             "also dump search-space size table and knobs list")
    parser.add_argument("--baseline-override", action="store_true",
                        help="Use per-app/device `baseline_override` values from APPS (e.g. HD-Clustering's "
                             "slowest redperf-off config that just reaches NMI 0.67) instead of the default "
                             "seed_time baseline. Off by default.")
    parser.add_argument("--series-filter", type=str, default=None, metavar="SUFFIXES",
                        help="Comma-separated db_suffixes to run (e.g. 'redperf_endonly,redperf_off'). "
                             "Default: run all series for the app.")
    parser.add_argument("--db-timestamp", type=str, default=None, metavar="STAMP",
                        help="Reuse an existing DB timestamp (e.g. '20260816_224322') instead of "
                             "minting a new one. Use with --series-filter to rerun only some "
                             "series into an existing set, keeping the others intact.")
    parser.add_argument("--print-best-configs", action="store_true",
                        help="Print best-config summary (speedup, accuracy, dominant knobs) "
                             "for the end-only condition across all apps and both GPU/CPU")
    args = parser.parse_args()

    global VENV_PYTHON, DB_DIR, DB_COPY_DIR, USE_BASELINE_OVERRIDE
    USE_BASELINE_OVERRIDE = args.baseline_override
    if args.venv_python:
        VENV_PYTHON = args.venv_python
        os.environ['HDC_VENV_PYTHON'] = args.venv_python
    if args.db_dir:
        DB_DIR = args.db_dir
        os.makedirs(DB_DIR, exist_ok=True)
    if args.db_copy_dir:
        DB_COPY_DIR = args.db_copy_dir

    apps_to_run = list(APPS.keys()) if args.app == "all" else [args.app]

    # ------------------------------------------------------------------
    # Matrix plot mode
    # ------------------------------------------------------------------
    if args.plot_matrix:
        print("HDC Reduction Perforation — Matrix Plot + Search Space Table + Knobs List")
        cutoffs = {a: (args.accuracy_cutoff or APPS[a].get("plot_cutoff") or APPS[a]["accuracy_target"]) for a in apps_to_run}
        ss_path    = os.path.join(OUT_DIR, "redperf_search_spaces.txt")
        knobs_path = os.path.join(OUT_DIR, "redperf_knobs.txt")
        h_path = os.path.join(OUT_DIR, "redperf_matrix.pdf")
        plot_matrix(apps_to_run, ["gpu", "cpu"], cutoffs, h_path, transpose=False)
        v_path = os.path.join(OUT_DIR, "redperf_matrix_vertical.pdf")
        plot_matrix(apps_to_run, ["gpu", "cpu"], cutoffs, v_path, transpose=True)
        dump_search_space_table(apps_to_run, ["gpu", "cpu"], ss_path)
        dump_knobs_list(apps_to_run, knobs_path)
        return 0

    # ------------------------------------------------------------------
    # Print best configs mode
    # ------------------------------------------------------------------
    if args.print_best_configs:
        print("HDC ApproxHDC — Best Configurations (end-only condition)")
        cutoffs = {a: (args.accuracy_cutoff or APPS[a].get("plot_cutoff") or APPS[a]["accuracy_target"]) for a in apps_to_run}
        for device in ["gpu", "cpu"]:
            for app_name in apps_to_run:
                app_cfg    = APPS[app_name]
                app_series = app_cfg.get("series", SERIES)
                cutoff     = cutoffs[app_name]
                # Find the end-only series entry
                endonly_label = next(
                    (lbl for lbl in [s[3] for s in app_series] if "end-only" in lbl), None
                )
                if endonly_label is None:
                    print(f"\n[{device.upper()}] {app_cfg['label_prefix']}: no end-only series")
                    continue
                try:
                    series = latest_dbs(app_cfg["db_prefix"], device=device, series=app_series)
                except FileNotFoundError:
                    print(f"\n[{device.upper()}] {app_cfg['label_prefix']}: no DB found")
                    continue

                endonly_series = [(db, lbl, c, ls) for db, lbl, c, ls in series
                                  if "end-only" in lbl]
                if not endonly_series:
                    continue

                db_path = endonly_series[0][0]
                baseline = resolve_baseline(app_name, device, series)
                if baseline is None:
                    print(f"\n[{device.upper()}] {app_cfg['label_prefix']}: no seed time")
                    continue

                df = _load_df_with_configs(db_path, cutoff)
                if df.empty:
                    print(f"\n[{device.upper()}] {app_cfg['label_prefix']}: no valid configs")
                    continue

                df = df.sort_values("collection_date").reset_index(drop=True)
                t0 = df["collection_date"].iloc[0]
                df["hours"]   = (df["collection_date"] - t0).dt.total_seconds() / 3600
                df["speedup"] = baseline / df["time"]
                best_row = df.loc[df["speedup"].idxmax()]
                speedup  = best_row["speedup"]
                accuracy = best_row["accuracy"]
                hours    = best_row["hours"]
                cfg      = best_row["config"] or {}

                print(f"\n[{device.upper()}] {app_cfg['label_prefix']}  "
                      f"speedup={speedup:.2f}×  accuracy={accuracy:.4f}  "
                      f"found_at={hours:.2f}h")
                for k, v in sorted(cfg.items()):
                    # Shorten LLVM-mangled op names
                    k_short = k.split("+")[-1] if "+" in k else k
                    print(f"    {k_short} = {v}")
        return 0

    # ------------------------------------------------------------------
    # Plot-only mode
    # ------------------------------------------------------------------
    if args.plot_only:
        print("HDC Reduction Perforation — Plot Only")
        print(f"Apps: {', '.join(apps_to_run)}\n")
        for app_name in apps_to_run:
            cutoff = args.accuracy_cutoff or APPS[app_name]["accuracy_target"]
            app_series = APPS[app_name].get("series", SERIES)
            try:
                series = latest_dbs(APPS[app_name]["db_prefix"], device=args.device,
                                    series=app_series)
                print(f"Using ({app_name}):")
                for db_path, label, _, _ in series:
                    print(f"  {label}: {db_path}")
                print()
                plot_app(app_name, series, cutoff, device=args.device,
                         annotate=args.annotate_configs)
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
        return 0

    # ------------------------------------------------------------------
    # Tuning mode
    # ------------------------------------------------------------------
    _series_filter = args.series_filter.split(",") if args.series_filter else None
    def _count_series(a):
        s = APPS[a].get("series", SERIES)
        if _series_filter:
            s = [x for x in s if x[2] in _series_filter]
        return len(s)
    n_runs = sum(_count_series(a) for a in apps_to_run)
    total_h = args.stop_after * n_runs // 3600
    print("HDC Autotuning — Reduction Perforation Comparison")
    print(f"Started:    {datetime.datetime.now()}")
    print(f"Apps:       {', '.join(apps_to_run)}")
    print(f"Device:     {args.device.upper()}")
    print(f"Budget:     {args.stop_after // 3600}h per run × {n_runs} runs = ~{total_h}h total")
    print(f"DBs:        {DB_DIR}/\n")

    timestamp = args.db_timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.db_timestamp:
        print(f"Reusing DB timestamp: {timestamp}")
    tune_results = {}

    series_filter = args.series_filter.split(",") if args.series_filter else None
    for app_name in apps_to_run:
        tune_results[app_name] = run_app(app_name, args.device, args.stop_after, timestamp,
                                         accuracy_target_override=args.accuracy_target,
                                         series_filter=series_filter)

    # Summary
    print(f"\n{'='*80}")
    print("Tuning Summary")
    print(f"{'='*80}")
    overall_ok = True
    for app_name, r in tune_results.items():
        print(f"  {app_name}:")
        for db_suffix, info in r.items():
            ok = info["rc"] == 0
            overall_ok = overall_ok and ok
            print(f"    {db_suffix:20s} — {'SUCCESS' if ok else 'FAILED'}  →  {info['db']}")
    print(f"{'='*80}\n")

    # Generate plots
    print("Generating plots...\n")
    for app_name, r in tune_results.items():
        app_series = APPS[app_name].get("series", SERIES)
        series = [(info["db"], lbl, color, ls)
                  for (_, _hint, db_suffix, lbl, color, ls), info
                  in zip(app_series, r.values())
                  if os.path.exists(info["db"])]
        if series:
            cutoff = args.accuracy_cutoff or APPS[app_name]["accuracy_target"]
            plot_app(app_name, series, cutoff, device=args.device,
                     annotate=args.annotate_configs)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
