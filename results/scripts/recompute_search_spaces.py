#!/usr/bin/env python3
"""Recompute ApproxHDC search-space sizes freshly by invoking each
autotune.py with --print-search-space-size (no tuning).

--device does not affect the OpenTuner manipulator in any of the apps
(it only switches make targets at eval time), so CPU and GPU numbers
are identical per (app, series). Output has one column per app.

Run:
    source approx-tuner/venv/bin/activate
    python3 results/scripts/recompute_search_spaces.py

Do NOT run while a tuner is active in any of the app dirs — this script
invokes `make collect-approx -B` and would race a concurrent build.
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_redperf import APPS, SERIES, regenerate_config_out, VENV_PYTHON, OUT_DIR


def capture(app_dir, autotune, base_flags, extra_flags):
    res = subprocess.run(
        [VENV_PYTHON, autotune, "--print-search-space-size"] + base_flags + extra_flags,
        cwd=app_dir, capture_output=True, text=True, timeout=600,
    )
    m = re.search(r'10\^\{[\d.]+\}', res.stdout)
    if not m:
        sys.stderr.write(f"  ! no match for {app_dir} {extra_flags}\n")
        sys.stderr.write(f"  stdout tail: {res.stdout[-400:]}\n")
        sys.stderr.write(f"  stderr tail: {res.stderr[-400:]}\n")
        return "unknown"
    return m.group(0)


def main():
    apps = ["clustering", "classification", "graph", "hashtable"]
    rows = {}
    for app in apps:
        cfg = APPS[app]
        app_dir = cfg["dir"]
        autotune = os.path.join(app_dir, "autotune.py")
        base_flags = cfg["base_flags"]
        make_vars = cfg.get("collect_approx_make_vars", [])
        series = cfg.get("series", SERIES)
        print(f"\n=== {cfg['label_prefix']} ===")
        for extra_flags, hint, suffix, label, *_ in series:
            regenerate_config_out(app_dir, make_vars, hint)
            ss = capture(app_dir, autotune, base_flags, extra_flags)
            print(f"  {label:22s} {ss}")
            rows.setdefault(label, {})[app] = ss

    headers = [APPS[a]["label_prefix"] for a in apps]
    col_w = max(18, max(len(h) for h in headers) + 2)
    out_path = os.path.join(OUT_DIR, "redperf_search_spaces.txt")
    with open(out_path, "w") as f:
        f.write("ApproxHDC — Approximation Search Space Sizes (fresh recompute)\n")
        f.write("=" * 80 + "\n")
        f.write("Note: --device does not affect the OpenTuner manipulator, so CPU and GPU\n")
        f.write("numbers are identical per (app, condition). One column per application.\n\n")
        f.write(f"{'Condition':<28}" + "".join(f"{h:<{col_w}}" for h in headers) + "\n")
        f.write("-" * (28 + col_w * len(headers)) + "\n")
        ordered_labels = []
        for _, _, _, lbl, *_ in SERIES:
            if lbl in rows and lbl not in ordered_labels:
                ordered_labels.append(lbl)
        for lbl in ordered_labels:
            f.write(f"{lbl:<28}")
            for a in apps:
                val = rows[lbl].get(a, "—")
                f.write(f"{val:<{col_w}}")
            f.write("\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
