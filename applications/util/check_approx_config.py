#!/usr/bin/env python3
"""Guard against silently applying a stale config_in.json to a build.

config_in.json is written at run time by autotune.py and consumed by the
compiler's -hdc-set-approx-knobs pass on *every* subsequent build, including
hand-run `make` invocations. A file left behind by a finished tuning run
therefore silently approximates binaries that nobody asked to approximate.

This has produced badly wrong measurements: a leftover entry perforating
inference cossim to 11.4% of its reduction moved HD-Classification from
0.9044 to 0.7826 accuracy on an otherwise identical build, with no indication
anything was applied.

Policy: a non-identity config may only be applied when the caller explicitly
says so. autotune.py passes APPROX_CONFIG=1; interactive builds must either
opt in the same way or reset the file.

Usage:  check_approx_config.py <config_in.json> <APPROX_CONFIG>

Exits 0 to allow the build, 1 with an explanation to stop it.
"""

import json
import os
import sys


def is_identity(red_perf):
    """True if this red_perf applies no approximation at all."""
    return (
        float(red_perf.get("base", 0.0)) == 0.0
        and int(red_perf.get("stride", 1)) == 1
        and float(red_perf.get("length", 1.0)) == 1.0
    )


def main():
    if len(sys.argv) < 3:
        print("usage: check_approx_config.py <config_in.json> <APPROX_CONFIG>",
              file=sys.stderr)
        return 2

    path, approved = sys.argv[1], sys.argv[2].strip()

    # The tuner asked for this config; applying it is the whole point.
    if approved == "1":
        return 0

    # No config at all is fine -- the pass is a no-op.
    if not os.path.exists(path):
        return 0

    try:
        with open(path) as f:
            entries = json.load(f)
    except (ValueError, OSError) as e:
        print(f"\n  ERROR: {path} exists but could not be parsed ({e}).\n"
              f"  Delete it or regenerate it before building.\n", file=sys.stderr)
        return 1

    offenders = []
    for entry in entries or []:
        name = entry.get("parent_func", "?")
        for approx in entry.get("approximations", []):
            rp = approx.get("red_perf")
            if rp is not None and not is_identity(rp):
                offenders.append((name, rp))

    if not offenders:
        return 0

    print(f"""
  ==========================================================================
  REFUSING TO BUILD: stale approximation config

  {path}
  contains {len(offenders)} non-identity approximation(s), most likely left
  behind by a previous tuning run. Building now would silently apply them and
  produce measurements that look like a clean baseline but are not.
""", file=sys.stderr)
    for name, rp in offenders[:6]:
        print(f"    {name}:  base={rp.get('base')} "
              f"stride={rp.get('stride')} length={rp.get('length')}",
              file=sys.stderr)
    if len(offenders) > 6:
        print(f"    ... and {len(offenders) - 6} more", file=sys.stderr)
    print(f"""
  Choose one:

    1. Measuring a clean baseline -- reset the config to identity:
         python3 {os.path.relpath(__file__)} --reset {path}

    2. You really do want these approximations applied:
         make <target> APPROX_CONFIG=1 ...

    3. Running the tuner -- it passes APPROX_CONFIG=1 itself, so this
       message means it was invoked without that flag.
  ==========================================================================
""", file=sys.stderr)
    return 1


def reset(path):
    """Rewrite every red_perf in place to the no-approximation values."""
    with open(path) as f:
        entries = json.load(f)
    n = 0
    for entry in entries or []:
        for approx in entry.get("approximations", []):
            if "red_perf" in approx:
                approx["red_perf"] = {"base": 0.0, "stride": 1, "length": 1.0}
                n += 1
    with open(path, "w") as f:
        json.dump(entries, f)
    print(f"Reset {n} approximation(s) to identity in {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--reset":
        sys.exit(reset(sys.argv[2]))
    sys.exit(main())
