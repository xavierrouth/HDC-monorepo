#!/usr/bin/env python3
"""
Run baseline (no-approx) accuracy check for HD-Classification and HD-Clustering
across all configured datasets.

Usage:
  python3 baseline_accuracy.py [--app classification|clustering|all] [--device cpu|gpu]
"""
import argparse
import json
import os
import subprocess
import sys

MONOREPO = "/home/xrouth2/hdc/HDC-monorepo"

CLASSIFICATION_DIR = f"{MONOREPO}/applications/HD-Classification"
CLUSTERING_DIR     = f"{MONOREPO}/applications/HD-Clustering"
GRAPH_DIR          = f"{MONOREPO}/applications/HD-Graph"

CLASSIFICATION_DATASETS = {
    "ISOLET": {
        "X_train": f"{MONOREPO}/datasets/classification/isolet_nq_trainX.bin",
        "y_train": f"{MONOREPO}/datasets/classification/isolet_nq_trainY.bin",
        "X_test":  f"{MONOREPO}/datasets/classification/isolet_nq_testX.bin",
        "y_test":  f"{MONOREPO}/datasets/classification/isolet_nq_testY.bin",
        "n_feat": 617, "n_sample": 6238, "n_test": 1559, "n_class": 26, "n_feat_pad": 624,
    },
    "UCIHAR": {
        "X_train": f"{MONOREPO}/datasets/classification/ucihar_trainX.bin",
        "y_train": f"{MONOREPO}/datasets/classification/ucihar_trainY.bin",
        "X_test":  f"{MONOREPO}/datasets/classification/ucihar_testX.bin",
        "y_test":  f"{MONOREPO}/datasets/classification/ucihar_testY.bin",
        "n_feat": 568, "n_sample": 6213, "n_test": 1554, "n_class": 12, "n_feat_pad": 568,
    },
}

CLUSTERING_DATASETS = {
    "ISOLET": {
        "X": f"{MONOREPO}/datasets/clustering/isolet_train_trainX.bin",
        "Y": f"{MONOREPO}/datasets/clustering/isolet_train_trainY.bin",
        "n_feat": 624, "n_sample": 6238, "n_center": 26,
    },
    "MNIST": {
        "X": f"{MONOREPO}/datasets/clustering/mnist_testX.bin",
        "Y": f"{MONOREPO}/datasets/clustering/mnist_testY.bin",
        "n_feat": 784, "n_sample": 10000, "n_center": 10,
    },
    "UCIHAR": {
        "X": f"{MONOREPO}/datasets/clustering/ucihar_trainX.bin",
        "Y": f"{MONOREPO}/datasets/clustering/ucihar_trainY.bin",
        "n_feat": 568, "n_sample": 6213, "n_center": 12,
    },
}

TRAIN_ITERATIONS    = 20
CLASSIFICATION_DIM  = 2048
CLUSTERING_DIM      = 1024
GRAPH_DIM           = 2048
HAMMING_DIST        = 0

VENV_PYTHON = "/home/xrouth2/hdc/HDC-monorepo/approx-tuner/venv/bin/python3.11"


def run(cmd, cwd=None, env=None):
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode


def write_empty_config(path):
    with open(path, "w") as f:
        json.dump([], f)


def check_classification(device="cpu"):
    results = {}
    app_dir = CLASSIFICATION_DIR
    config_in = f"{app_dir}/config/config_in.json"
    make_target = f"host-{device}-approx"
    executable  = f"./{make_target}"
    env = os.environ.copy()
    env["CONFIG_IN_PATH"]  = config_in
    env["CONFIG_OUT_PATH"] = f"{app_dir}/config/config_out.json"

    os.makedirs(f"{app_dir}/output", exist_ok=True)

    for name, ds in CLASSIFICATION_DATASETS.items():
        print(f"\n=== Classification / {name} ===")
        write_empty_config(config_in)
        run(["make", f"--directory={app_dir}", "clean"], env=env)
        ret = run(["make", f"--directory={app_dir}", make_target, "-B",
                   f"ENCODING_DIM={CLASSIFICATION_DIM}",
                   f"HAMMING_DIST={HAMMING_DIST}",
                   "DATA_TYPE=float",
                   "BINARIZE_ENCODING=0",
                   f"N_FEAT={ds['n_feat']}",
                   f"N_SAMPLE={ds['n_sample']}",
                   f"N_TEST={ds['n_test']}",
                   f"N_CLASS={ds['n_class']}",
                   f"N_FEAT_PAD={ds['n_feat_pad']}"],
                  env=env)
        if ret != 0:
            print(f"  BUILD FAILED")
            results[name] = None
            continue

        acc_file = f"{app_dir}/output/test_accuracy.txt"
        open(acc_file, "w").close()  # clear stale value
        run([executable, str(TRAIN_ITERATIONS),
             ds["X_train"], ds["y_train"], ds["X_test"], ds["y_test"]],
            cwd=app_dir, env=env)

        try:
            with open(acc_file) as f:
                acc = float(f.readline().strip())
            print(f"  accuracy: {acc:.4f}")
            results[name] = acc
        except Exception as e:
            print(f"  could not read accuracy: {e}")
            results[name] = None

    return results


def check_clustering(device="cpu"):
    results = {}
    app_dir = CLUSTERING_DIR
    config_in = f"{app_dir}/config/config_in.json"
    make_target = f"host-{device}-approx"
    executable  = f"./{make_target}"
    env = os.environ.copy()
    env["CONFIG_IN_PATH"]  = config_in
    env["CONFIG_OUT_PATH"] = f"{app_dir}/config/config_out.json"

    os.makedirs(f"{app_dir}/output", exist_ok=True)

    for name, ds in CLUSTERING_DATASETS.items():
        print(f"\n=== Clustering / {name} ===")
        write_empty_config(config_in)
        run(["make", f"--directory={app_dir}", "clean"], env=env)
        ret = run(["make", f"--directory={app_dir}", make_target, "-B",
                   f"ENCODING_DIM={CLUSTERING_DIM}",
                   f"HAMMING_DIST={HAMMING_DIST}",
                   "DATA_TYPE=float",
                   f"N_FEAT={ds['n_feat']}",
                   f"N_SAMPLE={ds['n_sample']}",
                   f"N_CENTER={ds['n_center']}"],
                  env=env)
        if ret != 0:
            print(f"  BUILD FAILED")
            results[name] = None
            continue

        run([executable, str(TRAIN_ITERATIONS), ds["X"], ds["Y"]],
            cwd=app_dir, env=env)

        score_file  = f"{app_dir}/score.txt"
        score_script = f"{app_dir}/mutual_info.py"
        predictions  = f"{app_dir}/output/out.txt"
        with open(score_file, "w") as f:
            subprocess.run([VENV_PYTHON, score_script, predictions],
                           stdout=f, stderr=subprocess.STDOUT, env=env)
        try:
            with open(score_file) as f:
                acc = float(f.readline().strip())
            print(f"  accuracy (NMI): {acc:.4f}")
            results[name] = acc
        except Exception as e:
            print(f"  could not read accuracy: {e}")
            results[name] = None

    return results


def check_graph(device="cpu"):
    app_dir = GRAPH_DIR
    config_in = f"{app_dir}/config/config_in.json"
    make_target = f"host-{device}-approx"
    executable  = f"./{make_target}"
    env = os.environ.copy()
    env["CONFIG_IN_PATH"]  = config_in
    env["CONFIG_OUT_PATH"] = f"{app_dir}/config/config_out.json"

    print(f"\n=== Graph / Cora ===")
    write_empty_config(config_in)
    run(["make", f"--directory={app_dir}", "clean"], env=env)
    ret = run(["make", f"--directory={app_dir}", make_target, "-B",
               f"ENCODING_DIM={GRAPH_DIM}",
               f"HAMMING_DIST={HAMMING_DIST}",
               "DATA_TYPE=float",
               "BINARIZE=0"],
              env=env)
    if ret != 0:
        print("  BUILD FAILED")
        return {"Cora": None}

    acc_file = f"{app_dir}/test_accuracy.txt"
    open(acc_file, "w").close()
    run([executable, "1"], cwd=app_dir, env=env)

    try:
        with open(acc_file) as f:
            acc = float(f.readline().strip())
        print(f"  accuracy: {acc:.4f}")
        return {"Cora": acc}
    except Exception as e:
        print(f"  could not read accuracy: {e}")
        return {"Cora": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", choices=["classification", "clustering", "graph", "all"],
                        default="all")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()

    all_results = {}

    if args.app in ("classification", "all"):
        all_results["classification"] = check_classification(args.device)

    if args.app in ("clustering", "all"):
        all_results["clustering"] = check_clustering(args.device)

    if args.app in ("graph", "all"):
        all_results["graph"] = check_graph(args.device)

    print("\n\n=== BASELINE ACCURACY SUMMARY ===")
    for app, res in all_results.items():
        for dataset, acc in res.items():
            val = f"{acc:.4f}" if acc is not None else "FAILED"
            print(f"  {app:16s} {dataset:12s}  {val}")
