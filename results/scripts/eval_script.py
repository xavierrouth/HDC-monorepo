#!/usr/bin/env python3
"""
Evaluation script to run HD-Clustering autotuning for both CPU and GPU configurations.
"""

import subprocess
import os
import sys

# Configuration
APP_DIR = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Clustering"
AUTOTUNE_SCRIPT = os.path.join(APP_DIR, "autotune.py")

# Database path - store in results directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "dbs/tuning.db")

# Dataset paths
DATASET_X = "../../datasets/clustering/isolet_train_trainX.bin"
DATASET_Y = "../../datasets/clustering/isolet_train_trainY.bin"

# Tuning parameters
ACCURACY_TARGET = 0.62
TEST_LIMIT = 100  # Number of configurations to test

# Ensure database directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def run_autotune(device, dataset_x, dataset_y, accuracy_target, test_limit):
    """
    Run autotuning for specified device configuration.
    
    Args:
        device: 'cpu' or 'gpu'
        dataset_x: Path to X dataset file
        dataset_y: Path to Y dataset file
        accuracy_target: Target accuracy threshold
        test_limit: Maximum number of configurations to test
    """
    print(f"\n{'='*80}")
    print(f"Running HD-Clustering Autotuning - Device: {device.upper()}")
    print(f"{'='*80}\n")
    
    cmd = [
        "python3",
        AUTOTUNE_SCRIPT,
        "--dataset-x", dataset_x,
        "--dataset-y", dataset_y,
        "--device", device,
        "--accuracy-target", str(accuracy_target),
        "--test-limit", str(test_limit),
        "--no-dups",  # OpenTuner flag to avoid duplicate configurations
        "--database", DB_PATH
    ]
    
    # Create label from the full command
    label = " ".join(cmd)
    cmd.extend(["--label", label])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, cwd=APP_DIR, check=True)
        print(f"\n{device.upper()} autotuning completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n{device.upper()} autotuning failed with error code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n{device.upper()} autotuning interrupted by user")
        return 1

def main():
    """Main execution function."""
    print("HD-Clustering Autotuning Evaluation Script")
    print("=" * 80)
    
    # Run CPU autotuning
    cpu_result = run_autotune(
        device="cpu",
        dataset_x=DATASET_X,
        dataset_y=DATASET_Y,
        accuracy_target=ACCURACY_TARGET,
        test_limit=TEST_LIMIT
    )
    
    # Run GPU autotuning
    gpu_result = run_autotune(
        device="gpu",
        dataset_x=DATASET_X,
        dataset_y=DATASET_Y,
        accuracy_target=ACCURACY_TARGET,
        test_limit=TEST_LIMIT
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}")
    print(f"CPU Result: {'SUCCESS' if cpu_result == 0 else 'FAILED'}")
    print(f"GPU Result: {'SUCCESS' if gpu_result == 0 else 'FAILED'}")
    print(f"{'='*80}\n")
    
    return 0 if (cpu_result == 0 and gpu_result == 0) else 1

if __name__ == "__main__":
    sys.exit(main())
