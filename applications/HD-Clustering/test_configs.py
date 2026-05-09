#!/usr/bin/env python3
"""
Test HD-Clustering with different DATA_TYPE and HAMMING_DIST configurations.
Tests all 4 combinations and saves results to a text file.
"""

import subprocess
import json
import os
import datetime

# Configuration
ROOT_DIR = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Clustering"
CONFIG_PATH = os.path.join(ROOT_DIR, "config/custom_config_in.json")
AUTOTUNE_SCRIPT = os.path.join(ROOT_DIR, "autotune.py")
RESULTS_FILE = os.path.join(ROOT_DIR, "config_test_results.txt")

# Dataset paths (relative to application directory)
DATASET_X = "../../datasets/clustering/isolet_train_trainX.bin"
DATASET_Y = "../../datasets/clustering/isolet_train_trainY.bin"

# Test configurations: (DATA_TYPE, HAMMING_DIST)
# DATA_TYPE: 0 = int, 1 = float
# HAMMING_DIST: 0 = standard, 1 = approximation, 2 = another variant
test_configs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
]

def create_config(data_type, hamming_dist):
    """Create a custom config JSON with specified DATA_TYPE and HAMMING_DIST."""
    config = {
        "CUSTOM+DATA_TYPE": data_type,
        "CUSTOM+HAMMING_DIST": hamming_dist,
        "CUSTOM+TRAIN_ITERATIONS": 10,
        "GLOBAL_EncodingDim": 1024,
        "Z15clustering_nodeIL+llvm.hpvm.hdc.hamming.distance+0+hmhv+RedPerfBase": 0.0,
        "Z15clustering_nodeIL+llvm.hpvm.hdc.hamming.distance+0+hmhv+RedPerfExtent": 1.0,
        "Z15clustering_nodeIL+llvm.hpvm.hdc.hamming.distance+0+hmhv+RedPerfStride": 0.0001
    }
    return config

def run_test(data_type, hamming_dist):
    """Run autotune.py with direct-eval for given configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: DATA_TYPE={data_type}, HAMMING_DIST={hamming_dist}")
    print(f"{'='*80}\n")
    
    # Create config file
    config = create_config(data_type, hamming_dist)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config file: {CONFIG_PATH}")
    print(f"Config: {json.dumps(config, indent=2)}\n")
    
    # Run autotune.py with --direct-eval
    cmd = [
        "python3",
        AUTOTUNE_SCRIPT,
        "--dataset-x", DATASET_X,
        "--dataset-y", DATASET_Y,
        "--device", "cpu",
        "--direct-eval", CONFIG_PATH
    ]
    
    print(f"Running command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        return {
            'data_type': data_type,
            'hamming_dist': hamming_dist,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'data_type': data_type,
            'hamming_dist': hamming_dist,
            'returncode': -1,
            'stdout': '',
            'stderr': 'TIMEOUT: Test exceeded 5 minutes'
        }
    except Exception as e:
        return {
            'data_type': data_type,
            'hamming_dist': hamming_dist,
            'returncode': -1,
            'stdout': '',
            'stderr': f'ERROR: {str(e)}'
        }

def main():
    """Run all test configurations and save results."""
    print(f"HD-Clustering Configuration Testing")
    print(f"Started at: {datetime.datetime.now()}")
    print(f"Results will be saved to: {RESULTS_FILE}\n")
    
    results = []
    
    # Run all tests
    for data_type, hamming_dist in test_configs:
        result = run_test(data_type, hamming_dist)
        results.append(result)
        
        # Print immediate feedback
        if result['returncode'] == 0:
            print(f"✓ Test completed successfully")
        else:
            print(f"✗ Test failed with return code {result['returncode']}")
    
    # Write results to file
    print(f"\n{'='*80}")
    print(f"Writing results to {RESULTS_FILE}")
    print(f"{'='*80}\n")
    
    with open(RESULTS_FILE, 'w') as f:
        f.write(f"HD-Clustering Configuration Test Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Test Date: {datetime.datetime.now()}\n")
        f.write(f"Dataset: {DATASET_X}, {DATASET_Y}\n")
        f.write(f"Device: CPU\n")
        f.write(f"{'='*80}\n\n")
        
        for result in results:
            f.write(f"\n{'='*80}\n")
            f.write(f"Configuration: DATA_TYPE={result['data_type']}, HAMMING_DIST={result['hamming_dist']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Return Code: {result['returncode']}\n\n")
            
            f.write(f"--- STDOUT ---\n")
            f.write(result['stdout'])
            f.write(f"\n\n--- STDERR ---\n")
            f.write(result['stderr'])
            f.write(f"\n\n")
    
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Completed at: {datetime.datetime.now()}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for r in results if r['returncode'] == 0)
    failed = len(results) - successful
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == '__main__':
    main()
