#!/usr/bin/env python3
"""
Test HD-Graph with different ENCODING_DIM, DATA_TYPE, HAMMING_DIST, and BINARIZE configurations.
Runs each configuration via autotune.py --direct-eval and saves results to a text file.
"""

import subprocess
import json
import os
import datetime

# Configuration
ROOT_DIR = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Graph"
CONFIG_PATH = os.path.join(ROOT_DIR, "config/custom_config_in.json")
AUTOTUNE_SCRIPT = os.path.join(ROOT_DIR, "autotune.py")
RESULTS_FILE = os.path.join(ROOT_DIR, "config_test_results.txt")

# Encoding dimensions to test
ENCODING_DIMS = [256, 512, 1024, 2048, 4096]

# Data types to test (C++ type names passed as -DDATA_TYPE=...)
DATA_TYPES = ["int8_t", "int16_t", "int32_t", "float"]

# HAMMING_DIST values: 0 = cosine similarity, 1 = hamming distance
HAMMING_DISTS = [0, 1]

# BINARIZE values: 0 = no binarization, 1 = binarize
BINARIZE_VALS = [0, 1]


def create_config(encoding_dim, data_type, hamming_dist, binarize):
    """Create a config JSON with specified ENCODING_DIM, DATA_TYPE, HAMMING_DIST, and BINARIZE."""
    config = {
        "CUSTOM+DATA_TYPE": data_type,
        "CUSTOM+HAMMING_DIST": hamming_dist,
        "CUSTOM+BINARIZE": binarize,
        "GLOBAL_EncodingDim": encoding_dim,
    }
    return config


def run_test(encoding_dim, data_type, hamming_dist, binarize):
    """Run autotune.py with direct-eval for given configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: ENCODING_DIM={encoding_dim}, DATA_TYPE={data_type}, HAMMING_DIST={hamming_dist}, BINARIZE={binarize}")
    print(f"{'='*80}\n")

    # Create config file
    config = create_config(encoding_dim, data_type, hamming_dist, binarize)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created config file: {CONFIG_PATH}")
    print(f"Config: {json.dumps(config, indent=2)}\n")

    # Run autotune.py with --direct-eval
    cmd = [
        "python3",
        AUTOTUNE_SCRIPT,
        "--device", "gpu",
        "--direct-eval", CONFIG_PATH
    ]

    print(f"Running command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        return {
            'encoding_dim': encoding_dim,
            'data_type': data_type,
            'hamming_dist': hamming_dist,
            'binarize': binarize,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'encoding_dim': encoding_dim,
            'data_type': data_type,
            'hamming_dist': hamming_dist,
            'binarize': binarize,
            'returncode': -1,
            'stdout': '',
            'stderr': 'TIMEOUT: Test exceeded 10 minutes'
        }
    except Exception as e:
        return {
            'encoding_dim': encoding_dim,
            'data_type': data_type,
            'hamming_dist': hamming_dist,
            'binarize': binarize,
            'returncode': -1,
            'stdout': '',
            'stderr': f'ERROR: {str(e)}'
        }


def main():
    """Run all test configurations and save results."""
    print(f"HD-Graph Configuration Testing")
    print(f"Started at: {datetime.datetime.now()}")
    print(f"Results will be saved to: {RESULTS_FILE}\n")

    results = []

    # Run all tests (all combinations of ENCODING_DIM x DATA_TYPE x HAMMING_DIST x BINARIZE)
    for encoding_dim in ENCODING_DIMS:
        for data_type in DATA_TYPES:
            for hamming_dist in HAMMING_DISTS:
                for binarize in BINARIZE_VALS:
                    result = run_test(encoding_dim, data_type, hamming_dist, binarize)
                    results.append(result)

                    # Print immediate feedback
                    if result['returncode'] == 0:
                        print(f"  -> Test completed successfully")
                    else:
                        print(f"  -> Test failed with return code {result['returncode']}")

    # Write results to file
    print(f"\n{'='*80}")
    print(f"Writing results to {RESULTS_FILE}")
    print(f"{'='*80}\n")

    with open(RESULTS_FILE, 'w') as f:
        f.write(f"HD-Graph Configuration Test Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Test Date: {datetime.datetime.now()}\n")
        f.write(f"Device: GPU\n")
        f.write(f"{'='*80}\n\n")

        for result in results:
            f.write(f"\n{'='*80}\n")
            f.write(f"Configuration: ENCODING_DIM={result['encoding_dim']}, DATA_TYPE={result['data_type']}, HAMMING_DIST={result['hamming_dist']}, BINARIZE={result['binarize']}\n")
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