#!/usr/bin/env python3
"""
Sweep experiment for HD-Classification: varying RedPerfExtent and EncodingDim.
Stores results in a SQLite database along with configuration details.
"""

import subprocess
import json
import os
import datetime
import sqlite3
import itertools
import argparse
from pathlib import Path

# Configuration
ROOT_DIR = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Classification"
CONFIG_PATH = os.path.join(ROOT_DIR, "config/sweep_config.json")
AUTOTUNE_SCRIPT = os.path.join(ROOT_DIR, "autotune.py")
DB_PATH = os.path.join(ROOT_DIR, "results", "sweep_results.db")

# Sweep parameters
ENCODING_DIMS = [16, 32, 64, 128, 256, 512]
REDPERF_EXTENTS = [0.25, 0.5, 0.75, 1.0]
REDPERF_BASES = [0.0, 0.125, 0.25, 0.375, 0.5]
TRAIN_ITERATIONS = 5
DEVICE = "cpu"  # Can be changed to "gpu"

# RedPerf parameter names (from baseline config)
TRAINING_REDPERF_BASE = "Z18training_root_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfBase"
TRAINING_REDPERF_STRIDE = "Z18training_root_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfStride"
TRAINING_REDPERF_EXTENT = "Z18training_root_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfExtent"
INFERENCE_REDPERF_BASE = "Z14inference_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfBase"
INFERENCE_REDPERF_STRIDE = "Z14inference_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfStride"
INFERENCE_REDPERF_EXTENT = "Z14inference_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfExtent"


def init_database():
    """Initialize SQLite database with results table."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sweep_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            device TEXT,
            encoding_dim INTEGER,
            training_redperf_base REAL,
            training_redperf_extent REAL,
            inference_redperf_base REAL,
            inference_redperf_extent REAL,
            train_iterations INTEGER,
            time_seconds REAL,
            accuracy REAL,
            energy_joules REAL,
            compile_time_seconds REAL,
            training_time_seconds REAL,
            inference_time_seconds REAL,
            success BOOLEAN,
            error_message TEXT,
            config_json TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")


def create_config(encoding_dim, training_base, training_extent, inference_base, inference_extent, train_iterations):
    """Create a configuration dictionary."""
    config = {
        TRAINING_REDPERF_BASE: training_base,
        TRAINING_REDPERF_STRIDE: 0.0001,
        TRAINING_REDPERF_EXTENT: training_extent,
        INFERENCE_REDPERF_BASE: inference_base,
        INFERENCE_REDPERF_STRIDE: 0.0001,
        INFERENCE_REDPERF_EXTENT: inference_extent,
        "GLOBAL_EncodingDim": encoding_dim,
        "CUSTOM+TRAIN_ITERATIONS": train_iterations
    }
    return config


def run_experiment(encoding_dim, training_base, training_extent, inference_base, inference_extent, train_iterations):
    """Run a single experiment with given parameters."""
    print(f"\n{'='*80}")
    print(f"Experiment: EncodingDim={encoding_dim}, "
          f"TrainingBase={training_base}, TrainingExtent={training_extent}, "
          f"InferenceBase={inference_base}, InferenceExtent={inference_extent}")
    print(f"{'='*80}\n")
    
    # Create config file
    config = create_config(encoding_dim, training_base, training_extent, inference_base, inference_extent, train_iterations)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config file: {CONFIG_PATH}")
    
    # Run autotune.py with --direct-eval
    cmd = [
        "python3",
        AUTOTUNE_SCRIPT,
        "--device", DEVICE,
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
        
        # Parse the result from stdout
        stdout = result.stdout
        stderr = result.stderr
        
        # Try to extract results from the output
        time_val = None
        accuracy_val = None
        energy_val = None
        compile_time_val = None
        training_time_val = None
        inference_time_val = None
        
        # Look for "Result: {..." in stdout
        if "Result:" in stdout:
            try:
                result_line = stdout[stdout.find("Result:") + 7:].strip()
                result_dict = eval(result_line.split('\n')[0])
                time_val = result_dict.get('time')
                accuracy_val = result_dict.get('accuracy')
                energy_val = result_dict.get('energy')
                compile_time_val = result_dict.get('compile_time')
                training_time_val = result_dict.get('training_time')
                inference_time_val = result_dict.get('inference_time')
            except Exception as e:
                print(f"Warning: Could not parse result: {e}")
        
        return {
            'encoding_dim': encoding_dim,
            'training_base': training_base,
            'training_extent': training_extent,
            'inference_base': inference_base,
            'inference_extent': inference_extent,
            'train_iterations': train_iterations,
            'time': time_val,
            'accuracy': accuracy_val,
            'energy': energy_val,
            'compile_time': compile_time_val,
            'training_time': training_time_val,
            'inference_time': inference_time_val,
            'success': result.returncode == 0 and time_val is not None,
            'error_message': stderr if result.returncode != 0 else None,
            'config_json': json.dumps(config),
            'stdout': stdout,
            'stderr': stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'encoding_dim': encoding_dim,
            'training_base': training_base,
            'training_extent': training_extent,
            'inference_base': inference_base,
            'inference_extent': inference_extent,
            'train_iterations': train_iterations,
            'time': None,
            'accuracy': None,
            'energy': None,
            'compile_time': None,
            'training_time': None,
            'inference_time': None,
            'success': False,
            'error_message': 'TIMEOUT: Test exceeded 10 minutes',
            'config_json': json.dumps(config),
            'stdout': '',
            'stderr': 'TIMEOUT'
        }
    except Exception as e:
        return {
            'encoding_dim': encoding_dim,
            'training_base': training_base,
            'training_extent': training_extent,
            'inference_base': inference_base,
            'inference_extent': inference_extent,
            'train_iterations': train_iterations,
            'time': None,
            'accuracy': None,
            'energy': None,
            'compile_time': None,
            'training_time': None,
            'inference_time': None,
            'success': False,
            'error_message': f'ERROR: {str(e)}',
            'config_json': json.dumps(config),
            'stdout': '',
            'stderr': str(e)
        }


def save_result_to_db(result):
    """Save a single result to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO sweep_results (
            device, encoding_dim, training_redperf_base, training_redperf_extent, 
            inference_redperf_base, inference_redperf_extent,
            train_iterations, time_seconds, accuracy, energy_joules,
            compile_time_seconds, training_time_seconds, inference_time_seconds,
            success, error_message, config_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        DEVICE,
        result['encoding_dim'],
        result['training_base'],
        result['training_extent'],
        result['inference_base'],
        result['inference_extent'],
        result['train_iterations'],
        result['time'],
        result['accuracy'],
        result['energy'],
        result['compile_time'],
        result['training_time'],
        result['inference_time'],
        result['success'],
        result['error_message'],
        result['config_json']
    ))
    
    conn.commit()
    conn.close()


def main():
    """Run all sweep experiments."""
    parser = argparse.ArgumentParser(description='HD-Classification Sweep Experiment')
    parser.add_argument('--hold-inference', action='store_true',
                        help='Hold inference base at 0.0 and extent at 1.0 (only sweep training parameters)')
    parser.add_argument('--independent-inference', action='store_true',
                        help='Sweep inference parameters independently from training (full Cartesian product)')
    args = parser.parse_args()

    print(f"HD-Classification Sweep Experiment")
    print(f"{'='*80}")
    print(f"Started at: {datetime.datetime.now()}")
    print(f"Database: {DB_PATH}")
    print(f"Device: {DEVICE}")

    if args.hold_inference and args.independent_inference:
        print("Error: Cannot use both --hold-inference and --independent-inference")
        return

    if args.hold_inference:
        print(f"\nMode: Holding inference base=0.0, extent=1.0 (sweeping training only)")
        # Generate all combinations with fixed inference parameters
        experiments = [(dim, t_base, t_extent, 0.0, 1.0)
                       for dim in ENCODING_DIMS
                       for t_base in REDPERF_BASES
                       for t_extent in REDPERF_EXTENTS]
    elif args.independent_inference:
        print(f"\nMode: Full sweep of training and inference parameters independently")
        # Generate all combinations (Cartesian product)
        experiments = list(itertools.product(ENCODING_DIMS, REDPERF_BASES, REDPERF_EXTENTS, REDPERF_BASES, REDPERF_EXTENTS))
    else:
        print(f"\nMode: Inference base and extent match training base and extent (DEFAULT)")
        # Generate combinations where inference matches training (DEFAULT)
        experiments = [(dim, base, extent, base, extent)
                       for dim in ENCODING_DIMS
                       for base in REDPERF_BASES
                       for extent in REDPERF_EXTENTS]

    total_experiments = len(experiments)

    print(f"\nSweep Parameters:")
    print(f"  Encoding Dimensions: {ENCODING_DIMS}")
    print(f"  Training RedPerf Bases: {REDPERF_BASES}")
    print(f"  Training RedPerf Extents: {REDPERF_EXTENTS}")
    if args.hold_inference:
        print(f"  Inference RedPerf Base: 0.0 (fixed)")
        print(f"  Inference RedPerf Extent: 1.0 (fixed)")
    elif args.independent_inference:
        print(f"  Inference RedPerf Bases: {REDPERF_BASES} (independent)")
        print(f"  Inference RedPerf Extents: {REDPERF_EXTENTS} (independent)")
    else:
        print(f"  Inference RedPerf Bases: Match training")
        print(f"  Inference RedPerf Extents: Match training")
    print(f"  Train Iterations: {TRAIN_ITERATIONS}")
    print(f"\nTotal experiments: {total_experiments}")
    print(f"{'='*80}\n")

    # Initialize database
    init_database()

    results = []
    successful = 0
    failed = 0

    # Run all experiments
    for idx, (encoding_dim, training_base, training_extent, inference_base, inference_extent) in enumerate(experiments, 1):
        print(f"\n[{idx}/{total_experiments}] ", end='')

        result = run_experiment(encoding_dim, training_base, training_extent, inference_base, inference_extent, TRAIN_ITERATIONS)
        results.append(result)
        
        # Save to database immediately
        save_result_to_db(result)
        
        # Print immediate feedback
        if result['success']:
            successful += 1
            print(f"✓ Success: time={result['time']:.2f}s, accuracy={result['accuracy']:.4f}")
        else:
            failed += 1
            print(f"✗ Failed: {result['error_message'][:100] if result['error_message'] else 'Unknown error'}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Completed at: {datetime.datetime.now()}")
    print(f"\nResults Summary:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nResults saved to database: {DB_PATH}")
    print(f"\nTo query results, use:")
    print(f"  sqlite3 {DB_PATH}")
    print(f"  SELECT encoding_dim, training_redperf_extent, inference_redperf_extent, ")
    print(f"         time_seconds, accuracy FROM sweep_results WHERE success=1;")


if __name__ == '__main__':
    main()
