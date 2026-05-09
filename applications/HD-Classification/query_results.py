#!/usr/bin/env python3
"""
Query and analyze results from HD-Classification sweep experiments.
"""

import sqlite3
import pandas as pd
import argparse
import os

DB_PATH = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Classification/results/sweep_results.db"


def get_all_results():
    """Fetch all results from database."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT id, timestamp, device, encoding_dim, 
               training_redperf_extent, inference_redperf_extent,
               train_iterations, time_seconds, accuracy, energy_joules,
               compile_time_seconds, training_time_seconds, inference_time_seconds,
               success, error_message
        FROM sweep_results
        ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_successful_results():
    """Fetch only successful results."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT id, timestamp, device, encoding_dim, 
               training_redperf_extent, inference_redperf_extent,
               train_iterations, time_seconds, accuracy, energy_joules,
               training_time_seconds, inference_time_seconds
        FROM sweep_results
        WHERE success = 1
        ORDER BY accuracy DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def print_summary():
    """Print summary statistics of all experiments."""
    df = get_all_results()
    
    if df.empty:
        print("No results found in database.")
        return
    
    print(f"\n{'='*80}")
    print(f"HD-CLASSIFICATION SWEEP RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Database: {DB_PATH}")
    print(f"Total experiments: {len(df)}")
    print(f"Successful: {df['success'].sum()}")
    print(f"Failed: {(~df['success']).sum()}")
    
    # Successful results statistics
    success_df = df[df['success'] == 1]
    
    if not success_df.empty:
        print(f"\n{'='*80}")
        print(f"SUCCESSFUL EXPERIMENTS STATISTICS")
        print(f"{'='*80}\n")
        
        print("Accuracy:")
        print(f"  Min: {success_df['accuracy'].min():.4f}")
        print(f"  Max: {success_df['accuracy'].max():.4f}")
        print(f"  Mean: {success_df['accuracy'].mean():.4f}")
        print(f"  Std: {success_df['accuracy'].std():.4f}")
        
        print("\nExecution Time (seconds):")
        print(f"  Min: {success_df['time_seconds'].min():.2f}")
        print(f"  Max: {success_df['time_seconds'].max():.2f}")
        print(f"  Mean: {success_df['time_seconds'].mean():.2f}")
        print(f"  Std: {success_df['time_seconds'].std():.2f}")
        
        print("\nEnergy (joules):")
        print(f"  Min: {success_df['energy_joules'].min():.2f}")
        print(f"  Max: {success_df['energy_joules'].max():.2f}")
        print(f"  Mean: {success_df['energy_joules'].mean():.2f}")
        print(f"  Std: {success_df['energy_joules'].std():.2f}")
        
        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS BY ACCURACY")
        print("="*80 + "\n")
        
        top_accuracy = success_df.nlargest(10, 'accuracy')
        for idx, row in top_accuracy.iterrows():
            print(f"Rank {top_accuracy.index.get_loc(idx) + 1}:")
            print(f"  Encoding Dim: {row['encoding_dim']}")
            print(f"  Training Extent: {row['training_redperf_extent']:.2f}")
            print(f"  Inference Extent: {row['inference_redperf_extent']:.2f}")
            print(f"  Accuracy: {row['accuracy']:.4f}")
            print(f"  Time: {row['time_seconds']:.2f}s")
            print(f"  Energy: {row['energy_joules']:.2f}J")
            print()
        
        print("="*80)
        print("TOP 10 CONFIGURATIONS BY SPEED (lowest time)")
        print("="*80 + "\n")
        
        top_speed = success_df.nsmallest(10, 'time_seconds')
        for idx, row in top_speed.iterrows():
            print(f"Rank {top_speed.index.get_loc(idx) + 1}:")
            print(f"  Encoding Dim: {row['encoding_dim']}")
            print(f"  Training Extent: {row['training_redperf_extent']:.2f}")
            print(f"  Inference Extent: {row['inference_redperf_extent']:.2f}")
            print(f"  Time: {row['time_seconds']:.2f}s")
            print(f"  Accuracy: {row['accuracy']:.4f}")
            print(f"  Energy: {row['energy_joules']:.2f}J")
            print()
        
        print("="*80)
        print("TOP 10 CONFIGURATIONS BY ENERGY EFFICIENCY (lowest energy)")
        print("="*80 + "\n")
        
        top_energy = success_df.nsmallest(10, 'energy_joules')
        for idx, row in top_energy.iterrows():
            print(f"Rank {top_energy.index.get_loc(idx) + 1}:")
            print(f"  Encoding Dim: {row['encoding_dim']}")
            print(f"  Training Extent: {row['training_redperf_extent']:.2f}")
            print(f"  Inference Extent: {row['inference_redperf_extent']:.2f}")
            print(f"  Energy: {row['energy_joules']:.2f}J")
            print(f"  Time: {row['time_seconds']:.2f}s")
            print(f"  Accuracy: {row['accuracy']:.4f}")
            print()


def export_to_csv(output_file):
    """Export all results to CSV file."""
    df = get_all_results()
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} results to {output_file}")


def query_by_encoding_dim(encoding_dim):
    """Query results for a specific encoding dimension."""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT encoding_dim, training_redperf_extent, inference_redperf_extent,
               time_seconds, accuracy, energy_joules
        FROM sweep_results
        WHERE success = 1 AND encoding_dim = {encoding_dim}
        ORDER BY accuracy DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\nResults for Encoding Dimension = {encoding_dim}:")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Query HD-Classification sweep results")
    parser.add_argument('--summary', action='store_true',
                       help='Print summary statistics')
    parser.add_argument('--export-csv', type=str, metavar='FILE',
                       help='Export results to CSV file')
    parser.add_argument('--encoding-dim', type=int,
                       help='Query results for specific encoding dimension')
    parser.add_argument('--list', action='store_true',
                       help='List all results')
    
    args = parser.parse_args()
    
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        print("Run sweep_experiment.py first to generate results.")
        return
    
    if args.summary:
        print_summary()
    elif args.export_csv:
        export_to_csv(args.export_csv)
    elif args.encoding_dim:
        query_by_encoding_dim(args.encoding_dim)
    elif args.list:
        df = get_all_results()
        print(df.to_string())
    else:
        # Default: show summary
        print_summary()


if __name__ == '__main__':
    main()
