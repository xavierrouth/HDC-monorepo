#!/usr/bin/env python3
"""
Plot speedup vs accuracy for HD-Clustering tuning runs (CPU and GPU).
Reads from tuning.db and creates separate plots for each device type.
"""

import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser(description="Plot speedup vs accuracy for HD-Clustering tuning runs")
parser.add_argument("--db", type=str, default="dbs/tuning.db", 
                    help="Path to tuning database (default: dbs/tuning.db)")
parser.add_argument("--output-dir", type=str, default="attachments", 
                    help="Output directory for plots (default: attachments)")
parser.add_argument("--annotate", action="store_true", 
                    help="Annotate points with configuration IDs")
parser.add_argument("--baseline-first", action="store_true",
                    help="Use first tuning run entry as baseline instead of highest accuracy")
parser.add_argument("--plot-history", action="store_true",
                    help="Plot history of maximum speedup over time above accuracy threshold")
parser.add_argument("--accuracy-threshold", type=float, default=None,
                    help="Minimum accuracy threshold for speedup history (default: baseline - 0.05)")
parser.add_argument("--history-by-configs", action="store_true",
                    help="Plot history by number of configs instead of elapsed time (default: by time)")
args = parser.parse_args()

# Connect to database
db_path = args.db
if not os.path.exists(db_path):
    print(f"Error: Database not found at {db_path}")
    exit(1)

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

conn = sqlite3.connect(db_path)

# Get all tuning runs
query_runs = "SELECT id, name FROM tuning_run"
runs_df = pd.read_sql_query(query_runs, conn)

print(f"Found {len(runs_df)} tuning runs:")
for _, row in runs_df.iterrows():
    print(f"  Run ID {row['id']}: {row['name']}")

# Process each tuning run
for _, run_row in runs_df.iterrows():
    tuning_run_id = run_row['id']
    run_name = run_row['name']
    
    # Extract device type from run name
    if '--device cpu' in run_name:
        device = 'CPU'
    elif '--device gpu' in run_name:
        device = 'GPU'
    else:
        device = 'Unknown'
    
    # Get results for this run
    query = f"""
    SELECT collection_date, accuracy, time, configuration_id
    FROM result
    WHERE tuning_run_id = {tuning_run_id}
    ORDER BY collection_date
    """
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print(f"\nNo results found for Run ID {tuning_run_id} ({device})")
        continue
    
    print(f"\n=== Run ID {tuning_run_id} ({device}) ===")
    print(f"Total results: {len(df)}")
    
    # Find baseline: either first entry or highest accuracy
    if args.baseline_first:
        baseline_idx = 0
        baseline_row = df.iloc[baseline_idx]
        baseline_label = 'first entry'
    else:
        baseline_idx = df['accuracy'].idxmax()
        baseline_row = df.loc[baseline_idx]
        baseline_label = 'max accuracy'
    
    baseline_time = baseline_row['time']
    baseline_accuracy = baseline_row['accuracy']
    
    # Set accuracy threshold
    if args.accuracy_threshold is not None:
        accuracy_threshold = args.accuracy_threshold
    else:
        accuracy_threshold = baseline_accuracy - 0.05
    
    # Calculate speedup relative to baseline
    df['speedup'] = baseline_time / df['time']
    
    print(f"Baseline ({baseline_label}):")
    print(f"  Configuration ID: {baseline_row['configuration_id']}")
    print(f"  Time: {baseline_time:.4f}s")
    print(f"  Accuracy: {baseline_accuracy:.4f}")
    print(f"Accuracy threshold: {accuracy_threshold:.4f}")
    print(f"Accuracy threshold: {accuracy_threshold:.4f}")
    print(f"Speedup range: {df['speedup'].min():.2f}x - {df['speedup'].max():.2f}x")
    print(f"Average speedup: {df['speedup'].mean():.2f}x")
    
    # Find configurations with significant speedup while maintaining reasonable accuracy
    # (e.g., accuracy >= 90% of baseline and speedup > 1.5x)
    threshold_90 = 0.9 * baseline_accuracy
    fast_configs = df[(df['accuracy'] >= threshold_90) & (df['speedup'] > 1.5)]
    if not fast_configs.empty:
        print(f"\nConfigurations with speedup > 1.5x and accuracy >= {threshold_90:.4f}:")
        for _, row in fast_configs.iterrows():
            print(f"  Config {row['configuration_id']}: {row['speedup']:.2f}x speedup, {row['accuracy']:.4f} accuracy")
    
    # Plot speedup history if requested
    if args.plot_history:
        # Track maximum speedup over time for configs meeting accuracy threshold
        max_speedup_history = []
        config_history = []
        x_values = []
        
        # Get start time for elapsed time calculation
        start_time = pd.to_datetime(df.iloc[0]['collection_date'])
        
        for i in range(len(df)):
            # Consider all configs up to this point
            df_so_far = df.iloc[:i+1]
            # Filter by accuracy threshold
            valid_configs = df_so_far[df_so_far['accuracy'] >= accuracy_threshold]
            
            if len(valid_configs) > 0:
                max_speedup = valid_configs['speedup'].max()
                max_config_id = valid_configs.loc[valid_configs['speedup'].idxmax(), 'configuration_id']
            else:
                max_speedup = 1.0  # No valid config yet, baseline has speedup of 1.0
                max_config_id = baseline_row['configuration_id']
            
            max_speedup_history.append(max_speedup)
            config_history.append(max_config_id)
            
            # Calculate x value based on mode
            if args.history_by_configs:
                x_values.append(i + 1)  # Number of configurations
            else:
                # Elapsed time in hours
                current_time = pd.to_datetime(df.iloc[i]['collection_date'])
                elapsed_seconds = (current_time - start_time).total_seconds()
                elapsed_hours = elapsed_seconds / 3600.0
                x_values.append(elapsed_hours)
        
        # Create history plot
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, max_speedup_history, linewidth=2, color='darkblue', marker='o', markersize=3)
        
        # Set labels based on mode
        if args.history_by_configs:
            plt.xlabel('Number of Configurations Evaluated', fontsize=12)
            x_label_str = 'configs'
        else:
            plt.xlabel('Elapsed Time (Hours)', fontsize=12)
            x_label_str = 'time'
        
        plt.ylabel('Maximum Speedup', fontsize=12)
        plt.title(f'HD-Clustering: Maximum Speedup History ({device})\nAccuracy ≥ {accuracy_threshold:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        
        # Add horizontal line at 1.0x
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
        
        # Add text box with final statistics
        final_speedup = max_speedup_history[-1]
        final_config = config_history[-1]
        final_x = x_values[-1]
        
        stats_text = f'Baseline: {baseline_time:.2f}s, {baseline_accuracy:.4f}\n'
        stats_text += f'Accuracy threshold: {accuracy_threshold:.4f}\n'
        if args.history_by_configs:
            stats_text += f'Final max speedup: {final_speedup:.2f}x (config {final_config})\n'
            stats_text += f'Total configs: {int(final_x)}'
        else:
            # Format time as H:MM:SS
            total_seconds = final_x * 3600
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            time_str = f'{hours}:{minutes:02d}:{seconds:02d}'
            
            stats_text += f'Final max speedup: {final_speedup:.2f}x (config {final_config})\n'
            stats_text += f'Total time: {time_str} ({final_x:.2f}h)'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.legend()
        plt.tight_layout()
        
        # Save history plot
        baseline_suffix = "_first" if args.baseline_first else "_maxacc"
        x_suffix = "_configs" if args.history_by_configs else "_time"
        history_file = os.path.join(args.output_dir, 
                                    f'hd_clustering_speedup_history_{device.lower()}{baseline_suffix}{x_suffix}.pdf')
        plt.savefig(history_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved speedup history plot to: {history_file}")
        plt.close()
        
        # Print some statistics about the speedup progression
        configs_above_threshold = len(df[df['accuracy'] >= accuracy_threshold])
        print(f"\nSpeedup History Statistics:")
        print(f"  Configurations meeting accuracy threshold: {configs_above_threshold}/{len(df)}")
        print(f"  Final maximum speedup: {final_speedup:.2f}x (config {final_config})")
        if args.history_by_configs:
            print(f"  Total configs evaluated: {int(final_x)}")
        else:
            # Format time as H:MM:SS
            total_seconds = final_x * 3600
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            time_str = f'{hours}:{minutes:02d}:{seconds:02d}'
            print(f"  Total elapsed time: {time_str} ({final_x:.2f} hours)")
        print(f"  Speedup improvement: {final_speedup:.2f}x / 1.0x = {final_speedup:.2f}x better than baseline")
    
    # Create scatter plot
    if not args.plot_history:
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['speedup'], df['accuracy'], c='blue', alpha=0.6, 
                    edgecolors='k', linewidths=0.5, s=50)
    # Create scatter plot
    if not args.plot_history:
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['speedup'], df['accuracy'], c='blue', alpha=0.6, 
                    edgecolors='k', linewidths=0.5, s=50)
        
        # Mark the baseline with a red X
        baseline_speedup = df.loc[df['configuration_id'] == baseline_row['configuration_id'], 'speedup'].values[0]
        plt.scatter([baseline_speedup], [baseline_accuracy], marker='x', c='red', 
                    s=200, linewidths=3, label=f'Baseline ({baseline_label})', zorder=5)
        
        # Annotate points if requested
        if args.annotate:
            for _, r in df.iterrows():
                plt.annotate(f"{r['configuration_id']}", 
                            (r['speedup'], r['accuracy']),
                            xytext=(4, 4), textcoords='offset points', 
                            fontsize=7, alpha=0.7)
        
        plt.xlabel('Speedup (baseline_time / time)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'HD-Clustering: Accuracy vs Speedup ({device})', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend()
        
        # Add text box with statistics
        stats_text = f'Baseline: {baseline_time:.2f}s, {baseline_accuracy:.4f}\n'
        stats_text += f'Speedup range: {df["speedup"].min():.2f}x - {df["speedup"].max():.2f}x\n'
        stats_text += f'Avg speedup: {df["speedup"].mean():.2f}x'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        baseline_suffix = "_first" if args.baseline_first else "_maxacc"
        output_file = os.path.join(args.output_dir, f'hd_clustering_speedup_vs_accuracy_{device.lower()}{baseline_suffix}.pdf')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved plot to: {output_file}")
        plt.close()

conn.close()
print("\n" + "="*80)
print("Done!")
print("="*80)
