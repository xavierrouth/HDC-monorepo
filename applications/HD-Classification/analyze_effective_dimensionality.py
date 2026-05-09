#!/usr/bin/env python3
"""
Analyze effective dimensionality vs accuracy for HD-Classification sweep results.
Effective dimensionality = encoding_dim * redperf_extent
"""

import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

DB_PATH = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Classification/results/sweep_results.db"
OUTPUT_DIR = "/home/xrouth2/hdc/HDC-monorepo/applications/HD-Classification/results"


def load_data():
    """Load successful results from database and calculate effective dimensionalities."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT encoding_dim, training_redperf_base, training_redperf_extent, 
               inference_redperf_base, inference_redperf_extent,
               time_seconds, accuracy, energy_joules,
               training_time_seconds, inference_time_seconds
        FROM sweep_results
        WHERE success = 1
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Calculate effective dimensionalities mirroring C++ calculation:
    # end = min(base + extent, 1.0)
    # effective_dim = encoding_dim * (end - base)
    df['training_end'] = df[['training_redperf_base', 'training_redperf_extent']].apply(
        lambda row: min(row['training_redperf_base'] + row['training_redperf_extent'], 1.0), axis=1)
    df['inference_end'] = df[['inference_redperf_base', 'inference_redperf_extent']].apply(
        lambda row: min(row['inference_redperf_base'] + row['inference_redperf_extent'], 1.0), axis=1)
    
    df['effective_training_dim'] = df['encoding_dim'] * (df['training_end'] - df['training_redperf_base'])
    df['effective_inference_dim'] = df['encoding_dim'] * (df['inference_end'] - df['inference_redperf_base'])
    
    # Filter: Remove results with effective dim > 2000 or accuracy < 0.5
    initial_count = len(df)
    df = df[(df['effective_training_dim'] <= 2000) & 
            (df['effective_inference_dim'] <= 2000) & 
            (df['accuracy'] >= 0.5)]
    filtered_count = initial_count - len(df)
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} results (effective_dim > 2000 or accuracy < 0.5)")
    
    return df


def plot_effective_dim_vs_accuracy():
    """Create plots showing relationship between effective dimensionality and accuracy."""
    df = load_data()
    
    if df.empty:
        print("No data found in database.")
        return
    
    print(f"Loaded {len(df)} successful experiments")
    print(f"\nEffective Training Dimensionality range: {df['effective_training_dim'].min():.0f} - {df['effective_training_dim'].max():.0f}")
    print(f"Effective Inference Dimensionality range: {df['effective_inference_dim'].min():.0f} - {df['effective_inference_dim'].max():.0f}")
    print(f"Accuracy range: {df['accuracy'].min():.4f} - {df['accuracy'].max():.4f}")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Training Effective Dimensionality vs Accuracy
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(df['effective_training_dim'], df['accuracy'], 
                          c=df['encoding_dim'], cmap='viridis', 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    ax1.set_xlabel('Effective Training Dimensionality (Dim × Extent)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training Effective Dimensionality vs Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Encoding Dim', fontsize=10)
    
    # Plot 2: Inference Effective Dimensionality vs Accuracy
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['effective_inference_dim'], df['accuracy'], 
                          c=df['encoding_dim'], cmap='viridis', 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    ax2.set_xlabel('Effective Inference Dimensionality (Dim × Extent)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Inference Effective Dimensionality vs Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Encoding Dim', fontsize=10)
    
    # Plot 3: Training vs Inference Effective Dimensionality (colored by accuracy)
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(df['effective_training_dim'], df['effective_inference_dim'], 
                          c=df['accuracy'], cmap='RdYlGn', 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    ax3.set_xlabel('Effective Training Dimensionality', fontsize=12)
    ax3.set_ylabel('Effective Inference Dimensionality', fontsize=12)
    ax3.set_title('Training vs Inference Effective Dimensionality (colored by accuracy)', 
                 fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3, linestyle='--')
    # Add diagonal line
    max_val = max(df['effective_training_dim'].max(), df['effective_inference_dim'].max())
    ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Equal dims')
    ax3.legend()
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Accuracy', fontsize=10)
    
    # Plot 4: Combined effective dimensionality vs accuracy
    ax4 = axes[1, 1]
    df['combined_effective_dim'] = df['effective_training_dim'] + df['effective_inference_dim']
    scatter4 = ax4.scatter(df['combined_effective_dim'], df['accuracy'], 
                          c=df['encoding_dim'], cmap='plasma', 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    ax4.set_xlabel('Combined Effective Dimensionality (Training + Inference)', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Combined Effective Dimensionality vs Accuracy', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Encoding Dim', fontsize=10)
    
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, 'effective_dimensionality_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    plt.close()


def plot_heatmaps():
    """Create heatmaps showing accuracy for different effective dimensionality combinations."""
    df = load_data()
    
    if df.empty:
        print("No data found in database.")
        return
    
    # Create pivot tables for heatmaps
    # We'll bin the effective dimensionalities for better visualization
    df['train_dim_bin'] = pd.cut(df['effective_training_dim'], bins=10)
    df['inf_dim_bin'] = pd.cut(df['effective_inference_dim'], bins=10)
    
    # Group by bins and calculate mean accuracy
    pivot_data = df.groupby(['train_dim_bin', 'inf_dim_bin'])['accuracy'].mean().reset_index()
    
    # For each encoding dimension, create a heatmap
    encoding_dims = sorted(df['encoding_dim'].unique())
    
    fig, axes = plt.subplots(1, len(encoding_dims), figsize=(6*len(encoding_dims), 5))
    if len(encoding_dims) == 1:
        axes = [axes]
    
    for idx, enc_dim in enumerate(encoding_dims):
        df_dim = df[df['encoding_dim'] == enc_dim].copy()
        
        # Create pivot table for this encoding dimension
        pivot = df_dim.pivot_table(
            values='accuracy', 
            index='training_redperf_extent', 
            columns='inference_redperf_extent',
            aggfunc='mean'
        )
        
        ax = axes[idx]
        im = ax.imshow(pivot, cmap='RdYlGn', aspect='auto', vmin=df['accuracy'].min(), vmax=df['accuracy'].max())
        ax.set_title(f'Encoding Dim = {enc_dim}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Inference RedPerf Extent', fontsize=10)
        ax.set_ylabel('Training RedPerf Extent', fontsize=10)
        
        # Set tick labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{y:.2f}' for y in pivot.index])
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                if not pd.isna(pivot.iloc[i, j]):
                    text = ax.text(j, i, f'{pivot.iloc[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Accuracy')
    
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, 'accuracy_heatmaps_by_encoding_dim.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {output_file}")
    plt.close()


def plot_approximation_vs_true_dimensionality():
    """Compare approximated dimensions vs true dimensions at the same effective value.
    
    For example, compare:
    - encoding_dim=1024 with extent=0.5 (effective=512) 
    - vs encoding_dim=512 with extent=1.0 (effective=512, true dimension)
    """
    df = load_data()
    
    if df.empty:
        print("No data found in database.")
        return
    
    # Only consider configurations where inference extent = 1.0
    df_inf1 = df # df[df['inference_redperf_extent'] == 1.0].copy()
    
    # Separate into two groups: training extent=1.0 (true dimension) vs extent<1.0 (approximated)
    df_true = df_inf1[df_inf1['training_redperf_extent'] == 1.0].copy()
    df_approx = df_inf1[df_inf1['training_redperf_extent'] < 1.0].copy()
    
    print(f"\nFiltered to inference extent=1.0 only")
    print(f"Found {len(df_true)} results with training extent=1.0 (true dimension)")
    print(f"Found {len(df_approx)} results with training extent<1.0 (approximated)")
    
    # For plotting, we want to match effective training dimensions (inference is always 1.0)
    # Match key is just the effective training dimension
    
    # Create unique keys for matching: just effective_training_dim
    df_true['match_key'] = df_true['effective_training_dim']
    df_approx['match_key'] = df_approx['effective_training_dim']
    
    # Find matching configurations (same effective training dim, both with inference extent=1.0)
    true_keys = set(df_true['match_key'])
    approx_keys = set(df_approx['match_key'])
    common_keys = sorted(true_keys.intersection(approx_keys))
    
    if not common_keys:
        print("No overlapping configurations between true and approximated results.")
        return
    
    print(f"Found {len(common_keys)} matching effective training dimensions (all with inference extent=1.0)")
    
    # Create comparison plot with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Plot 1: Side-by-side comparison for each effective dimension
    ax1 = axes[0]
    
    true_accuracies = []
    approx_accuracies = []
    eff_dim_labels = []
    
    for eff_train_dim in common_keys:
        true_acc = df_true[df_true['match_key'] == eff_train_dim]['accuracy'].values
        approx_acc = df_approx[df_approx['match_key'] == eff_train_dim]['accuracy'].values
        
        if len(true_acc) > 0 and len(approx_acc) > 0:
            true_accuracies.append(true_acc.mean())
            approx_accuracies.append(approx_acc.mean())
            eff_dim_labels.append(int(eff_train_dim))
    
    x = np.arange(len(eff_dim_labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, true_accuracies, width, label='True Dim (extent=1.0)', 
                    color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, approx_accuracies, width, label='Approximated (extent<1.0)', 
                    color='coral', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Effective Training Dimensionality', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('True vs Approximated Training Dimensionality\n(Inference Extent = 1.0 for All)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(eff_dim_labels, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Scatter plot showing accuracy difference
    ax2 = axes[1]
    
    # For each matching configuration, plot the difference (all have inference extent=1.0)
    diff_data = []
    for eff_train_dim in common_keys:
        true_rows = df_true[df_true['match_key'] == eff_train_dim]
        approx_rows = df_approx[df_approx['match_key'] == eff_train_dim]
        
        for _, true_row in true_rows.iterrows():
            for _, approx_row in approx_rows.iterrows():
                diff_data.append({
                    'effective_train_dim': eff_train_dim,
                    'true_accuracy': true_row['accuracy'],
                    'approx_accuracy': approx_row['accuracy'],
                    'accuracy_diff': true_row['accuracy'] - approx_row['accuracy'],
                    'true_encoding_dim': true_row['encoding_dim'],
                    'approx_encoding_dim': approx_row['encoding_dim'],
                    'approx_extent': approx_row['training_redperf_extent']
                })
    
    diff_df = pd.DataFrame(diff_data)
    
    # Scatter plot colored by training approximation extent
    scatter = ax2.scatter(diff_df['effective_train_dim'], diff_df['accuracy_diff'], 
                         c=diff_df['approx_extent'], cmap='viridis',
                         s=100, alpha=0.6, edgecolors='k', linewidths=0.5)
    
    # Add zero line
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No difference')
    
    ax2.set_xlabel('Effective Training Dimensionality', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Difference (True - Approximated)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Impact: True vs Approximated Training Dim\n(Inference Extent = 1.0, Positive = True Better)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.legend()
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Training Approximation Extent', fontsize=10)
    
    # Plot 3: Absolute accuracy scatter plot colored by training extent (ALL results)
    ax3 = axes[2]
    
    # Collect ALL data points for plotting (not just matching dimensions)
    all_approx_dims = []
    all_approx_acc = []
    all_approx_extent = []
    all_true_dims = []
    all_true_acc = []
    
    # Get ALL approximated results (regardless of matching)
    for _, row in df_approx.iterrows():
        all_approx_dims.append(row['effective_training_dim'])
        all_approx_acc.append(row['accuracy'])
        all_approx_extent.append(row['training_redperf_extent'])
    
    # Get ALL true results
    for _, row in df_true.iterrows():
        all_true_dims.append(row['effective_training_dim'])
        all_true_acc.append(row['accuracy'])
    
    # Plot approximated results (colored by training extent)
    scatter_approx = ax3.scatter(all_approx_dims, all_approx_acc, 
                                c=all_approx_extent, cmap='viridis',
                                s=100, alpha=0.6, edgecolors='k', linewidths=0.5,
                                marker='o')
    
    # Plot true dimension results (all have extent=1.0)
    scatter_true = ax3.scatter(all_true_dims, all_true_acc, 
                              color='red', s=120, alpha=0.8, edgecolors='k', linewidths=1.5,
                              marker='s')
    
    ax3.set_xlabel('Effective Training Dimensionality', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Absolute Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Accuracy: True vs Approximated Training Dim\n(Inference Extent = 1.0)', 
                  fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3, linestyle='--')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='k', label='Approximated (colored by extent)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=10, markeredgecolor='k', label='True (extent=1.0)')
    ]
    ax3.legend(handles=legend_elements)
    
    cbar3 = plt.colorbar(scatter_approx, ax=ax3)
    cbar3.set_label('Training Approximation Extent', fontsize=10)
    
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, 'true_vs_approximated_dimensionality.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_file}")
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"TRUE vs APPROXIMATED DIMENSIONALITY COMPARISON")
    print(f"{'='*80}\n")
    
    if len(diff_df) > 0:
        mean_diff = diff_df['accuracy_diff'].mean()
        std_diff = diff_df['accuracy_diff'].std()
        max_diff = diff_df['accuracy_diff'].max()
        min_diff = diff_df['accuracy_diff'].min()
        
        print(f"Accuracy Difference Statistics (True - Approximated):")
        print(f"  Mean:     {mean_diff:+.4f}")
        print(f"  Std Dev:  {std_diff:.4f}")
        print(f"  Max:      {max_diff:+.4f}")
        print(f"  Min:      {min_diff:+.4f}")
        print(f"\nInterpretation:")
        if abs(mean_diff) < 0.01:
            print("  → Approximation gives nearly identical accuracy to true dimension!")
        elif mean_diff > 0:
            print("  → True dimension (extent=1.0) gives better accuracy on average")
        else:
            print("  → Approximation gives better accuracy on average (surprising!)")


def print_statistics():
    """Print statistical analysis of effective dimensionality."""
    df = load_data()
    
    if df.empty:
        print("No data found in database.")
        return
    
    print(f"\n{'='*80}")
    print(f"EFFECTIVE DIMENSIONALITY ANALYSIS")
    print(f"{'='*80}\n")
    
    # Correlation analysis
    print("Correlation with Accuracy:")
    print(f"  Training Effective Dim:   {df['effective_training_dim'].corr(df['accuracy']):.4f}")
    print(f"  Inference Effective Dim:  {df['effective_inference_dim'].corr(df['accuracy']):.4f}")
    print(f"  Encoding Dim:             {df['encoding_dim'].corr(df['accuracy']):.4f}")
    print(f"  Training RedPerf Extent:  {df['training_redperf_extent'].corr(df['accuracy']):.4f}")
    print(f"  Inference RedPerf Extent: {df['inference_redperf_extent'].corr(df['accuracy']):.4f}")
    
    # Find optimal configurations
    print(f"\n{'='*80}")
    print(f"TOP 10 CONFIGURATIONS BY ACCURACY")
    print(f"{'='*80}\n")
    
    top_configs = df.nlargest(10, 'accuracy')
    for idx, row in top_configs.iterrows():
        print(f"Rank {top_configs.index.get_loc(idx) + 1}:")
        print(f"  Encoding Dim: {row['encoding_dim']}")
        print(f"  Training Extent: {row['training_redperf_extent']:.2f} (Effective: {row['effective_training_dim']:.0f})")
        print(f"  Inference Extent: {row['inference_redperf_extent']:.2f} (Effective: {row['effective_inference_dim']:.0f})")
        print(f"  Accuracy: {row['accuracy']:.4f}")
        print(f"  Time: {row['time_seconds']:.2f}s")
        print()
    
    # Analysis by encoding dimension
    print(f"{'='*80}")
    print(f"ACCURACY BY ENCODING DIMENSION")
    print(f"{'='*80}\n")
    
    for enc_dim in sorted(df['encoding_dim'].unique()):
        df_dim = df[df['encoding_dim'] == enc_dim]
        print(f"Encoding Dim = {enc_dim}:")
        print(f"  Mean Accuracy: {df_dim['accuracy'].mean():.4f}")
        print(f"  Max Accuracy:  {df_dim['accuracy'].max():.4f}")
        print(f"  Best Training Extent:  {df_dim.loc[df_dim['accuracy'].idxmax(), 'training_redperf_extent']:.2f}")
        print(f"  Best Inference Extent: {df_dim.loc[df_dim['accuracy'].idxmax(), 'inference_redperf_extent']:.2f}")
        print()


def main():
    """Main analysis function."""
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        print("Run sweep_experiment.py first to generate results.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Analyzing effective dimensionality vs accuracy...")
    
    # Generate plots
    plot_effective_dim_vs_accuracy()
    plot_heatmaps()
    plot_approximation_vs_true_dimensionality()
    
    # Print statistics
    print_statistics()
    
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
