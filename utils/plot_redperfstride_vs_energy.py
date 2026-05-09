import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pickle
import zlib
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot RedPerfStride vs Energy controlled for EncodingDim.")
parser.add_argument("tuning_run_id", type=int, help="Tuning run ID to analyze")
parser.add_argument("accuracy_cutoff", type=float, help="accuracy cutoff")

args = parser.parse_args()
accuracy_cutoff = args.accuracy_cutoff
tuning_run_id = args.tuning_run_id

# Connect to your SQLite database
db_name = "sim"
conn = sqlite3.connect(f'/home/xrouth2/hdc/HDC-monorepo/approx-tuner/opentuner.db/miranda.cs.illinois.edu.db')  # change to your database path
cursor = conn.cursor()

# Query all records
query = f"""
SELECT collection_date, configuration_id, accuracy, time, energy
FROM result
WHERE tuning_run_id = {tuning_run_id}
ORDER BY collection_date
"""
df = pd.read_sql_query(query, conn)

# Extract configurations and decode EncodingDim and RedPerfStride
def extract_config_field(config_id, field_suffix):
    cursor.execute(f"SELECT data FROM Configuration WHERE id = {config_id}")
    row = cursor.fetchone()
    if row:
        data_json = row[0]
        decompressed_data = zlib.decompress(data_json)
        config = pickle.loads(decompressed_data)
        for key, value in config.items():
            if key.endswith(field_suffix):
                return value
    return None

df['EncodingDim'] = df['configuration_id'].apply(lambda id: extract_config_field(id, 'EncodingDim'))
df['RedPerfStride'] = df['configuration_id'].apply(lambda id: extract_config_field(id, 'RedPerfStride'))

# Drop rows where required fields could not be extracted
df = df.dropna(subset=['EncodingDim', 'RedPerfStride'])

df['collection_date'] = pd.to_datetime(df['collection_date'])

# Add an accuracy filter
df = df[df['accuracy'] >= accuracy_cutoff]

# Select the baseline configuration (first data point)
baseline_row = df.iloc[0]
baseline_energy = baseline_row['energy']

# Compute relative energy
df['rel_energy'] = baseline_energy / df['energy']

# Compute relative speedup
baseline_time = baseline_row['time']
df['rel_speedup'] = baseline_time / df['time']

# Group by EncodingDim and create a grid of subplots for all groups
unique_dims = sorted(df['EncodingDim'].unique())
num_dims = len(unique_dims)

# Create a figure with 2 columns and rows equal to the number of unique EncodingDim values
fig, axes = plt.subplots(num_dims, 2, figsize=(16, 6 * num_dims), squeeze=False)

for i, dim in enumerate(unique_dims):
    subset = df[df['EncodingDim'] == dim]
    
    # Plot RedPerfStride vs relative energy, colored by accuracy
    scatter_energy = axes[i, 0].scatter(
        subset['RedPerfStride'], subset['rel_energy'], c=subset['accuracy'], cmap='viridis', alpha=0.8
    )
    axes[i, 0].set_xlabel('RedPerfStride')
    axes[i, 0].set_ylabel('Relative Energy (baseline normalized)')
    axes[i, 0].set_title(f'RedPerfStride vs Relative Energy (EncodingDim={dim})')
    axes[i, 0].grid(True, alpha=0.3)
    
    # Plot RedPerfStride vs relative speedup, colored by accuracy
    scatter_speedup = axes[i, 1].scatter(
        subset['RedPerfStride'], subset['rel_speedup'], c=subset['accuracy'], cmap='viridis', alpha=0.8
    )
    axes[i, 1].set_xlabel('RedPerfStride')
    axes[i, 1].set_ylabel('Relative Speedup (baseline normalized)')
    axes[i, 1].set_title(f'RedPerfStride vs Relative Speedup (EncodingDim={dim})')
    axes[i, 1].grid(True, alpha=0.3)

# Add a single colorbar for accuracy
cbar = fig.colorbar(scatter_energy, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('Accuracy', rotation=270, labelpad=15)

# Adjust layout to prevent overlap and save the plot
plt.tight_layout()
plt.savefig(f'misc/redperfstride_vs_rel_energy_speedup_all_dims_{tuning_run_id}.png', dpi=300, bbox_inches='tight')
plt.show()
