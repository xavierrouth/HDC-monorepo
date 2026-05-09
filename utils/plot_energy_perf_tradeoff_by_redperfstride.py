import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pickle
import zlib
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import argparse
import matplotlib.colors as mcolors

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot energy vs performance tradeoff colored by RedPerfStride.")
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

df['collection_date'] = pd.to_datetime(df['collection_date'])
df = df[df['accuracy'] >= accuracy_cutoff]

# Extract configurations and decode RedPerfStride
def extract_redperfstride(config_id):
    cursor.execute(f"SELECT data FROM Configuration WHERE id = {config_id}")
    row = cursor.fetchone()
    if row:
        data_json = row[0]
        decompressed_data = zlib.decompress(data_json)
        config = pickle.loads(decompressed_data)
        for key, value in config.items():
            if key.endswith('RedPerfStride'):
                return value
    return None

df['RedPerfStride'] = df['configuration_id'].apply(extract_redperfstride)

# Drop rows where RedPerfStride could not be extracted
df = df.dropna(subset=['RedPerfStride'])

## Compute speedup using the first data point as baseline
baseline_row = df.iloc[0]
baseline_time = baseline_row['time']
baseline_energy = baseline_row['energy']
df['speedup'] = baseline_time / df['time']
df['rel_energy'] = baseline_energy / df['energy']

# Create the scatter plot with RedPerfStride as color
plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    df['rel_energy'], df['speedup'],
    c=df['RedPerfStride'], cmap='viridis', alpha=0.8, s=50
)

plt.scatter([1.0], [baseline_time / baseline_time], c='red', s=120, marker='x', linewidth=2, label='Baseline')

# Add colorbar to show RedPerfStride scale
colorbar = plt.colorbar(scatter)
colorbar.set_label('RedPerfStride', rotation=270, labelpad=15)

# Customize the plot
plt.xlabel('Energy Decrease (relative to baseline)')
plt.ylabel('Speedup (relative to baseline)')
plt.title(f'Energy vs Speedup (colored by RedPerfStride), run {tuning_run_id}')
plt.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Save the plot to a file
plt.savefig(f'misc/energy_perf_rel_curve_by_redperfstride_{tuning_run_id}.png', dpi=300, bbox_inches='tight')

# Print some statistics
print(f"Baseline time: {baseline_time:.4f}")
print(f"Baseline energy: {baseline_energy:.4f}")
print(f"Max speedup: {df['speedup'].max():.2f}x")
print(f"Min speedup: {df['speedup'].min():.2f}x")
print(f"Average speedup: {df['speedup'].mean():.2f}x")
print(f"Min relative energy: {df['rel_energy'].min():.3f}")
print(f"Max relative energy: {df['rel_energy'].max():.3f}")
