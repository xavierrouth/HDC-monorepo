import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import zlib
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot RedPerfStride vs Relative Speedup controlled for EncodingDim.")
parser.add_argument("tuning_run_id", type=int, help="Tuning run ID to analyze")
parser.add_argument("accuracy_cutoff", type=float, help="accuracy cutoff")
args = parser.parse_args()
accuracy_cutoff = args.accuracy_cutoff
tuning_run_id = args.tuning_run_id

# Connect to your SQLite database
conn = sqlite3.connect('/home/xrouth2/hdc/HDC-monorepo/approx-tuner/opentuner.db/miranda.cs.illinois.edu.db')
cursor = conn.cursor()

# Query required records (ignore energy)
query = f"""
SELECT collection_date, configuration_id, accuracy, time
FROM result
WHERE tuning_run_id = {tuning_run_id}
ORDER BY collection_date
"""
df = pd.read_sql_query(query, conn)

# Extract configurations and decode EncodingDim and RedPerfStride
def extract_config_field(config_id, field_suffix):
    cursor.execute(f"SELECT data FROM Configuration WHERE id = {config_id}")
    row = cursor.fetchone()
    if not row:
        return None
    data_json = row[0]
    try:
        # breakpoint()
        # decompressed_data = zlib.decompress(data_json)
        # breakpoint()
        config = pickle.loads(data_json)
        for key, value in config.items():
            if key.endswith(field_suffix):
                return value
    except Exception:
        return None
    return None

df['EncodingDim'] = df['configuration_id'].apply(lambda cid: extract_config_field(cid, 'EncodingDim'))
df['RedPerfStride'] = df['configuration_id'].apply(lambda cid: extract_config_field(cid, 'STRIDE'))

# Drop rows where required fields could not be extracted
df = df.dropna(subset=['EncodingDim', 'RedPerfStride'])

df['collection_date'] = pd.to_datetime(df['collection_date'])

# Accuracy filter
df = df[df['accuracy'] >= accuracy_cutoff]

if df.empty:
    print("No data after filtering/extraction.")
    conn.close()
    raise SystemExit(1)

# Select the baseline configuration (first data point)
baseline_row = df.iloc[0]
baseline_time = baseline_row['time']

# Compute relative speedup
df['rel_speedup'] = baseline_time / df['time']

# Group by EncodingDim and create subplots
unique_dims = sorted(df['EncodingDim'].unique())
num_dims = len(unique_dims)

fig, axes = plt.subplots(num_dims, 1, figsize=(12, 4 * num_dims), squeeze=False)

for i, dim in enumerate(unique_dims):
    subset = df[df['EncodingDim'] == dim]
    sc = axes[i, 0].scatter(
        subset['RedPerfStride'],
        subset['rel_speedup'],
        c=subset['accuracy'],
        cmap='viridis',
        alpha=0.85,
        s=55
    )
    axes[i, 0].set_xlabel('RedPerfStride')
    axes[i, 0].set_ylabel('Relative Speedup (baseline normalized)')
    axes[i, 0].set_title(f'RedPerfStride vs Relative Speedup (EncodingDim={dim})')
    axes[i, 0].grid(True, alpha=0.3)

# Single colorbar for accuracy
cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('Accuracy', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(f'misc/redperfstride_vs_rel_speedup_all_dims_{tuning_run_id}.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Baseline time: {baseline_time:.4f}")
print(f"Max relative speedup: {df['rel_speedup'].max():.2f}x")
print(f"Min relative speedup: {df['rel_speedup'].min():.2f}x")
print(f"Average relative speedup: {df['rel_speedup'].mean():.2f}x")

conn.close()
