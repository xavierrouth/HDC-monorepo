import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import zlib
import argparse
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser(description="Plot accuracy vs speedup colored by GLOBAL_EncodingDim (baseline = highest accuracy).")
parser.add_argument("tuning_run_id", type=int, help="Tuning run ID to analyze")
parser.add_argument("accuracy_cutoff", type=float, help="Minimum accuracy to include")
args = parser.parse_args()
tuning_run_id = args.tuning_run_id
accuracy_cutoff = args.accuracy_cutoff

conn = sqlite3.connect('/home/xrouth2/hdc/HDC-monorepo/approx-tuner/opentuner.db/miranda.cs.illinois.edu.db')
cursor = conn.cursor()

query = f"""
SELECT collection_date, configuration_id, accuracy, time
FROM result
WHERE tuning_run_id = {tuning_run_id}
ORDER BY collection_date
"""
df = pd.read_sql_query(query, conn)

if df.empty:
    print("No data for tuning_run_id:", tuning_run_id)
    conn.close()
    raise SystemExit(1)

df['collection_date'] = pd.to_datetime(df['collection_date'])
df = df[df['accuracy'] >= accuracy_cutoff]

def extract_encoding_dim(config_id):
    cursor.execute(f"SELECT data FROM Configuration WHERE id = {config_id}")
    row = cursor.fetchone()
    # breakpoint()
    if not row:
        return None
    data_blob = row[0]
    try:
        # Sometimes data is not compressed? If the config is small enough?
        # data_blob = zlib.decompress(data_blob)
        config = pickle.loads(data_blob)
        return config.get('GLOBAL_EncodingDim', None)
    except Exception:
        return None

df['GLOBAL_EncodingDim'] = df['configuration_id'].apply(extract_encoding_dim)
df = df.dropna(subset=['GLOBAL_EncodingDim'])

if df.empty:
    print("No rows after extracting GLOBAL_EncodingDim.")
    conn.close()
    raise SystemExit(1)

# breakpoint()
# Baseline = highest accuracy row (first occurrence)
# baseline_idx = df['accuracy'].idxmax()
baseline_idx = 0
baseline_row = df.loc[baseline_idx]
baseline_time = baseline_row['time']
baseline_accuracy = baseline_row['accuracy']

df['speedup'] = baseline_time / df['time']  # speedup relative to best accuracy cfg

# Colormap for powers of two
discrete_cmap = mcolors.ListedColormap(plt.cm.viridis(np.linspace(0, 1, 11)))
bounds = [2**i for i in range(1, 12)]
norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)

plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    df['speedup'], df['accuracy'],
    c=df['GLOBAL_EncodingDim'], cmap=discrete_cmap, norm=norm,
    alpha=0.85, s=55, edgecolors='k', linewidths=0.4
)

# Baseline marker at speedup=1, accuracy=baseline_accuracy
plt.scatter([1.0], [baseline_accuracy], c='red', marker='x', s=120, linewidths=2, label='Baseline (max accuracy)')

colorbar = plt.colorbar(scatter, ticks=bounds)
colorbar.set_label('GLOBAL_EncodingDim', rotation=270, labelpad=15)
colorbar.ax.set_yticklabels([str(b) for b in bounds])

plt.xlabel('Speedup (baseline_time / time)')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs Speedup (tuning_run_id={tuning_run_id}, cutoff ≥ {accuracy_cutoff})')
plt.grid(alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()

plt.show()
plt.savefig(f'misc/accuracy_speedup_by_config_{tuning_run_id}.png', dpi=300, bbox_inches='tight')

print(f"Baseline (max accuracy) time: {baseline_time:.4f}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(f"Max speedup: {df['speedup'].max():.2f}x")
print(f"Min speedup: {df['speedup'].min():.2f}x")
print(f"Average speedup: {df['speedup'].mean():.2f}x")
print(f"Accuracy range: {df['accuracy'].min():.4f} - {df['accuracy'].max():.4f}")
conn.close()
