import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Plot accuracy vs speedup for a tuning run.")
parser.add_argument("tuning_run_id", type=int, help="Tuning run ID to analyze")
parser.add_argument("--annotate", action="store_true", help="Annotate points with configuration ids")
args = parser.parse_args()

tuning_run_id = args.tuning_run_id

conn = sqlite3.connect('/home/xrouth2/hdc/HDC-monorepo/approx-tuner/opentuner.db/miranda.cs.illinois.edu.db')
query = f"""
SELECT collection_date, accuracy, time, configuration_id
FROM result
WHERE tuning_run_id = {tuning_run_id}
ORDER BY collection_date
"""
df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("No data found for tuning_run_id:", tuning_run_id)
    raise SystemExit(1)

# Use the highest-accuracy row (first occurrence) as the baseline
baseline_idx = df['accuracy'].idxmax()
baseline_idx = 0
baseline_row = df.loc[baseline_idx]
baseline_time = baseline_row['time']
baseline_accuracy = baseline_row['accuracy']

df['speedup'] = baseline_time / df['time']

plt.figure(figsize=(10, 6))
plt.scatter(df['speedup'], df['accuracy'], c='blue', alpha=0.6, edgecolors='k', linewidths=0.3)

# Mark the baseline (max accuracy) with a red X at speedup=1.0
plt.scatter([1.0], [baseline_accuracy], marker='x', c='red', s=100, linewidths=2)

if args.annotate:
    for _, r in df.iterrows():
        plt.annotate(f"{r['configuration_id']}", (r['speedup'], r['accuracy']),
                     xytext=(4, 4), textcoords='offset points', fontsize=7)

plt.xlabel('Speedup (baseline_time / time)')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs Speedup (tuning_run_id={tuning_run_id})')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig(f'misc/accuracy_vs_speedup_{tuning_run_id}.png', dpi=300, bbox_inches='tight')

print(f"Baseline time: {baseline_time:.4f}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(f"Max speedup: {df['speedup'].max():.2f}x")
print(f"Min speedup: {df['speedup'].min():.2f}x")
print(f"Average speedup: {df['speedup'].mean():.2f}x")
