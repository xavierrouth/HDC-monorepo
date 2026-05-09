import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pickle
import zlib
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot cumulative distribution function of speedup.")
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
SELECT collection_date, accuracy, time, configuration_id
FROM result
WHERE tuning_run_id = {tuning_run_id}
ORDER BY collection_date
"""
df = pd.read_sql_query(query, conn)
conn.close()

df['collection_date'] = pd.to_datetime(df['collection_date'])
df = df[df['accuracy'] >= accuracy_cutoff]

## Compute speedup using the first data point as baseline
# Speedup = baseline_time / current_time
if False:
    baseline_accuracy = df['accuracy'].max()  # First time value as baseline
    baseline_row = df[df['accuracy'] == baseline_accuracy].iloc[0]  # Row with max accuracy
    baseline_time = baseline_row['time']  # Accuracy corresponding to max time
else:
    baseline_row = df.iloc[0]  # First data point as baseline
    baseline_time = baseline_row['time']
    baseline_accuracy = baseline_row['accuracy']

df['speedup'] = baseline_time / df['time']

first_time = df['collection_date'].min()
df['hours_since_start'] = (df['collection_date'] - first_time).dt.total_seconds() / 3600

# Calculate cumulative max speedup at each time point
df = df.sort_values('hours_since_start')
df['max_speedup'] = df['speedup'].cummax()

# Filter to only keep rows where a new max speedup is achieved
df['is_new_max'] = df['max_speedup'] > df['max_speedup'].shift(1).fillna(0)
new_max_df = df[df['is_new_max']]

# Create the scatter plot
plt.figure(figsize=(12, 6))
plt.plot(new_max_df['hours_since_start'], new_max_df['max_speedup'], 
           color='blue', marker='o', alpha=0.7, label='Max Speedup Achieved')

# Add speedup and configuration_id text labels with alternating vertical positions
y_offsets = [10, 25, 40, 10, 25, 40]  # Alternate between different vertical offsets
for idx, (i, row) in enumerate(new_max_df.iterrows()):
    y_offset = y_offsets[idx % len(y_offsets)]
    plt.annotate(f'{row["accuracy"]:.2f}x\n(cfg {row["configuration_id"]})', 
                (row['hours_since_start'], row['max_speedup']),
                xytext=(5, y_offset), textcoords='offset points', 
                fontsize=8, alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='gray'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', lw=0.5))


# Customize the plot
plt.xlabel('Time since start (hours)')
plt.ylabel('Max Speedup')
plt.title(f'Maximum Speedup Achieved Over Time (Accuracy ≥ {accuracy_cutoff})')
plt.grid(True, alpha=0.3)
plt.legend()

# Format x-axis dates for better readability
def format_hours_minutes(x, pos):
    hours = int(x)
    minutes = int((x - hours) * 60)
    return f"{hours}h{minutes:02d}m"

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_hours_minutes))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Save the plot to a file
plt.savefig(f'misc/max_speedup_over_time_{tuning_run_id}.png', dpi=300, bbox_inches='tight')

# Print some speedup statistics
print(f"Baseline time: {baseline_time:.4f}")
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Max speedup achieved: {df['max_speedup'].iloc[-1]:.2f}x")
print(f"Min speedup: {df['speedup'].min():.2f}x")
print(f"Average speedup: {df['speedup'].mean():.2f}x")
print(f"Number of new max speedups achieved: {len(new_max_df)}")