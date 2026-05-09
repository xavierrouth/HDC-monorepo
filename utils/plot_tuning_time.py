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
parser = argparse.ArgumentParser(description="Plot tuning time vs accuracy.")
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
SELECT collection_date, accuracy, time
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

# Create the scatter plot with speedup as color
plt.figure(figsize=(12, 6))
scatter = plt.scatter(df['hours_since_start'], df['accuracy'], 
                     c=df['speedup'], cmap='viridis', alpha=0.7, s=50)

# Add speedup text labels next to each point
for i, row in df.iterrows():
    plt.annotate(f'{row["speedup"]:.2f}x', 
                (row['hours_since_start'], row['accuracy']),
                xytext=(5, 5), textcoords='offset points', 
                fontsize=8, alpha=0.8)

# Highlight the baseline points (where speedup = 1.0)
baseline_points = df[df['speedup'] == 1.0]
plt.scatter(baseline_points['hours_since_start'], baseline_points['accuracy'], 
           c='red', s=100, marker='x', linewidth=3, label='Baseline (Max Time)')
           
# Add colorbar to show speedup scale
colorbar = plt.colorbar(scatter)
colorbar.set_label('Speedup', rotation=270, labelpad=15)

# Customize the plot
plt.xlabel('Time since start (hours)')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Collection Date (colored by Speedup)')
plt.grid(True, alpha=0.3)

# Format x-axis dates for better readability
def format_hours_minutes(x, pos):
    hours = int(x)
    minutes = int((x - hours) * 60)
    return f"{hours}h{minutes:02d}m"

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_hours_minutes))
# plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Save the plot to a file
plt.savefig(f'misc/accuracy_colored_by_speedup_{tuning_run_id}.png', dpi=300, bbox_inches='tight')

# Print some speedup statistics
print(f"Baseline time: {baseline_time:.4f}")
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Max speedup: {df['speedup'].max():.2f}x")
print(f"Min speedup: {df['speedup'].min():.2f}x")
print(f"Average speedup: {df['speedup'].mean():.2f}x")