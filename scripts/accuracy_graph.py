import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

n_threads = sys.argv[1]
t_time = sys.argv[2]
run_dir = sys.argv[3]

# 1. Load CSV file
df = pd.read_csv(os.path.join(run_dir, "accuracy.csv"))

with open(os.path.join(run_dir, "config.json"), 'r') as f:
    config = json.load(f)

# 2. Create an instance number column based on the row index (starting at 1)
# This preserves the order of the instances as they appear in the file
df['instance_number'] = range(1, len(df) + 1)

final_accuracy = df["global_accuracy"].iloc[-1]

# 3. Create the plot
plt.figure(figsize=(12, 6))

# Plotting global_accuracy against the instance number
plt.plot(df['instance_number'],
         df['global_accuracy'],
         marker='o',  # Add points
         linestyle='-',  # Connect with lines
         color='#2c7bb6',  # Nice blue color
         markersize=3,  # Size of the dots
         linewidth=1,  # Width of the line
         alpha=0.8)  # Transparency

drift_indices = df[df['drift_detected'] == True]['instance_number']

for i, x_pos in enumerate(drift_indices):
    # Add the vertical line
    plt.axvline(x=x_pos, color='red', linestyle='--', alpha=0.6, linewidth=1.5,
                label='Drift Detected' if i == 0 else "")

# If drifts exist, add a legend to explain the red dashed lines
if not drift_indices.empty:
    plt.legend(loc='upper left')

# 4. Formatting the chart
plt.title('Global Accuracy Trend', fontsize=14, fontweight='bold')
plt.xlabel('Instance Number', fontsize=12)
plt.ylabel('Global Accuracy', fontsize=12)
plt.ylim(bottom=0)

# Set limits for y-axis if needed (e.g., from 0 to 1 for percentage)
plt.ylim(0, 1.05)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# 5. Add text box with thread count and time
text_content = f'threads: {n_threads}\ntime: {t_time} s\nthroughput: {len(df) / float(t_time):.2f} ins/s\nfinal accuracy: {final_accuracy:.4f}\n\n'
text_content += "\n".join([f"{key}: {value}" for key, value in config.items()])

# Place in bottom-right (x=0.95, y=0.05)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.95, 0.05, text_content,
         transform=plt.gca().transAxes,
         fontsize=12,
         family='monospace',
         verticalalignment='bottom',
         horizontalalignment='right',
         bbox=props)

# annotate the final accuracy value at the end of the line
plt.annotate(f'{final_accuracy:.4f}',
             xy=(df['instance_number'].iloc[-1], final_accuracy),
             xytext=(10, 0),
             textcoords='offset points',
             va='center',
             color='red',
             fontweight='bold')

# Tight layout to prevent labels from being cut off
plt.tight_layout()

# 6. Show or Save the plot
plt.savefig(os.path.join(run_dir, "accuracy.png"), dpi=300)
# plt.show()
