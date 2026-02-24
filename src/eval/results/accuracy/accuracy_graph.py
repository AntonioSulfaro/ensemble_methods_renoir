import json
import sys

import matplotlib.pyplot as plt
import pandas as pd

csv_path = sys.argv[1]
n_threads = sys.argv[2]
t_time = sys.argv[3]
exec_config = json.loads(sys.argv[4])

# 1. Load CSV file
df = pd.read_csv(csv_path)

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

# 4. Formatting the chart
plt.title('Global Accuracy Trend', fontsize=14, fontweight='bold')
plt.xlabel('Instance Number', fontsize=12)
plt.ylabel('Global Accuracy', fontsize=12)

# Set limits for y-axis if needed (e.g., from 0 to 1 for percentage)
plt.ylim(0, 1.05)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# 5. Add text box with thread count and time
text_content = f'threads: {n_threads}\ntime: {t_time} s\nthroughput: {len(df) / float(t_time):.2f} ins/s\nfinal accuracy: {final_accuracy:.4f}\n\n'
text_content += "\n".join([f"{key}: {value}" for key, value in exec_config.items()])

# Place in bottom-right (x=0.95, y=0.05)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.95, 0.05, text_content,
         transform=plt.gca().transAxes,
         fontsize=12,
         family='monospace',
         verticalalignment='bottom',
         horizontalalignment='right',
         bbox=props)

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
plt.savefig(csv_path.split('.')[0] + '.png', dpi=300)
# plt.show()
