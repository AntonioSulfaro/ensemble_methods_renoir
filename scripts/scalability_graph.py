import matplotlib.pyplot as plt
import pandas as pd

# 1. Load your CSV file
file_path = 'src/eval/results/scalability/master_log'
df = pd.read_csv(file_path + '.csv')

# filter for the latest configuration used
config_columns = [
    'ensemble_type', 'drift_detection', 'n_trees', 'max_bins',
    'n_min', 'delta', 'tau', 'range_r', 'features_patch', 'lambda'
]

# 3. Get the configuration used in the VERY LAST run
latest_config = df.iloc[-1][config_columns]
filtered_df = df.copy()
for col in config_columns:
    filtered_df = filtered_df[filtered_df[col] == latest_config[col]]
filtered_df = filtered_df.sort_values('n_threads')

# 2. Create the plot with dual axes
fig, ax1 = plt.subplots(figsize=(12, 7))

# --- Primary Axis: Time ---
color_time = '#2c7bb6'  # Blue
ax1.set_xlabel('Number of Threads', fontsize=12)
ax1.set_ylabel('Execution Time [s]', fontsize=12, color=color_time)
line1 = ax1.plot(filtered_df['n_threads'], filtered_df['time'],
                 marker='o', linestyle='-', color=color_time,
                 linewidth=2, label='Time (s)')
ax1.tick_params(axis='y', labelcolor=color_time)

# --- Secondary Axis: Throughput ---
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color_thru = '#d7191c'  # Red
ax2.set_ylabel('Throughput [instances/s]', fontsize=12, color=color_thru)
# Calculate throughput: Total Instances (1e6) / Time
filtered_df['throughput'] = 1e6 / filtered_df['time']
line2 = ax2.plot(filtered_df['n_threads'], filtered_df['throughput'],
                 marker='s', linestyle='--', color=color_thru,
                 linewidth=2, label='Throughput')
ax2.tick_params(axis='y', labelcolor=color_thru)

# 3. Formatting
plt.title(f"Scalability Analysis: {filtered_df.iloc[-1]['ensemble_type']} (Trees: {filtered_df.iloc[-1]['n_trees']})",
          fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)

# Combined Legend for both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# Add configuration details in a text box
text_content = "\n".join([f"{key}: {value}" for key, value in latest_config.items()])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.95, 0.20, text_content,
         transform=plt.gca().transAxes,
         fontsize=12,
         family='monospace',
         verticalalignment='bottom',
         horizontalalignment='right',
         bbox=props)

# 4. Save
plt.tight_layout()
plt.savefig(file_path + '_' + df.iloc[-1]['id'] + '.png', dpi=300)
