import matplotlib.pyplot as plt
import pandas as pd

# 1. Load and Filter
file_path = 'results/scalability/master_log'
df = pd.read_csv(file_path + '.csv')

config_columns = [
    'dataset', 'ensemble_type', 'drift_detection', 'n_trees',
    'max_bins', 'n_min', 'delta', 'tau', 'range_r', 'features_patch', 'lambda'
]

latest_config = df.iloc[-1][config_columns]
filtered_df = df.copy()
for col in config_columns:
    filtered_df = filtered_df[filtered_df[col] == latest_config[col]]

# Ensure we are sorted by threads for line plotting
filtered_df = filtered_df.sort_values('n_threads').reset_index()

# 2. Calculate Scalability Metrics
n_instances = filtered_df.iloc[-1]['n_instances']
filtered_df['throughput'] = n_instances / filtered_df['time']

# Speedup = Throughput(N) / Throughput(Base)
base_throughput = filtered_df.loc[filtered_df['n_threads'].idxmin(), 'throughput']
filtered_df['speedup'] = filtered_df['throughput'] / base_throughput

# 3. Create Plot
fig, ax1 = plt.subplots(figsize=(12, 7))

# --- Primary Axis: Throughput ---
color_thru = '#2c7bb6' # Professional Blue
ax1.set_xlabel('Number of Threads (Cores)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Throughput (instances/s)', fontsize=12, color=color_thru)

line1 = ax1.plot(filtered_df['n_threads'], filtered_df['throughput'],
                 marker='o', markersize=8, color=color_thru,
                 linewidth=3, label='Actual Throughput')
ax1.tick_params(axis='y', labelcolor=color_thru)

# --- Secondary Axis: Speedup ---
ax2 = ax1.twinx()
color_speed = '#d7191c' # Professional Red
ax2.set_ylabel('Speedup (x-fold)', fontsize=12, color=color_speed)

# Plot Actual Speedup
line2 = ax2.plot(filtered_df['n_threads'], filtered_df['speedup'],
                 marker='s', linestyle='--', color=color_speed,
                 linewidth=2, label='Measured Speedup')

# Plot Ideal Speedup (Linear)
threads = filtered_df['n_threads']
line3 = ax2.plot(threads, threads / threads.min(),
                 linestyle=':', color='gray', alpha=0.7, label='Ideal Scaling')

ax2.tick_params(axis='y', labelcolor=color_speed)

# 4. Professional Formatting
plt.title(f"Scalability: {latest_config['ensemble_type']} on {latest_config['dataset']}\n",
          fontsize=16, fontweight='bold')

# Metadata Subtitle
meta_text = f"Trees: {latest_config['n_trees']} | Lambda: {latest_config['lambda']} | Features: {latest_config['features_patch']}"
fig.text(0.5, 0.88, meta_text, ha='center', fontsize=10, color='#555555')

ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xticks(filtered_df['n_threads']) # Ensure every thread count is marked

# Combined Legend
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True)

# Annotate Efficiency on the last point
final_efficiency = (filtered_df['speedup'].iloc[-1] / (filtered_df['n_threads'].iloc[-1] / filtered_df['n_threads'].iloc[0])) * 100
ax2.annotate(f'Efficiency: {final_efficiency:.1f}%',
             xy=(filtered_df['n_threads'].iloc[-1], filtered_df['speedup'].iloc[-1]),
             xytext=(-100, 10), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", color=color_speed),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_speed, lw=1),
             fontsize=10, fontweight='bold', color=color_speed)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_path}_{df.iloc[-1]['id']}.png", dpi=300)