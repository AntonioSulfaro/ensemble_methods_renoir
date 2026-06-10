import matplotlib.pyplot as plt
import pandas as pd

algo_type = "ht"  # ht or amf
back = 0

# 1. Load and Filter
file_path = 'results/scalability/'
df = pd.read_csv(file_path + f'master_log_{algo_type}.csv')

config_columns = ['dataset', 'n_trees', 'voting', 'ensemble_type']

config_columns += [
    'drift_detection', 'n_min', 'delta', 'tau', 'features_patch', 'lambda', 'numeric_estimator'
] if algo_type == 'ht' else ['step', 'dirichlet']

latest_config = df.iloc[-1 - (6 * back)][config_columns]
filtered_df = df.copy()
for col in config_columns:
    filtered_df = filtered_df[filtered_df[col] == latest_config[col]]

# Ensure we are sorted by threads for line plotting
filtered_df = filtered_df.sort_values('n_threads').reset_index()

# 2. Calculate Scalability Metrics
filtered_df['throughput'] = filtered_df['n_instances'] / filtered_df['time']

# Speedup = Throughput(N) / Throughput(Base)
baseline_threads = filtered_df['n_threads'].min()
base_throughput = filtered_df.loc[
    filtered_df['n_threads'] == baseline_threads, 'throughput'
].iloc[0]
filtered_df['speedup'] = filtered_df['throughput'] / base_throughput

# 3. Create Plot
fig, ax1 = plt.subplots(figsize=(12, 7))

# --- Primary Axis: Throughput ---
color_thru = '#2c7bb6'  # Blue
ax1.set_xlabel('Number of Threads (Cores)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Throughput (instances/s)', fontsize=12, color=color_thru)

line1 = ax1.plot(filtered_df['n_threads'], filtered_df['throughput'],
                 marker='o', markersize=8, color=color_thru,
                 linewidth=3, label='Actual Throughput')
ax1.tick_params(axis='y', labelcolor=color_thru)

# --- Secondary Axis: Speedup ---
ax2 = ax1.twinx()
color_speed = '#d7191c'  # Red
ax2.set_ylabel('Speedup (x-fold)', fontsize=12, color=color_speed)

# Plot Actual Speedup
line2 = ax2.plot(filtered_df['n_threads'], filtered_df['speedup'],
                 marker='s', linestyle='--', color=color_speed,
                 linewidth=2, label='Measured Speedup')

# Plot Ideal Speedup (Linear)
threads = filtered_df['n_threads']
line3 = ax2.plot(threads, threads / threads.min(),
                 linestyle=':', color='gray', alpha=0.7, label='Ideal Speedup')

ax2.tick_params(axis='y', labelcolor=color_speed)

# 4. Formatting
# Comment for thesis image
# plt.title(f"Scalability: {latest_config['ensemble_type']} on {latest_config['dataset']}\n",
#           fontsize=16, fontweight='bold')

# Shared top-level params
research_params = {
    "Trees": latest_config["n_trees"],
    "Voting": latest_config["voting"],
}

if algo_type == "amf":
    research_params["Step"] = latest_config["step"]
    research_params["Dirichlet"] = latest_config["dirichlet"]
else:
    # Srp or Arf logic
    features_val = latest_config["features_patch"]
    research_params["Lambda"] = latest_config["lambda"]
    research_params["Features"] = f"{features_val:.0%}" if isinstance(features_val, (float, int)) else "Full"
meta_text = "  |  ".join([f"{k}: {v}" for k, v in research_params.items()])

# Metadata Subtitle
# Fontsize: 10 standard, 12 thesis
fig.text(0.5, 0.88, meta_text, ha='center', fontsize=12, color='#555555')

ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xticks(filtered_df['n_threads'])  # Ensure every thread count is marked

# Combined Legend
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True)

# Annotate Efficiency
for i in range(1, 3):
    final_efficiency = (filtered_df['speedup'].iloc[-i] / (
            filtered_df['n_threads'].iloc[-i] / filtered_df['n_threads'].iloc[0])) * 100
    ax2.annotate(f'Speedup: {filtered_df['speedup'].iloc[-i]:.1f}\nEfficiency: {final_efficiency:.1f}%',
                 xy=(filtered_df['n_threads'].iloc[-i], filtered_df['speedup'].iloc[-i]),
                 xytext=(-100, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color=color_speed),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_speed, lw=1),
                 fontsize=10, fontweight='bold', color=color_speed)

final_throughput = filtered_df['throughput'].iloc[-1]
print(f"throughput: {final_throughput}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{file_path}{df.iloc[-1 - (6 * back)]['id']}_{latest_config['ensemble_type']}.png", dpi=300)
