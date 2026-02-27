import json
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) < 4:
    print("Usage: python script.py <n_threads> <t_time> <run_dir>")
    sys.exit(1)

# Setup and Data Loading
n_threads = sys.argv[1]
t_time = float(sys.argv[2])
run_dir = sys.argv[3]

df = pd.read_csv(os.path.join(run_dir, "accuracy.csv"))
with open(os.path.join(run_dir, "config.json"), 'r') as f:
    config = json.load(f)

# Reverse-Engineer Windowed Accuracy (Prequential)
# Window size
window_size = max(500, min(len(df) // 100, 10000))
df['instance_number'] = range(1, len(df) + 1)

# Calculate total correct hits at each point
df['total_correct'] = (df['global_accuracy'] * df['instance_number']).round()

# Calculate hits within the sliding window
df['window_hits'] = df['total_correct'].diff(periods=window_size)
df['prequential_accuracy'] = df['window_hits'] / window_size

# Filter Config for Plotting (Only show what matters for research)
# We hide hyperparameters that are usually constant (like delta/tau)
research_params = {
    "Model": config.get("ensemble_type"),
    "Trees": config.get("n_trees"),
    "Lambda": config.get("lambda"),
    "Voting": config.get("voting"),
    "Features": config.get("features_patch", "Full")
}
config_str = "  |  ".join([f"{k}: {v}" for k, v in research_params.items()])

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))

# Plot 1: The Global Accuracy (Faded background)
ax.plot(df['instance_number'], df['global_accuracy'],
        color='green', alpha=0.7, label='Cumulative Accuracy', linestyle='--')

plot_step = max(1, len(df) // 2000)

# Plot 2: The Prequential Accuracy (The "Real-time" performance)
ax.plot(df['instance_number'][::plot_step], df['prequential_accuracy'][::plot_step],
        color='#2c7bb6', linewidth=2, label=f'Prequential Accuracy (Window={window_size})')

# Annotate Drifts
drift_indices = df[df['drift_detected'] == True]['instance_number']
for i, x_pos in enumerate(drift_indices):
    ax.axvline(x=x_pos, color='#d7191c', linestyle='--', alpha=0.6, linewidth=1.5, label='Drift' if i == 0 else None)

ax.set_xlabel('Instances Seen', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax.set_ylim(min(df['prequential_accuracy'].dropna().min() - 0.05, 0), 1.05)
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='lower left', frameon=True)

throughput = len(df) / t_time
plt.suptitle(f"Distributed Ensemble Performance: {config.get('dataset', 'Dataset')}",
             fontsize=16, fontweight='bold', y=0.98)
ax.set_title(f"Threads: {n_threads}  •  Throughput: {throughput:.2f} instances/s  •  Time: {t_time}s\n{config_str}",
             fontsize=10, color='#444444', pad=10)

# Annotate final windowed performance
final_global = df['global_accuracy'].iloc[-1]
ax.annotate(f'Final Accuracy: {final_global:.2%}',
            xy=(df['instance_number'].iloc[-1], final_global),
            xytext=(15, -10), textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#2c7bb6", lw=1.5),
            fontsize=10, fontweight='bold', color='#2c7bb6',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='#2c7bb6'))
# print(final_global)

# Clean Legend
ax.legend(loc='lower right', frameon=True, shadow=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(run_dir, "accuracy.png"), dpi=300)