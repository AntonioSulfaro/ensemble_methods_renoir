import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python avg_depth_graph.py <run_dir>")
    sys.exit(1)

run_dir = sys.argv[1]

df = pd.read_csv(os.path.join(run_dir, "accuracy.csv"))
with open(os.path.join(run_dir, "config.json"), "r") as f:
    config = json.load(f)

# Add instance number for x‑axis
df["instance_number"] = range(1, len(df) + 1)

research_params = {
    "Model": config.get("ensemble_type"),
    "Trees": config.get("n_trees"),
    "Lambda": config.get("lambda"),
    "Features": config.get("features_patch", "Full"),
}
config_str = "  |  ".join([f"{k}: {v}" for k, v in research_params.items()])

fig, ax = plt.subplots(figsize=(14, 7))

# Plot average tree depth (raw, per instance)
plot_step = max(1, len(df) // 100)
ax.plot(
    df["instance_number"][::plot_step],
    df["avg_depth"][::plot_step],
    color="#2c7bb6",
    marker="o",
    markerfacecolor="#2c7bb6",
    markeredgecolor="white",
    markeredgewidth=0.5,
    linewidth=1.5,
    label="Average Tree Depth (every 1% of instances)",
)

# Mark drifts
drift_indices = df[df["drift_detected"] == True]["instance_number"]
for i, x_pos in enumerate(drift_indices):
    ax.axvline(
        x=x_pos,
        color="#d7191c",
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label="Drift" if i == 0 else None,
    )

# Labels, limits, grid
ax.set_xlabel("Instances Seen", fontsize=11, fontweight="bold")
ax.set_ylabel("Average Tree Depth", fontsize=11, fontweight="bold")
ax.set_ylim(bottom=0)
ax.grid(True, which="both", linestyle=":", alpha=0.5)

# Throughput and title
plt.suptitle(
    f"Tree Depth Evolution – {config.get('dataset', 'Dataset')}",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
ax.set_title(
    f"{config_str}",
    fontsize=10,
    color="#444444",
    pad=10,
)

# Legend (clean)
ax.legend(loc="upper left", frameon=True, shadow=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(run_dir, "depth.png"), dpi=300)
