import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set MAX_THREADS based on OS
MAX_THREADS = 32 if platform.system() == 'Linux' else 8


def plot_pareto_frontier(x, y, ax, color='gray'):
    """Draws a line connecting the best performing points."""
    # Sort by x (throughput)
    sorted_indices = np.argsort(x)
    x_s, y_s = x[sorted_indices], y[sorted_indices]

    pareto_x, pareto_y = [x_s[0]], [y_s[0]]
    for i in range(1, len(x_s)):
        if y_s[i] > max(pareto_y):  # Only add if it improves accuracy
            pareto_x.append(x_s[i])
            pareto_y.append(y_s[i])

    ax.step(pareto_x, pareto_y, where='post', linestyle='--', color=color, alpha=0.4, label='_nolegend_')


def analyze_experiments(base_results_path, master_log_path):
    if not os.path.exists(master_log_path):
        print("Master log not found!")
        return

    master_df = pd.read_csv(master_log_path)
    comparison_data = []

    # 1. Collect data
    for _, row in master_df.iterrows():
        if row['n_threads'] != MAX_THREADS:
            continue

        run_id = row['id']
        run_dir = os.path.join(base_results_path, run_id)
        accuracy_file = os.path.join(run_dir, "accuracy.csv")

        if os.path.exists(accuracy_file):
            acc_df = pd.read_csv(accuracy_file)
            final_acc = acc_df['global_accuracy'].iloc[-1]

            total_instances = acc_df['instance_id'].iloc[-1] + 1
            throughput = total_instances / row['time']

            comparison_data.append({
                'dataset': row['dataset'],
                'ensemble': row['ensemble_type'],
                'n_trees': row['n_trees'],
                'drift_detection': row['drift_detection'],
                'max_bins': row['max_bins'],
                'n_min': row['n_min'],
                'features_patch': row['features_patch'],
                'lambda': row['lambda'],
                'accuracy': final_acc,
                'throughput': throughput,
            })

    if not comparison_data:
        print(f"No valid data found for {MAX_THREADS} threads.")
        return

    summary_df = pd.DataFrame(comparison_data)

    # 2. Plotting per Dataset
    for dataset_name, dataset_group in summary_df.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Professional color palette
        colors = {'srp': '#2c7bb6', 'arf': '#d7191c'}

        for ensemble_name, ensemble_group in dataset_group.groupby('ensemble'):
            # Encoding variables into markers
            # Size = n_trees | Shape = Drift Detection
            drift_on = ensemble_group[ensemble_group['drift_detection'] == True]
            drift_off = ensemble_group[ensemble_group['drift_detection'] == False]

            c = colors.get(ensemble_name.lower(), 'black')

            # Plot Drift Enabled (Circles)
            ax.scatter(drift_on['throughput'], drift_on['accuracy'],
                       s=drift_on['n_trees'] * 8, c=c, label=f"{ensemble_name.upper()} (Drift On)",
                       alpha=0.8, edgecolors='none')

            # Plot Drift Disabled (Squares)
            ax.scatter(drift_off['throughput'], drift_off['accuracy'],
                       s=drift_off['n_trees'] * 8, c=c, marker='s', label=f"{ensemble_name.upper()} (Drift Off)",
                       alpha=0.5, edgecolors='none')

            # Draw Pareto Frontier for this specific ensemble
            plot_pareto_frontier(ensemble_group['throughput'].values,
                                 ensemble_group['accuracy'].values, ax, c)

        # Labels for specific configuration points (Lambda and Patching)
        # We only label points on the frontier to avoid overlapping text
        for i, row in dataset_group.iterrows():
            # Basic logic: Only label if it's in the top 15% of accuracy for its ensemble
            if row['accuracy'] > dataset_group['accuracy'].quantile(0.85):
                ax.text(row['throughput'], row['accuracy'] + 0.005,
                        f"λ:{row['lambda']}, F:{row['features_patch']}",
                        fontsize=7, ha='center', alpha=0.7)

        # Axis Formatting
        ax.set_title(f"Throughput-Accuracy Pareto Analysis: {dataset_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Throughput (instances/s)", fontsize=11)
        ax.set_ylabel("Global Accuracy", fontsize=11)

        # Axes forced to start from 0
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.0)

        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='lower right', frameon=True, fontsize=9)

        plt.tight_layout()

        # Save unique file for each dataset
        safe_name = str(dataset_name).replace(" ", "_").replace("/", "_")
        output_plot = f"results/comparison_{safe_name}.png"
        plt.savefig(output_plot, dpi=300)
        plt.close()  # Close figure to free memory for the next dataset
        print(f"Comparison plot for {dataset_name} saved to {output_plot}")


if __name__ == "__main__":
    analyze_experiments("results/runs/", "results/scalability/master_log_ht.csv")  # or _amf
