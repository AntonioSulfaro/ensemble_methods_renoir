import os

import matplotlib.pyplot as plt
import pandas as pd

MAX_THREADS = 8


def analyze_experiments(base_results_path, master_log_path):
    # 1. Load the master log
    if not os.path.exists(master_log_path):
        print("Master log not found!")
        return

    master_df = pd.read_csv(master_log_path)

    # We will store the final results here
    comparison_data = []

    # 2. Iterate through each run recorded in the master log
    for _, row in master_df.iterrows():
        if row['n_threads'] != MAX_THREADS:
            continue
        run_id = row['id']
        run_dir = os.path.join(base_results_path, run_id)
        accuracy_file = os.path.join(run_dir, "accuracy.csv")

        if os.path.exists(accuracy_file):
            # Load accuracy data
            acc_df = pd.read_csv(accuracy_file)
            # Get the last recorded accuracy (final performance)
            final_acc = acc_df['global_accuracy'].iloc[-1]

            # Calculate throughput: (Total Instances / Total Time)
            total_instances = acc_df['instance_id'].iloc[-1] + 1
            throughput = total_instances / row['time']

            comparison_data.append({
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

    # 3. Plotting
    summary_df = pd.DataFrame(comparison_data)
    plt.figure(figsize=(10, 7))

    for name, group in summary_df.groupby('ensemble'):
        plt.scatter(group['throughput'], group['accuracy'], label=name, s=100, alpha=0.7)

        # Annotate points with all parameters
        for i in range(len(group)):
            # Extract the metadata for this specific point
            row = group.iloc[i]

            # Create a multi-line label
            # We use \n to keep the label vertical and narrow
            label = (
                f"T:{row['n_trees']}\n"
                f"Drift:{row['drift_detection']}\n"
                f"Bins:{row['max_bins']}\n"
                f"n_min:{row['n_min']}\n"
                f"Patch:{row['features_patch']}\n"
                f"λ:{row['lambda']}"
            )

            plt.annotate(
                label,
                (group['throughput'].iat[i], group['accuracy'].iat[i]),
                xytext=(5, 5),  # Shift text 5pts right and up from the point
                textcoords='offset points',
                fontsize=7,  # Small font to avoid clutter
                alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.2)  # Light background
            )

    plt.title("Accuracy vs. Throughput: Finding the Sweet Spot")
    plt.xlabel("Throughput (Instances/sec)")
    plt.ylabel("Final Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    output_plot = "src/eval/results/comparison_summary.png"
    plt.savefig(output_plot)
    print(f"Comparison plot saved to {output_plot}")


if __name__ == "__main__":
    analyze_experiments("src/eval/results/runs/", "src/eval/results/scalability/master_log.csv")
