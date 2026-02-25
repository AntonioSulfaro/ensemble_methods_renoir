import os
import platform

import matplotlib.pyplot as plt
import pandas as pd

# Set MAX_THREADS based on OS
MAX_THREADS = 16 if platform.system() == 'Linux' else 8


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
                'dataset': row['dataset'],  # Ensure this column exists in your CSV
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
    # We group by dataset so each one gets its own independent graph
    for dataset_name, dataset_group in summary_df.groupby('dataset'):
        plt.figure(figsize=(12, 8))

        # Plot each ensemble (SRP, ARF) with a different color within this dataset
        for ensemble_name, ensemble_group in dataset_group.groupby('ensemble'):
            plt.scatter(
                ensemble_group['throughput'],
                ensemble_group['accuracy'],
                label=ensemble_name,
                s=120,
                alpha=0.7,
                edgecolors='w'
            )

            # Annotations
            for i in range(len(ensemble_group)):
                row = ensemble_group.iloc[i]
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
                    (ensemble_group['throughput'].iat[i], ensemble_group['accuracy'].iat[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=7,
                    alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.15)
                )

        # Formatting
        plt.title(f"Accuracy vs Throughput - Dataset: {dataset_name}\n({MAX_THREADS} Threads)",
                  fontsize=14, fontweight='bold')
        plt.xlabel("Throughput (Instances/sec)")
        plt.ylabel("Final Accuracy")

        # Ensure graphs start from 0 as requested
        plt.xlim(left=0)
        plt.ylim(bottom=0, top=1.05)  # Accuracy usually caps at 1.0

        plt.legend(title="Ensemble Type")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Save unique file for each dataset
        safe_name = str(dataset_name).replace(" ", "_").replace("/", "_")
        output_plot = f"src/eval/results/comparison_{safe_name}.png"
        plt.savefig(output_plot, dpi=300)
        plt.close()  # Close figure to free memory for the next dataset
        print(f"Comparison plot for {dataset_name} saved to {output_plot}")


if __name__ == "__main__":
    analyze_experiments("src/eval/results/runs/", "src/eval/results/scalability/master_log.csv")
