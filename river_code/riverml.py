import csv
import time
import os
from river import forest
from river import metrics
from river import evaluate
from pathlib import Path

DATASET_NAMES = [
    "RBF_m_minmax_onehot.arff"
]
SEEDS_TO_TEST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TREES_TO_TEST = 1
STEP_VAL = 1.0
DIRICHLET_VAL = 0.5
USE_AGGREGATION = True
INPUT_DIR = Path("datasets/scale")
OUTPUT_DIR = Path("river_result")

def load_arff_to_memory(filepath):
    attributes = []
    categorical_maps = {}
    dataset_in_memory = []
    in_data_section = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if not in_data_section:
                if line.lower().startswith('@data'):
                    in_data_section = True
                elif line.lower().startswith('@attribute'):
                    parts = line.split(maxsplit=2)
                    attr_name = parts[1].strip("'\"")
                    attr_type = parts[2]
                    if attr_type.lower() in ('numeric', 'real', 'integer', 'continuous'):
                        attributes.append((attr_name, 'numeric'))
                    elif attr_type.startswith('{') and attr_type.endswith('}'):
                        categories = [c.strip(" '\"") for c in attr_type[1:-1].split(',')]
                        cat_map = {cat: i for i, cat in enumerate(categories)}
                        categorical_maps[attr_name] = cat_map
                        attributes.append((attr_name, 'categorical'))
                    else:
                        attributes.append((attr_name, 'string'))
            else:
                values = line.split(',')
                x = {}
                for (attr_name, attr_type), val in zip(attributes, values):
                    val = val.strip(" '\"")
                    if val == '?':
                        x[attr_name] = None
                    elif attr_type == 'numeric':
                        x[attr_name] = float(val)
                    elif attr_type == 'categorical':
                        x[attr_name] = categorical_maps[attr_name][val]
                    else:
                        x[attr_name] = val
                target_col = attributes[-1][0]
                y = x.pop(target_col)
                dataset_in_memory.append((x, int(y) if isinstance(y, (float, str)) and y.isdigit() else y))
    return dataset_in_memory

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASET_NAMES:
        file_path = INPUT_DIR / dataset_name

        if not file_path.exists():
            print(f"[!] Skipping: {dataset_name} not found in {INPUT_DIR}")
            continue

        csv_filename = OUTPUT_DIR / f"{file_path.stem}_benchmark.csv"

        print(f"\n{'='*60}")
        print(f"LOADING DATASET: {dataset_name}")
        print(f"{'='*60}")

        dataset = load_arff_to_memory(file_path)
        n_samples = len(dataset)
        print(f"Loaded {n_samples} samples. Starting {len(SEEDS_TO_TEST)} seeds...")

        csv_headers = ["dataset used", "throughput", "accuracy", "step", "dirichlet", "seed"]
        file_exists = csv_filename.exists()

        with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(csv_headers)

            for current_seed in SEEDS_TO_TEST:
                print(f" -> Seed {current_seed}...", end=" ", flush=True)

                model = forest.AMFClassifier(
                    n_estimators=TREES_TO_TEST,
                    step=STEP_VAL,
                    dirichlet=DIRICHLET_VAL,
                    use_aggregation=USE_AGGREGATION,
                    seed=current_seed
                )
                metric = metrics.Accuracy()

                start_time = time.time()

                final_metric = evaluate.progressive_val_score(
                    dataset=dataset,
                    model=model,
                    metric=metric
                )

                end_time = time.time()
                elapsed_time = end_time - start_time

                throughput = n_samples / elapsed_time if elapsed_time > 0 else 0
                final_accuracy = final_metric.get()

                writer.writerow([
                    dataset_name,
                    round(throughput, 2),
                    round(final_accuracy, 4),
                    STEP_VAL,
                    DIRICHLET_VAL,
                    current_seed
                ])
                csv_file.flush()
                print(f"Done (Acc: {final_accuracy:.4f}, Speed: {throughput:,.2f}/s)")

    print(f"\nBatch Benchmark Complete! Results stored in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()