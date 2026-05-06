import os
import time
import numpy as np
import pandas as pd
from scipy.io import arff
from onelearn import AMFClassifier

DATASET_DIR = os.path.join("datasets", "scale")
OUTPUT_DIR = "river_result"

FILE_NAMES = [
    "AGR_a_minmax_onehot.arff",
    "AGR_g_minmax_onehot.arff",
    "LED_a_minmax_onehot.arff",
    "LED_g_minmax_onehot.arff",
    "RBF_f_minmax_onehot.arff",
    "RBF_m_minmax_onehot.arff"
]

SEEDS_TO_TEST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TREES_TO_TEST = 1
DIRICHLET_VAL = 0.5
STEP_VAL = 1.0
USE_AGGREGATION = True

def evaluate_dataset(file_path, current_seed):
    dataset_name = os.path.basename(file_path)

    try:
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"  [!] Failed to load {file_path}: {e}")
        return None

    target_col = df.columns[-1]

    X = np.ascontiguousarray(df.drop(columns=[target_col]).values, dtype=np.float32)
    y = np.ascontiguousarray(df[target_col].values, dtype=np.float32)

    n_classes = len(np.unique(y))
    n_samples = X.shape[0]
    n_features = X.shape[1]

    amf = AMFClassifier(
        n_classes=n_classes,
        n_estimators=TREES_TO_TEST,
        random_state=current_seed,
        use_aggregation=USE_AGGREGATION,
        dirichlet=DIRICHLET_VAL,
        step=STEP_VAL
    )

    correct_predictions = 0
    start_time = time.time()

    for i in range(n_samples):
        X_i = X[i:i+1]
        y_i = y[i:i+1]

        if i > 0:
            prediction = amf.predict_proba(X_i).argmax(axis=1)[0]
            if prediction == y_i[0]:
                correct_predictions += 1

        amf.partial_fit(X_i, y_i)

    elapsed_time = time.time() - start_time

    final_accuracy = correct_predictions / n_samples if n_samples > 0 else 0
    instances_per_sec = n_samples / elapsed_time if elapsed_time > 0 else 0

    return {
        "Dataset": dataset_name,
        "Samples": n_samples,
        "Features": n_features,
        "Classes": n_classes,
        "Accuracy": final_accuracy,
        "Instances_per_Second": instances_per_sec,
        "Trees": TREES_TO_TEST,
        "Seed": current_seed,
        "Dirichlet": DIRICHLET_VAL,
        "Step": STEP_VAL
    }

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in FILE_NAMES:
        file_path = os.path.join(DATASET_DIR, filename)

        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' could not be found.")
            continue

        dataset_name = os.path.basename(file_path)
        print(f"\nEvaluating dataset: {dataset_name}")

        dataset_results = []

        for seed in SEEDS_TO_TEST:
            result = evaluate_dataset(file_path, current_seed=seed)

            if result is not None:
                dataset_results.append(result)
                print(f"  [✓] Seed {seed} | Accuracy: {result['Accuracy']*100:.2f}% | Speed: {result['Instances_per_Second']:.0f} inst/sec")

        if dataset_results:
            results_df = pd.DataFrame(dataset_results)
            csv_filename = os.path.join(OUTPUT_DIR, f"{dataset_name}_results.csv")
            results_df.to_csv(csv_filename, index=False)
            print(f"Saved individual results to '{csv_filename}'")