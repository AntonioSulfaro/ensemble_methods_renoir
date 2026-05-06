import os
import pandas as pd
import numpy as np
import arff
import io
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path

def write_dense_preprocessed_arff(df, relation_name, output_path, target_col, num_classes):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"@relation {relation_name}_preprocessed\n\n")

        for col in df.columns:
            if col == target_col:
                classes_str = ",".join([str(i) for i in range(num_classes)])
                f.write(f"@attribute '{col}' {{{classes_str}}}\n")
            else:
                f.write(f"@attribute '{col}' numeric\n")

        f.write("\n@data\n")

        df.to_csv(f, header=False, index=False, lineterminator='\n')

def process_arff_file(file_path, output_dir):
    print(f"Processing: {file_path} ...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.endswith(','):
                stripped = stripped[:-1]
            cleaned_lines.append(stripped)

        cleaned_content = "\n".join(cleaned_lines)
        dataset = arff.load(io.StringIO(cleaned_content))

        columns = [attr[0] for attr in dataset['attributes']]
        df = pd.DataFrame(dataset['data'], columns=columns)

    except Exception as e:
        print(f"  [!] Failed to load {file_path}: {e}")
        return

    df.fillna(0, inplace=True)

    target_col = df.columns[-1]
    y = df.pop(target_col)

    continuous = df.select_dtypes(include=[np.number]).columns.tolist()
    discrete = df.select_dtypes(exclude=[np.number]).columns.tolist()

    processed_dfs = []

    if continuous:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[continuous].astype("float32"))
        df_continuous = pd.DataFrame(scaled_data, columns=continuous, index=df.index)
        processed_dfs.append(df_continuous)

    if discrete:
        df_discrete = pd.get_dummies(df[discrete], prefix_sep="#", dtype=np.float32)
        processed_dfs.append(df_discrete)

    if not processed_dfs:
        print(f"  [!] No features found in {file_path}.")
        return

    X_df = pd.concat(processed_dfs, axis=1)

    le = LabelEncoder()
    y = y.astype(str)
    y_encoded = le.fit_transform(y)
    y_df = pd.DataFrame({target_col: y_encoded}, index=df.index)
    num_classes = len(le.classes_)

    final_df = pd.concat([X_df, y_df], axis=1)

    base_name = file_path.stem
    output_filename = f"{base_name}_minmax_onehot.arff"
    output_path = output_dir / output_filename

    write_dense_preprocessed_arff(final_df, base_name, output_path, target_col, num_classes)

    print(f"  [✓] Saved (Dense) to: {output_path} ")
    print(f"      Classes: {list(le.classes_)} -> {list(range(num_classes))}\n")

if __name__ == "__main__":
    input_dirs = [Path("datasets/synthetic")]
    output_dir = Path("datasets/scale")

    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_process = []
    for d in input_dirs:
        if d.exists() and d.is_dir():
            for file_path in d.glob("*.arff"):
                if "_minmax_onehot" not in file_path.name:
                    files_to_process.append(file_path)

    if not files_to_process:
        print("No raw .arff files found in the synthetic directory.")
    else:
        print(f"Found {len(files_to_process)} files to process.\n")
        for file in files_to_process:
            process_arff_file(file, output_dir)
        print("Batch processing complete!")