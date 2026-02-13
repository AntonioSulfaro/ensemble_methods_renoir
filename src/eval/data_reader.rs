use std::fs::File;
use std::io::{BufRead, BufReader};
use flate2::read::MultiGzDecoder;
use std::sync::Arc;
use crate::forest_results::ForestTask;
use crate::data_structures::Instance;
use std::collections::HashMap;

/// Reads an ARFF file (optionally gzipped) and converts it into
/// a vector of ForestTasks with instance id and the number of classes.
pub fn read_arff_to_tasks(path: &str) -> (Vec<(usize, ForestTask)>, usize) {
    let file = File::open(path).expect("Error opening .arff file");
    let reader: Box<dyn BufRead> = if path.ends_with(".gz") {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut in_data = false;
    let mut tasks = Vec::new();
    let mut label_map: HashMap<String, usize> = HashMap::new();
    let mut next_label_id = 0;
    let mut n_features_total: usize = 0;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => continue,
        };

        if line.is_empty() || line.starts_with('%') { continue; }
        let lower = line.to_lowercase();

        if !in_data {
            if lower.starts_with("@attribute") && !lower.contains("class") {
                n_features_total += 1;
                continue;
            }
            if lower.starts_with("@data") {
                in_data = true;
                continue;
            }
            if !line.starts_with('@') { in_data = true; }
        }

        if in_data {
            let instance = if line.starts_with('{') {
                // --- Sparse Parsing ---
                let content = &line[1..line.len() - 1];
                let parts: Vec<&str> = content.split(',').map(|s| s.trim()).collect();
                let mut features = vec![0.0; n_features_total];
                let mut label_val: Option<usize> = None;

                for part in parts {
                    let kv: Vec<&str> = part.split_whitespace().collect();
                    if kv.len() != 2 { continue; }
                    let idx = kv[0].parse::<usize>().unwrap_or(0);
                    let val_str = kv[1];

                    if idx == n_features_total {
                        label_val = Some(*label_map.entry(val_str.to_string()).or_insert_with(|| {
                            let id = next_label_id; next_label_id += 1; id
                        }));
                    } else if idx < n_features_total {
                        features[idx] = val_str.parse::<f64>().unwrap_or(0.0);
                    }
                }
                Instance { features, label: label_val }
            } else {
                // --- Dense Parsing ---
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() < n_features_total { continue; }

                let features: Vec<f64> = parts[0..n_features_total]
                    .iter()
                    .map(|s| s.parse::<f64>().unwrap_or(0.0))
                    .collect();

                let label_val = parts.get(n_features_total).map(|&s| {
                    if s == "?" { return None; } // ARFF standard for missing label
                    Some(*label_map.entry(s.to_string()).or_insert_with(|| {
                        let id = next_label_id; next_label_id += 1; id
                    }))
                }).flatten();

                Instance { features, label: label_val }
            };

            // --- Wrap in ForestTask and assign ID ---
            let instance_id = tasks.len();
            let task = match instance.label {
                Some(_) => ForestTask::Train(Arc::new(instance)),
                None => ForestTask::Predict {
                    instance_id,
                    instance: Arc::new(instance),
                },
            };
            tasks.push((instance_id, task));
        }
    }

    let num_classes = label_map.len();
    (tasks, num_classes)
}