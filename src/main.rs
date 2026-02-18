mod data_structures;
mod forest_utils;
mod srp;
mod tree;
mod eval {
    pub mod data_reader;
    pub mod evaluation;
}

use crate::eval::data_reader::read_arff;
use crate::eval::evaluation::ExperimentResult;
use data_structures::Instance;
use rand::RngExt;
use rand_distr::{Distribution, Poisson};
use renoir::{RuntimeConfig, StreamContext};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;
use tree::HoeffdingTree;

// --- FOREST CONSTANTS ---
const N_TREE: usize = 10;
const MAX_BINS: usize = 10;
// --- HOEFFDING TREE CONSTANTS ---
const N_MIN: usize = 20; // Minimum samples before split evaluation
const DELTA: f64 = 1e-7; // Confidence for Hoeffding bound
const TAU: f64 = 1e-4; // Tie threshold
const RANGE_R: f64 = 1.0; // Range of Gini coefficient
// --- SRP CONSTANTS ---
const FEATURES_PATCH: f64 = 0.6; // Percentage of features for subspace
const LAMBDA: f64 = 1.0;
// --- DATASET CONSTANTS ---
const N_CLASSES: usize = 2; // Number of classes
const N_FEATURES: usize = 100; // Total number of features

/// Generate stream of instances from synthetic data
fn generate_stream_data(count: usize) -> Vec<(usize, Arc<Instance>)> {
    let mut rng = rand::rng();

    (0..count)
        .map(|id| {
            let f0 = rng.random_range(-2.0..2.0);
            let label = if f0 > 0.0 { 1 } else { 0 };

            (
                id,
                Arc::new(Instance {
                    features: vec![f0],
                    label: Some(label),
                }),
            )
        })
        .collect()
}

fn main() {
    let (config, _args) = RuntimeConfig::from_args();
    let env = StreamContext::new(config);

    // 1. CREATE DATA STREAM
    let (data, num_classes) = read_arff("dense_100f_100k.arff");

    let global_start = Instant::now();

    // 2. REPLICATE TO ALL TREES
    let instances = env
        .stream_iter(data.into_iter())
        .flat_map(move |(instance_id, instance)| {
            // start timer for each instance
            let start_offset = global_start.elapsed();
            (0..N_TREE).map(move |tree_id| (tree_id, instance_id, instance.clone(), start_offset))
        });

    let feature_subspaces = srp::generate_feature_subspaces(N_FEATURES, FEATURES_PATCH, N_TREE);

    // 3. PROCESS IN PARALLEL PER TREE
    let results = instances.group_by(|(tree_id, ..)| *tree_id).rich_map({
        // State maintained per partition (per tree)
        let subspace = feature_subspaces.clone();
        let mut tree: Option<HoeffdingTree> = None;
        let poisson = Poisson::new(LAMBDA).unwrap();

        move |(tree_id, (_orig_tree_id, instance_id, instance, start_offset))| {
            let tree = tree.get_or_insert(HoeffdingTree::new(
                subspace[*tree_id].clone(),
                N_MIN,
                DELTA,
                TAU,
            ));

            // predict
            let predicted_class = tree.predict(&instance);

            //TODO drift detection logic
            // let is_correct = predicted_class == instance.label;

            // train
            let mut rng = rand::rng();
            let k = poisson.sample(&mut rng) as usize;
            if k > 0 {
                tree.train(&instance, k);
            }

            (instance_id, predicted_class, instance.label, start_offset)
        }
    });

    // 4. RE-AGGREGATE BY INSTANCE
    let final_predictions = results
        .drop_key()
        .group_by(|(instance_id, ..)| *instance_id)
        .rich_map_transient({
            let mut entry = None;

            move |(inst_id, (_key, class_prediction, actual_label, start_offset))| {
                let (count, votes) = entry.get_or_insert((0, vec![0; N_CLASSES]));

                *count += 1;

                if let Some(class) = class_prediction {
                    votes[class] += 1;
                }

                // check fragmentation target
                if *count == N_TREE {
                    let now_offset = global_start.elapsed();

                    let latency_duration = now_offset.saturating_sub(start_offset);
                    let latency_micros = latency_duration.as_micros();

                    // Determine winner (Majority Vote)
                    let winner = votes
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, count)| count)
                        .map(|(class_id, _)| class_id);

                    ControlFlow::Break(Some((*inst_id, winner, actual_label, latency_micros)))
                } else {
                    ControlFlow::Continue(None) // Still waiting for more trees to report
                }
            }
        })
        .filter_map(|(_, x)| x) // Remove the 'None' values from the stream
        .drop_key()
        // 5. COMPUTE GLOBAL ACCURACY ON THE FLY
        .rich_map({
            let mut total_correct = 0;
            let mut total_processed = 0;

            move |(inst_id, winner, actual, latency)| {
                total_processed += 1;
                if winner == actual {
                    total_correct += 1;
                }

                ExperimentResult {
                    instance_id: inst_id,
                    actual_class: actual,
                    predicted_class: winner,
                    latency_micros: latency,
                    global_accuracy: total_correct as f64 / total_processed as f64,
                }
            }
        })
        .collect_vec();

    env.execute_blocking();

    // 6. SAVE RESULTS TO CSV
    if let Some(mut final_data) = final_predictions.get() {
        final_data.sort_by_key(|res| res.latency_micros);

        let file_path = "results.csv";
        let mut wtr = csv::Writer::from_path(file_path).expect("Unable to create CSV file");

        for res in final_data {
            wtr.serialize(res).expect("Error serializing result");
        }

        wtr.flush().expect("Error flushing CSV");
        println!("Successfully saved results to {}", file_path);
    } else {
        eprintln!("No results were collected.");
    }
}
