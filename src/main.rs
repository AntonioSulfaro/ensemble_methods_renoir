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
use renoir::{Replication, RuntimeConfig, StreamContext};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;
use tree::HoeffdingTree;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

// --- FOREST CONSTANTS ---
const N_TREE: usize = 10; // Number of trees in the ensemble
const MAX_BINS: usize = 128; // Max bins for numeric features in Hoeffding Tree
// --- HOEFFDING TREE CONSTANTS ---
const N_MIN: usize = 200; // Minimum samples before split evaluation
const DELTA: f64 = 1e-7; // Confidence for Hoeffding bound
const TAU: f64 = 0.05; // Tie threshold
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
    let env = StreamContext::new(config.clone());

    // Determine output file name based on runtime configuration
    let (locality, threads) = match &config {
        RuntimeConfig::Local(local_cfg) => ("l", local_cfg.parallelism),
        RuntimeConfig::Remote(_) => (
            "r",
            std::thread::available_parallelism()
                .map(|q| q.get() as u64)
                .unwrap_or(1),
        ),
    };
    let file_path = format!("src/eval/results_{}{}.csv", locality, threads);

    // 1. CREATE DATA STREAM
    let (data, num_classes) = read_arff("src/eval/dense_100f_1M.arff");

    let global_start = Instant::now();

    // 2. REPLICATE TO ALL TREES
    let instances = env
        .stream_iter(data.into_iter())
        .flat_map(move |(instance_id, instance)| {
            (0..N_TREE).map(move |tree_id| (tree_id, instance_id, instance.clone()))
        });

    let feature_subspaces = srp::generate_feature_subspaces(N_FEATURES, FEATURES_PATCH, N_TREE);

    // 3. PROCESS IN PARALLEL PER TREE
    let results = instances
        .group_by(|(tree_id, ..)| *tree_id)
        .rich_map({
            // State maintained per partition (per tree)
            let all_subspaces = feature_subspaces;
            let mut tree: Option<HoeffdingTree> = None;
            let poisson = Poisson::new(LAMBDA).unwrap();

            move |(tree_id, (_orig_tree_id, instance_id, instance))| {
                let tree = tree.get_or_insert_with(|| {
                    let my_subspace = all_subspaces[*tree_id].clone();
                    HoeffdingTree::new(my_subspace, N_MIN, DELTA, TAU)
                });

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

                (instance_id, predicted_class, instance.label)
            }
        })
        .drop_key()
        .group_by(|(instance_id, ..)| *instance_id)
        .rich_map_transient({
            let mut entry = None;

            move |(inst_id, (_key, class_prediction, actual_label))| {
                let (count, votes) = entry.get_or_insert((0, vec![0; N_CLASSES]));

                *count += 1;

                if let Some(class) = class_prediction {
                    votes[class] += 1;
                }

                // check fragmentation target
                if *count == N_TREE {
                    // Determine winner (Majority Vote)
                    let winner = votes
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, count)| count)
                        .map(|(class_id, _)| class_id);

                    ControlFlow::Break(Some((*inst_id, winner, actual_label)))
                } else {
                    ControlFlow::Continue(None) // Still waiting for more trees to report
                }
            }
        })
        .filter_map(|(_, x)| x) // Remove the 'None' values from the stream
        .drop_key()
        .repartition_by(Replication::One, |_| 0)
        // 5. COMPUTE GLOBAL ACCURACY ON THE FLY
        .rich_map({
            let mut total_correct = 0;
            let mut total_processed = 0;

            move |(inst_id, winner, actual)| {
                total_processed += 1;
                if winner == actual {
                    total_correct += 1;
                }

                ExperimentResult {
                    instance_id: inst_id,
                    actual_class: actual,
                    predicted_class: winner,
                    global_accuracy: total_correct as f64 / total_processed as f64,
                }
            }
        })
        .write_csv(|_| file_path.into(), false);

    env.execute_blocking();

    let total_time = global_start.elapsed();
    println!("Execution time: {} ms", total_time.as_millis());
    print!("Throughput: {} instances/s", 1e5 / total_time.as_secs_f64());
}
