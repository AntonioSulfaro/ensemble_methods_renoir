mod data_structures;
mod tree;
mod srp;
mod forest_results;

use crate::forest_results::{AggregatedPrediction, ForestResult, ForestTask};
use data_structures::Instance;
use rand::Rng;
use renoir::{RuntimeConfig, StreamContext};
use rand_distr::{Poisson, Distribution};
use std::collections::HashMap;
use tree::VFDT;

// --- FOREST CONSTANTS ---
const N_TREE: usize = 10;
const MAX_BINS: usize = 10;
// --- HOEFFDING TREE CONSTANTS ---
const N_MIN: usize = 20;        // Minimum samples before split evaluation
const DELTA: f64 = 1e-7;        // Confidence for Hoeffding bound
const TAU: f64 = 1e-4;          // Tie threshold
const RANGE_R: f64 = 1.0;       // Range of Gini coefficient
// --- SRP CONSTANTS ---
const N_FEATURES_PATCH: usize = 10;   // Number of features per patch
const LAMBDA: f64 = 1.0;
// --- DATASET CONSTANTS ---
const N_CLASSES: usize = 2;           // Number of classes
const N_FEATURES: usize = 100;        // Total number of features

/// Generate mixed stream of labeled (80%) and unlabeled (20%) instances
fn generate_stream_data(count: usize) -> Vec<(usize, ForestTask)> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   Streaming Random Forest with Hoeffding Trees (Renoir)   ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ Trees: {:43}                                              ║", N_TREE);
    println!("║ Training data: ~80% (labeled)                             ║");
    println!("║ Inference data: ~20% (unlabeled)                          ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let mut rng = rand::rng();

    (0..count).map(|id| {
        let f0 = rng.random_range(-2.0..2.0);

        // 80% training (labeled), 20% inference (unlabeled)
        if rng.random_bool(0.8) {
            let label = if f0 > 0.0 { 1 } else { 0 };
            (id, ForestTask::Train(Instance {
                features: vec![f0],
                label: Some(label),
            }))
        } else {
            (id, ForestTask::Predict {
                instance_id: id,
                instance: Instance {
                    features: vec![f0],
                    label: None,
                }
            })
        }
    }).collect()
}

// send id, fragmentation -> data and number of trees receiving it -> end when received
// this to avoid master presence (pure dataflow)

// [instance, tid, seq_n (instance_id), fragmentation, k bagging]
//flat_map
//group_by(tid)
//rich_map //processing tree
//group_by(seq_n)
//rich_map //inference aggregation

fn main() {
    let (config, _args) = RuntimeConfig::from_args();
    let env = StreamContext::new(config);

    // 1. CREATE DATA STREAM
    let data = generate_stream_data(1000);
    let poisson = Poisson::new(LAMBDA).unwrap();

    // 2. REPLICATE TO ALL TREES
    // TODO: wrap task in an Arc (Atomic Reference Count) so you are only cloning a pointer.
    let tasks = env.stream_iter(data.into_iter())
        .flat_map(move |(instance_id, task)| {
            let mut rng = rand::rng();
            let mut assignments = Vec::new();

            match &task {
                ForestTask::Train(_) => {
                    // Apply Bagging: k can be 0 (skip), 1, 2...
                    for tree_id in 0..N_TREE {
                        let k = poisson.sample(&mut rng) as usize;
                        if k > 0 {
                            assignments.push((tree_id, k));
                        }
                    }
                },
                ForestTask::Predict { .. } => {
                    // No Bagging for Inference: Send k=1 to ALL trees
                    for tree_id in 0..N_TREE {
                        assignments.push((tree_id, 1));
                    }
                }
            }

            let fragmentation = assignments.len();
            assignments.into_iter().map(move |(tree_id, k)| {
                (tree_id, instance_id, fragmentation, k, task.clone())
            })
        });

    let feature_subspaces = srp::generate_feature_subspaces(N_FEATURES, N_FEATURES_PATCH, N_TREE);

    // 3. PROCESS IN PARALLEL PER TREE
    // Group by tree_id: each partition maintains its own tree
    let results = tasks
        .group_by(|(tree_id, _instance_id, _fragmentation, _k, _task)| *tree_id)
        .rich_map({
            // State maintained per partition (per tree)
            let mut local_trees: HashMap<usize, VFDT> = HashMap::new();

            move |(tree_id, (_orig_tree_id, _instance_id, fragmentation, k, task))| {
                // Initialize tree if needed
                let tree = local_trees
                    .entry(*tree_id)
                    .or_insert_with(|| {
                        let subspace = feature_subspaces[*tree_id].clone();
                        VFDT::new(subspace, N_MIN, DELTA, TAU)
                    });

                // Process the task
                match task {
                    ForestTask::Train(inst) => {
                        tree.train(inst, k);

                        ForestResult::Trained {
                            tree_id: *tree_id,
                            nodes: tree.nodes.len(),
                        }
                    },
                    ForestTask::Predict { instance_id, instance } => {
                        let predicted_class = tree.predict(&instance);

                        ForestResult::Prediction {
                            instance_id,
                            tree_id: *tree_id,
                            predicted_class,
                            fragmentation,
                        }
                    }
                }
            }
        });

    let final_predictions = results
        .unkey()
        .map(|(_tid, res)| res)
        // 4a. Filter only predictions (ignore Trained variants)
        // TODO remove when implementing concept drift detection
        .filter_map(|res| match res {
            ForestResult::Prediction { instance_id, predicted_class, fragmentation, .. } =>
                Some((instance_id, (predicted_class, fragmentation))),
            _ => None,
        })
        // 4b. Align predictions for the same instance
        .group_by(|(instance_id, _)| *instance_id)

        // 5. Aggregate logic
        .rich_map({
            // State: Map<InstanceID, (Count, VotesHistogram)>
            let mut pending_votes: HashMap<usize, (usize, usize, HashMap<Option<usize>, usize>)> = HashMap::new();

            move |(inst_id, (_key, (class_prediction, frag_target)))| {
                let entry = pending_votes.entry(*inst_id).or_insert((0, frag_target, HashMap::new()));

                // Increment total votes received for this instance
                entry.0 += 1;
                // Record the specific vote
                *entry.2.entry(class_prediction).or_insert(0) += 1;

                // check fragmentation target
                if entry.0 == entry.1 {
                    let (_, count, votes) = pending_votes.remove(&inst_id).unwrap();

                    // Determine winner (Majority Vote)
                    let final_winner = votes.iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(class, _)| *class)
                        .flatten();

                    Some(AggregatedPrediction {
                        instance_id: *inst_id,
                        predicted_class: final_winner,
                        votes, //for debug
                        n_trees: count,
                    })
                } else {
                    None // Still waiting for more trees to report
                }
            }
        })
        .filter_map(|(_, x)| x); // Remove the 'None' values from the stream

    env.execute_blocking();
}