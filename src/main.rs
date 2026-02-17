mod data_structures;
mod forest_utils;
mod srp;
mod tree;
mod eval {
    pub mod data_reader;
}

use crate::eval::data_reader::read_arff;
use crate::forest_utils::AggregatedPrediction;
use data_structures::Instance;
use rand::RngExt;
use rand_distr::{Distribution, Poisson};
use renoir::{RuntimeConfig, StreamContext};
use std::ops::ControlFlow;
use std::sync::Arc;
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
const N_FEATURES_PATCH: usize = 10; // Number of features per patch
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
    let (data, num_classes) = read_arff("dense_100f_100k.arff");

    // 2. REPLICATE TO ALL TREES
    let instances = env
        .stream_iter(data.into_iter())
        .flat_map(move |(instance_id, instance)| {
            (0..N_TREE).map(move |tree_id| (tree_id, instance_id, instance.clone()))
        });

    let feature_subspaces = srp::generate_feature_subspaces(N_FEATURES, N_FEATURES_PATCH, N_TREE);

    // 3. PROCESS IN PARALLEL PER TREE
    // Group by tree_id: each partition maintains its own tree
    let results = instances.group_by(|(tree_id, ..)| *tree_id).rich_map({
        // State maintained per partition (per tree)
        let subspace = feature_subspaces.clone();
        let mut tree: Option<HoeffdingTree> = None;
        let poisson = Poisson::new(LAMBDA).unwrap();

        move |(tree_id, (_orig_tree_id, instance_id, instance))| {
            let tree = tree.get_or_insert(HoeffdingTree::new(
                subspace[*tree_id].clone(),
                N_MIN,
                DELTA,
                TAU,
            ));

            let predicted_class = tree.predict(&instance);

            //TODO drift detection logic
            // let is_correct = predicted_class == instance.label;

            let mut rng = rand::rng();
            let k = poisson.sample(&mut rng) as usize;
            if k > 0 {
                tree.train(&instance, k);
            }

            (instance_id, predicted_class)
        }
    });

    let final_predictions = results
        .drop_key()
        .group_by(|(instance_id, _)| *instance_id)
        // 5. Aggregate logic
        .rich_map_transient({
            let mut entry = None;

            move |(inst_id, (_key, class_prediction))| {
                let (count, votes) = entry.get_or_insert((0, vec![0; N_CLASSES]));

                // Increment total votes received for this instance
                *count += 1;

                if let Some(class) = class_prediction {
                    votes[class] += 1;
                }

                // check fragmentation target
                if *count == N_TREE {
                    // Determine winner (Majority Vote)
                    let final_winner = votes
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, count)| count)
                        .map(|(class_id, _)| class_id);

                    ControlFlow::Break(Some(AggregatedPrediction {
                        instance_id: *inst_id,
                        predicted_class: final_winner,
                        votes: votes.clone(), //for debug
                        n_trees: *count,
                    }))
                } else {
                    ControlFlow::Continue(None) // Still waiting for more trees to report
                }
            }
        })
        .filter_map(|(_, x)| x) // Remove the 'None' values from the stream
        .for_each(drop);

    env.execute_blocking();
}
