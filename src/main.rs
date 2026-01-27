mod data_structures;
mod tree;
mod srp;

use std::collections::HashMap;
use rand::Rng;
use renoir::{RuntimeConfig, StreamContext};
use data_structures::Instance;
use tree::VFDT;

fn get_synthetic_data(count: usize) -> Vec<Instance> {
    let mut rng = rand::thread_rng();
    (0..count).map(|idx| {
        // Simple logic: if f0 > 0, label is 1, else 0
        let f0 = rng.gen_range(-2.0..2.0);
        let label = if f0 > 0.0 { 1 } else { 0 };

        Instance {
            features: vec![f0],
            label: Some(label),
        }
    }).collect()
}

// --- CONSTANTS ---
const N_TREE: usize = 10;
//  -- for the Hoeffding Bound --

const DELTA: f64 = 1e-7;        //Confidence
const TAU: f64 = 1e-4;          //Tie threshold
const RANGE_R : f64 = 1.0;      //Range of Gini coefficient

fn main() {
    let (config, _args) = RuntimeConfig::from_args();
    let env = StreamContext::new(config);

    // 1. Create a stream of (TreeID, Instance)
    // We duplicate each instance for every tree in the forest
    let data = get_synthetic_data(1000);
    let instances = env.stream_iter(data.into_iter())
        .flat_map(move |inst| {
            (0..N_TREE).map(move |tid| (tid, inst.clone()))
        });

    // 2. Distributed Training
    let trained_forest = instances
        .group_by(|(tid, _inst)| *tid)
        .rich_map({
            let mut local_trees: HashMap<usize, VFDT> = HashMap::new();
            let n_min = 20;

            move |args: (&usize, (usize, Instance))| {
                let (tid, (_orig_tid, inst)) = args;

                let tree = local_trees
                    .entry(*tid)
                    .or_insert_with(|| VFDT::new(n_min, DELTA, TAU));

                tree.train(inst);

                tree.nodes.len()
            }
        });

    trained_forest.for_each(|(tid, count)| {
        println!("Tree {} updated. Nodes: {}", tid, count);
    });

    env.execute_blocking();
}