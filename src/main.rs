mod data_structures;
mod exec_config;
mod forest_utils;
mod srp;
mod tree;

mod eval {
    pub mod data_reader;
    pub mod evaluation;
}

use crate::eval::data_reader::read_arff;
use crate::eval::evaluation::ExperimentResult;
use rand_distr::{Distribution, Poisson};
use renoir::{Replication, RuntimeConfig, StreamContext};
use std::ops::ControlFlow;
use std::process::Command;
use std::time::Instant;
use tree::HoeffdingTree;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DRAW_ACCURACY_GRAPH: bool = true;
const DRAW_SCALABILITY_GRAPH: bool = true;

fn main() {
    let (config, _args) = RuntimeConfig::from_args();
    let env = StreamContext::new(config.clone());
    let config_str =
        std::fs::read_to_string("exec_config.toml").expect("Failed to read config.toml");
    let exec_config: exec_config::ExecConfig =
        toml::from_str(&config_str).expect("Failed to parse config.toml");

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
    let base_path = "src/eval/";
    let file_path = format!(
        "{}results/accuracy/accuracy_{}{}.csv",
        base_path, locality, threads
    );

    // 1. CREATE DATA STREAM
    let (data, n_classes, n_features) = read_arff(&format!("{}dense_100f_1M.arff", base_path));

    let global_start = Instant::now();

    // 2. REPLICATE TO ALL TREES
    let instances = env
        .stream_iter(data.into_iter())
        .flat_map(move |(instance_id, instance)| {
            (0..exec_config.n_trees).map(move |tree_id| (tree_id, instance_id, instance.clone()))
        });

    // if (exec_config.ensemble_type == "srp")
    let feature_subspaces = srp::generate_feature_subspaces(
        n_features,
        exec_config.features_patch,
        exec_config.n_trees,
    );

    // 3. PROCESS IN PARALLEL PER TREE
    instances
        .group_by(|(tree_id, ..)| *tree_id)
        .rich_map({
            // State maintained per partition (per tree)
            let all_subspaces = feature_subspaces;
            let mut tree: Option<HoeffdingTree> = None;
            let poisson = Poisson::new(exec_config.lambda).unwrap();

            move |(tree_id, (_orig_tree_id, instance_id, instance))| {
                let tree = tree.get_or_insert_with(|| {
                    let my_subspace = all_subspaces[*tree_id].clone();
                    HoeffdingTree::new(
                        my_subspace,
                        exec_config.n_min,
                        exec_config.delta,
                        exec_config.tau,
                        n_classes,
                        exec_config.max_bins,
                        exec_config.range_r,
                    )
                });

                // predict
                let predicted_class = tree.predict(&instance);

                if exec_config.drift_detection {
                    // let is_correct = predicted_class == instance.label;
                    //TODO drift detection logic
                }

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
                let (count, votes) = entry.get_or_insert((0, vec![0; n_classes]));

                *count += 1;

                if let Some(class) = class_prediction {
                    votes[class] += 1;
                }

                // check fragmentation target
                if *count == exec_config.n_trees {
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

    let total_time = global_start.elapsed().as_secs_f64();
    println!("Execution time: {} s", total_time);
    print!("Throughput: {} instances/s", 1e6 / total_time);

    if DRAW_ACCURACY_GRAPH {
        Command::new("py")
            .arg(format!("{}results/accuracy/accuracy_graph.py", base_path))
            .arg(format!(
                "{}results/accuracy/accuracy_{}{}.csv",
                base_path, locality, threads
            ))
            .arg(threads.to_string())
            .arg(format!("{:.2}", total_time))
            .arg(serde_json::to_string(&exec_config).unwrap())
            .status()
            .expect("Failed to execute Python script");
    }
}
