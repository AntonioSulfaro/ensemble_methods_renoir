mod adaptive_tree;
mod adwin;
mod data_structures;
mod exec_config;
mod forest_utils;
mod srp;
mod tree;

mod eval {
    pub mod data_reader;
    pub mod evaluation;
}

use crate::adaptive_tree::AdaptiveLearner;
use crate::eval::data_reader::read_arff;
use crate::eval::evaluation::ExperimentResult;
use crate::exec_config::ExecConfig;
use chrono::Local;
use rand_distr::{Distribution, Poisson};
use renoir::{Replication, RuntimeConfig, StreamContext};
use std::fs::OpenOptions;
use std::ops::ControlFlow;
use std::process::Command;
use std::time::Instant;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let (renoir_config, _args) = RuntimeConfig::from_args();
    let config_str =
        std::fs::read_to_string("exec_config.json").expect("Failed to read json configurations");
    let exec_config: ExecConfig =
        serde_json::from_str(&config_str).expect("JSON was not well-formatted");

    // Determine output file name based on runtime configuration
    let (locality, threads) = match &renoir_config {
        RuntimeConfig::Local(local_cfg) => ("l", local_cfg.parallelism),
        RuntimeConfig::Remote(_) => (
            "r",
            std::thread::available_parallelism()
                .map(|q| q.get() as u64)
                .unwrap_or(1),
        ),
    };

    let timestamp = Local::now().format("%m%d_%H%M%S").to_string();
    let run_id = format!(
        "{}_{}_{}{}",
        timestamp,
        exec_config.dataset.replace("_", ""),
        locality,
        threads
    );
    let run_dir = format!("src/eval/results/runs/{}/", run_id);
    let accuracy_csv_path = format!("{}accuracy.csv", run_dir);

    // Create the directory
    std::fs::create_dir_all(&run_dir).unwrap();

    std::fs::write(format!("{}config.json", run_dir), &config_str).unwrap();

    // Start renoir environment
    let env = StreamContext::new(renoir_config);

    // 1. CREATE DATA STREAM
    let (data, n_classes, n_features) =
        read_arff(format!("datasets/{}.arff", exec_config.dataset).as_str());
    let n_instances = data.len();
    println!("Starting the computation");

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
            let mut learner: Option<AdaptiveLearner> = None;
            let poisson = Poisson::new(exec_config.lambda).unwrap();

            move |(tree_id, (_orig_tree_id, instance_id, instance))| {
                let learner = learner.get_or_insert_with(|| {
                    let my_subspace = all_subspaces[*tree_id].clone();
                    AdaptiveLearner::new(
                        my_subspace,
                        exec_config.n_min,
                        exec_config.delta,
                        exec_config.tau,
                        n_classes,
                        exec_config.max_bins,
                        exec_config.range_r,
                        exec_config.adwin_delta_warning,
                        exec_config.adwin_delta_drift,
                    )
                });

                // predict
                let predicted_class = learner.predict(&instance);

                // train
                let mut rng = rand::rng();
                let k = poisson.sample(&mut rng) as usize;
                let mut drift_detected = false;
                if k > 0 {
                    if exec_config.drift_detection {
                        let is_correct = predicted_class == instance.label;
                        drift_detected = learner.train_adaptive(&instance, k, is_correct);
                    } else {
                        learner.tree.train(&instance, k);
                    }
                }

                (instance_id, predicted_class, instance.label, drift_detected)
            }
        })
        .drop_key()
        .group_by(|(instance_id, ..)| *instance_id)
        .rich_map_transient({
            let mut entry = None;

            move |(inst_id, (_key, class_prediction, actual_label, drift_detected))| {
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

                    ControlFlow::Break(Some((*inst_id, winner, actual_label, drift_detected)))
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

            move |(inst_id, winner, actual, drift_detected)| {
                total_processed += 1;
                if winner == actual {
                    total_correct += 1;
                }

                ExperimentResult {
                    instance_id: inst_id,
                    actual_class: actual,
                    predicted_class: winner,
                    global_accuracy: total_correct as f64 / total_processed as f64,
                    drift_detected,
                }
            }
        })
        .write_csv(|_| accuracy_csv_path.into(), false);

    env.execute_blocking();

    let total_time = global_start.elapsed().as_secs_f64();
    println!("Finishing the computation");

    // draw accuracy graph
    Command::new(if cfg!(windows) { "py" } else { "python3" })
        .arg("scripts/accuracy_graph.py")
        .arg(threads.to_string())
        .arg(format!("{:.2}", total_time))
        .arg(&run_dir)
        .status()
        .expect("Failed to execute Python script");

    let mut scalability_f = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open("src/eval/results/scalability/master_log.csv")
        .unwrap();
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(&mut scalability_f);

    wtr.serialize((
        &run_id,
        threads,
        format!("{:.2}", total_time),
        n_instances,
        &exec_config,
    ))
    .unwrap();
    wtr.flush().unwrap();
}
