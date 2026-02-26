use crate::run::RunContext;
use anyhow::Context;
use rand_distr::{Distribution, Poisson};
use renoir::Replication;
use std::ops::ControlFlow;

/// Build the renoir pipeline, execute it blocking and return the (possibly-updated) RunContext and elapsed seconds.
pub fn process_stream(ctx: RunContext) -> anyhow::Result<(RunContext, f64)> {
    use crate::eval::evaluation::InstanceResult;
    use crate::learners::adaptive::AdaptiveLearner;
    use crate::learners::forest_utils;

    let RunContext {
        env,
        run_dir,
        accuracy_csv_path,
        run_id,
        threads,
        config,
        n_instances,
        n_classes,
        n_features,
        final_patch,
        feature_subspaces,
        data,
    } = ctx;

    let env = env.context("stream context was already taken or not provided in RunContext")?;

    let global_start = std::time::Instant::now();

    let feature_subspaces_for_closure = feature_subspaces.clone();
    let config_for_closure = config.clone();
    let accuracy_csv_path_clone = accuracy_csv_path.clone();

    // build the stream: replicate each instance to all trees
    let instances = env
        .stream_iter(data.into_iter())
        .flat_map(move |(instance_id, instance)| {
            (0..config_for_closure.n_trees)
                .map(move |tree_id| (tree_id, instance_id, instance.clone()))
        });

    instances
        .group_by(|(tree_id, ..)| *tree_id)
        .rich_map({
            // per-partition (per-tree) state
            let all_subspaces = feature_subspaces_for_closure.clone();
            let mut learner: Option<AdaptiveLearner> = None;
            let poisson = Poisson::new(config_for_closure.lambda)?;

            // capture config_for_closure by move as well (it is cloned above)
            move |(tree_id, (_orig_tree_id, instance_id, instance))| {
                let learner = learner.get_or_insert_with(|| {
                    let my_subspace = all_subspaces[*tree_id].clone();
                    AdaptiveLearner::new(
                        my_subspace,
                        config_for_closure.n_min,
                        config_for_closure.delta,
                        config_for_closure.tau,
                        n_classes,
                        config_for_closure.max_bins,
                        config_for_closure.range_r,
                        config_for_closure.adwin_delta_warning,
                        config_for_closure.adwin_delta_drift,
                    )
                });

                // predict
                let predicted_class = learner.predict(&instance);

                // train
                let mut rng = rand::rng();
                let k = poisson.sample(&mut rng) as usize;
                let mut drift_detected = false;
                if k > 0 {
                    if config_for_closure.drift_detection {
                        let is_correct = predicted_class == instance.label;
                        drift_detected = learner.train_adaptive(&instance, k, is_correct);
                    } else {
                        learner.tree.train(&instance, k);
                    }
                }

                (
                    instance_id,
                    predicted_class,
                    instance.label,
                    drift_detected,
                    learner.detector.warning.error_rate(),
                )
            }
        })
        .drop_key()
        .group_by(|(instance_id, ..)| *instance_id)
        .rich_map_transient({
            // This closure needs access to config_for_closure as well.
            let config_for_closure = config_for_closure.clone();
            let mut entry = None;

            move |(inst_id, (_key, class_prediction, actual_label, drift_detected, error_rate))| {
                let (count, votes) = entry.get_or_insert((0, vec![0.0; n_classes]));

                *count += 1;

                if let Some(class) = class_prediction {
                    let weight = match config_for_closure.voting {
                        crate::learners::forest_utils::VotingStrategy::Majority => 1.0,
                        crate::learners::forest_utils::VotingStrategy::Weighted => 1.0 - error_rate,
                    };
                    votes[class] += weight;
                }

                // check fragmentation target
                if *count == config_for_closure.n_trees {
                    let winner =
                        forest_utils::aggregate_vote(votes, *count, config_for_closure.n_trees);
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

                InstanceResult {
                    instance_id: inst_id,
                    actual_class: actual,
                    predicted_class: winner,
                    global_accuracy: total_correct as f64 / total_processed as f64,
                    drift_detected,
                }
            }
        })
        .write_csv(|_| accuracy_csv_path_clone.into(), false);

    env.execute_blocking();

    let total_time = global_start.elapsed().as_secs_f64();

    let returned_ctx = RunContext {
        env: None, // stream context consumed by execution
        run_dir,
        accuracy_csv_path,
        run_id,
        threads,
        config,
        n_instances,
        n_classes,
        n_features,
        final_patch,
        feature_subspaces,
        data: Vec::new(),
    };

    Ok((returned_ctx, total_time))
}
