use crate::eval::evaluation::InstanceResult;
use crate::learners::forest_utils;
use crate::learners::online_learner::OnlineLearner;
use crate::run::{ResultContext, RunContext};
use anyhow::Context;
use rand_distr::{Distribution, Poisson};
use renoir::Replication;
use std::ops::ControlFlow;

// TODO look at tree depth limit of MOA

/// Build the renoir pipeline, execute it blocking and return the (possibly-updated) RunContext and elapsed seconds.
pub fn process_stream(ctx: RunContext) -> anyhow::Result<(ResultContext, f64)> {
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
        patch_size,
        data,
    } = ctx;

    let env = env.context("stream context was already taken or not provided in RunContext")?;

    let global_start = std::time::Instant::now();

    let config_for_closure = config.clone();
    let accuracy_csv_path_clone = accuracy_csv_path.clone();

    // build the stream: replicate each instance to all trees
    let instances = env
        .stream_iter(data)
        .flat_map(move |(instance_id, instance)| {
            (0..config_for_closure.n_trees)
                .map(move |tree_id| (tree_id, instance_id, instance.clone()))
        });

    instances
        .group_by(|(tree_id, ..)| *tree_id)
        .rich_map({
            // per-partition (per-tree) state
            let mut learner: Option<OnlineLearner> = None;
            let poisson = Poisson::new(config_for_closure.lambda)?;

            move |(_tree_id, (_orig_tree_id, instance_id, instance))| {
                let learner = learner.get_or_insert_with(|| {
                    OnlineLearner::new(
                        config_for_closure.ensemble_type,
                        patch_size,
                        config_for_closure.n_min,
                        config_for_closure.delta,
                        config_for_closure.tau,
                        n_classes,
                        n_features,
                        config_for_closure.max_bins,
                        config_for_closure.adwin_delta_warning,
                        config_for_closure.adwin_delta_drift,
                        config_for_closure.numeric_estimator,
                        config_for_closure.drift_detection,
                    )
                });

                // predict
                let (predicted_class, depth) = learner.predict(&instance);

                // Get full vote vector (all class probabilities)
                let votes = learner.get_votes_for_instance(&instance);

                // train
                let mut rng = rand::rng();
                let k = poisson.sample(&mut rng) as usize;

                let is_correct = predicted_class == instance.label;
                let drift_detected = learner.train(&instance, k, is_correct);

                (
                    instance_id,
                    votes,
                    instance.label,
                    drift_detected,
                    learner.cumulative_accuracy(),
                    depth,
                )
            }
        })
        .drop_key()
        .group_by(|(instance_id, ..)| *instance_id)
        .rich_map_transient({
            let config_for_closure = config_for_closure.clone();
            let mut entry = None;

            move |(inst_id, (_key, tree_votes, actual_label, drift_detected, accuracy, depth))| {
                let (count, combined_votes, depth_sum) =
                    entry.get_or_insert((0, vec![0.0; n_classes].into_boxed_slice(), 0.0));

                *count += 1;
                *depth_sum += depth as f64;

                // Apply accuracy weight for weighted voting
                let mut weighted_votes = tree_votes.clone();

                if let forest_utils::VotingStrategy::Weighted = config_for_closure.voting {
                    // Apply accuracy weight to normalized votes
                    forest_utils::apply_accuracy_weight(&mut weighted_votes, accuracy);
                }

                // Accumulate weighted votes
                for (i, &vote) in weighted_votes.iter().enumerate() {
                    combined_votes[i] += vote;
                }

                // check fragmentation target
                if *count == config_for_closure.n_trees {
                    let winner = forest_utils::aggregate_vote(
                        &combined_votes,
                        *count,
                        config_for_closure.n_trees,
                    );
                    let avg_depth = *depth_sum / *count as f64;
                    ControlFlow::Break(Some((
                        *inst_id,
                        winner,
                        actual_label,
                        drift_detected,
                        avg_depth,
                    )))
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

            move |(inst_id, winner, actual, drift_detected, avg_depth)| {
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
                    avg_depth,
                }
            }
        })
        .write_csv(|_| accuracy_csv_path_clone.into(), false);

    env.execute_blocking();

    let total_time = global_start.elapsed().as_secs_f64();

    let returned_ctx = ResultContext {
        run_dir,
        run_id,
        threads,
        config,
        n_instances,
    };

    Ok((returned_ctx, total_time))
}
