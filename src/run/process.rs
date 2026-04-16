use crate::eval::evaluation::InstanceResult;
use crate::learners::online_learner::{create_learner, OnlineLearner, OnlineLearnerTrait};
use crate::learners::{forest_utils, VotingStrategy};
use crate::run::{ResultContext, RunContext};
use anyhow::Context;
use renoir::Replication;
use std::ops::ControlFlow;

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
        data,
    } = ctx;

    let env = env.context("stream context was already taken or not provided in RunContext")?;

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

            move |(tree_id, (_orig_tree_id, instance_id, instance))| {
                let learner = learner.get_or_insert_with(|| {
                    create_learner(
                        &config_for_closure.algorithm,
                        n_classes,
                        n_features,
                        *tree_id as u64,
                    )
                });

                // predict
                let (predicted_class, votes, depth) = learner.predict(&instance);

                // train
                let is_correct = predicted_class == instance.label;
                let drift_detected = learner.train(&instance, is_correct);

                (
                    instance_id,
                    votes,
                    instance.label,
                    drift_detected,
                    learner.cumulative_accuracy(),
                    predicted_class,
                    depth,
                )
            }
        })
        .drop_key()
        .group_by(|(instance_id, ..)| *instance_id)
        .rich_map_transient({
            let mut entry = None;

            move |(
                inst_id,
                (_key, tree_votes, actual_label, drift_detected, accuracy, predicted_class, depth),
            )| {
                let (count, combined_votes, depth_sum) =
                    entry.get_or_insert((0, vec![0.0; n_classes].into_boxed_slice(), 0.0));

                *count += 1;
                *depth_sum += depth as f64;

                match config_for_closure.voting {
                    VotingStrategy::Soft => {
                        for (i, &vote) in tree_votes.iter().enumerate() {
                            combined_votes[i] += vote;
                        }
                    }

                    VotingStrategy::Weighted => {
                        let mut weighted_votes = tree_votes.clone();
                        forest_utils::apply_accuracy_weight(&mut weighted_votes, accuracy);

                        for (i, &vote) in weighted_votes.iter().enumerate() {
                            combined_votes[i] += vote;
                        }
                    }

                    VotingStrategy::Majority => {
                        if let Some(p_class) = predicted_class {
                            combined_votes[p_class] += 1.0;
                        }
                    }
                }

                // check fragmentation target
                if *count == config_for_closure.n_trees {
                    let winner = forest_utils::aggregate_vote(&combined_votes);
                    let avg_depth = *depth_sum / *count as f64;
                    ControlFlow::Break(Some((
                        *inst_id,
                        winner == actual_label,
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
        .map({
            move |(inst_id, is_correct, drift_detected, avg_depth)| InstanceResult {
                instance_id: inst_id,
                is_correct,
                drift_detected,
                avg_depth,
            }
        })
        .write_csv(|_| accuracy_csv_path_clone.into(), false);

    let global_start = std::time::Instant::now();

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
