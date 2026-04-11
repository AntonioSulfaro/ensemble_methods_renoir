use crate::learners::forest_utils::FeatureSubspace;
use crate::learners::NumericEstimatorType;
use crate::tree::tree_utils::{argmax_f64, evaluate_split, make_stats, naive_bayes_votes};
use crate::tree::{HoeffdingNode, HoeffdingNodeKind, NodeId, SplitTest};
use crate::Instance;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomPatchesTree {
    pub nodes: Vec<HoeffdingNode>,
    pub feature_subspace: Arc<FeatureSubspace>,
    pub n_min: usize,
    pub delta: f64,
    pub tau: f64,
    pub n_classes: usize,
    pub max_bins: usize,
    pub estimator_type: NumericEstimatorType,
    total_instances_seen: usize,
}

impl RandomPatchesTree {
    pub fn new(
        feature_subspace: Arc<FeatureSubspace>,
        n_min: usize,
        delta: f64,
        tau: f64,
        n_classes: usize,
        max_bins: usize,
        estimator_type: NumericEstimatorType,
    ) -> Self {
        let nodes = vec![HoeffdingNode {
            kind: HoeffdingNodeKind::Leaf {
                total_samples: 0,
                class_counts: vec![0; n_classes].into_boxed_slice(),
                weight_seen_at_last_split: 0,
                feature_stats: make_stats(
                    feature_subspace.len(),
                    n_classes,
                    max_bins,
                    estimator_type,
                ),
                mc_correct_weight: 0.0,
                nb_correct_weight: 0.0,
            },
        }];
        RandomPatchesTree {
            nodes,
            feature_subspace,
            n_min,
            delta,
            tau,
            n_classes,
            max_bins,
            estimator_type,
            total_instances_seen: 0,
        }
    }

    pub fn route(&self, inst: &Instance) -> NodeId {
        let mut curr = 0usize;
        loop {
            match &self.nodes[curr].kind {
                HoeffdingNodeKind::Leaf { .. } => return curr,
                HoeffdingNodeKind::Internal { test, left, right } => {
                    let global_f = self.feature_subspace[test.feature_id];
                    curr = if inst.features[global_f] <= test.threshold {
                        *left
                    } else {
                        *right
                    };
                }
            }
        }
    }

    /// Returns (leaf_id, depth) for the instance.
    /// Depth is the number of internal nodes traversed from root to leaf.
    pub fn route_with_depth(&self, inst: &Instance) -> (NodeId, usize) {
        let mut curr = 0usize;
        let mut depth = 0;
        loop {
            match &self.nodes[curr].kind {
                HoeffdingNodeKind::Leaf { .. } => return (curr, depth),
                HoeffdingNodeKind::Internal { test, left, right } => {
                    depth += 1;
                    let global_f = self.feature_subspace[test.feature_id];
                    curr = if inst.features[global_f] <= test.threshold {
                        *left
                    } else {
                        *right
                    };
                }
            }
        }
    }

    /// Returns (predicted class, depth) for the instance.
    /// Depth is the number of internal nodes traversed from root to leaf.
    /// Votes are the probability distribution over classes
    pub fn predict(&self, inst: &Instance) -> (Option<usize>, Vec<f64>, usize) {
        let (leaf_id, depth) = self.route_with_depth(inst);

        if let HoeffdingNodeKind::Leaf {
            class_counts,
            feature_stats,
            mc_correct_weight,
            nb_correct_weight,
            ..
        } = &self.nodes[leaf_id].kind
        {
            let votes = if mc_correct_weight > nb_correct_weight {
                // --- Majority-class Logic ---
                let total: f64 = class_counts.iter().map(|&c| c as f64).sum();
                if total > 0.0 {
                    class_counts.iter().map(|&c| c as f64 / total).collect()
                } else {
                    vec![1.0 / self.n_classes as f64; self.n_classes]
                }
            } else {
                // --- Naive Bayes Logic ---
                let local_vals: Vec<f64> = (0..self.feature_subspace.len())
                    .map(|lf| inst.features[self.feature_subspace[lf]])
                    .collect();
                let scores =
                    naive_bayes_votes(feature_stats, class_counts, &local_vals, self.n_classes);
                crate::tree::tree_utils::scores_to_votes(&scores)
            };

            // Deriving prediction from the votes (equivalent to argmax)
            let pred = argmax_f64(&votes);

            (pred, votes, depth)
        } else {
            // Fallback for non-leaf nodes (should theoretically not be reached if tree is valid)
            let uniform_votes = vec![1.0 / self.n_classes as f64; self.n_classes];
            (None, uniform_votes, depth)
        }
    }

    pub fn train(&mut self, inst: &Instance, k: usize) {
        self.total_instances_seen += k;
        let label = inst.label.expect("Training requires a label");
        let leaf_id = self.route(inst);

        // ── NBAdaptive: score both predictors before updating stats ──────────
        let (mc_correct, nb_correct) = {
            if let HoeffdingNodeKind::Leaf {
                class_counts,
                feature_stats,
                ..
            } = &self.nodes[leaf_id].kind
            {
                // MC prediction
                let mc_pred = class_counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, c)| c)
                    .map(|(id, _)| id);
                let mc_ok = mc_pred == Some(label);

                // NB prediction
                let local_vals: Vec<f64> = (0..self.feature_subspace.len())
                    .map(|lf| inst.features[self.feature_subspace[lf]])
                    .collect();
                let votes =
                    naive_bayes_votes(feature_stats, class_counts, &local_vals, self.n_classes);
                let nb_ok = argmax_f64(&votes) == Some(label);

                (mc_ok, nb_ok)
            } else {
                (false, false)
            }
        };

        let (ready, samples_at_leaf) = {
            let node = self.nodes.get_mut(leaf_id).unwrap();
            if let HoeffdingNodeKind::Leaf {
                total_samples,
                class_counts,
                feature_stats,
                weight_seen_at_last_split,
                mc_correct_weight,
                nb_correct_weight,
            } = &mut node.kind
            {
                // Update NBAdaptive counters
                if mc_correct {
                    *mc_correct_weight += k as f64;
                }
                if nb_correct {
                    *nb_correct_weight += k as f64;
                }

                *total_samples += k;
                class_counts[label] += k as u32;
                for (local_f, stats) in feature_stats.iter_mut().enumerate() {
                    let val = inst.features[self.feature_subspace[local_f]];
                    stats.update(val, label, k, self.n_classes);
                }
                let ready = *total_samples - *weight_seen_at_last_split >= self.n_min;

                (ready, *total_samples)
            } else {
                (false, 0)
            }
        };

        if ready {
            if let HoeffdingNodeKind::Leaf {
                feature_stats,
                class_counts,
                ..
            } = &self.nodes[leaf_id].kind
            {
                if let Some((fid, threshold, ..)) = evaluate_split(
                    feature_stats,
                    class_counts,
                    samples_at_leaf,
                    self.delta,
                    self.tau,
                    self.n_classes,
                ) {
                    self.apply_split(leaf_id, fid, threshold);
                } else {
                    // No split: reset now so we wait another n_min before retrying
                    if let HoeffdingNodeKind::Leaf {
                        weight_seen_at_last_split,
                        total_samples,
                        ..
                    } = &mut self.nodes[leaf_id].kind
                    {
                        *weight_seen_at_last_split = *total_samples;
                    }
                }
            }
        }
    }

    pub fn reset_tree(&mut self, new_subspace: Arc<FeatureSubspace>) {
        self.total_instances_seen = 0;
        self.feature_subspace = new_subspace;
        self.nodes.clear();
        self.nodes.push(HoeffdingNode {
            kind: HoeffdingNodeKind::Leaf {
                total_samples: 0,
                class_counts: vec![0; self.n_classes].into_boxed_slice(),
                weight_seen_at_last_split: 0,
                feature_stats: make_stats(
                    self.feature_subspace.len(),
                    self.n_classes,
                    self.max_bins,
                    self.estimator_type,
                ),
                mc_correct_weight: 0.0,
                nb_correct_weight: 0.0,
            },
        });
    }

    fn apply_split(&mut self, leaf_id: NodeId, fid: usize, threshold: f64) {
        let left_id = self.nodes.len();
        let right_id = self.nodes.len() + 1;
        let subspace_len = self.feature_subspace.len();

        for _ in 0..2 {
            self.nodes.push(HoeffdingNode {
                kind: HoeffdingNodeKind::Leaf {
                    total_samples: 0,
                    class_counts: vec![0; self.n_classes].into_boxed_slice(),
                    weight_seen_at_last_split: 0,
                    mc_correct_weight: 0.0,
                    nb_correct_weight: 0.0,
                    feature_stats: make_stats(
                        subspace_len,
                        self.n_classes,
                        self.max_bins,
                        self.estimator_type,
                    ),
                },
            });
        }

        self.nodes[leaf_id].kind = HoeffdingNodeKind::Internal {
            test: SplitTest {
                feature_id: fid,
                threshold,
            },
            left: left_id,
            right: right_id,
        };
    }
}
