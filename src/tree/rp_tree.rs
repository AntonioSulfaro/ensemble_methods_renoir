use crate::learners::forest_utils::{random_subspace, FeatureSubspace};
use crate::learners::NumericEstimatorType;
use crate::tree::tree_utils::{argmax_f64, evaluate_split, make_stats, naive_bayes_votes};
use crate::tree::{Node, NodeId, NodeKind, SplitTest};
use crate::Instance;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomPatchesTree {
    pub nodes: Vec<Node>,
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
        let nodes = vec![Node {
            kind: NodeKind::Leaf {
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
                NodeKind::Leaf { .. } => return curr,
                NodeKind::Internal { test, left, right } => {
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
                NodeKind::Leaf { .. } => return (curr, depth),
                NodeKind::Internal { test, left, right } => {
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
    pub fn predict(&self, inst: &Instance) -> (Option<usize>, usize) {
        let (leaf_id, depth) = self.route_with_depth(inst);

        let pred = if let NodeKind::Leaf {
            class_counts,
            feature_stats,
            mc_correct_weight,
            nb_correct_weight,
            ..
        } = &self.nodes[leaf_id].kind
        {
            if mc_correct_weight > nb_correct_weight {
                // Majority-class prediction
                class_counts
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, c)| c)
                    .map(|(id, _)| id)
            } else {
                // Naive Bayes prediction
                let local_vals: Vec<f64> = (0..self.feature_subspace.len())
                    .map(|lf| inst.features[self.feature_subspace[lf]])
                    .collect();
                let votes =
                    naive_bayes_votes(feature_stats, class_counts, &local_vals, self.n_classes);
                argmax_f64(&votes)
            }
        } else {
            None
        };
        (pred, depth)
    }

    pub fn train(&mut self, inst: &Instance, k: usize) {
        self.total_instances_seen += k;
        let label = inst.label.expect("Training requires a label");
        let leaf_id = self.route(inst);

        // ── NBAdaptive: score both predictors before updating stats ──────────
        let (mc_correct, nb_correct) = {
            if let NodeKind::Leaf {
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
            if let NodeKind::Leaf {
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
            if let NodeKind::Leaf {
                feature_stats,
                class_counts,
                ..
            } = &self.nodes[leaf_id].kind
            {
                if let Some((fid, threshold, left_dist, right_dist)) = evaluate_split(
                    feature_stats,
                    class_counts,
                    samples_at_leaf,
                    self.delta,
                    self.tau,
                    self.n_classes,
                ) {
                    self.apply_split(leaf_id, fid, threshold, left_dist, right_dist);
                } else {
                    // No split: reset now so we wait another n_min before retrying
                    if let NodeKind::Leaf {
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

    pub fn reset_tree(&mut self, n_features: usize) {
        self.total_instances_seen = 0;
        self.feature_subspace = random_subspace(n_features, self.feature_subspace.len());
        self.nodes.clear();
        self.nodes.push(Node {
            kind: NodeKind::Leaf {
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

    fn apply_split(
        &mut self,
        leaf_id: NodeId,
        fid: usize,
        threshold: f64,
        left_dist: Vec<u32>,
        right_dist: Vec<u32>,
    ) {
        let left_id = self.nodes.len();
        let right_id = self.nodes.len() + 1;
        let subspace_len = self.feature_subspace.len();

        // Compute initial total_samples from the distributions
        let left_total: usize = left_dist.iter().map(|&c| c as usize).sum();
        let right_total: usize = right_dist.iter().map(|&c| c as usize).sum();

        self.nodes.push(Node {
            kind: NodeKind::Leaf {
                total_samples: left_total,
                class_counts: left_dist.into_boxed_slice(),
                weight_seen_at_last_split: left_total,
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
        self.nodes.push(Node {
            kind: NodeKind::Leaf {
                total_samples: right_total,
                class_counts: right_dist.into_boxed_slice(),
                weight_seen_at_last_split: right_total,
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

        self.nodes[leaf_id].kind = NodeKind::Internal {
            test: SplitTest {
                feature_id: fid,
                threshold,
            },
            left: left_id,
            right: right_id,
        };
    }
}
