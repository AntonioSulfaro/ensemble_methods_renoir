use crate::learners::forest_utils::{random_subspace, FeatureSubspace};
use crate::learners::NumericEstimatorType;
use crate::tree::tree_utils::{evaluate_split, make_stats};
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
                class_counts: vec![0; n_classes],
                weight_seen_at_last_split: 0,
                feature_stats: make_stats(
                    feature_subspace.len(),
                    n_classes,
                    max_bins,
                    estimator_type,
                ),
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
        let pred = if let NodeKind::Leaf { class_counts, .. } = &self.nodes[leaf_id].kind {
            class_counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, c)| c)
                .map(|(id, _)| id)
        } else {
            None
        };
        (pred, depth)
    }

    pub fn train(&mut self, inst: &Instance, k: usize) {
        self.total_instances_seen += k;
        let label = inst.label.expect("Training requires a label");
        let leaf_id = self.route(inst);

        let (ready, samples_at_leaf) = {
            let node = self.nodes.get_mut(leaf_id).unwrap();
            if let NodeKind::Leaf {
                total_samples,
                class_counts,
                feature_stats,
                weight_seen_at_last_split,
            } = &mut node.kind
            {
                *total_samples += k;
                class_counts[label] += k;
                for (local_f, stats) in feature_stats.iter_mut().enumerate() {
                    let val = inst.features[self.feature_subspace[local_f]];
                    stats.update(val, label, k, self.n_classes);
                }
                let ready = *total_samples - *weight_seen_at_last_split >= self.n_min;
                if ready {
                    *weight_seen_at_last_split = *total_samples;
                }
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
                if let Some((fid, threshold)) = evaluate_split(
                    feature_stats,
                    class_counts,
                    samples_at_leaf,
                    self.delta,
                    self.tau,
                    self.n_classes,
                ) {
                    self.apply_split(leaf_id, fid, threshold);
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
                class_counts: vec![0; self.n_classes],
                weight_seen_at_last_split: 0,
                feature_stats: make_stats(
                    self.feature_subspace.len(),
                    self.n_classes,
                    self.max_bins,
                    self.estimator_type,
                ),
            },
        });
    }

    fn apply_split(&mut self, leaf_id: NodeId, fid: usize, threshold: f64) {
        let left_id = self.nodes.len();
        let right_id = self.nodes.len() + 1;
        let subspace_len = self.feature_subspace.len();

        for _ in 0..2 {
            self.nodes.push(Node {
                kind: NodeKind::Leaf {
                    total_samples: 0,
                    class_counts: vec![0; self.n_classes],
                    weight_seen_at_last_split: 0,
                    feature_stats: make_stats(
                        subspace_len,
                        self.n_classes,
                        self.max_bins,
                        self.estimator_type,
                    ),
                },
            });
        }

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
