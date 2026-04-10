use crate::learners::forest_utils::random_subspace;
use crate::learners::NumericEstimatorType;
use crate::tree::tree_utils::{argmax_f64, evaluate_split, make_stats, naive_bayes_votes};
use crate::tree::{NodeId, NodeWithPatch, NodeWithPatchKind, SplitTest};
use crate::Instance;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Adaptive Random Tree – each leaf has its own random feature subspace.
/// Internal nodes store a global feature id (obtained from the leaf’s subspace at split time).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRandomTree {
    pub nodes: Vec<NodeWithPatch>,
    pub n_global_features: usize, // total number of attributes in the dataset
    pub subspace_size: usize,     // number of features considered at each leaf
    pub n_min: usize,
    pub delta: f64,
    pub tau: f64,
    pub n_classes: usize,
    pub max_bins: usize,
    pub estimator_type: NumericEstimatorType,
    total_instances_seen: usize,
    rng_seed: u64,
    #[serde(skip)]
    rng: Option<SmallRng>,
}

impl AdaptiveRandomTree {
    fn rng(&mut self) -> &mut SmallRng {
        if self.rng.is_none() {
            self.rng = Some(SmallRng::seed_from_u64(self.rng_seed));
        }
        self.rng.as_mut().unwrap()
    }

    /// Creates a new tree with a root leaf that has a freshly drawn random subspace.
    pub fn new(
        n_global_features: usize,
        subspace_size: usize,
        n_min: usize,
        delta: f64,
        tau: f64,
        n_classes: usize,
        max_bins: usize,
        estimator_type: NumericEstimatorType,
        seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let nodes = vec![NodeWithPatch {
            kind: NodeWithPatchKind::Leaf {
                total_samples: 0,
                class_counts: vec![0; n_classes].into_boxed_slice(),
                weight_seen_at_last_split: 0,
                feature_stats: make_stats(subspace_size, n_classes, max_bins, estimator_type),
                feature_subspace: random_subspace(n_global_features, subspace_size, &mut rng),
                mc_correct_weight: 0.0,
                nb_correct_weight: 0.0,
            },
        }];
        AdaptiveRandomTree {
            nodes,
            n_global_features,
            subspace_size,
            n_min,
            delta,
            tau,
            n_classes,
            max_bins,
            estimator_type,
            total_instances_seen: 0,
            rng_seed: seed,
            rng: Some(rng),
        }
    }

    /// Routes an instance to a leaf.
    /// Internal nodes use the global feature id stored in their split test.
    pub fn route(&self, inst: &Instance) -> NodeId {
        let mut curr = 0usize;
        loop {
            match &self.nodes[curr].kind {
                NodeWithPatchKind::Leaf { .. } => return curr,
                NodeWithPatchKind::Internal { test, left, right } => {
                    curr = if inst.features[test.feature_id] <= test.threshold {
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
                NodeWithPatchKind::Leaf { .. } => return (curr, depth),
                NodeWithPatchKind::Internal { test, left, right } => {
                    depth += 1;
                    curr = if inst.features[test.feature_id] <= test.threshold {
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

        let subspace = match &self.nodes[leaf_id].kind {
            NodeWithPatchKind::Leaf {
                feature_subspace, ..
            } => feature_subspace.clone(),
            _ => unreachable!(),
        };

        if let NodeWithPatchKind::Leaf {
            class_counts,
            feature_stats,
            mc_correct_weight,
            nb_correct_weight,
            ..
        } = &self.nodes[leaf_id].kind
        {
            let votes = if mc_correct_weight > nb_correct_weight {
                // Majority-class: convert class counts to probabilities
                let total: f64 = class_counts.iter().map(|&c| c as f64).sum();
                if total > 0.0 {
                    class_counts.iter().map(|&c| c as f64 / total).collect()
                } else {
                    vec![1.0 / self.n_classes as f64; self.n_classes]
                }
            } else {
                // Naive Bayes: return normalized scores
                let local_vals: Vec<f64> = (0..self.subspace_size)
                    .map(|lf| inst.features[subspace[lf]])
                    .collect();
                let scores =
                    naive_bayes_votes(feature_stats, class_counts, &local_vals, self.n_classes);
                crate::tree::tree_utils::scores_to_votes(&scores)
            };

            let pred = argmax_f64(&votes);

            (pred, votes, depth)
        } else {
            let uniform_votes = vec![1.0 / self.n_classes as f64; self.n_classes];
            (None, uniform_votes, depth)
        }
    }

    /// Updates the tree with one (or `k` copies of) training instance.
    pub fn train(&mut self, inst: &Instance, k: usize) {
        self.total_instances_seen += k;
        let label = inst.label.expect("Training requires a label");
        let leaf_id = self.route(inst);

        let subspace = match &self.nodes[leaf_id].kind {
            NodeWithPatchKind::Leaf {
                feature_subspace, ..
            } => feature_subspace.clone(),
            _ => unreachable!(),
        };

        let (mc_correct, nb_correct) = {
            if let NodeWithPatchKind::Leaf {
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
                let local_vals: Vec<f64> = (0..self.subspace_size)
                    .map(|lf| inst.features[subspace[lf]])
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
            if let NodeWithPatchKind::Leaf {
                total_samples,
                class_counts,
                feature_stats,
                weight_seen_at_last_split,
                mc_correct_weight,
                nb_correct_weight,
                ..
            } = &mut node.kind
            {
                if mc_correct {
                    *mc_correct_weight += k as f64;
                }
                if nb_correct {
                    *nb_correct_weight += k as f64;
                }

                *total_samples += k;
                class_counts[label] += k as u32;

                // Update feature statistics using the leaf's subspace to obtain global feature values.
                for (local_f, stats) in feature_stats.iter_mut().enumerate() {
                    let global_f = subspace[local_f];
                    let val = inst.features[global_f];
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
            if let NodeWithPatchKind::Leaf {
                feature_stats,
                class_counts,
                feature_subspace,
                ..
            } = &self.nodes[leaf_id].kind
            {
                if let Some((local_fid, threshold, ..)) = evaluate_split(
                    feature_stats,
                    class_counts,
                    samples_at_leaf,
                    self.delta,
                    self.tau,
                    self.n_classes,
                ) {
                    // Convert local feature id to global using the leaf's subspace.
                    let global_fid = feature_subspace[local_fid];
                    self.apply_split(leaf_id, global_fid, threshold);
                }
            }
        }
    }

    /// Resets the tree to a single leaf with a fresh random subspace.
    pub fn reset_tree(&mut self) {
        self.total_instances_seen = 0;
        self.nodes.clear();
        let subspace = random_subspace(self.n_global_features, self.subspace_size, self.rng());
        self.nodes.push(NodeWithPatch {
            kind: NodeWithPatchKind::Leaf {
                total_samples: 0,
                class_counts: vec![0; self.n_classes].into_boxed_slice(),
                weight_seen_at_last_split: 0,
                feature_stats: make_stats(
                    self.subspace_size,
                    self.n_classes,
                    self.max_bins,
                    self.estimator_type,
                ),
                feature_subspace: subspace,
                mc_correct_weight: 0.0,
                nb_correct_weight: 0.0,
            },
        });
    }

    /// Converts a leaf into an internal node.
    /// The leaf's subspace is discarded; the internal node stores the split test with global feature id.
    /// Two new leaves are created, each with a fresh random subspace.
    fn apply_split(&mut self, leaf_id: NodeId, global_fid: usize, threshold: f64) {
        let left_id = self.nodes.len();
        let right_id = self.nodes.len() + 1;

        // Generate new random subspaces for the children.
        let left_subspace = random_subspace(self.n_global_features, self.subspace_size, self.rng());
        let right_subspace =
            random_subspace(self.n_global_features, self.subspace_size, self.rng());

        // Push the two new leaves.
        for subspace in [left_subspace, right_subspace] {
            self.nodes.push(NodeWithPatch {
                kind: NodeWithPatchKind::Leaf {
                    total_samples: 0,
                    class_counts: vec![0; self.n_classes].into_boxed_slice(),
                    weight_seen_at_last_split: 0,
                    feature_stats: make_stats(
                        self.subspace_size,
                        self.n_classes,
                        self.max_bins,
                        self.estimator_type,
                    ),
                    feature_subspace: subspace,
                    mc_correct_weight: 0.0,
                    nb_correct_weight: 0.0,
                },
            });
        }

        // Replace the original leaf with an internal node (its subspace is no longer needed).
        self.nodes[leaf_id].kind = NodeWithPatchKind::Internal {
            test: SplitTest {
                feature_id: global_fid,
                threshold,
            },
            left: left_id,
            right: right_id,
        };
    }
}
