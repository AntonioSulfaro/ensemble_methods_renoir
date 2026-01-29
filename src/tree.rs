use crate::data_structures::{FeatureMapper, Instance, LocalStats};
use crate::srp::FeatureSubspace;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::{MAX_BINS, RANGE_R};

pub type NodeId = usize;

/// Split test structure
/// feature_id: index of the feature that split
/// threshold: threshold for the split
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitTest {
    pub feature_id: usize,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    Internal {
        test: SplitTest,
        left: NodeId,
        right: NodeId,
    },
    Leaf {
        total_samples: usize,
        class_counts: HashMap<usize, usize>,
        feature_stats: Vec<LocalStats>, // Histograms for each feature
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub kind: NodeKind,
}

/// Very Fast Decision Tree (VFDT) structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VFDT {
    pub nodes: Vec<Node>,
    pub feature_subspace: FeatureSubspace,
    pub next_id: NodeId,
    pub n_min: usize,
    pub delta: f64,
    pub tau: f64,
}

impl VFDT {
    pub fn new(feature_subspace: FeatureSubspace, n_min: usize, delta: f64, tau: f64) -> Self {
        let mut nodes = Vec::new();
        nodes.insert(0, Node {
            kind: NodeKind::Leaf {
                total_samples: 0,
                class_counts: HashMap::new(),
                feature_stats: (0..feature_subspace.len()).map(|_| LocalStats::new(MAX_BINS))
                    .collect(),
            }
        });
        VFDT { nodes, feature_subspace, next_id: 1, n_min, delta, tau}
    }

    /// Route an instance through the tree to find the leaf node
    pub fn route(&self, inst: &Instance) -> NodeId {
        let mut curr: usize = 0;
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

    /// Predict the class label for an instance
    /// Returns the majority class at the leaf node reached
    pub fn predict(&self, inst: &Instance) -> Option<usize> {
        let leaf_id = self.route(inst);

        if let Some(node) = self.nodes.get(leaf_id) {
            if let NodeKind::Leaf { class_counts, .. } = &node.kind {
                // Return the majority class
                class_counts.iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(class, _)| *class)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Train the tree with a labeled instance
    /// Updates statistics at the leaf node and evaluates splits
    pub fn train(&mut self, inst: Instance, k: usize) {
        let label = inst.label.expect("Training requires a label");
        let leaf_id = self.route(&inst);

        let (ready_to_evaluate, samples_at_leaf) = {
            let node = self.nodes.get_mut(leaf_id).expect("Leaf must exist");
            if let NodeKind::Leaf { total_samples, class_counts, feature_stats } = &mut node.kind {
                // weight the instance k times (bagging)
                *total_samples += k;
                *class_counts.entry(label).or_insert(0) += k;

                for (local_f, stats) in feature_stats.iter_mut().enumerate() {
                    let global_f = self.feature_subspace[local_f];
                    let val = inst.features[global_f];
                    stats.update(val, label, k);
                }

                // Return true if we hit the N_MIN threshold
                (*total_samples >= self.n_min && *total_samples % self.n_min == 0, *total_samples)
            } else {
                (false, 0)
            }
        };

        if ready_to_evaluate {
            let feature_stats = if let NodeKind::Leaf { feature_stats, .. } = &self.nodes[leaf_id].kind {
                feature_stats
            } else {
                return;
            };

            if let Some((fid, threshold)) = self.evaluate_split(feature_stats, samples_at_leaf) {
                self.apply_split(leaf_id, fid, threshold);
            }
        }
    }

    /// Evaluate the best split for the given statistics at a leaf node
    /// Returns Some((feature_id, threshold)) if a split is decided, else None
    fn evaluate_split(&self, stats: &[LocalStats], n: usize) -> Option<(usize, f64)> {
        let mut best_fid = 0;
        let mut best_score = f64::INFINITY; // We want to minimize Gini
        let mut best_threshold = 0.0;

        let mut second_best_score = f64::INFINITY;

        for (fid, f_stat) in stats.iter().enumerate() {
            // Find the best threshold for this specific feature
            if let Some((score, threshold)) = self.calculate_best_gini_for_feature(f_stat) {
                if score < best_score {
                    second_best_score = best_score;
                    best_score = score;
                    best_fid = fid;
                    best_threshold = threshold;
                } else if score < second_best_score {
                    second_best_score = score;
                }
            }
        }

        // Hoeffding Bound Calculation
        let epsilon = ((RANGE_R * RANGE_R * (1.0 / self.delta).ln()) / (2.0 * n as f64)).sqrt();

        // Split if the difference is greater than the bound, or if the bound is tiny (tie)
        if (second_best_score - best_score) > epsilon || epsilon < self.tau {
            Some((best_fid, best_threshold))
        } else {
            None
        }
    }

    /// Calculate the best Gini impurity and threshold for a given feature's statistics
    fn calculate_best_gini_for_feature(&self, f_stat: &LocalStats) -> Option<(f64, f64)> {
        let bins = &f_stat.histogram.bins;
        if bins.len() < 2 { return None; }

        let mut best_score = f64::INFINITY;
        let mut best_threshold = 0.0;

        // Try splitting between every adjacent pair of bins
        for i in 0..bins.len() - 1 {
            let threshold = (bins[i].mean + bins[i+1].mean) / 2.0;

            // Calculate Gini for this specific split
            let mut left_counts = HashMap::new();
            let mut right_counts = HashMap::new();
            let mut n_left = 0;
            let mut n_right = 0;

            for (j, bin) in bins.iter().enumerate() {
                let target = if j <= i { &mut left_counts } else { &mut right_counts };
                let target_n = if j <= i { &mut n_left } else { &mut n_right };

                for (&label, &count) in &bin.by_label {
                    *target.entry(label).or_insert(0) += count as usize;
                    *target_n += count as usize;
                }
            }

            let gini = self.compute_split_gini(left_counts, n_left, right_counts, n_right);
            if gini < best_score {
                best_score = gini;
                best_threshold = threshold;
            }
        }

        Some((best_score, best_threshold))
    }

    /// Compute the Gini impurity for a proposed split
    fn compute_split_gini(&self, left: HashMap<usize, usize>, n_l: usize, right: HashMap<usize, usize>, n_r: usize) -> f64 {
        let total = (n_l + n_r) as f64;
        let gini_l = 1.0 - left.values().map(|&c| (c as f64 / n_l as f64).powi(2)).sum::<f64>();
        let gini_r = 1.0 - right.values().map(|&c| (c as f64 / n_r as f64).powi(2)).sum::<f64>();

        (n_l as f64 / total) * gini_l + (n_r as f64 / total) * gini_r
    }

    /// Apply the split to the tree, converting the leaf node into an internal node
    fn apply_split(&mut self, leaf_id: NodeId, fid: usize, threshold: f64) {
        let left_id = self.nodes.len();
        let right_id = self.nodes.len() + 1;

        // 1. Create the new children leaves
        for _ in 0..2 {
            self.nodes.push(Node {
                kind: NodeKind::Leaf {
                    total_samples: 0,
                    class_counts: HashMap::new(),
                    feature_stats: (0..self.feature_subspace.len())
                        .map(|_| LocalStats::new(MAX_BINS))
                        .collect(),
                }
            });
        }

        // 2. Transform the current leaf into an Internal node
        // We use get_mut because we know leaf_id is valid
        if let Some(node) = self.nodes.get_mut(leaf_id) {
            node.kind = NodeKind::Internal {
                test: SplitTest { feature_id: fid, threshold },
                left: left_id,
                right: right_id,
            };
        }
    }
}