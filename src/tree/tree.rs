use crate::data::structures::{GaussianFeatureStats, Histogram, LocalStats};
use crate::learners::{srp, FeatureSubspace, NumericEstimatorType};
use crate::tree::{Node, NodeId, NodeKind, SplitTest};
use crate::Instance;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoeffdingTree {
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

fn entropy(counts: &[u64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    counts.iter().fold(0.0, |acc, &c| {
        if c == 0 {
            acc
        } else {
            let p = c as f64 / n_f;
            acc - p * p.log2()
        }
    })
}

fn entropy_f(weights: &[f64], total: f64) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    weights.iter().fold(0.0, |acc, &w| {
        if w <= 0.0 {
            acc
        } else {
            let p = w / total;
            acc - p * p.log2()
        }
    })
}

impl HoeffdingTree {
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
                feature_stats: Self::make_stats(
                    feature_subspace.len(),
                    n_classes,
                    max_bins,
                    estimator_type,
                ),
            },
        }];
        HoeffdingTree {
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

    fn make_stats(
        n_features: usize,
        n_classes: usize,
        max_bins: usize,
        est: NumericEstimatorType,
    ) -> Vec<LocalStats> {
        (0..n_features)
            .map(|_| match est {
                NumericEstimatorType::Histogram => LocalStats::new_histogram(max_bins),
                NumericEstimatorType::Gaussian => LocalStats::new_gaussian(n_classes),
            })
            .collect()
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

    pub fn predict(&self, inst: &Instance) -> Option<usize> {
        let leaf_id = self.route(inst);
        if let NodeKind::Leaf { class_counts, .. } = &self.nodes[leaf_id].kind {
            class_counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, c)| c)
                .map(|(id, _)| id)
        } else {
            None
        }
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
                if let Some((fid, threshold)) =
                    self.evaluate_split(feature_stats, class_counts, samples_at_leaf)
                {
                    self.apply_split(leaf_id, fid, threshold);
                }
            }
        }
    }

    pub fn reset_tree(&mut self, n_features: usize) {
        self.total_instances_seen = 0;
        self.feature_subspace = srp::random_subspace(n_features, self.feature_subspace.len());
        self.nodes.clear();
        self.nodes.push(Node {
            kind: NodeKind::Leaf {
                total_samples: 0,
                class_counts: vec![0; self.n_classes],
                weight_seen_at_last_split: 0,
                feature_stats: Self::make_stats(
                    self.feature_subspace.len(),
                    self.n_classes,
                    self.max_bins,
                    self.estimator_type,
                ),
            },
        });
    }

    fn evaluate_split(
        &self,
        stats: &[LocalStats],
        class_counts: &[usize],
        n: usize,
    ) -> Option<(usize, f64)> {
        let total_counts: Vec<u64> = class_counts.iter().map(|&c| c as u64).collect();
        let parent_entropy = entropy(&total_counts, n);

        // Pure node — no split can help
        if parent_entropy <= 0.0 {
            return None;
        }

        let mut best_fid = 0;
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_threshold = 0.0;
        let mut second_best_gain = f64::NEG_INFINITY;

        for (fid, f_stat) in stats.iter().enumerate() {
            if let Some((gain, threshold)) =
                self.best_gain_for_feature(f_stat, &total_counts, n, parent_entropy)
            {
                if gain > best_gain {
                    second_best_gain = best_gain; // old best becomes second
                    best_gain = gain;
                    best_fid = fid;
                    best_threshold = threshold;
                } else if gain > second_best_gain {
                    second_best_gain = gain;
                }
            }
        }

        // No feature produced any gain
        if best_gain == f64::NEG_INFINITY {
            return None;
        }

        // Standard Hoeffding bound: sqrt(ln(1/delta) / 2n)
        let epsilon = ((1.0 / self.delta).ln() / (2.0 * n as f64)).sqrt();

        // If second_best_gain is still NEG_INFINITY (only one feature had gain),
        // the difference is +INF, which always exceeds epsilon → correct, always split.
        let gain_diff = best_gain - second_best_gain; // NEG_INFINITY subtraction → +INF

        if gain_diff > epsilon || epsilon < self.tau {
            Some((best_fid, best_threshold))
        } else {
            None
        }
    }

    fn best_gain_for_feature(
        &self,
        stat: &LocalStats,
        total_counts: &[u64],
        n_total: usize,
        parent_entropy: f64,
    ) -> Option<(f64, f64)> {
        match stat {
            LocalStats::Histogram { stats } => {
                self.best_gain_histogram(stats, total_counts, n_total, parent_entropy)
            }
            LocalStats::Gaussian { stats } => {
                self.best_gain_gaussian(stats, n_total as f64, parent_entropy)
            }
        }
    }

    // ── Histogram path ──────────────────────────────────────

    fn best_gain_histogram(
        &self,
        hist: &Histogram,
        total_counts: &[u64],
        n_total: usize,
        parent_entropy: f64,
    ) -> Option<(f64, f64)> {
        let bins = &hist.bins;
        if bins.len() < 2 {
            return None;
        }

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_threshold = 0.0;
        let mut left_counts = vec![0u64; self.n_classes];
        let mut n_left = 0usize;

        for i in 0..bins.len() - 1 {
            let bin = &bins[i];
            n_left += bin.total;
            for (c, &cnt) in bin.by_label.iter().enumerate() {
                if c < self.n_classes {
                    left_counts[c] += cnt;
                }
            }
            let n_right = n_total.saturating_sub(n_left);
            if n_left == 0 || n_right == 0 {
                continue;
            }

            let right_counts: Vec<u64> = total_counts
                .iter()
                .zip(left_counts.iter())
                .map(|(&t, &l)| t.saturating_sub(l))
                .collect();

            let gain = parent_entropy
                - (n_left as f64 / n_total as f64) * entropy(&left_counts, n_left)
                - (n_right as f64 / n_total as f64) * entropy(&right_counts, n_right);

            if gain > best_gain {
                best_gain = gain;
                best_threshold = (bins[i].mean + bins[i + 1].mean) / 2.0;
            }
        }

        if best_gain == f64::NEG_INFINITY {
            None
        } else {
            Some((best_gain, best_threshold))
        }
    }

    // Gaussian path ─────────────────────────────────────────────

    fn best_gain_gaussian(
        &self,
        gstats: &GaussianFeatureStats,
        n_total: f64,
        parent_entropy: f64,
    ) -> Option<(f64, f64)> {
        let split_points = gstats.split_points(10); // MOA default: 10 bins
        if split_points.is_empty() {
            return None;
        }

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_threshold = 0.0;

        for threshold in split_points {
            let (left, right) = gstats.class_dists_at_split(threshold);
            let n_left: f64 = left.iter().sum();
            let n_right: f64 = right.iter().sum();
            if n_left < 1.0 || n_right < 1.0 {
                continue;
            }

            let gain = parent_entropy
                - (n_left / n_total) * entropy_f(&left, n_left)
                - (n_right / n_total) * entropy_f(&right, n_right);

            if gain > best_gain {
                best_gain = gain;
                best_threshold = threshold;
            }
        }

        if best_gain == f64::NEG_INFINITY {
            None
        } else {
            Some((best_gain, best_threshold))
        }
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
                    feature_stats: Self::make_stats(
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

    pub fn n_splits(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Internal { .. }))
            .count()
    }
}
