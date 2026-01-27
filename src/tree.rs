use crate::data_structures::{Instance, LocalStats};
use std::collections::HashMap;

pub type NodeId = usize;

pub struct SplitTest {
    pub feature_id: usize,
    pub threshold: f64,
}

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

pub struct Node {
    pub kind: NodeKind,
}

pub struct VFDT {
    pub nodes: HashMap<NodeId, Node>,
    pub next_id: NodeId,
    pub n_min: usize,
    pub delta: f64,
    pub tau: f64,
}

impl VFDT {
    pub fn new(n_min: usize, delta: f64, tau: f64) -> Self {
        let mut nodes = HashMap::new();
        nodes.insert(0, Node {
            kind: NodeKind::Leaf {
                total_samples: 0,
                class_counts: HashMap::new(),
                feature_stats: Vec::new(),
            }
        });
        VFDT { nodes, next_id: 1, n_min, delta, tau }
    }

    pub fn route(&self, inst: &Instance) -> NodeId {
        let mut curr = 0;
        loop {
            match &self.nodes[&curr].kind {
                NodeKind::Leaf { .. } => return curr,
                NodeKind::Internal { test, left, right } => {
                    curr = if inst.features[test.feature_id] <= test.threshold { *left } else { *right };
                }
            }
        }
    }

    pub fn train(&mut self, inst: Instance) {
        let label = inst.label.expect("Training requires a label");
        let leaf_id = self.route(&inst);

        let (ready_to_evaluate, samples_at_leaf) = {
            let node = self.nodes.get_mut(&leaf_id).expect("Leaf must exist");
            if let NodeKind::Leaf { total_samples, class_counts, feature_stats } = &mut node.kind {
                *total_samples += 1;
                *class_counts.entry(label).or_insert(0) += 1;

                if feature_stats.is_empty() {
                    for _ in 0..inst.features.len() {
                        feature_stats.push(LocalStats::new(32));
                    }
                }

                for (i, &val) in inst.features.iter().enumerate() {
                    feature_stats[i].update(val, label);
                }

                // Return true if we hit the N_MIN threshold
                (*total_samples >= self.n_min && *total_samples % self.n_min == 0, *total_samples)
            } else {
                (false, 0)
            }
        };

        if ready_to_evaluate {
            let feature_stats = if let NodeKind::Leaf { feature_stats, .. } = &self.nodes[&leaf_id].kind {
                feature_stats
            } else {
                return;
            };

            if let Some((fid, threshold)) = self.evaluate_split(feature_stats, samples_at_leaf) {
                self.apply_split(leaf_id, fid, threshold);
            }
        }
    }

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
        // R is the range of the random variable (for Gini, max diff is 1.0)
        let r = 1.0;
        let epsilon = ((r * r * (1.0 / self.delta).ln()) / (2.0 * n as f64)).sqrt();

        // Split if the difference is greater than the bound, or if the bound is tiny (tie)
        if (second_best_score - best_score) > epsilon || epsilon < self.tau {
            Some((best_fid, best_threshold))
        } else {
            None
        }
    }

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

    fn compute_split_gini(&self, left: HashMap<usize, usize>, n_l: usize, right: HashMap<usize, usize>, n_r: usize) -> f64 {
        let total = (n_l + n_r) as f64;
        let gini_l = 1.0 - left.values().map(|&c| (c as f64 / n_l as f64).powi(2)).sum::<f64>();
        let gini_r = 1.0 - right.values().map(|&c| (c as f64 / n_r as f64).powi(2)).sum::<f64>();

        (n_l as f64 / total) * gini_l + (n_r as f64 / total) * gini_r
    }

    fn apply_split(&mut self, leaf_id: NodeId, fid: usize, threshold: f64) {
        let left = self.next_id;
        let right = self.next_id + 1;
        self.next_id += 2;

        for id in [left, right] {
            self.nodes.insert(id, Node {
                kind: NodeKind::Leaf {
                    total_samples: 0, class_counts: HashMap::new(), feature_stats: Vec::new()
                }
            });
        }

        if let Some(node) = self.nodes.get_mut(&leaf_id) {
            node.kind = NodeKind::Internal {
                test: SplitTest { feature_id: fid, threshold },
                left, right,
            };
        }
        println!("Split Leaf {} on Feature {} at {}", leaf_id, fid, threshold);
    }
}