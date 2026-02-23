use crate::srp::FeatureSubspace;
use serde::{Deserialize, Serialize};

/// Data instance structure
/// features: vector of feature values
/// label: optional class label (None for unlabeled instances)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub features: Vec<f64>,
    pub label: Option<usize>,
}

/// Bin structure for histogram
/// mean: mean value of the bin
/// total: total count of instances in the bin
/// by_label: count of instances per class label in the bin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bin {
    pub mean: f64,
    pub total: usize,
    pub by_label: Vec<u64>,
}

impl Bin {
    fn new(value: f64, class: usize, k: usize, n_classes: usize) -> Self {
        let mut by_label = vec![0; n_classes];
        by_label[class] = k as u64;
        Bin {
            mean: value,
            total: k,
            by_label,
        }
    }

    fn add(&mut self, value: f64, class: usize, k: usize) {
        let new_total = self.total + k;
        self.mean = (self.mean * self.total as f64 + value * k as f64) / new_total as f64;
        self.total = new_total;
        self.by_label[class] += k as u64;
    }
}

/// Histogram structure with bin merging
/// bins: vector of bins
/// max_bins: maximum number of bins allowed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    pub bins: Vec<Bin>,
    pub max_bins: usize,
}

impl Histogram {
    pub fn new(max_bins: usize) -> Self {
        Histogram {
            bins: Vec::with_capacity(max_bins + 1),
            max_bins,
        }
    }

    pub fn update(&mut self, value: f64, class: usize, k: usize, n_classes: usize) {
        let res = self
            .bins
            .binary_search_by(|b| b.mean.partial_cmp(&value).unwrap());

        let insert_idx = match res {
            Ok(idx) => {
                self.bins[idx].add(value, class, k);
                return; // exact match, no structural change, no merge needed
            }
            Err(idx) => {
                if idx < self.bins.len() && (self.bins[idx].mean - value).abs() < 1e-9 {
                    self.bins[idx].add(value, class, k);
                    return;
                } else if idx > 0 && (self.bins[idx - 1].mean - value).abs() < 1e-9 {
                    self.bins[idx - 1].add(value, class, k);
                    return;
                }
                self.bins.insert(idx, Bin::new(value, class, k, n_classes));
                idx
            }
        };

        if self.bins.len() > self.max_bins {
            // Only check pairs adjacent to the new bin rather than full scan.
            // Candidates: (insert_idx-1, insert_idx) and (insert_idx, insert_idx+1)
            let best_i = self.merge_candidate_near(insert_idx);
            self.merge_at(best_i, n_classes);
        }
    }

    /// Check the (up to 2) pairs adjacent to `idx` and return the index
    /// of the pair with the smallest gap. Falls back to full scan if needed
    /// (only happens at boundaries).
    fn merge_candidate_near(&self, idx: usize) -> usize {
        let len = self.bins.len();
        debug_assert!(len >= 2);

        // Collect candidate pair indices: left pair and right pair
        let mut best_i = 0;
        let mut min_dist = f64::INFINITY;

        // Only examine the (up to 2) pairs touching the new bin
        let start = idx.saturating_sub(1);
        let end = (idx + 1).min(len - 1);

        for i in start..end {
            let dist = self.bins[i + 1].mean - self.bins[i].mean;
            if dist < min_dist {
                min_dist = dist;
                best_i = i;
            }
        }
        best_i
    }

    fn merge_at(&mut self, i: usize, n_classes: usize) {
        let b2 = self.bins.remove(i + 1);
        let b1 = &mut self.bins[i];
        let total_new = b1.total + b2.total;
        b1.mean = (b1.mean * b1.total as f64 + b2.mean * b2.total as f64) / total_new as f64;
        b1.total = total_new;
        for c in 0..n_classes {
            b1.by_label[c] += b2.by_label[c];
        }
    }
}

/// Local statistics structure
/// total: total number of instances
/// histogram: histogram of feature values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalStats {
    pub total: usize,
    pub histogram: Histogram,
}

impl LocalStats {
    pub fn new(max_bins: usize) -> Self {
        Self {
            total: 0,
            histogram: Histogram::new(max_bins),
        }
    }
    pub fn update(&mut self, value: f64, class: usize, k: usize, n_classes: usize) {
        self.total += k;
        self.histogram.update(value, class, k, n_classes);
    }
}

pub trait FeatureMapper {
    fn len(&self) -> usize;
    fn global_index(&self, local_idx: usize) -> usize;
}

impl FeatureMapper for FeatureSubspace {
    fn len(&self) -> usize {
        self.len()
    }

    fn global_index(&self, local_idx: usize) -> usize {
        self[local_idx]
    }
}
