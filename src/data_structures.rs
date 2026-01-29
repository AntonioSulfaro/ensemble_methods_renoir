use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::N_CLASSES;
use crate::srp::FeatureSubspace;

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
        Bin { mean: value, total: k, by_label }
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
        Histogram { bins: Vec::new(), max_bins }
    }

    pub fn update(&mut self, value: f64, class: usize, k: usize) {
        // 1. Try to find an existing bin with the same mean
        if let Some(bin) = self.bins.iter_mut().find(|b| (b.mean - value).abs() < 1e-9) {
            bin.add(value, class, k);
        } else {
            // 2. Or create a new bin with weight k
            self.bins.push(Bin::new(value, class, k, N_CLASSES));
            self.bins.sort_by(|a, b| a.mean.partial_cmp(&b.mean).expect("NaN in histogram"));
        }

        // 3. Maintenance
        if self.bins.len() > self.max_bins {
            self.merge_closest();
        }
    }

    fn merge_closest(&mut self) {
        if self.bins.len() < 2 { return; }

        let mut best_i = 0;
        let mut min_dist = f64::INFINITY;

        // Find the pair with the smallest difference in means
        for i in 0..self.bins.len() - 1 {
            let dist = self.bins[i+1].mean - self.bins[i].mean;
            if dist < min_dist {
                min_dist = dist;
                best_i = i;
            }
        }

        // Remove the two bins to be merged
        let b1 = self.bins.remove(best_i);
        let b2 = self.bins.remove(best_i);

        let total = b1.total + b2.total;
        // Weighted average: (μ1*n1 + μ2*n2) / (n1 + n2)
        let mean = (b1.mean * b1.total as f64 + b2.mean * b2.total as f64) / total as f64;

        let mut by_label = b1.by_label;
        for (i, count) in b2.by_label.into_iter().enumerate() {
            by_label[i] += count;
        }

        // Insert the new merged bin back at the same position
        // Since it's a weighted mean of two sorted means, it will still be in order
        self.bins.insert(best_i, Bin { mean, total, by_label });
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
        Self { total: 0, histogram: Histogram::new(max_bins) }
    }
    pub fn update(&mut self, value: f64, class: usize, k: usize) {
        self.total += k;
        self.histogram.update(value, class, k);
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
