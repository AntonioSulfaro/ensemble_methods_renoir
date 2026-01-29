use std::collections::HashMap;
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
    pub by_label: HashMap<usize, u64>,
}

impl Bin {
    fn new(value: f64, class: usize) -> Self {
        let mut by_label = HashMap::new();
        by_label.insert(class, 1);
        Bin { mean: value, total: 1, by_label }
    }

    fn add(&mut self, value: f64, class: usize) {
        self.mean = (self.mean * self.total as f64 + value) / (self.total as f64 + 1.0);
        self.total += 1;
        *self.by_label.entry(class).or_insert(0) += 1;
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

    pub fn update(&mut self, value: f64, class: usize) {
        if let Some(bin) = self.bins.iter_mut().find(|b| (b.mean - value).abs() < 1e-9) {
            bin.add(value, class);
        } else {
            self.bins.push(Bin::new(value, class));
            self.bins.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
        }
        if self.bins.len() > self.max_bins {
            self.merge_closest();
        }
    }

    fn merge_closest(&mut self) {
        let mut best_i = 0;
        let mut min_dist = f64::INFINITY;
        for i in 0..self.bins.len() - 1 {
            let dist = self.bins[i+1].mean - self.bins[i].mean;
            if dist < min_dist {
                min_dist = dist;
                best_i = i;
            }
        }
        let b1 = self.bins.remove(best_i);
        let b2 = self.bins.remove(best_i);
        let total = b1.total + b2.total;
        let mean = (b1.mean * b1.total as f64 + b2.mean * b2.total as f64) / total as f64;
        let mut by_label = b1.by_label;
        for (c, count) in b2.by_label {
            *by_label.entry(c).or_insert(0) += count;
        }
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
    pub fn update(&mut self, value: f64, class: usize) {
        self.total += 1;
        self.histogram.update(value, class);
    }
}