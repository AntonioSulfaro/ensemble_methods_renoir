use serde::{Deserialize, Serialize};

// ── Instance ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub features: Vec<f64>,
    pub label: Option<usize>,
}

// ── Histogram-based estimator ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bin {
    pub mean: f64,
    pub total: usize,
    pub by_label: Box<[u32]>,
}

impl Bin {
    fn new(value: f64, class: usize, k: usize, n_classes: usize) -> Self {
        let mut by_label = vec![0u32; n_classes].into_boxed_slice();
        by_label[class] = k as u32;
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
        self.by_label[class] += k as u32;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    pub bins: Vec<Bin>,
    pub max_bins: usize,
}

impl Histogram {
    pub fn new(max_bins: usize) -> Self {
        Histogram {
            bins: Vec::new(), //Vec::with_capacity(max_bins + 1)
            max_bins,
        }
    }

    pub fn update(&mut self, value: f64, class: usize, k: usize, n_classes: usize) {
        let res = self
            .bins
            .binary_search_by(|b| b.mean.partial_cmp(&value).unwrap());
        match res {
            Ok(idx) => {
                self.bins[idx].add(value, class, k);
            }
            Err(idx) => {
                if idx < self.bins.len() && (self.bins[idx].mean - value).abs() < 1e-9 {
                    self.bins[idx].add(value, class, k);
                } else if idx > 0 && (self.bins[idx - 1].mean - value).abs() < 1e-9 {
                    self.bins[idx - 1].add(value, class, k);
                } else {
                    self.bins.insert(idx, Bin::new(value, class, k, n_classes));
                    if self.bins.len() > self.max_bins {
                        let best_i = self.find_closest_pair();
                        self.merge_at(best_i, n_classes);
                    }
                }
            }
        }
    }

    fn find_closest_pair(&self) -> usize {
        let len = self.bins.len();
        let mut best_i = 0;
        let mut min_dist = f64::INFINITY;
        for i in 0..len - 1 {
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

// ── Gaussian estimator ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GaussianEstimator {
    pub weight: f64,
    pub mean: f64,
    variance_sum: f64, // Welford's M2
}

impl GaussianEstimator {
    pub fn update(&mut self, value: f64, weight: f64) {
        self.weight += weight;
        let delta = value - self.mean;
        self.mean += delta * weight / self.weight;
        let delta2 = value - self.mean;
        self.variance_sum += weight * delta * delta2;
    }

    fn std_dev(&self) -> f64 {
        if self.weight <= 1.0 {
            return 0.0;
        }
        (self.variance_sum / (self.weight - 1.0)).sqrt()
    }

    /// Fraction of weight estimated to fall strictly below `value`.
    pub fn prob_below(&self, value: f64) -> f64 {
        if self.weight == 0.0 {
            return 0.0;
        }
        let std = self.std_dev();
        if std < 1e-10 {
            return if value > self.mean { 1.0 } else { 0.0 };
        }
        (1.0 + erf((value - self.mean) / (std * std::f64::consts::SQRT_2))) / 2.0
    }

    pub fn pdf(&self, value: f64) -> f64 {
        if self.weight <= 1.0 {
            return 0.0;
        }
        let std = self.std_dev();
        if std < 1e-10 {
            // degenerate distribution: all mass at the mean
            return if (value - self.mean).abs() < 1e-10 {
                1.0
            } else {
                0.0
            };
        }
        let z = (value - self.mean) / std;
        (-0.5 * z * z).exp() / (std * (2.0 * std::f64::consts::PI).sqrt())
    }
}

/// Abramowitz & Stegun erf approximation, max error ≈ 1.5e-7
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianFeatureStats {
    pub estimators: Box<[GaussianEstimator]>, // one per class
    pub min_val: f64,
    pub max_val: f64,
}

impl GaussianFeatureStats {
    pub fn new(n_classes: usize) -> Self {
        Self {
            estimators: vec![GaussianEstimator::default(); n_classes].into_boxed_slice(),
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
        }
    }

    pub fn update(&mut self, value: f64, class: usize, k: usize) {
        if value < self.min_val {
            self.min_val = value;
        }
        if value > self.max_val {
            self.max_val = value;
        }
        if class < self.estimators.len() {
            self.estimators[class].update(value, k as f64);
        }
    }

    /// 10 evenly-spaced split candidates in (min, max) — matches MOA's numBins=10 default.
    pub fn split_points(&self, n: usize) -> Vec<f64> {
        if self.min_val >= self.max_val {
            return vec![];
        }
        let range = self.max_val - self.min_val;
        (0..n)
            .map(|i| self.min_val + range / (n + 1) as f64 * (i + 1) as f64)
            .filter(|&v| v > self.min_val && v < self.max_val)
            .collect()
    }

    /// Estimated (left_weights_per_class, right_weights_per_class) for a threshold.
    pub fn class_dists_at_split(&self, threshold: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.estimators.len();
        let mut left = vec![0.0f64; n];
        let mut right = vec![0.0f64; n];
        for (c, est) in self.estimators.iter().enumerate() {
            if est.weight == 0.0 {
                continue;
            }
            let p = est.prob_below(threshold);
            left[c] = est.weight * p;
            right[c] = est.weight * (1.0 - p);
        }
        (left, right)
    }
}

// ── Unified LocalStats — the type stored in NodeKind::Leaf ───────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LocalStats {
    Histogram { stats: Histogram },
    Gaussian { stats: GaussianFeatureStats },
}

impl LocalStats {
    pub fn new_histogram(max_bins: usize) -> Self {
        LocalStats::Histogram {
            stats: Histogram::new(max_bins),
        }
    }

    pub fn new_gaussian(n_classes: usize) -> Self {
        LocalStats::Gaussian {
            stats: GaussianFeatureStats::new(n_classes),
        }
    }

    pub fn update(&mut self, value: f64, class: usize, k: usize, n_classes: usize) {
        match self {
            LocalStats::Histogram { stats } => stats.update(value, class, k, n_classes),
            LocalStats::Gaussian { stats } => stats.update(value, class, k),
        }
    }

    pub fn prob_of_value_given_class(&self, value: f64, class: usize) -> f64 {
        match self {
            LocalStats::Gaussian { stats } => {
                if class < stats.estimators.len() {
                    stats.estimators[class].pdf(value)
                } else {
                    0.0
                }
            }
            LocalStats::Histogram { stats } => {
                let bins = &stats.bins;
                if bins.is_empty() {
                    return 0.0;
                }
                // Find the nearest bin by mean
                let idx = bins
                    .binary_search_by(|b| b.mean.partial_cmp(&value).unwrap())
                    .unwrap_or_else(|i| {
                        if i == 0 {
                            0
                        } else if i >= bins.len() {
                            bins.len() - 1
                        } else {
                            let d_left = (bins[i - 1].mean - value).abs();
                            let d_right = (bins[i].mean - value).abs();
                            if d_left <= d_right { i - 1 } else { i }
                        }
                    });
                let bin = &bins[idx];
                let bin_class_count = if class < bin.by_label.len() {
                    bin.by_label[class] as f64
                } else {
                    return 0.0;
                };
                let total_class: f64 = bins
                    .iter()
                    .map(|b| {
                        if class < b.by_label.len() {
                            b.by_label[class] as f64
                        } else {
                            0.0
                        }
                    })
                    .sum();
                if total_class <= 0.0 {
                    0.0
                } else {
                    bin_class_count / total_class
                }
            }
        }
    }
}
