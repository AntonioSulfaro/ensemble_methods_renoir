use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const MIN_WINDOW_CHECK: usize = 10; // minimum window size to start checking
const MIN_WIN_DENOM: usize = 5; // used in denominator of epsilon (mintMinWinLength)
const MAX_BUCKETS: usize = 5; // maximum buckets per level
const MIN_CLOCK: usize = 32; // check every 32 steps

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adwin {
    levels: Vec<VecDeque<Bucket>>, // each level i stores buckets covering 2^i elements
    total_count: usize,            // total number of elements in window
    total_sum: f64,                // sum of all values
    total_var: f64,                // total sum of squared deviations (incremental)
    pub delta: f64,                // confidence parameter
    clock: usize,                  // counter for periodic checks
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Bucket {
    count: usize,  // number of elements in this bucket (always a power of two)
    sum: f64,      // sum of values
    variance: f64, // sum of squared deviations inside this bucket
}

impl Adwin {
    pub fn new(delta: f64) -> Self {
        Self {
            levels: Vec::new(),
            total_count: 0,
            total_sum: 0.0,
            total_var: 0.0,
            delta,
            clock: 0,
        }
    }

    /// Insert a new value and return `true` if a change was detected.
    pub fn add(&mut self, value: f64) -> bool {
        self.insert_element(value);
        self.clock += 1;
        if self.clock % MIN_CLOCK == 0 && self.total_count > MIN_WINDOW_CHECK {
            self.detect_change()
        } else {
            false
        }
    }

    /// Insert one element into the window (steps 1‑2 of ADWIN).
    fn insert_element(&mut self, value: f64) {
        // Update global statistics incrementally
        let old_count = self.total_count;
        let old_sum = self.total_sum;
        self.total_count += 1;
        self.total_sum += value;

        if old_count > 0 {
            let mean_old = old_sum / old_count as f64;
            // Correct online update for sum of squared deviations
            let inc_var = old_count as f64 * (value - mean_old).powi(2) / self.total_count as f64;
            self.total_var += inc_var;
        }

        // Insert a new bucket at level 0 (size 1, variance 0)
        if self.levels.is_empty() {
            self.levels.push(VecDeque::new());
        }
        self.levels[0].push_back(Bucket {
            count: 1,
            sum: value,
            variance: 0.0,
        });

        // Merge buckets if any level overflows
        self.compress_buckets();
    }

    /// Merge buckets when a level contains more than MAX_BUCKETS.
    fn compress_buckets(&mut self) {
        let mut level = 0;
        while level < self.levels.len() {
            if self.levels[level].len() <= MAX_BUCKETS {
                level += 1;
                continue;
            }

            // Merge the two oldest buckets at this level
            let b1 = self.levels[level].pop_front().unwrap();
            let b2 = self.levels[level].pop_front().unwrap();

            let n1 = b1.count;
            let n2 = b2.count;
            let u1 = b1.sum / n1 as f64;
            let u2 = b2.sum / n2 as f64;

            let inc_var = (n1 * n2) as f64 * (u1 - u2).powi(2) / (n1 + n2) as f64;

            let merged = Bucket {
                count: n1 + n2,
                sum: b1.sum + b2.sum,
                variance: b1.variance + b2.variance + inc_var,
            };

            // Ensure next level exists
            if level + 1 >= self.levels.len() {
                self.levels.push(VecDeque::new());
            }
            self.levels[level + 1].push_back(merged);
        }
    }

    /// Core change detection. Returns `true` if at least one drift was found.
    fn detect_change(&mut self) -> bool {
        let mut changed = false;
        loop {
            let mut found = false;
            let mut n0 = 0usize;
            let mut sum0 = 0.0f64;

            // Scan from the oldest (highest level, front) to the newest
            'outer: for li in (0..self.levels.len()).rev() {
                for bi in 0..self.levels[li].len() {
                    let bucket = &self.levels[li][bi];
                    n0 += bucket.count;
                    sum0 += bucket.sum;
                    let n1 = self.total_count - n0;

                    // Both parts must be large enough (strictly > MIN_WIN_DENOM+1, i.e. ≥7)
                    if n0 > MIN_WIN_DENOM + 1 && n1 > MIN_WIN_DENOM + 1 {
                        let mean0 = sum0 / n0 as f64;
                        let mean1 = (self.total_sum - sum0) / n1 as f64;
                        let diff = (mean0 - mean1).abs();

                        let n = self.total_count as f64;
                        let dd = (4.0 * n * n / self.delta).ln();
                        let m = 1.0 / n0 as f64 + 1.0 / n1 as f64;
                        let epsilon = (m * dd / 2.0).sqrt();

                        if diff > epsilon {
                            found = true;
                            changed = true;
                            self.delete_oldest_bucket();
                            break 'outer;
                        }
                    }
                }
            }
            if !found {
                break;
            }
        }
        changed
    }

    /// Remove the oldest bucket from the window and update global statistics.
    fn delete_oldest_bucket(&mut self) {
        if self.levels.is_empty() {
            return;
        }
        let last_lev = self.levels.len() - 1;
        let bucket = self.levels[last_lev].pop_front().unwrap();
        let bucket_size = bucket.count;

        // Update totals *before* computing the variance contribution (MOA order)
        self.total_count -= bucket_size;
        self.total_sum -= bucket.sum;
        let new_mean = self.total_sum / self.total_count as f64;
        let u1 = bucket.sum / bucket_size as f64;
        let inc_var = bucket.variance
            + (bucket_size * self.total_count) as f64 * (u1 - new_mean).powi(2)
                / (bucket_size + self.total_count) as f64;
        self.total_var -= inc_var;
        self.total_var = self.total_var.max(0.0); // avoid tiny negatives due to rounding

        // Remove empty level
        if self.levels[last_lev].is_empty() {
            self.levels.pop();
        }
    }

    /// Current estimate of the mean (e.g., error rate).
    pub fn estimation(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.total_sum / self.total_count as f64
        }
    }

    /// Current window width.
    pub fn width(&self) -> usize {
        self.total_count
    }
}

// ── Dual-threshold drift monitor ─────────────────────────────────────────────

/// The signal returned to the caller after each observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftSignal {
    /// Nothing noteworthy — keep training normally.
    None,
    /// Warning level crossed: spin up / keep training a background model.
    Warning,
    /// Drift level crossed: promote background model, reset subspace.
    Drift,
}

/// Wraps two ADWIN detectors (warning + drift) and owns the notion of whether
/// a background learner is currently active.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualAdwin {
    /// Looser threshold — triggers "start background model".
    pub warning: Adwin,
    /// Stricter threshold — triggers "promote background model".
    pub drift: Adwin,
}

impl DualAdwin {
    pub fn new(delta_warning: f64, delta_drift: f64) -> Self {
        Self {
            warning: Adwin::new(delta_warning),
            drift: Adwin::new(delta_drift),
        }
    }

    /// Feed one error observation (0.0 = correct, 1.0 = wrong).
    /// Returns the strongest signal detected this step.
    pub fn add(&mut self, error: f64) -> DriftSignal {
        let drift_fired = self.drift.add(error);
        let warn_fired = self.warning.add(error);

        if drift_fired {
            DriftSignal::Drift
        } else if warn_fired {
            DriftSignal::Warning
        } else {
            DriftSignal::None
        }
    }

    pub fn reset_warning(&mut self, delta_warning: f64) {
        self.warning = Adwin::new(delta_warning);
    }
}
