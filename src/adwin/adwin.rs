use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adwin {
    levels: Vec<VecDeque<Bucket>>,
    total_count: usize,
    total_sum: f64,
    pub delta: f64,
    ln_delta: f64,
    min_window_size: usize,
    max_buckets: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Bucket {
    count: usize,
    sum: f64,
}

impl Adwin {
    pub fn new(delta: f64) -> Self {
        Self {
            levels: Vec::new(),
            total_count: 0,
            total_sum: 0.0,
            delta,
            ln_delta: delta.ln(),
            min_window_size: 10,
            max_buckets: 2,
        }
    }

    /// Returns true if a change was detected.
    pub fn add(&mut self, value: f64) -> bool {
        self.insert_bucket(value);
        self.detect_change()
    }

    fn insert_bucket(&mut self, value: f64) {
        if self.levels.is_empty() {
            self.levels.push(VecDeque::new());
        }

        self.levels[0].push_back(Bucket {
            count: 1,
            sum: value,
        });
        self.total_count += 1;
        self.total_sum += value;

        let mut level = 0;

        loop {
            if self.levels[level].len() <= self.max_buckets {
                break;
            }

            let b1 = self.levels[level].pop_front().unwrap();
            let b2 = self.levels[level].pop_front().unwrap();

            let merged = Bucket {
                count: b1.count + b2.count,
                sum: b1.sum + b2.sum,
            };

            level += 1;

            if self.levels.len() <= level {
                self.levels.push(VecDeque::new());
            }

            self.levels[level].push_back(merged);
        }
    }

    fn detect_change(&mut self) -> bool {
        let mut changed = false;

        loop {
            let mut n0 = 0usize;
            let mut sum0 = 0.0;
            let mut n1 = self.total_count;
            let mut sum1 = self.total_sum;

            let mut cut_found = false;

            // iterate from oldest → newest
            for level in self.levels.iter().rev() {
                for bucket in level {
                    n0 += bucket.count;
                    sum0 += bucket.sum;

                    n1 -= bucket.count;
                    sum1 -= bucket.sum;

                    if n0 >= self.min_window_size && n1 >= self.min_window_size {
                        let diff = (sum0 / n0 as f64) - (sum1 / n1 as f64);

                        if diff.abs() > self.epsilon(n0, n1) {
                            changed = true;
                            cut_found = true;
                            break;
                        }
                    }
                }
                if cut_found {
                    self.drop_oldest_bucket();
                    break;
                }
            }

            if !cut_found {
                break;
            }
        }
        changed
    }

    fn drop_oldest_bucket(&mut self) {
        for level in self.levels.iter_mut().rev() {
            if let Some(b) = level.pop_front() {
                self.total_count -= b.count;
                self.total_sum -= b.sum;
                break;
            }
        }
        while self.levels.last().map_or(false, |l| l.is_empty()) {
            self.levels.pop();
        }
    }

    fn epsilon(&self, n0: usize, n1: usize) -> f64 {
        let (n0, n1) = (n0 as f64, n1 as f64);
        let n = n0 + n1;
        let m = 1.0 / (1.0 / n0 + 1.0 / n1);
        ((1.0 / (2.0 * m)) * ((4.0 * n).ln() - self.ln_delta)).sqrt()
    }

    pub fn error_rate(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.total_sum / self.total_count as f64
        }
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
    /// True once a warning has fired, and we are waiting for either drift or
    /// for the warning to clear naturally.
    in_warning: bool,
}

impl DualAdwin {
    pub fn new(delta_warning: f64, delta_drift: f64) -> Self {
        Self {
            warning: Adwin::new(delta_warning),
            drift: Adwin::new(delta_drift),
            in_warning: false,
        }
    }

    /// Feed one error observation (0.0 = correct, 1.0 = wrong).
    /// Returns the strongest signal detected this step.
    pub fn add(&mut self, error: f64) -> DriftSignal {
        let drift_fired = self.drift.add(error);
        let warn_fired = self.warning.add(error);

        if drift_fired {
            // Full drift: reset everything, including warning state.
            self.in_warning = false;
            DriftSignal::Drift
        } else if warn_fired {
            self.in_warning = true;
            DriftSignal::Warning
        } else {
            // Even if nothing new fired, we may still be in an active warning
            // period from a previous step.
            if self.in_warning {
                DriftSignal::Warning
            } else {
                DriftSignal::None
            }
        }
    }

    pub fn in_warning(&self) -> bool {
        self.in_warning
    }

    /// Call this if the background model was successfully promoted so we can
    /// reset the warning flag.
    pub fn clear_warning(&mut self) {
        self.in_warning = false;
    }
}
