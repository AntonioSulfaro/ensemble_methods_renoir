use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adwin {
    buckets: VecDeque<AdwinBucket>,
    total_count: usize,
    total_sum: f64,
    delta: f64,
    min_window_size: usize,
    max_buckets: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdwinBucket {
    count: usize,
    sum: f64,
    size_pow: u32, // Represents 2^size_pow elements
}

impl Adwin {
    pub fn new(delta: f64) -> Self {
        Self {
            buckets: VecDeque::new(),
            total_count: 0,
            total_sum: 0.0,
            delta,
            min_window_size: 10,
            max_buckets: 2,
        }
    }

    pub fn add(&mut self, value: f64) -> bool {
        self.insert_bucket(value);
        self.detect_change()
    }

    fn insert_bucket(&mut self, value: f64) {
        // 1. Add a new bucket of size 1 (2^0)
        self.buckets.push_back(AdwinBucket {
            count: 1,
            sum: value,
            size_pow: 0,
        });
        self.total_count += 1;
        self.total_sum += value;

        // 2. Compress: Merge buckets if we have more than max_buckets of the same size
        let mut i = 0;
        while i < self.buckets.len() {
            let count_same_size = self
                .buckets
                .iter()
                .filter(|b| b.size_pow == self.buckets[i].size_pow)
                .count();

            if count_same_size > self.max_buckets {
                // Find first two buckets of this size and merge them
                let first_idx = self
                    .buckets
                    .iter()
                    .position(|b| b.size_pow == self.buckets[i].size_pow)
                    .unwrap();
                let b1 = self.buckets.remove(first_idx).unwrap();
                let b2 = self.buckets.remove(first_idx).unwrap();

                let merged = AdwinBucket {
                    count: b1.count + b2.count,
                    sum: b1.sum + b2.sum,
                    size_pow: b1.size_pow + 1,
                };
                self.buckets.insert(first_idx, merged);
                // Continue checking from this size
            } else {
                break;
            }
            i += 1;
        }
    }

    fn detect_change(&mut self) -> bool {
        let mut changed = false;

        // ADWIN checks all possible sub-window partitions
        // In the bucketed version, we check at every bucket boundary
        loop {
            let mut n0 = 0usize;
            let mut sum0 = 0.0f64;
            let mut n1 = self.total_count;
            let mut sum1 = self.total_sum;
            let mut exit_loop = true;

            // Iterate through buckets (from oldest to newest) to find a split point
            // that violates the Hoeffding bound
            for i in 0..(self.buckets.len() - 1) {
                let b = &self.buckets[i];
                n0 += b.count;
                sum0 += b.sum;
                n1 -= b.count;
                sum1 -= b.sum;

                if n1 >= self.min_window_size && n0 >= self.min_window_size {
                    let diff = (sum0 / n0 as f64) - (sum1 / n1 as f64);
                    if diff.abs() > self.calculate_epsilon(n0, n1) {
                        // Drift detected! Drop the oldest bucket and re-check
                        self.drop_oldest();
                        changed = true;
                        exit_loop = false;
                        break;
                    }
                }
            }

            if exit_loop {
                break;
            }
        }
        changed
    }

    fn drop_oldest(&mut self) {
        if let Some(oldest) = self.buckets.pop_front() {
            self.total_count -= oldest.count;
            self.total_sum -= oldest.sum;
        }
    }

    fn calculate_epsilon(&self, n0: usize, n1: usize) -> f64 {
        let n0 = n0 as f64;
        let n1 = n1 as f64;
        let n = n0 + n1;

        // m is the harmonic mean of n0 and n1
        let m = 1.0 / (1.0 / n0 + 1.0 / n1);
        let delta_prime = self.delta / n; // Correcting for multiple comparisons

        ((1.0 / (2.0 * m)) * (1.0 / delta_prime).ln()).sqrt()
    }

    pub fn error_rate(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.total_sum / self.total_count as f64
        }
    }
}
