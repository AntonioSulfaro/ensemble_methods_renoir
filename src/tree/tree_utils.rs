use crate::data::structures::{GaussianFeatureStats, Histogram, LocalStats};
use crate::learners::NumericEstimatorType;

pub fn entropy(counts: &[u64], n: usize) -> f64 {
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

pub fn entropy_f(weights: &[f64], total: f64) -> f64 {
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

pub fn naive_bayes_votes(
    feature_stats: &[LocalStats],
    class_counts: &[u32],
    local_feature_values: &[f64],
    n_classes: usize,
) -> Vec<f64> {
    let total: f64 = class_counts.iter().map(|&c| c as f64).sum();
    if total <= 0.0 {
        // No data yet — uniform prior
        return vec![1.0 / n_classes as f64; n_classes];
    }

    // Start with the class prior P(c)
    let mut scores: Vec<f64> = class_counts.iter().map(|&c| c as f64 / total).collect();

    // Multiply in per-feature likelihoods P(x_j | c)
    for (local_f, stat) in feature_stats.iter().enumerate() {
        let val = local_feature_values[local_f];
        for c in 0..n_classes {
            scores[c] *= stat.prob_of_value_given_class(val, c);
        }
    }
    scores
}

/// argmax of a float slice (ties broken by lower index).
pub fn argmax_f64(scores: &[f64]) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

pub fn make_stats(
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

pub fn evaluate_split(
    stats: &[LocalStats],
    class_counts: &Box<[u32]>,
    n: usize,
    delta: f64,
    tau: f64,
    n_classes: usize,
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
            best_gain_for_feature(f_stat, &total_counts, n, parent_entropy, n_classes)
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
    let epsilon = ((1.0 / delta).ln() / (2.0 * n as f64)).sqrt();

    // If second_best_gain is still NEG_INFINITY (only one feature had gain),
    // the difference is +INF, which always exceeds epsilon → correct, always split.
    let gain_diff = best_gain - second_best_gain; // NEG_INFINITY subtraction → +INF

    if gain_diff > epsilon || epsilon < tau {
        Some((best_fid, best_threshold))
    } else {
        None
    }
}

fn best_gain_for_feature(
    stat: &LocalStats,
    total_counts: &[u64],
    n_total: usize,
    parent_entropy: f64,
    n_classes: usize,
) -> Option<(f64, f64)> {
    match stat {
        LocalStats::Histogram { stats } => {
            best_gain_histogram(stats, total_counts, n_total, parent_entropy, n_classes)
        }
        LocalStats::Gaussian { stats } => best_gain_gaussian(stats, n_total as f64, parent_entropy),
    }
}

// ── Histogram path ──────────────────────────────────────

fn best_gain_histogram(
    hist: &Histogram,
    total_counts: &[u64],
    n_total: usize,
    parent_entropy: f64,
    n_classes: usize,
) -> Option<(f64, f64)> {
    let bins = &hist.bins;
    if bins.len() < 2 {
        return None;
    }

    let mut best_gain = f64::NEG_INFINITY;
    let mut best_threshold = 0.0;
    let mut left_counts = vec![0u64; n_classes];
    let mut n_left = 0u64;

    for i in 0..bins.len() - 1 {
        let bin = &bins[i];
        n_left += bin.total as u64;
        for (c, &cnt) in bin.by_label.iter().enumerate() {
            if c < n_classes {
                left_counts[c] += cnt as u64;
            }
        }
        let n_right = (n_total as u64).saturating_sub(n_left);
        if n_left == 0 || n_right == 0 {
            continue;
        }

        let right_counts: Vec<u64> = total_counts
            .iter()
            .zip(left_counts.iter())
            .map(|(&t, &l)| t.saturating_sub(l))
            .collect();

        let gain = parent_entropy
            - (n_left as f64 / n_total as f64) * entropy(&left_counts, n_left as usize)
            - (n_right as f64 / n_total as f64) * entropy(&right_counts, n_right as usize);

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
    gstats: &GaussianFeatureStats,
    n_total: f64,
    parent_entropy: f64,
) -> Option<(f64, f64)> {
    let split_points = gstats.split_points(10);
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
