use rand::prelude::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
pub enum VotingStrategy {
    #[serde(alias = "majority", alias = "MAJORITY", alias = "HARD", alias = "hard")]
    Majority,
    #[serde(alias = "soft", alias = "SOFT")]
    Soft,
    #[serde(alias = "weighted", alias = "WEIGHTED")]
    Weighted,
}
impl Default for VotingStrategy {
    fn default() -> Self {
        Self::Soft
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum HTEnsembleType {
    Srp,
    Arf,
}

impl Default for HTEnsembleType {
    fn default() -> Self {
        Self::Srp
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum NumericEstimatorType {
    #[serde(alias = "histogram", alias = "HISTOGRAM")]
    Histogram,
    #[serde(alias = "gaussian", alias = "GAUSSIAN")]
    Gaussian,
}

impl Default for NumericEstimatorType {
    fn default() -> Self {
        Self::Histogram
    }
}

pub type FeatureSubspace = Vec<usize>;

pub fn random_subspace(
    n_features: usize,
    features_patch: usize,
    rng: &mut impl Rng,
) -> Arc<FeatureSubspace> {
    let mut feats: Vec<usize> = (0..n_features).collect();
    feats.shuffle(rng);
    feats.truncate(features_patch);
    feats.sort();
    Arc::new(feats)
}

/// Aggregate votes from multiple trees into a single prediction.
pub fn aggregate_vote(votes: &[f64]) -> Option<usize> {
    votes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(class_id, _)| class_id)
}

/// Normalize a vote vector so that all values sum to 1.0.
///
/// If all votes are zero or negative, returns a uniform distribution.
pub fn normalize_votes(votes: &mut [f64]) {
    let sum: f64 = votes.iter().sum();
    if sum > 0.0 {
        for vote in votes.iter_mut() {
            *vote /= sum;
        }
    } else {
        // Uniform distribution if no votes
        let uniform = 1.0 / votes.len() as f64;
        for vote in votes.iter_mut() {
            *vote = uniform;
        }
    }
}

/// Apply accuracy weight to a normalized vote vector.
///
/// Follows MOA's StreamingRandomPatches.getVotesForInstance():
/// Each normalized probability is multiplied by the tree's accuracy.
pub fn apply_accuracy_weight(votes: &mut [f64], accuracy: f64) {
    let clamped_accuracy = accuracy.clamp(0.0, 1.0);
    for vote in votes.iter_mut() {
        *vote *= clamped_accuracy;
    }
}
