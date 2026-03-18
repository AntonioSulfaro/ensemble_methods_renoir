use rand::prelude::SliceRandom;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
pub enum VotingStrategy {
    #[serde(alias = "majority", alias = "MAJORITY")]
    Majority,
    #[serde(alias = "weighted", alias = "WEIGHTED")]
    Weighted,
}
impl Default for VotingStrategy {
    fn default() -> Self {
        Self::Majority
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum EnsembleType {
    Srp,
    Arf,
}

impl Default for EnsembleType {
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

pub fn random_subspace(n_features: usize, features_patch: usize) -> Arc<FeatureSubspace> {
    let mut feats: Vec<usize> = (0..n_features).collect();
    feats.shuffle(&mut rand::rng());
    feats.truncate(features_patch);
    feats.sort();
    Arc::new(feats)
}

pub fn aggregate_vote(votes: &[f64], count: usize, n_trees: usize) -> Option<usize> {
    if count < n_trees {
        return None;
    }
    votes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(class_id, _)| class_id)
}
