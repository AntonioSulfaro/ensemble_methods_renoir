use serde::{Deserialize, Serialize};

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

#[derive(Serialize, Deserialize, Debug, Clone)]
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
