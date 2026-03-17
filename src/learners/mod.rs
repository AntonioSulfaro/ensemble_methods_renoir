pub mod adaptive;
pub mod arf;
pub mod base;
pub mod forest_utils;
pub mod srp;

pub use adaptive::SrpLearner;
pub use arf::ArfLearner;
pub use base::BaseLearner;
pub use forest_utils::{aggregate_vote, EnsembleType, NumericEstimatorType, VotingStrategy};
pub use srp::FeatureSubspace;

use crate::Instance;
use serde::{Deserialize, Serialize};

/// Algorithm-dispatch wrapper used by the streaming pipeline.
///
/// Choosing `Srp` incurs zero ARF overhead (no ADWIN, no background tree).
/// Choosing `Arf` enables full drift detection and background-tree swapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Learner {
    Srp(SrpLearner),
    Arf(ArfLearner),
}

impl Learner {
    pub fn predict(&self, inst: &Instance) -> Option<usize> {
        match self {
            Learner::Srp(l) => l.predict(inst),
            Learner::Arf(l) => l.predict(inst),
        }
    }

    /// Train and return `true` on drift (ARF only).
    pub fn train(&mut self, inst: &Instance, k: usize, is_correct: bool) -> bool {
        match self {
            Learner::Srp(l) => l.train(inst, k, is_correct),
            Learner::Arf(l) => l.train(inst, k, is_correct),
        }
    }

    pub fn prequential_accuracy(&self) -> f64 {
        match self {
            Learner::Srp(l) => l.prequential_accuracy(),
            Learner::Arf(l) => l.prequential_accuracy(),
        }
    }
}
