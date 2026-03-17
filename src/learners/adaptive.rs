use crate::learners::base::BaseLearner;
use crate::learners::forest_utils::NumericEstimatorType;
use crate::learners::FeatureSubspace;
use crate::tree::HoeffdingTree;
use crate::Instance;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Lightweight SRP ensemble slot: a single HoeffdingTree with a fixed feature subspace.
///
/// No drift detector and no background tree — zero overhead compared to ARF.
/// Training simply forwards the Poisson-weighted instance to the tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrpLearner {
    pub tree: HoeffdingTree,
    prequential_correct: usize,
    prequential_n: usize,
}

impl SrpLearner {
    pub fn new(
        feature_subspace: Arc<FeatureSubspace>,
        n_min: usize,
        delta: f64,
        tau: f64,
        n_classes: usize,
        max_bins: usize,
        estimator_type: NumericEstimatorType,
    ) -> Self {
        let tree = HoeffdingTree::new(
            feature_subspace,
            n_min,
            delta,
            tau,
            n_classes,
            max_bins,
            estimator_type,
        );
        Self {
            tree,
            prequential_correct: 0,
            prequential_n: 0,
        }
    }
}

impl BaseLearner for SrpLearner {
    fn predict(&self, inst: &Instance) -> Option<usize> {
        self.tree.predict(inst)
    }

    /// Train on `inst` with Poisson weight `k`.
    ///
    /// `is_correct` is recorded for prequential accuracy; no drift
    /// detection is performed.  Always returns `false`.
    fn train(&mut self, inst: &Instance, k: usize, is_correct: bool) -> bool {
        self.prequential_n += 1;
        if is_correct {
            self.prequential_correct += 1;
        }
        if k > 0 {
            self.tree.train(inst, k);
        }
        false
    }

    fn prequential_accuracy(&self) -> f64 {
        if self.prequential_n == 0 {
            return 0.0;
        }
        self.prequential_correct as f64 / self.prequential_n as f64
    }
}
