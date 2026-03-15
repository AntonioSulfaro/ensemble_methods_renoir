use crate::adwin::{DriftSignal, DualAdwin};
use crate::learners::forest_utils::NumericEstimatorType;
use crate::learners::srp::random_subspace;
use crate::learners::FeatureSubspace;
use crate::tree::HoeffdingTree;
use crate::Instance;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A single ensemble slot: primary HoeffdingTree + dual ADWIN + optional background learner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearner {
    pub tree: HoeffdingTree,
    pub detector: DualAdwin,
    pub background: Option<HoeffdingTree>,
    n_min: usize,
    delta: f64,
    tau: f64,
    n_classes: usize,
    n_features: usize,
    max_bins: usize,
    pub prequential_correct: usize,
    prequential_n: usize,
    estimator_type: NumericEstimatorType,
}

impl AdaptiveLearner {
    pub fn new(
        feature_subspace: Arc<FeatureSubspace>,
        n_min: usize,
        delta: f64,
        tau: f64,
        n_classes: usize,
        n_features: usize,
        max_bins: usize,
        adwin_delta_warning: f64,
        adwin_delta_drift: f64,
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
            detector: DualAdwin::new(adwin_delta_warning, adwin_delta_drift),
            background: None,
            n_min,
            delta,
            tau,
            n_classes,
            n_features,
            max_bins,
            prequential_correct: 0,
            prequential_n: 0,
            estimator_type,
        }
    }

    /// Feed one labeled instance. Returns true if full drift was detected.
    pub fn train(
        &mut self,
        inst: &Instance,
        k: usize,
        drift_detection: bool,
        is_correct: bool,
    ) -> bool {
        self.update_prequential(is_correct);

        if k == 0 {
            return false;
        }

        self.tree.train(inst, k);

        if !drift_detection {
            return false;
        }

        if let Some(bg) = &mut self.background {
            bg.train(inst, k);
        }

        let error = if is_correct { 0.0 } else { 1.0 };
        let mut drift_fired = false;

        match self.detector.add(error) {
            DriftSignal::None => {}

            DriftSignal::Warning => {
                if self.background.is_none() {
                    self.background = Some(self.new_tree());
                    self.detector.reset_warning(self.detector.warning.delta);
                }
            }

            DriftSignal::Drift => {
                // println!("{}", self.detector.drift.error_rate());
                match self.background.take() {
                    Some(bg) => self.tree = bg,
                    None => self.tree.reset_tree(self.n_features),
                }
                self.detector =
                    DualAdwin::new(self.detector.warning.delta, self.detector.drift.delta);
                self.prequential_n = 0;
                self.prequential_correct = 0;
                drift_fired = true;
            }
        }

        drift_fired
    }

    pub fn predict(&self, inst: &Instance) -> Option<usize> {
        self.tree.predict(inst)
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Create a fresh HoeffdingTree with a newly sampled random subspace.
    fn new_tree(&self) -> HoeffdingTree {
        let subspace = random_subspace(self.n_features, self.tree.feature_subspace.len());
        HoeffdingTree::new(
            subspace,
            self.n_min,
            self.delta,
            self.tau,
            self.n_classes,
            self.max_bins,
            self.estimator_type,
        )
    }

    pub fn update_prequential(&mut self, is_correct: bool) {
        self.prequential_n += 1;
        if is_correct {
            self.prequential_correct += 1;
        }
    }

    pub fn prequential_accuracy(&self) -> f64 {
        if self.prequential_n == 0 {
            return 0.0;
        }
        self.prequential_correct as f64 / self.prequential_n as f64
    }
}
