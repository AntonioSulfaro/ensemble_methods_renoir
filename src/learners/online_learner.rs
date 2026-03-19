use crate::adwin::{DriftSignal, DualAdwin};
use crate::learners::forest_utils::{random_subspace, NumericEstimatorType};
use crate::learners::EnsembleType;
use crate::tree::ar_tree::AdaptiveRandomTree;
use crate::tree::RandomPatchesTree;
use crate::Instance;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeKind {
    RandomPatches(RandomPatchesTree),
    AdaptiveRandom(AdaptiveRandomTree),
}

impl TreeKind {
    pub fn train(&mut self, inst: &Instance, k: usize) {
        match self {
            TreeKind::RandomPatches(t) => t.train(inst, k),
            TreeKind::AdaptiveRandom(t) => t.train(inst, k),
        }
    }

    pub fn predict(&self, inst: &Instance) -> (Option<usize>, usize) {
        match self {
            TreeKind::RandomPatches(t) => t.predict(inst),
            TreeKind::AdaptiveRandom(t) => t.predict(inst),
        }
    }

    /// Reset the tree. For RandomPatches, the global feature count is needed;
    /// for AdaptiveRandom it is ignored (the tree stores its own).
    pub fn reset_tree(&mut self, n_features: usize) {
        match self {
            TreeKind::RandomPatches(t) => t.reset_tree(n_features),
            TreeKind::AdaptiveRandom(t) => t.reset_tree(),
        }
    }

    /// Return the number of features used in each node (subspace size).
    pub fn subspace_size(&self) -> usize {
        match self {
            TreeKind::RandomPatches(t) => t.feature_subspace.len(),
            TreeKind::AdaptiveRandom(t) => t.subspace_size,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftConfig {
    Disabled,
    Enabled {
        detector: DualAdwin,
        background: Option<TreeKind>,
    },
}

/// A single ensemble slot: primary HoeffdingTree + dual ADWIN + optional background learner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearner {
    pub tree: TreeKind,
    pub drift_config: DriftConfig,
    ensemble_type: EnsembleType,
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

impl OnlineLearner {
    pub fn new(
        ensemble_type: EnsembleType,
        patch_size: usize,
        n_min: usize,
        delta: f64,
        tau: f64,
        n_classes: usize,
        n_features: usize,
        max_bins: usize,
        adwin_delta_warning: f64,
        adwin_delta_drift: f64,
        estimator_type: NumericEstimatorType,
        enable_drift: bool,
    ) -> Self {
        let tree = match ensemble_type {
            EnsembleType::Srp => TreeKind::RandomPatches(RandomPatchesTree::new(
                random_subspace(n_features, patch_size),
                n_min,
                delta,
                tau,
                n_classes,
                max_bins,
                estimator_type,
            )),
            EnsembleType::Arf => TreeKind::AdaptiveRandom(AdaptiveRandomTree::new(
                n_features,
                patch_size,
                n_min,
                delta,
                tau,
                n_classes,
                max_bins,
                estimator_type,
            )),
        };
        let drift_config = if enable_drift {
            DriftConfig::Enabled {
                detector: DualAdwin::new(adwin_delta_warning, adwin_delta_drift),
                background: None,
            }
        } else {
            DriftConfig::Disabled
        };
        Self {
            tree,
            drift_config,
            ensemble_type,
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
    pub fn train(&mut self, inst: &Instance, k: usize, is_correct: bool) -> bool {
        self.update_prequential(is_correct);

        if k == 0 {
            return false;
        }

        self.tree.train(inst, k);

        let mut create_background = false;
        let mut drift_fired = false;

        match &mut self.drift_config {
            DriftConfig::Disabled => return false,

            DriftConfig::Enabled {
                detector,
                background,
            } => {
                if let Some(bg) = background {
                    bg.train(inst, k);
                }

                let error = if is_correct { 0.0 } else { 1.0 };

                match detector.add(error) {
                    DriftSignal::None => {}

                    DriftSignal::Warning => {
                        if background.is_none() {
                            create_background = true;
                            detector.reset_warning(detector.warning.delta);
                        }
                    }

                    DriftSignal::Drift => {
                        match background.take() {
                            Some(bg) => self.tree = bg,
                            None => self.tree.reset_tree(self.n_features),
                        }
                        *detector = DualAdwin::new(detector.warning.delta, detector.drift.delta);
                        self.prequential_n = 0;
                        self.prequential_correct = 0;
                        drift_fired = true;
                    }
                }
            }
        }

        if create_background {
            let new_tree = self.new_tree();

            if let DriftConfig::Enabled { background, .. } = &mut self.drift_config {
                *background = Some(new_tree);
            }
        }

        drift_fired
    }

    pub fn predict(&self, inst: &Instance) -> (Option<usize>, usize) {
        self.tree.predict(inst)
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Create a fresh tree with a newly sampled random subspace.
    fn new_tree(&self) -> TreeKind {
        match self.ensemble_type {
            EnsembleType::Srp => TreeKind::RandomPatches(RandomPatchesTree::new(
                random_subspace(self.n_features, self.tree.subspace_size()),
                self.n_min,
                self.delta,
                self.tau,
                self.n_classes,
                self.max_bins,
                self.estimator_type,
            )),
            EnsembleType::Arf => TreeKind::AdaptiveRandom(AdaptiveRandomTree::new(
                self.n_features,
                self.tree.subspace_size(),
                self.n_min,
                self.delta,
                self.tau,
                self.n_classes,
                self.max_bins,
                self.estimator_type,
            )),
        }
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
