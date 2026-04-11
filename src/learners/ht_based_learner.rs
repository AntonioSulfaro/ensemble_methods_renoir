use crate::adwin::{DriftSignal, DualAdwin};
use crate::config::config::HTConfig;
use crate::learners::forest_utils::{
    random_subspace, FeatureSubspace, HTEnsembleType, NumericEstimatorType,
};
use crate::learners::online_learner::OnlineLearnerTrait;
use crate::tree::ar_tree::AdaptiveRandomTree;
use crate::tree::RandomPatchesTree;
use crate::Instance;
use rand::distr::Distribution;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::Poisson;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

    pub fn predict(&self, inst: &Instance) -> (Option<usize>, Vec<f64>, usize) {
        match self {
            TreeKind::RandomPatches(t) => t.predict(inst),
            TreeKind::AdaptiveRandom(t) => t.predict(inst),
        }
    }

    /// Reset the tree. For RandomPatches, the global feature count is needed;
    /// for AdaptiveRandom it is ignored (the tree stores its own).
    pub fn reset_tree(&mut self, new_subspace: Arc<FeatureSubspace>) {
        match self {
            TreeKind::RandomPatches(t) => t.reset_tree(new_subspace),
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
pub struct HTBasedLearner {
    pub tree: TreeKind,
    pub drift_config: DriftConfig,
    rng_seed: u64,
    #[serde(skip)]
    rng: Option<SmallRng>,
    lambda: f64,
    ensemble_type: HTEnsembleType,
    n_min: usize,
    delta: f64,
    tau: f64,
    n_classes: usize,
    n_features: usize,
    max_bins: usize,
    estimator_type: NumericEstimatorType,
    pub cumulative_correct: usize,
    pub cumulative_n: usize,
}

impl OnlineLearnerTrait for HTBasedLearner {
    fn train(&mut self, inst: &Instance, is_correct: bool) -> bool {
        self.train(inst, is_correct)
    }

    fn predict(&self, inst: &Instance) -> (Option<usize>, Vec<f64>, usize) {
        self.predict(inst)
    }

    fn cumulative_accuracy(&self) -> f64 {
        self.cumulative_accuracy()
    }
}

impl HTBasedLearner {
    fn rng(&mut self) -> &mut SmallRng {
        if self.rng.is_none() {
            self.rng = Some(SmallRng::seed_from_u64(self.rng_seed));
        }
        self.rng.as_mut().unwrap()
    }

    pub fn new(
        cfg: &HTConfig,
        ensemble_type: HTEnsembleType,
        n_classes: usize,
        n_features: usize,
        seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let tree = match ensemble_type {
            HTEnsembleType::Srp => TreeKind::RandomPatches(RandomPatchesTree::new(
                random_subspace(
                    n_features,
                    Self::patch_size(cfg.features_patch, n_features),
                    &mut rng,
                ),
                cfg.n_min,
                cfg.delta,
                cfg.tau,
                n_classes,
                cfg.max_bins,
                cfg.numeric_estimator,
            )),
            HTEnsembleType::Arf => TreeKind::AdaptiveRandom(AdaptiveRandomTree::new(
                n_features,
                Self::patch_size(cfg.features_patch, n_features),
                cfg.n_min,
                cfg.delta,
                cfg.tau,
                n_classes,
                cfg.max_bins,
                cfg.numeric_estimator,
                rng.next_u64(),
            )),
        };
        let drift_config = if cfg.drift_detection {
            DriftConfig::Enabled {
                detector: DualAdwin::new(cfg.adwin_delta_warning, cfg.adwin_delta_drift),
                background: None,
            }
        } else {
            DriftConfig::Disabled
        };
        Self {
            tree,
            drift_config,
            rng_seed: seed,
            rng: Some(rng),
            lambda: cfg.lambda,
            ensemble_type,
            n_min: cfg.n_min,
            delta: cfg.delta,
            tau: cfg.tau,
            n_classes,
            n_features,
            max_bins: cfg.max_bins,
            cumulative_correct: 0,
            cumulative_n: 0,
            estimator_type: cfg.numeric_estimator,
        }
    }

    fn patch_size(features_patch: Option<f64>, n_features: usize) -> usize {
        match features_patch {
            Some(p) => (n_features as f64 * p).ceil() as usize,
            None => (n_features as f64).sqrt().floor() as usize + 1,
        }
    }

    /// Feed one labeled instance. Returns true if full drift was detected.
    pub fn train(&mut self, inst: &Instance, is_correct: bool) -> bool {
        self._update_cumulative(is_correct);

        let k = {
            let poisson = Poisson::new(self.lambda).unwrap();
            let rng = self.rng();
            poisson.sample(rng) as usize
        };

        if k == 0 {
            return false;
        }

        self.tree.train(inst, k);

        let mut create_background = false;
        let mut reset_tree = false;
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
                        }
                    }

                    DriftSignal::Drift => {
                        let (dw, dd) = (detector.warning.delta, detector.drift.delta);
                        match background.take() {
                            Some(bg) => self.tree = bg,
                            None => {
                                reset_tree = true;
                            }
                        }
                        *detector = DualAdwin::new(dw, dd);
                        self.cumulative_correct = 0;
                        self.cumulative_n = 0;
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

        if reset_tree {
            let new_subspace =
                random_subspace(self.n_features, self.tree.subspace_size(), self.rng());
            self.tree.reset_tree(new_subspace)
        }

        drift_fired
    }

    pub fn predict(&self, inst: &Instance) -> (Option<usize>, Vec<f64>, usize) {
        self.tree.predict(inst)
    }

    pub fn cumulative_accuracy(&self) -> f64 {
        if self.cumulative_n == 0 {
            return 0.0;
        }
        self.cumulative_correct as f64 / self.cumulative_n as f64
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Create a fresh tree with a newly sampled random subspace.
    fn new_tree(&mut self) -> TreeKind {
        match self.ensemble_type {
            HTEnsembleType::Srp => TreeKind::RandomPatches(RandomPatchesTree::new(
                random_subspace(self.n_features, self.tree.subspace_size(), self.rng()),
                self.n_min,
                self.delta,
                self.tau,
                self.n_classes,
                self.max_bins,
                self.estimator_type,
            )),
            HTEnsembleType::Arf => TreeKind::AdaptiveRandom(AdaptiveRandomTree::new(
                self.n_features,
                self.tree.subspace_size(),
                self.n_min,
                self.delta,
                self.tau,
                self.n_classes,
                self.max_bins,
                self.estimator_type,
                self.rng().next_u64(),
            )),
        }
    }

    fn _update_cumulative(&mut self, is_correct: bool) {
        self.cumulative_n += 1;
        if is_correct {
            self.cumulative_correct += 1;
        }
    }
}
