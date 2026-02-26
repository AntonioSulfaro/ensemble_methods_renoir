use crate::adwin::{DriftSignal, DualAdwin};
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
    steps_since_warning: usize,
    warning_patience: usize,
    n_min: usize,
    delta: f64,
    tau: f64,
    n_classes: usize,
    max_bins: usize,
    range_r: f64,
}

impl AdaptiveLearner {
    pub fn new(
        feature_subspace: Arc<FeatureSubspace>,
        n_min: usize,
        delta: f64,
        tau: f64,
        n_classes: usize,
        max_bins: usize,
        range_r: f64,
        adwin_delta_warning: f64,
        adwin_delta_drift: f64,
    ) -> Self {
        let tree = HoeffdingTree::new(
            feature_subspace,
            n_min,
            delta,
            tau,
            n_classes,
            max_bins,
            range_r,
        );
        Self {
            tree,
            detector: DualAdwin::new(adwin_delta_warning, adwin_delta_drift),
            background: None,
            n_min,
            delta,
            tau,
            n_classes,
            max_bins,
            range_r,
            steps_since_warning: 0,
            warning_patience: 300,
        }
    }

    /// Feed one labelled instance. Returns true if full drift was detected.
    pub fn train_adaptive(&mut self, inst: &Instance, k: usize, is_correct: bool) -> bool {
        let error = if is_correct { 0.0 } else { 1.0 };
        let mut drift_fired = false;

        match self.detector.add(error) {
            DriftSignal::None => {
                if self.background.is_some() {
                    self.steps_since_warning += 1;
                    if self.steps_since_warning >= self.warning_patience {
                        // Warning was noise — discard the contaminated background
                        self.background = None;
                        self.steps_since_warning = 0;
                    }
                }
            }

            DriftSignal::Warning => {
                // Lazily spin up background learner on the first warning tick.
                self.steps_since_warning = 0;
                if self.background.is_none() {
                    self.background = Some(self.new_tree());
                }
            }

            DriftSignal::Drift => {
                match self.background.take() {
                    Some(bg) => {
                        // Promote the already-trained background tree.
                        // It already has a fresh random subspace baked in.
                        self.tree = bg;
                    }
                    None => {
                        // Drift fired before a warning (independent windows) —
                        // cold reset with a new subspace.
                        self.tree.reset_tree();
                    }
                }
                self.detector =
                    DualAdwin::new(self.detector.warning.delta, self.detector.drift.delta);
                drift_fired = true;
                self.steps_since_warning = 0;
            }
        }

        // Always train the primary tree.
        self.tree.train(inst, k);

        // Train background in parallel if it exists.
        if let Some(bg) = &mut self.background {
            bg.train(inst, k);
        }

        drift_fired
    }

    pub fn predict(&self, inst: &Instance) -> Option<usize> {
        self.tree.predict(inst)
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Create a fresh HoeffdingTree with a newly sampled random subspace.
    fn new_tree(&self) -> HoeffdingTree {
        HoeffdingTree::new(
            self.tree.feature_subspace.clone(),
            self.n_min,
            self.delta,
            self.tau,
            self.n_classes,
            self.max_bins,
            self.range_r,
        )
    }
}
