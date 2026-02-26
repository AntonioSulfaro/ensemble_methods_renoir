use crate::learners::{EnsembleType, VotingStrategy};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
    pub dataset: String,
    #[serde(default)]
    pub ensemble_type: EnsembleType,
    pub drift_detection: bool,
    pub n_trees: usize,
    pub max_bins: usize,
    pub n_min: usize,
    pub delta: f64,
    pub tau: f64,
    pub range_r: f64,
    pub features_patch: Option<f64>,
    pub lambda: f64,
    pub adwin_delta_warning: f64,
    pub adwin_delta_drift: f64,
    #[serde(default)]
    pub voting: VotingStrategy,
}
