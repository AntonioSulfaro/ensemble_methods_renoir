use crate::learners::{NumericEstimatorType, VotingStrategy};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub dataset: String,
    pub n_trees: usize,
    #[serde(default)]
    pub voting: VotingStrategy,
    pub seed: usize,
    pub algorithm: AlgorithmConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")] // JSON: "type": "srp" / "arf" / "amf"
#[serde(rename_all = "lowercase")]
pub enum AlgorithmConfig {
    Srp(HTConfig),
    Arf(HTConfig),
    Amf(AmfConfig),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HTConfig {
    pub drift_detection: bool,
    pub max_bins: usize,
    pub n_min: usize,
    pub delta: f64,
    pub tau: f64,
    pub features_patch: Option<f64>,
    pub lambda: f64,
    pub adwin_delta_warning: f64,
    pub adwin_delta_drift: f64,
    #[serde(default)]
    pub numeric_estimator: NumericEstimatorType,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AmfConfig {
    pub step: f64,
    pub dirichlet: f64,
}
