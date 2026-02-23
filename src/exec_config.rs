use serde::Deserialize;

#[derive(Deserialize)]
pub struct ExecConfig {
    pub ensemble_type: String, // "srp" or "arf"
    pub drift_detection: bool,
    pub n_trees: usize,
    pub max_bins: usize,
    pub n_min: usize,
    pub delta: f64,
    pub tau: f64,
    pub range_r: f64,
    pub features_patch: f64,
    pub lambda: f64,
}
