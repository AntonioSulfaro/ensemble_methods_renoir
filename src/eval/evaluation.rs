use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub instance_id: usize,
    pub actual_class: Option<usize>,
    pub predicted_class: Option<usize>,
    pub latency_micros: u128,
    pub global_accuracy: f64,
}
