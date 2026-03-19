use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct InstanceResult {
    pub instance_id: usize,
    pub actual_class: Option<usize>,
    pub predicted_class: Option<usize>,
    pub global_accuracy: f64,
    pub drift_detected: bool,
    pub avg_depth: f64,
}
