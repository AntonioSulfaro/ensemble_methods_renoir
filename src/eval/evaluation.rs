use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct InstanceResult {
    pub instance_id: usize,
    pub is_correct: bool,
    pub drift_detected: bool,
    pub avg_depth: f64,
}
