use serde::{Deserialize, Serialize};

/// Final aggregated prediction from the forest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedPrediction {
    pub instance_id: usize,
    pub predicted_class: Option<usize>,
    pub votes: Vec<usize>,
    pub n_trees: usize,
}
