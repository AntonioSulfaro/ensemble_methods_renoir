use crate::data_structures::Instance;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Task type for processing in the forest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForestTask {
    /// Train with a labeled instance
    Train(Arc<Instance>),
    /// Predict with an unlabeled instance
    Predict {
        // TODO can I remove id and place it in the instance struct?
        instance_id: usize,
        instance: Arc<Instance>,
    },
}

/// Result from a tree processing a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForestResult {
    /// Training completed on a tree
    Trained { tree_id: usize, nodes: usize },
    /// Prediction from a single tree
    Prediction {
        instance_id: usize,
        tree_id: usize,
        predicted_class: Option<usize>,
        fragmentation: usize,
    },
}

/// Final aggregated prediction from the forest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedPrediction {
    pub instance_id: usize,
    pub predicted_class: Option<usize>,
    pub votes: Vec<usize>,
    pub n_trees: usize,
}
