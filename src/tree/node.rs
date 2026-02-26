use crate::data::structures::LocalStats;
use serde::{Deserialize, Serialize};

pub type NodeId = usize;

/// Split test structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitTest {
    pub feature_id: usize,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    Internal {
        test: SplitTest,
        left: NodeId,
        right: NodeId,
    },
    Leaf {
        total_samples: usize,
        class_counts: Vec<usize>,
        feature_stats: Vec<LocalStats>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub kind: NodeKind,
}
