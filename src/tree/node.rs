use crate::data::structures::LocalStats;
use crate::learners::forest_utils::FeatureSubspace;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
        class_counts: Box<[u32]>,
        weight_seen_at_last_split: usize,
        feature_stats: Vec<LocalStats>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub kind: NodeKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeWithPatchKind {
    Internal {
        test: SplitTest,
        left: NodeId,
        right: NodeId,
    },
    Leaf {
        total_samples: usize,
        class_counts: Box<[u32]>,
        weight_seen_at_last_split: usize,
        feature_stats: Vec<LocalStats>,
        feature_subspace: Arc<FeatureSubspace>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeWithPatch {
    pub kind: NodeWithPatchKind,
}
