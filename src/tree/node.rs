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
pub enum HoeffdingNodeKind {
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
        mc_correct_weight: f64,
        nb_correct_weight: f64,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoeffdingNode {
    pub kind: HoeffdingNodeKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HoeffdingNodeWithPatchKind {
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
        mc_correct_weight: f64,
        nb_correct_weight: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoeffdingNodeWithPatch {
    pub kind: HoeffdingNodeWithPatchKind,
}

// TODO is possible to merge with NodeId?
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct NodeIdStruct(pub usize);

// MondrianNode, if left and right is None, this is a leaf node, else this is a branch node
#[derive(Clone)]
pub struct MondrianNode {
    pub parent: Option<NodeIdStruct>,
    pub left: Option<NodeIdStruct>,
    pub right: Option<NodeIdStruct>,
    pub min_range: Option<Vec<f64>>,
    pub max_range: Option<Vec<f64>>,
    pub feature: Option<usize>,
    pub threshold: Option<f64>,
    pub time: f64,
    pub classes: Vec<f64>,
    pub weight: f64,
    pub log_weight: f64,
    pub n_samples: f64,
}

impl MondrianNode {
    pub fn new(
        parent: Option<NodeIdStruct>,
        left: Option<NodeIdStruct>,
        right: Option<NodeIdStruct>,
        time: f64,
        n_features: usize,
        n_classes: usize,
    ) -> Self {
        let classes = vec![0.0; n_classes];
        Self {
            parent,
            left,
            right,
            min_range: Some(vec![0.0; n_features]), // Filled with 0.0 (The Python bug)
            max_range: Some(vec![0.0; n_features]), // Filled with 0.0 (The Python bug)
            feature: None,
            threshold: None,
            time,
            classes,
            weight: 0.0,
            log_weight: 0.0,
            n_samples: 0.0,
        }
    }
}
