use crate::config::config::AlgorithmConfig;
use crate::learners::ht_based_learner::HTBasedLearner;
use crate::learners::HTEnsembleType;
use crate::tree::mondrian_tree::MondrianTree;
use crate::Instance;

pub trait OnlineLearnerTrait: Send + Sync + OnlineLearnerClone {
    /// Train on one labeled instance. Returns true if a full drift was detected.
    fn train(&mut self, inst: &Instance, is_correct: bool) -> bool;
    /// Predict on one instance. Returns (predicted_class, votes, depth of prediction)
    fn predict(&self, inst: &Instance) -> (Option<usize>, Vec<f64>, usize);
    /// Return the cumulative accuracy until that moment for weighted voting
    fn cumulative_accuracy(&self) -> f64;
}

pub trait OnlineLearnerClone {
    fn clone_box(&self) -> Box<dyn OnlineLearnerTrait>;
}

impl<T: Clone + OnlineLearnerTrait + 'static> OnlineLearnerClone for T {
    fn clone_box(&self) -> Box<dyn OnlineLearnerTrait> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn OnlineLearnerTrait> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub fn create_learner(
    algorithm: &AlgorithmConfig,
    n_classes: usize,
    n_features: usize,
    seed: u64,
) -> Box<dyn OnlineLearnerTrait> {
    match algorithm {
        AlgorithmConfig::Srp(cfg) => Box::new(HTBasedLearner::new(
            cfg,
            HTEnsembleType::Srp,
            n_classes,
            n_features,
            seed,
        )),
        AlgorithmConfig::Arf(cfg) => Box::new(HTBasedLearner::new(
            cfg,
            HTEnsembleType::Arf,
            n_classes,
            n_features,
            seed,
        )),
        AlgorithmConfig::Amf(cfg) => Box::new(MondrianTree::new(cfg, seed)),
    }
}
