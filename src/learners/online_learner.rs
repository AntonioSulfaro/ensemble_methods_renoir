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

#[derive(Clone)]
pub enum OnlineLearner {
    HT(HTBasedLearner),
    MT(MondrianTree),
}

impl OnlineLearnerTrait for OnlineLearner {
    fn train(&mut self, inst: &Instance, is_correct: bool) -> bool {
        match self {
            OnlineLearner::HT(l) => l.train(inst, is_correct),
            OnlineLearner::MT(l) => l.train(inst, is_correct),
        }
    }

    fn predict(&self, inst: &Instance) -> (Option<usize>, Vec<f64>, usize) {
        match self {
            OnlineLearner::HT(l) => l.predict(inst),
            OnlineLearner::MT(l) => l.predict(inst),
        }
    }

    fn cumulative_accuracy(&self) -> f64 {
        match self {
            OnlineLearner::HT(l) => l.cumulative_accuracy(),
            OnlineLearner::MT(l) => l.cumulative_accuracy(),
        }
    }
}

pub fn create_learner(
    algorithm: &AlgorithmConfig,
    n_classes: usize,
    n_features: usize,
    seed: u64,
) -> OnlineLearner {
    match algorithm {
        AlgorithmConfig::Srp(cfg) => OnlineLearner::HT(HTBasedLearner::new(
            cfg,
            HTEnsembleType::Srp,
            n_classes,
            n_features,
            seed,
        )),
        AlgorithmConfig::Arf(cfg) => OnlineLearner::HT(HTBasedLearner::new(
            cfg,
            HTEnsembleType::Arf,
            n_classes,
            n_features,
            seed,
        )),
        AlgorithmConfig::Amf(cfg) => {
            OnlineLearner::MT(MondrianTree::new(cfg, n_classes, n_features, seed))
        }
    }
}
