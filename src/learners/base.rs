use crate::Instance;

/// Common interface for ensemble learner slots.
///
/// Both [`SrpLearner`](super::adaptive::SrpLearner) and
/// [`ArfLearner`](super::arf::ArfLearner) implement this trait so that
/// [`process_stream`](crate::run::process::process_stream) can remain
/// algorithm-agnostic.
pub trait BaseLearner {
    /// Predict the class label for `inst`.
    fn predict(&self, inst: &Instance) -> Option<usize>;

    /// Train on `inst` with Poisson weight `k`.
    ///
    /// `is_correct` is whether the prediction made **before** this call
    /// matched the true label; it is used by ARF for drift detection.
    ///
    /// Returns `true` if a full drift was detected (ARF only; always
    /// `false` for SRP).
    fn train(&mut self, inst: &Instance, k: usize, is_correct: bool) -> bool;

    /// Prequential (test-then-train) accuracy accumulated so far.
    fn prequential_accuracy(&self) -> f64;
}
