use rand::seq::SliceRandom;
use std::sync::Arc;

pub type FeatureSubspace = Vec<usize>;

pub fn random_subspace(
    n_features: usize,
    features_patch: usize,
) -> Arc<FeatureSubspace> {
    let mut feats: Vec<usize> = (0..n_features).collect();
    feats.shuffle(&mut rand::rng());
    feats.truncate(features_patch);
    feats.sort();
    Arc::new(feats)
}
