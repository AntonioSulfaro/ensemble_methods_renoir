use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

pub type FeatureSubspace = Vec<usize>;

const SRP_SEED: u64 = 42;

pub fn generate_feature_subspaces(
    n_features: usize,
    features_patch: f64,
    n_trees: usize,
) -> Vec<Arc<FeatureSubspace>> {
    let mut rng = ChaCha8Rng::seed_from_u64(SRP_SEED);

    (0..n_trees)
        .map(|_| {
            let mut feats: Vec<usize> = (0..n_features).collect();
            feats.shuffle(&mut rng);
            feats.truncate(features_patch as usize);
            feats.sort();
            Arc::new(feats)
        })
        .collect()
}
