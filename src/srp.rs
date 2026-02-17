use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
// High quality and reproducible

pub(crate) type FeatureSubspace = Vec<usize>;

const SRP_SEED: u64 = 42;

pub(crate) fn generate_feature_subspaces(
    n_features: usize,
    n_features_patch: usize,
    n_trees: usize,
) -> Vec<Arc<FeatureSubspace>> {
    let mut rng = ChaCha8Rng::seed_from_u64(SRP_SEED);

    (0..n_trees)
        .map(|_| {
            let mut feats: Vec<usize> = (0..n_features).collect();
            feats.shuffle(&mut rng);
            feats.truncate(n_features_patch);
            feats.sort();
            Arc::new(feats)
        })
        .collect()
}
