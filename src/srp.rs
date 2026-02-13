use rand::rng;
use rand::seq::SliceRandom;

pub(crate) type FeatureSubspace = Vec<usize>;

//TODO fix seed
pub(crate) fn generate_feature_subspaces(
    n_features: usize,
    n_features_patch: usize,
    n_trees: usize,
) -> Vec<FeatureSubspace> {
    (0..n_trees)
        .map(|_| {
            let mut feats: Vec<usize> = (0..n_features).collect();
            feats.shuffle(&mut rng());
            feats.truncate(n_features_patch);
            feats
        })
        .collect()
}
