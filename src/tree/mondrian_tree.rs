use crate::config::config::AmfConfig;
use crate::learners::online_learner::OnlineLearnerTrait;
use crate::tree::{MondrianNode, NodeId};
use crate::Instance;
use rand::distr::weighted::WeightedIndex;
use rand::distr::{Distribution, Uniform};
use rand::prelude::SmallRng;
use rand::SeedableRng;
use rand_distr::Exp;

#[derive(Clone)]
pub struct MondrianTree {
    pub nodes: Vec<MondrianNode>,
    pub dirichlet: f64,
    pub step: f64,
    pub rng: SmallRng,
    pub root: Option<NodeId>,
    pub distances: Vec<f64>,
    pub n_features: usize,
    pub n_classes: usize,
    correct: usize,
    seen: usize,
    pub iteration: usize,
}

impl OnlineLearnerTrait for MondrianTree {
    fn train(&mut self, inst: &Instance, is_correct: bool) -> bool {
        // update cumulative accuracy
        // not used for soft voting
        self.seen += 1;
        if is_correct {
            self.correct += 1;
        }
        // train Mondrian
        self.learn_one(&inst.features, &inst.label);

        // no drift detection
        false
    }

    fn predict(&self, inst: &Instance) -> (Option<usize>, Vec<f64>, usize) {
        if self.root.is_none() {
            return (None, Vec::new(), 0);
        }

        let (probs, depth) = self.predict_prob_one(&inst.features);

        let pred = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i);

        (Some(1000), probs, depth)
    }

    fn cumulative_accuracy(&self) -> f64 {
        if self.seen == 0 {
            0.0
        } else {
            self.correct as f64 / self.seen as f64
        }
    }
}

impl MondrianTree {
    pub fn new(cfg: &AmfConfig, n_classes: usize, n_features: usize,seed: u64) -> Self {
        let distances = vec![0.0; n_features];
        let mut dirichlet = 0.01;
        if n_classes == 2  {
            dirichlet = 0.5;
        }
        Self {
            nodes: Vec::new(),
            dirichlet,
            step: cfg.step,
            rng: SmallRng::seed_from_u64(seed),
            root: None,
            distances,
            n_features,
            n_classes,
            correct: 0,
            seen: 0,
            iteration: 0,
        }
    }

    pub fn predict_one(&self, current_id: usize) -> Vec<f64> {
        let current = &self.nodes[current_id];
        let mut predictions = Vec::with_capacity(self.n_classes);
        let den = current.n_samples + self.dirichlet * self.n_classes as f64;
        for i in 0..self.n_classes{
            predictions.push((current.classes[i] + self.dirichlet) / den);
        }
        predictions
    }

    pub fn update_weight(&mut self, current_id: usize, y_index: usize) {
        let current = &self.nodes[current_id];
        let loss =  (current.classes[y_index] + self.dirichlet) / (current.n_samples + self.dirichlet * self.n_classes as f64);
        self.nodes[current_id].weight -= self.step * (-loss.ln());
    }

    pub fn update_downwards(&mut self, current_id: usize, y_index: usize, x: &[f64], update: bool) {
        {
            let current = &mut self.nodes[current_id];
            if current.n_samples == 0.0 {
                current.min_range = Some(x.to_vec());
                current.max_range = Some(x.to_vec());
            } else {
                current
                    .min_range
                    .as_mut()
                    .unwrap()
                    .iter_mut()
                    .zip(current.max_range.as_mut().unwrap().iter_mut())
                    .zip(x.iter())
                    .for_each(|((min, max), &val)| {
                        if val < *min {
                            *min = val;
                        } else if val > *max {
                            *max = val;
                        }
                    });
            }
            current.n_samples += 1.0;
        }
        if update {
            self.update_weight(current_id, y_index);
        }
        self.nodes[current_id].classes[y_index] += 1.0;
    }

    pub fn range_extensions(&mut self, current_id: usize, x: &[f64]) -> f64 {
        let current = &self.nodes[current_id];
        current
            .min_range
            .as_ref()
            .unwrap()
            .iter()
            .zip(current.max_range.as_ref().unwrap().iter())
            .zip(x.iter())
            .zip(self.distances.iter_mut())
            .map(|(((&min, &max), &val), dist)| {
                *dist = if val < min {
                    min - val
                } else if val > max {
                    val - max
                } else {
                    0.0
                };
                *dist
            })
            .sum()
    }

    pub fn compute_split_time(&mut self, current_id: usize, y_index: usize, sum: f64) -> f64 {
        let current = &self.nodes[current_id];
        let class_count = current.classes[y_index];
        if class_count == current.n_samples || sum <= 0.0 {
            return 0.0;
        }
        let exp = Exp::new(sum).unwrap();
        let split = current.time + exp.sample(&mut self.rng);
        if let Some(left_id) = current.left {
            if self.nodes[left_id].time < split {
                return 0.0;
            }
        }
        split
    }

    pub fn replant(&mut self, current_id: usize, from_id: usize, copy: bool) {
        let w = self.nodes[from_id].weight;
        let lw = self.nodes[from_id].log_weight;
        self.nodes[current_id].weight = w;
        self.nodes[current_id].log_weight = lw;
        if copy {
            let min_r = self.nodes[from_id].min_range.clone();
            let max_r = self.nodes[from_id].max_range.clone();
            let n = self.nodes[from_id].n_samples;
            let classes = self.nodes[from_id].classes.clone();
            let current = &mut self.nodes[current_id];
            current.min_range = min_r;
            current.max_range = max_r;
            current.n_samples = n;
            current.classes = classes;
        }
    }

    pub fn split(
        &mut self,
        current_id: usize,
        split_time: f64,
        feature: usize,
        threshold: f64,
        is_right: bool,
    ) {
        let (c_left, c_right, c_feature, c_threshold) = {
            let current = &self.nodes[current_id];
            (
                current.left,
                current.right,
                current.feature,
                current.threshold,
            )
        };
        let branch_id = self.nodes.len();
        let leaf_id = self.nodes.len() + 1;
        if c_left.is_some() {
            if is_right {
                let mut left =
                    MondrianNode::new(Some(current_id), c_left, c_right, split_time, self.n_features, self.n_classes);
                left.feature = c_feature;
                left.threshold = c_threshold;
                self.nodes.push(left);
                self.replant(branch_id, current_id, true);
                self.nodes[c_left.unwrap()].parent = Some(branch_id);
                self.nodes[c_right.unwrap()].parent = Some(branch_id);
                let right =
                    MondrianNode::new(Some(current_id), None, None, split_time, self.n_features, self.n_classes);
                self.nodes.push(right);
                self.nodes[current_id].left = Some(branch_id);
                self.nodes[current_id].right = Some(leaf_id);
            } else {
                let mut right =
                    MondrianNode::new(Some(current_id), c_left, c_right, split_time, self.n_features, self.n_classes);
                right.feature = c_feature;
                right.threshold = c_threshold;
                self.nodes.push(right);
                self.replant(branch_id, current_id, true);
                self.nodes[c_left.unwrap()].parent = Some(branch_id);
                self.nodes[c_right.unwrap()].parent = Some(branch_id);
                let left =
                    MondrianNode::new(Some(current_id), None, None, split_time, self.n_features, self.n_classes);
                self.nodes.push(left);
                self.nodes[current_id].left = Some(leaf_id);
                self.nodes[current_id].right = Some(branch_id);
            }
        } else {
            let left_id = self.nodes.len();
            let right_id = self.nodes.len() + 1;
            let left = MondrianNode::new(Some(current_id), None, None, split_time, self.n_features, self.n_classes);
            let right = MondrianNode::new(Some(current_id), None, None, split_time, self.n_features, self.n_classes);
            self.nodes.push(left);
            self.nodes.push(right);
            self.nodes[current_id].left = Some(left_id);
            self.nodes[current_id].right = Some(right_id);
            if is_right {
                self.replant(left_id, current_id, true);
            } else {
                self.replant(right_id, current_id, true);
            }
        }
        self.nodes[current_id].feature = Some(feature);
        self.nodes[current_id].threshold = Some(threshold);
    }

    pub fn go_downwards(&mut self, x: &[f64], y_index: usize) -> usize {
        if self.root.is_none() {
            let node_id = self.nodes.len();
            let node = MondrianNode::new(None, None, None, 0.0, self.n_features, self.n_classes);
            self.nodes.push(node);
            self.root = Some(node_id);
            self.update_downwards(node_id, y_index, x, false);
            node_id
        } else {
            let mut current_id = self.root.unwrap();

            loop {
                let sum = self.range_extensions(current_id, x);
                let split_time = self.compute_split_time(current_id, y_index, sum);

                if split_time > 0.0 {
                    let dist_idx = WeightedIndex::new(&self.distances)
                        .expect("Weights must be valid and sum to > 0");
                    let feature = dist_idx.sample(&mut self.rng);
                    let feature_value = x[feature];

                    let (range_min, range_max) = {
                        let current = &self.nodes[current_id];
                        (
                            current.min_range.as_ref().unwrap()[feature],
                            current.max_range.as_ref().unwrap()[feature],
                        )
                    };

                    let threshold = if feature_value > range_max {
                        Uniform::new(range_max, feature_value)
                            .unwrap()
                            .sample(&mut self.rng)
                    } else {
                        Uniform::new(feature_value, range_min)
                            .unwrap()
                            .sample(&mut self.rng)
                    };

                    let is_right = x[feature] > range_max;

                    self.split(current_id, split_time, feature, threshold, is_right);

                    self.update_downwards(current_id, y_index, x, true);

                    current_id = if is_right {
                        self.nodes[current_id].right.unwrap()
                    } else {
                        self.nodes[current_id].left.unwrap()
                    };

                    self.update_downwards(current_id, y_index, x, false);
                    return current_id;
                } else {
                    self.update_downwards(current_id, y_index, x, true);
                    let (right_child, left_child, feature_idx, threshold_val) = {
                        let current = &self.nodes[current_id];
                        (
                            current.right,
                            current.left,
                            current.feature,
                            current.threshold,
                        )
                    };

                    if right_child.is_none() {
                        return current_id;
                    } else {
                        if x[feature_idx.unwrap()] > threshold_val.unwrap() {
                            current_id = right_child.unwrap();
                        } else {
                            current_id = left_child.unwrap();
                        }
                    }
                }
            }
        }
    }

    pub fn update_weight_tree(&mut self, current_id: usize) {
        if let (Some(left), Some(right)) =
            (self.nodes[current_id].left, self.nodes[current_id].right)
        {
            let left_log = self.nodes[left].log_weight;
            let right_log = self.nodes[right].log_weight;
            let a = self.nodes[current_id].weight;
            let b = left_log + right_log;
            self.nodes[current_id].log_weight = if a > b {
                a + ((1.0 + (b - a).exp()) / 2.0).ln()
            } else {
                b + ((1.0 + (a - b).exp()) / 2.0).ln()
            };
        } else {
            self.nodes[current_id].log_weight = self.nodes[current_id].weight;
        }
    }

    pub fn go_upwards(&mut self, current_id: usize) {
        let mut current = current_id;
        loop {
            self.update_weight_tree(current);
            if self.nodes[current].parent.is_none() {
                break;
            } else {
                current = self.nodes[current].parent.unwrap();
            }
        }
    }

    pub fn learn_one(&mut self, x: &[f64], y: &Option<usize>) {
        let y_val = y.unwrap();
        let leaf = self.go_downwards(x, y_val);
        self.go_upwards(leaf);
    }

    pub fn traverse(&self, x: &[f64]) -> (usize, usize) {
        let mut current_id = self.root.unwrap();
        let mut depth = 0;

        loop {
            let current = &self.nodes[current_id];
            if current.left.is_none() {
                return (current_id, depth);
            } else {
                depth += 1;
                if x[current.feature.unwrap()] > current.threshold.unwrap() {
                    current_id = current.right.unwrap();
                } else {
                    current_id = current.left.unwrap();
                }
            }
        }
    }

    pub fn predict_prob_one(&self, x: &[f64]) -> (Vec<f64>, usize) {
        let mut scores: Vec<f64> = Vec::new();
        if self.root.is_none() {
            return (scores, 0);
        }
        let (leaf, depth) = self.traverse(x);
        let mut current_id = leaf;
        loop {
            if self.nodes[current_id].left.is_none() {
                scores = self.predict_one(current_id);
            } else {
                let w = (self.nodes[current_id].weight - self.nodes[current_id].log_weight).exp();
                let pred = self.predict_one(current_id);
                for i in 0..self.n_classes {
                    scores[i] = 0.5 * w * pred[i] + (1.0 - 0.5 * w) * scores[i];
                }
            }
            if self.nodes[current_id].parent.is_none() {
                break;
            } else {
                current_id = self.nodes[current_id].parent.unwrap();
            }
        }
        // let sum: f64 = scores.iter().sum();
        // if sum > 0.0 {
        //     for score in scores.iter_mut() {
        //         *score /= sum;
        //     }
        // }
        (scores, depth)
        // }
        // fn onelearn_partial_fit(&mut self, x: &[f64], y: &Option<usize>) {
        //     let leaf = self.onelearn_go_downwards(x, y.unwrap());
        //     self.onelearn.go_upwards(leaf);
        // }
        //
        // fn onelearn_go_downwards(&mut self, x: &[f64], y: usize) -> usize {
        //     if self.root.is_none() {
        //         onelearn_update_downwards(x, y, false);
        //         return 0;
        //     }
        //     else {
        //         let mut current_id = self.root.unwrap();
        //         loop {
        //             let split_time = onelearn_compute_split_time(current_id)
        //         }
        //     }
        // }
    }
}
