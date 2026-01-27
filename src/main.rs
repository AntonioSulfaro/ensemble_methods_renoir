mod data_structures;
mod tree;

use data_structures::Instance;
use tree::VFDT;

fn main() {
    let mut forest_member = VFDT::new(20, 1e-7, 1e-4);

    // Mock stream
    for i in 0..200 {
        let inst = Instance {
            features: vec![if i % 2 == 0 { -1.0 } else { 1.0 }],
            label: Some(if i % 2 == 0 { 0 } else { 1 }),
        };
        forest_member.train(inst);
    }
}