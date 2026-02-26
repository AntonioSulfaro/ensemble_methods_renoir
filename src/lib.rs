pub mod adwin;
pub mod config;
pub mod data;
pub mod eval;
pub mod learners;
pub mod run;
pub mod tree;

pub use crate::config::config::Config;
pub use crate::data::reader::read_arff;
pub use crate::data::structures::Instance;
