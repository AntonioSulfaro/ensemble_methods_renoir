pub mod reader;
pub mod structures;

// re-export frequently used items so callers do `crate::data::read_arff` or `crate::data::Instance`
pub use reader::read_arff;
pub use structures::Instance;
