//! Run orchestration module — thin public API that composes helpers.

pub mod prepare;
pub mod process;
pub mod report;

use crate::config::config::Config;
use crate::data::reader::ArffStreamIter;
use anyhow::Result;
use renoir::RuntimeConfig;

/// Context shared across run helpers.
pub struct RunContext {
    pub env: Option<renoir::StreamContext>,
    pub run_dir: String,
    pub accuracy_csv_path: String,
    pub run_id: String,
    pub threads: u64,
    pub config: Config,
    pub n_instances: usize,
    pub n_classes: usize,
    pub n_features: usize,
    pub patch_size: usize,
    pub data: ArffStreamIter,
}

pub struct ResultContext {
    pub run_dir: String,
    pub run_id: String,
    pub threads: u64,
    pub n_instances: usize,
    pub config: Config,
}

/// Public entrypoint used by the binary.
pub fn execute(renoir_config: RuntimeConfig, config: Config) -> Result<()> {
    let ctx = prepare::prepare_run(renoir_config, config)?;
    let (ctx, total_time) = process::process_stream(ctx)?;
    report::report_results(&ctx, total_time)?;
    Ok(())
}
