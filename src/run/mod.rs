//! Run orchestration module — thin public API that composes helpers.

pub mod prepare;
pub mod process;
pub mod report;

use crate::config::config::Config;
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
    pub final_patch: f64,
    pub data: Vec<(usize, std::sync::Arc<crate::data::structures::Instance>)>,
}

/// Public entrypoint used by the binary.
pub fn execute(renoir_config: RuntimeConfig, config: Config) -> Result<()> {
    let ctx = prepare::prepare_run(renoir_config, config)?;
    let (ctx, total_time) = process::process_stream(ctx)?;
    report::report_results(&ctx, total_time)?;
    Ok(())
}
