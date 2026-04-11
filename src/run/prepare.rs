use super::RunContext;
use crate::config::config::AlgorithmConfig;
use anyhow::Context;
use chrono::Local;
use renoir::{RuntimeConfig, StreamContext};

/// Prepare the run context: StreamContext, run directory, dataset, final patch & feature subspaces.
pub fn prepare_run(
    renoir_config: RuntimeConfig,
    mut config: crate::config::config::Config,
) -> anyhow::Result<RunContext> {
    // Determine locality and thread count
    let (locality, threads) = match &renoir_config {
        RuntimeConfig::Local(local_cfg) => ("l", local_cfg.parallelism),
        RuntimeConfig::Remote(_) => (
            "r",
            std::thread::available_parallelism()
                .map(|q| q.get() as u64)
                .unwrap_or(1),
        ),
    };

    // timestamped run id & directories
    let timestamp = Local::now().format("%m%d_%H%M%S").to_string();
    let run_id = format!(
        "{}_{}_{}{}",
        timestamp,
        config
            .dataset
            .split('/')
            .last()
            .unwrap_or("")
            .replace("_", ""),
        locality,
        threads
    );
    let run_dir = format!("results/runs/{}/", run_id);
    let accuracy_csv_path = format!("{}accuracy.csv", run_dir);

    // Start renoir environment
    let env = StreamContext::new(renoir_config);

    // Read dataset
    let (data, n_classes, n_features, n_instances) =
        crate::data::reader::read_arff(format!("datasets/{}.arff", config.dataset).as_str());

    match &mut config.algorithm {
        AlgorithmConfig::Srp(ht) | AlgorithmConfig::Arf(ht) => {
            let patch_size = match ht.features_patch {
                Some(p) => (n_features as f64 * p).ceil() as usize,
                None => (n_features as f64).sqrt().floor() as usize + 1,
            };
            ht.features_patch = Some(patch_size as f64 / n_features as f64);
        }
        AlgorithmConfig::Amf(_) => {}
    }

    std::fs::create_dir_all(&run_dir)
        .with_context(|| format!("creating run directory '{}'", run_dir))?;
    let updated_config_json =
        serde_json::to_string_pretty(&config).context("Failed to serialize updated config")?;
    std::fs::write(format!("{}config.json", run_dir), updated_config_json)
        .context("writing updated config to run dir")?;

    Ok(RunContext {
        env: Some(env),
        run_dir,
        accuracy_csv_path,
        run_id,
        threads,
        config,
        n_instances,
        n_classes,
        n_features,
        data,
    })
}
