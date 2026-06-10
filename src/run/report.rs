use crate::config::config::AlgorithmConfig;
use crate::run::ResultContext;
use anyhow::Context;
use csv::WriterBuilder;
use std::fs::OpenOptions;

/// Run post-processing: draw graph and append to master scalability log.
pub fn report_results(ctx: &ResultContext, total_time: f64) -> anyhow::Result<()> {
    // draw accuracy graph via Python script
    let py_bin = if cfg!(windows) { "py" } else { "python3" };
    let mut script_path = "scripts/".to_string();
    if ctx.is_remote {
        script_path = format!("ensemble_methods_renoir/{}", script_path);
    }

    let status = std::process::Command::new(py_bin)
        .arg(format!("{}accuracy_graph.py", script_path))
        .arg(ctx.threads.to_string())
        .arg(format!("{:.2}", total_time))
        .arg(&ctx.run_dir)
        .status()
        .context("Failed to execute Python script for drawing accuracy graph")?;

    if !status.success() {
        eprintln!("Python graph script returned non-zero exit code");
    }

    // append to master scalability CSV
    let log_name = match ctx.config.algorithm {
        AlgorithmConfig::Srp(_) | AlgorithmConfig::Arf(_) => "master_log_ht",
        AlgorithmConfig::Amf(_) => "master_log_amf",
    };

    let mut scalability_path = format!("results/scalability/{log_name}.csv");
    if ctx.is_remote {
        scalability_path = format!("ensemble_methods_renoir/{}", scalability_path);
    }
    let mut scalability_f = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(scalability_path)
        .context("opening scalability master log file")?;
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_writer(&mut scalability_f);

    wtr.serialize((
        &ctx.run_id,
        ctx.threads,
        format!("{:.2}", total_time),
        ctx.n_instances,
        ctx.n_instances as f64 / total_time,
        &ctx.config,
    ))
    .context("serializing scalability log row")?;
    wtr.flush().context("flushing scalability CSV writer")?;

    // compute average depth graph
    let status = std::process::Command::new(py_bin)
        .arg(format!("{}avg_depth_graph.py", script_path))
        .arg(&ctx.run_dir)
        .status()
        .context("Failed to execute Python script for drawing average tree depth graph")?;

    if !status.success() {
        eprintln!("Python graph script returned non-zero exit code");
    }

    Ok(())
}
