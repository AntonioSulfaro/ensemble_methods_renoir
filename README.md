## Introduction

This project implements streaming ensemble learning algorithms on [Renoir](https://github.com/deib-polimi/renoir), a Rust-based distributed data processing platform built on the dataflow paradigm. Renoir provides an ergonomic programming interface similar to Apache Flink.

We aim to support Streaming Random Patches (SRP) and Adaptive Random Forest (ARF), exploiting both multithreading within a machine and distributed execution across machines.

Our goal is to achieve high processing throughput while maintaining competitive predictive accuracy, and to evaluate scalability against other distributed stream processing frameworks.

## Quick Start

Run the following commands from the project root.

### Local Scalability Tests

1. Copy the contents of the desired JSON configuration file from `configs/` into `config.json`, then adjust the parameters as needed.

2. In `scalability_graph.py`, set `algo_type` to the algorithm you want to evaluate (`ht` or `amf`).

3. Run the scalability sweep:

   ```bash
   ./scripts/sweep_scalabnility.sh
   ```

   The script runs experiments with 1, 2, 4, 8, 16, and 32 threads, saves the results as CSV files in `results/scalability/`, and generates scalability plots.

### Remote Tests

1. Copy the contents of the desired JSON configuration file from `configs/` into `config.json`, then adjust the parameters as needed.

2. Configure the remote execution settings in `remote_config.toml`.

3. Build the executable:

   ```bash
   cargo build --profile profiling
   ```

4. Run the executable with the remote configuration:

   ```bash
   ./target/profiling/ensemble_methods_renoir -r ./remote_config.toml
   ```
