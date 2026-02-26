use anyhow::Result;
use ensemble_methods_renoir::config::config::Config;
use ensemble_methods_renoir::run;
use renoir::RuntimeConfig;

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<()> {
    let (renoir_config, _args) = RuntimeConfig::from_args();

    let config_str =
        std::fs::read_to_string("config.json").expect("Failed to read json configurations");
    let config: Config = serde_json::from_str(&config_str).expect("JSON was not well-formatted");

    run::execute(renoir_config, config)?;

    Ok(())
}
