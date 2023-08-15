extern crate log;
use env_logger::Builder;

pub fn initialize_logging_layer() {
    let mut builder = Builder::new();
    let log_level = std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    match log_level.to_lowercase().as_str() {
        "info" => builder.filter_level(log::LevelFilter::Info),
        "warn" => builder.filter_level(log::LevelFilter::Warn),
        "error" => builder.filter_level(log::LevelFilter::Error),
        "trace" => builder.filter_level(log::LevelFilter::Trace),
        "debug" => builder.filter_level(log::LevelFilter::Debug),
        "off" => builder.filter_level(log::LevelFilter::Off),
        _ => builder.filter_level(log::LevelFilter::Info),
    };

    if builder.try_init().is_ok() {
        log::info!("Initialized the logger ..")
    }
}
