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

    log_detected_x86_features();
}

fn log_detected_x86_features() {
    let mut features: Vec<String> = Vec::new();
    if is_x86_feature_detected!("avx") {
        features.push("AVX".to_string());
    }

    if is_x86_feature_detected!("avx2") {
        features.push("AVX2".to_string());
    }

    if is_x86_feature_detected!("avx512f") {
        features.push("AVX512F".to_string());
    }

    if is_x86_feature_detected!("fma") {
        features.push("FMA".to_string());
    }

    if features.is_empty() {
        log::info!("No selected CPU features detected ..");
    } else {
        log::info!("Detected CPU features: {:?}", features.join(", "));
    }
}