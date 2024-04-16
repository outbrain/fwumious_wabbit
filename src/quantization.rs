use half::f16;
use std::io;

const BY_X: usize = 2;
const NUM_BUCKETS: f32 = 65025.0;
const CRITICAL_WEIGHT_BOUND: f32 = 10.0; // naive detection of really bad weights, this should never get to prod.
const MEAN_SAMPLING_RATIO: usize = 10;
const MIN_PREC: f32 = 10_000.0;
const MAX_PREC: f32 = 10_000.0;

#[derive(Debug)]
struct WeightStat {
    min: f32,
    max: f32,
    mean: f32,
}

fn emit_weight_statistics(weights: &[f32]) -> WeightStat {
    let mut min_weight = weights[0];
    let mut max_weight = weights[0];
    let mut mean_weight = 0.0;
    let mut weight_counter = 0;

    for (enx, &weight) in weights.iter().enumerate() {
        max_weight = max_weight.max(weight);
        min_weight = min_weight.min(weight);

        if enx % MEAN_SAMPLING_RATIO == 0 {
            weight_counter += 1;
            mean_weight += weight;
        }
    }

    WeightStat {
        min: (min_weight * MIN_PREC).round() / MIN_PREC,
        max: (max_weight * MAX_PREC).round() / MAX_PREC,
        mean: mean_weight / weight_counter as f32,
    }
}

pub fn quantize_ffm_weights(weights: &[f32]) -> Vec<[u8; BY_X]> {
    let weight_statistics = emit_weight_statistics(weights);
    let weight_increment = (weight_statistics.max - weight_statistics.min) / NUM_BUCKETS;

    if weight_statistics.mean.abs() > CRITICAL_WEIGHT_BOUND {
        log::warn!("Identified a very skewed weight distribution indicating exploded weights, not serving that! Mean weight value: {}", weight_statistics.mean);
    }

    log::info!(
        "Weight values; min: {}, max: {}, mean: {}",
        weight_statistics.min,
        weight_statistics.max,
        weight_statistics.mean
    );

    let weight_increment_bytes = weight_increment.to_le_bytes();
    let min_val_bytes = weight_statistics.min.to_le_bytes();

    let mut v = Vec::<[u8; BY_X]>::with_capacity(weights.len() + 4);

    // Bytes are stored as pairs
    v.push([weight_increment_bytes[0], weight_increment_bytes[1]]);
    v.push([weight_increment_bytes[2], weight_increment_bytes[3]]);
    v.push([min_val_bytes[0], min_val_bytes[1]]);
    v.push([min_val_bytes[2], min_val_bytes[3]]);

    for &weight in weights {
        let weight_interval = ((weight - weight_statistics.min) / weight_increment).round();
        v.push(f16::to_le_bytes(f16::from_f32(weight_interval)));
    }

    assert_eq!(v.len() - 4, weights.len());

    v
}

pub fn dequantize_ffm_weights(
    input_bufreader: &mut dyn io::Read,
    reference_weights: &mut Vec<f32>,
) {
    let mut header: [u8; 8] = [0; 8];
    input_bufreader.read_exact(&mut header).unwrap();

    let weight_increment = f32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let weight_min = f32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    let mut weight_bytes: [u8; 2] = [0; 2];

    for weight_index in 0..reference_weights.len() {
        input_bufreader.read_exact(&mut weight_bytes).unwrap();

        let weight_interval = f16::from_le_bytes(weight_bytes);
        let final_weight = weight_min + weight_interval.to_f32() * weight_increment;
        reference_weights[weight_index] = final_weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_statistics() {
        let some_random_float_weights = [0.51, 0.12, 0.11, 0.1232, 0.6123, 0.23];
        let out_struct = emit_weight_statistics(&some_random_float_weights);
        assert_eq!(out_struct.mean, 0.51);
        assert_eq!(out_struct.max, 0.6123);
        assert_eq!(out_struct.min, 0.11);
    }

    #[test]
    fn test_quantize() {
        let some_random_float_weights = vec![0.51, 0.12, 0.11, 0.1232, 0.6123, 0.23];
        let output_weights = quantize_ffm_weights(&some_random_float_weights);
        assert_eq!(output_weights.len(), 10);
    }

    #[test]
    fn test_dequantize() {
        let mut reference_weights = vec![0.51, 0.12, 0.11, 0.1232, 0.6123, 0.23];
        let old_reference_weights = reference_weights.clone();
        let quantized_representation = quantize_ffm_weights(&reference_weights);
        let mut all_bytes: Vec<u8> = Vec::new();
        for el in quantized_representation {
            all_bytes.push(el[0]);
            all_bytes.push(el[1]);
        }
        let mut contents = io::Cursor::new(all_bytes);
        dequantize_ffm_weights(&mut contents, &mut reference_weights);

        let matching = old_reference_weights
            .iter()
            .zip(&reference_weights)
            .filter(|&(a, b)| a == b)
            .count();

        assert_ne!(matching, 0);

        let allowed_eps = 0.0001;
        let mut all_diffs = 0.0;
        for it in old_reference_weights.iter().zip(reference_weights.iter()) {
            let (old, new) = it;
            all_diffs += (old - new).abs();
        }
        assert!(all_diffs < allowed_eps);
    }

    #[test]
    fn test_large_values() {
        let weights = vec![-1e9, 1e9];
        let quantized = quantize_ffm_weights(&weights);
        let mut buffer = io::Cursor::new(quantized.into_iter().flatten().collect::<Vec<_>>());
        let mut dequantized = vec![0.0; weights.len()];
        dequantize_ffm_weights(&mut buffer, &mut dequantized);
        for (w, dw) in weights.iter().zip(&dequantized) {
            assert!(
                (w - dw).abs() / w.abs() < 0.1,
                "Relative error is too large"
            );
        }
    }

    #[test]
    #[ignore]
    fn test_performance() {
        let weights: Vec<f32> = (0..10_000_000).map(|x| x as f32).collect();
        let now = std::time::Instant::now();
        let quantized = quantize_ffm_weights(&weights);
        assert!(now.elapsed().as_millis() < 300);

        let mut buffer = io::Cursor::new(quantized.into_iter().flatten().collect::<Vec<_>>());
        let mut dequantized = vec![0.0; weights.len()];
        let now = std::time::Instant::now();
        dequantize_ffm_weights(&mut buffer, &mut dequantized);
        assert!(now.elapsed().as_millis() < 300);
    }
}
