use std::io;
use half::f16;

const BY_X: usize = 2;
const NUM_BUCKETS: f32 = 65025.0;
//const NUM_BUCKETS: f32 = 255.0;
const CRITICAL_WEIGHT_BOUND: f32 = 10.0; // naive detection of really bad weights, this should never get to prod.
const MEAN_SAMPLING_RATIO: usize = 10;
const MIN_PREC: f32 = 10_000.0;
const MAX_PREC: f32 = 10_000.0;


#[derive(Debug)]
struct WeightStat {
    min: f32,
    max: f32,
    mean: f32,
    std: f32	
}


fn emit_weight_statistics(weights: &[f32]) -> WeightStat {

    let mut min_weight = weights[0];
    let mut max_weight = weights[0];
    
    let mut mean_weight = 0.0;
    let mut weight_counter = 0;
    
    let mut square_sums = 0.0;
    let mut sum_squares = 0.0;
    let mut std_est = 0.0;

    for (enx, &weight) in weights.iter().enumerate() {
	max_weight = max_weight.max(weight);
	min_weight = min_weight.min(weight);

	if enx % MEAN_SAMPLING_RATIO == 0 {
	    weight_counter += 1;
	    mean_weight += weight;
	    square_sums += weight.powf(2.0);
	    sum_squares += weight;
	    std_est = (square_sums / weight_counter as f32 - (sum_squares / weight_counter as f32).powf(2.0)).sqrt(); // can be done a bit better with Knuth's formula, slower tho
	}
    }

    WeightStat {
	min: (min_weight * MIN_PREC).round() / MIN_PREC,
	max: (max_weight * MAX_PREC).round() / MAX_PREC,
	mean: mean_weight / weight_counter as f32,
	std: std_est,
    }
}


fn non_uniform_binner(weight_min: f32, weight_max: f32, weight_mean: f32, weight_std: f32) -> Vec<f32> {

    // lower and upper bounds need to be informed
    let focus_buckets_min = weight_mean - 0.1 * weight_min.abs();
    let focus_buckets_max = weight_mean + 0.1 * weight_max.abs();

    let mut bucket_dist_focus = (NUM_BUCKETS * 0.8).round();
    let mut bucket_dist_remainder = NUM_BUCKETS - bucket_dist_focus;

    // must be odd
    if bucket_dist_remainder as usize % 2 == 0 {
	bucket_dist_remainder = bucket_dist_remainder / 2.0;
    } else {
	bucket_dist_remainder = (bucket_dist_remainder - 1.0) / 2.0;
	bucket_dist_focus += 1.0;
    }

    // two side intervals + focus one
    let interval_first = (focus_buckets_min - weight_min) / bucket_dist_remainder;
    let interval_second = (focus_buckets_max - focus_buckets_min) / bucket_dist_focus;
    let interval_third = (weight_max - focus_buckets_max) / bucket_dist_remainder;

    let mut bucket_space = Vec::<f32>::with_capacity(NUM_BUCKETS as usize);

    let mut current_weight_value = weight_min;
    for _ in 0..bucket_dist_remainder as usize {
	current_weight_value += interval_first;
	bucket_space.push(current_weight_value);
    }
	
    for _ in 0..bucket_dist_focus as usize {
	current_weight_value += interval_second;
	bucket_space.push(current_weight_value);
    }
    
    for _ in 0..bucket_dist_remainder as usize {
	current_weight_value += interval_third;
	bucket_space.push(current_weight_value);
    }

    bucket_space
    
}

fn identify_bucket(weight: f32, bucket_space: &Vec<f32>) -> f32 {
    bucket_space.partition_point(|x| x < &weight) as f32
}

pub fn quantize_ffm_weights(weights: &[f32]) -> Vec<[u8; BY_X]> {
    let weight_statistics = emit_weight_statistics(weights);
    let weight_increment = (weight_statistics.max - weight_statistics.min) / NUM_BUCKETS;

    let quantized = non_uniform_binner(weight_statistics.min, weight_statistics.max, weight_statistics.mean, weight_statistics.std);

    if weight_statistics.mean.abs() > CRITICAL_WEIGHT_BOUND {
	log::warn!("Identified a very skewed weight distribution indicating exploded weights, not serving that! Mean weight value: {}", weight_statistics.mean);
    }

    log::info!("Weight values; min: {}, max: {}, mean: {}, std: {}", weight_statistics.min, weight_statistics.max, weight_statistics.mean, weight_statistics.std);

    let weight_increment_bytes = weight_increment.to_le_bytes();
    let min_val_bytes = weight_statistics.min.to_le_bytes();
    let max_val_bytes = weight_statistics.max.to_le_bytes();
    let mean_val_bytes = weight_statistics.mean.to_le_bytes();
    let std_val_bytes = weight_statistics.mean.to_le_bytes();

    let mut v = Vec::<[u8; BY_X]>::with_capacity(weights.len() + 10);

    v.push([weight_increment_bytes[0], weight_increment_bytes[1]]);
    v.push([weight_increment_bytes[2], weight_increment_bytes[3]]);
    
    v.push([min_val_bytes[0], min_val_bytes[1]]);
    v.push([min_val_bytes[2], min_val_bytes[3]]);
    
    v.push([max_val_bytes[0], max_val_bytes[1]]);
    v.push([max_val_bytes[2], max_val_bytes[3]]);

    v.push([mean_val_bytes[0], mean_val_bytes[1]]);
    v.push([mean_val_bytes[2], mean_val_bytes[3]]);

    v.push([std_val_bytes[0], std_val_bytes[1]]);
    v.push([std_val_bytes[2], std_val_bytes[3]]);   
    
    for &weight in weights {
	let weight_interval = identify_bucket(weight, &quantized);
//	let weight_interval = ((weight - weight_statistics.min) / weight_increment).round();
	v.push(f16::to_le_bytes(f16::from_f32(weight_interval)));
    }

    assert_eq!(v.len() - 10, weights.len());
    
    v
}

pub fn dequantize_ffm_weights(
    input_bufreader: &mut dyn io::Read,
    reference_weights: &mut Vec<f32>,
) {
    let mut header: [u8; 20] = [0; 20];
    input_bufreader.read_exact(&mut header).unwrap();

    let weight_increment = f32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let weight_min = f32::from_le_bytes([header[4], header[5], header[6], header[7]]);

    let weight_max = f32::from_le_bytes([header[8], header[9], header[10], header[11]]);

    let weight_mean = f32::from_le_bytes([header[12], header[13], header[14], header[15]]);

    let weight_std = f32::from_le_bytes([header[16], header[17], header[18], header[19]]);

    let quantized = non_uniform_binner(weight_min, weight_max, weight_mean, weight_std);
    
    let mut weight_bytes: [u8; 2] = [0; 2];

    for weight_index in 0..reference_weights.len() {
        input_bufreader.read_exact(&mut weight_bytes).unwrap();

        let weight_interval = f16::from_le_bytes(weight_bytes);
	let final_weight = quantized[weight_interval.to_f32() as usize];
//        let final_weight = weight_min + weight_interval.to_f32() * weight_increment;
        reference_weights[weight_index] = final_weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_statistics(){
	let some_random_float_weights = [0.51, 0.12, 0.11, 0.1232, 0.6123, 0.23];
	let out_struct = emit_weight_statistics(&some_random_float_weights);
	assert_eq!(out_struct.mean, 0.51);
	assert_eq!(out_struct.max, 0.6123);
	assert_eq!(out_struct.min, 0.11);
    }
    
    // #[test]
    // fn test_quantize() {
    //     let some_random_float_weights = vec![0.51, 0.12, 0.11, 0.1232, 0.6123, 0.23];
    //     let output_weights = quantize_ffm_weights(&some_random_float_weights);
    //     assert_eq!(output_weights.len(), 10);
    // }

    // #[test]
    // fn test_dequantize() {
    // 	let mut reference_weights = vec![0.51, 0.12, 0.11, 0.1232, 0.6123, 0.23];
    // 	let old_reference_weights = reference_weights.clone();
    // 	let quantized_representation = quantize_ffm_weights(&reference_weights);
    // 	let mut all_bytes: Vec<u8> = Vec::new();
    // 	for el in quantized_representation {
    // 	    all_bytes.push(el[0]);
    // 	    all_bytes.push(el[1]);
    // 	}
    // 	let mut contents = io::Cursor::new(all_bytes);
    // 	dequantize_ffm_weights(&mut contents, &mut reference_weights);

    // 	let matching = old_reference_weights.iter().zip(&reference_weights).filter(|&(a, b)| a == b).count();
	
    // 	assert_ne!(matching, 0);

    // 	let allowed_eps = 0.0001;
    // 	let mut all_diffs = 0.0;
    // 	for it in old_reference_weights.iter().zip(reference_weights.iter()) {
    // 	    let (old, new) = it;
    // 	    all_diffs += (old - new).abs();	    
    // 	}
    // 	assert!(all_diffs < allowed_eps);
    // }


    // #[test]
    // fn test_large_values() {
    // 	let weights = vec![-1e9, 1e9];
    // 	let quantized = quantize_ffm_weights(&weights);
    // 	let mut buffer = io::Cursor::new(quantized.into_iter().flatten().collect::<Vec<_>>());
    // 	let mut dequantized = vec![0.0; weights.len()];
    // 	dequantize_ffm_weights(&mut buffer, &mut dequantized);
    // 	for (w, dw) in weights.iter().zip(&dequantized) {
    //         assert!((w - dw).abs() / w.abs() < 0.1, "Relative error is too large");
    // 	}
    // }


    // #[test]
    // fn test_performance() {
    // 	let weights: Vec<f32> = (0..10_000_000).map(|x| x as f32).collect();
    // 	let now = std::time::Instant::now();
    // 	let quantized = quantize_ffm_weights(&weights);
    // 	assert!(now.elapsed().as_millis() < 300);
	
    // 	let mut buffer = io::Cursor::new(quantized.into_iter().flatten().collect::<Vec<_>>());
    // 	let mut dequantized = vec![0.0; weights.len()];
    // 	let now = std::time::Instant::now();
    // 	dequantize_ffm_weights(&mut buffer, &mut dequantized);
    // 	assert!(now.elapsed().as_millis() < 300);
    // }

    // #[test]
    // fn test_nonuniform() {
    // 	let quantized = non_uniform_binner(-0.42, 0.82, 0.0, 0.02);
    // 	assert!(quantized.len() == NUM_BUCKETS as usize);

    // 	let bucket = identify_bucket(-0.35, &quantized);
    // 	assert!(bucket == 1137.0);

    // 	let bucket = identify_bucket(0.23, &quantized);
    // 	assert!(bucket == 60229.0);

    // 	let now = std::time::Instant::now();

    // 	// these calls need to be ultra fast
    // 	for _ in 0..1_000_000 {
    // 	    identify_bucket(0.23, &quantized);
    // 	}
    // 	assert!(now.elapsed().as_millis() < 300);
    // }    
}
