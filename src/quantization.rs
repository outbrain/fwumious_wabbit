use std::io;
use half::f16;

const BY_X: usize = 2;
const NUM_BUCKETS: f32 = 65025.0;
const CRITICAL_WEIGHT_BOUND: f32 = 10.0; // naive detection of really bad weights, this should never get to prod.
const MEAN_SAMPLING_RATIO: usize = 10;


#[derive(Debug)]
struct WeightStat {
    min: f32,
    max: f32,
    mean: f32
}


fn emit_weight_statistics(weights: &[f32]) -> WeightStat {
    // Bound estimator for quantization range

    let init_weight = weights[0];
    let mut min_weight = init_weight;
    let mut max_weight = init_weight;
    let mut mean_weight = 0.0;
    let mut weight_counter = 0;

    for (enx, weight) in weights.iter().enumerate() {

	if *weight > max_weight {
	    max_weight = *weight;
	}

	if *weight < min_weight {
	    min_weight = *weight;
	}

	if enx % MEAN_SAMPLING_RATIO == 0 {
	    weight_counter += 1;
	    mean_weight += *weight;
	}

    }

    log::info!("Weight values; min: {}, max: {}, mean: {}", min_weight, max_weight, mean_weight / weight_counter as f32);

    WeightStat{min: min_weight, max: max_weight, mean: mean_weight}
}


pub fn quantize_ffm_weights(weights: &[f32]) -> Vec<[u8; BY_X]> {
    // Quantize float-based weights to three most significant bytes
    // To be more precise in terms of representation of ranges, we extend the weight object with a "header" that contains two floats required for proper dequantization -- this is computed on-the-fly, works better

    
    let weight_statistics = emit_weight_statistics(weights);

    // Cheap, yet important check
    if weight_statistics.mean > CRITICAL_WEIGHT_BOUND || weight_statistics.mean < -CRITICAL_WEIGHT_BOUND {
	panic!("Identified a very skewed weight distribution indicating exploded weights, not serving that! Mean weight value: {}", weight_statistics.mean);
    }

    // Uniform distribution within the relevant interval
    let weight_increment = (weight_statistics.max - weight_statistics.min) / NUM_BUCKETS;
    let mut v = Vec::<[u8; BY_X]>::with_capacity(weights.len());
    
    // Increment needs to be stored
    let weight_increment_bytes = weight_increment.to_le_bytes();    
    let deq_header1 = [weight_increment_bytes[0], weight_increment_bytes[1]];
    let deq_header2 = [weight_increment_bytes[2], weight_increment_bytes[3]];
    v.push(deq_header1);
    v.push(deq_header2);

    // Minimal value needs to be stored
    let min_val_bytes = weight_statistics.min.to_le_bytes();
    let deq_header3 = [min_val_bytes[0], min_val_bytes[1]];
    let deq_header4 = [min_val_bytes[2], min_val_bytes[3]];
    v.push(deq_header3);
    v.push(deq_header4);

    for weight in weights {

	let weight_interval = ((*weight - weight_statistics.min) / weight_increment).round();
	let weight_interval_bytes = f16::to_le_bytes(f16::from_f32(weight_interval));
        v.push(weight_interval_bytes);
	
    }

    // This is done during reading, so fine as a sanity here.
    assert_eq!(v.len() - 4, weights.len());
    
    v
}

pub fn dequantize_ffm_weights(
    input_bufreader: &mut dyn io::Read,
    reference_weights: &mut Vec<f32>,
) {
    // This function overwrites existing weights with dequantized ones from the input buffer.

    let mut header: [u8; 8] = [0; 8];
    let _ = input_bufreader.read_exact(&mut header);

    let mut incr_vec: [u8; 4] = [0; 4];
    let mut min_vec: [u8; 4] = [0; 4];

    for j in 0..4 {
	incr_vec[j] = header[j];
	min_vec[j] = header[j + 4];
    }

    let weight_increment = f32::from_le_bytes(incr_vec);
    let weight_min = f32::from_le_bytes(min_vec);    
    let mut weight_bytes: [u8; 2] = [0; 2];

    // All set, dequantize in a stream
    for weight_index in 0..reference_weights.len(){
	let _ = input_bufreader.read_exact(&mut weight_bytes);
	let weight_interval = f16::from_le_bytes(weight_bytes);
	let weight_interval_f32: f32 = weight_interval.to_f32();
	let final_weight = weight_min + weight_interval_f32 * weight_increment;
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

	let matching = old_reference_weights.iter().zip(&reference_weights).filter(|&(a, b)| a == b).count();
	
	assert_ne!(matching, 0);

	let allowed_eps = 0.0001;
	let mut all_diffs = 0.0;
	for it in old_reference_weights.iter().zip(reference_weights.iter()) {
	    let (old, new) = it;
	    all_diffs += (old - new).abs();	    
	}
	assert!(all_diffs < allowed_eps);
    }
}
