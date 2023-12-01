use std::io;
use std::slice;
//use half::bf16;

const BY_X: usize = 1;


pub fn quantize_ffm_weights_3by(weights: &[f32]) -> Vec<[u8; BY_X]> {
    // Quantize float-based weights to three most significant bytes

    let mut v = Vec::<[u8; BY_X]>::with_capacity(weights.len());
    for &weight in weights {
        let tmp_bytes = (weight).to_le_bytes();
        let mut out_ary: [u8; BY_X] = [0; BY_X];
        for k in 0..BY_X {
            out_ary[k] = tmp_bytes[k];
        }
        v.push(out_ary);
    }
    debug_assert_eq!(v.len(), weights.len());
    v
}

pub fn dequantize_ffm_weights_3by(
    input_bufreader: &mut dyn io::Read,
    reference_weights: &mut Vec<f32>,
) {
    // This function overwrites existing weights with dequantized ones from the input buffer.

    unsafe {
        let buf_view: &mut [u8] = slice::from_raw_parts_mut(
            reference_weights.as_mut_ptr() as *mut u8,
            reference_weights.len() * BY_X,
        );
        let _ = input_bufreader.read_exact(buf_view);

        let mut out_ary: [u8; 4] = [0; 4];
        for (chunk, float_ref) in buf_view.chunks(BY_X).zip(reference_weights.iter_mut()) {
            for k in 0..BY_X {
                out_ary[k] = chunk[k];
            }
	    let weight = f32::from_le_bytes(out_ary);
//	    let weight = bf16::to_f32(bf16::from_be_bytes(out_ary));
            *float_ref = weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_2by() {
        let some_random_float_weights = vec![0.51, 0.12, 0.11, 0.1232, 0.6123, 0.23];
        let output_weights = quantize_ffm_weights_3by(&some_random_float_weights);
        assert_eq!(output_weights[3], [72]);
    }
}
