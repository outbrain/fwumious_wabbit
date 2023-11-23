
use std::slice;
use std::io;


pub fn quantize_ffm_weights_3by(weights: &[f32]) -> Vec<[u8; 3]> {
    // Quantize float-based weights to three most significant bytes

    let mut v = Vec::<[u8; 3]>::with_capacity(weights.len());
    for &weight in weights {
        let tmp_bytes = weight.to_be_bytes();
        let tmp_vec = [tmp_bytes[0], tmp_bytes[1], tmp_bytes[2]];
        v.push(tmp_vec);
    }
    debug_assert_eq!(v.len(), weights.len());
    v
}


pub fn dequantize_ffm_weights_3by(input_bufreader: &mut dyn io::Read, reference_weights: &mut Vec<f32>) {
    // This function overwrites existing weights with dequantized ones from the input buffer.

    unsafe {
        let buf_view: &mut [u8] = slice::from_raw_parts_mut(
	    reference_weights.as_mut_ptr() as *mut u8,
	    reference_weights.len() * 3,
        );
        let _ = input_bufreader.read_exact(buf_view);

        let tmp_weights: Vec<u8> = buf_view
            .chunks(3)
            .flat_map(|chunk| chunk.iter().chain(std::iter::once(&0u8)).cloned())
            .collect();

        for (chunk, float_ref) in tmp_weights.chunks(4).zip(reference_weights.iter_mut()) {

 	    let mut out_ary: [u8; 4] = [0; 4];
	    out_ary[0] = chunk[0];
	    out_ary[1] = chunk[1];
	    out_ary[2] = chunk[2];
	    
            *float_ref = f32::from_be_bytes(out_ary);
        }
    }
}
