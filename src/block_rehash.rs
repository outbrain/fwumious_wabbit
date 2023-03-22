use crate::feature_buffer;
use std::convert::TryInto;
use crate::model_instance;
use rustc_hash::FxHashMap;
//use growable_bloom_filter::GrowableBloom;


pub fn identify_frequent_stream(
    freq_hash: &mut FxHashMap<u32, u32>, cache_size: usize) {
    ///    A method that in-place stores frequent hashes that are used during translation
    ///    Tried Bloom filters, too slow for now.

    if freq_hash.len() % cache_size == 0  && freq_hash.len() > 1 {
        let mut to_remove_vec: Vec<u32> = Vec::new();
        let mut cleaned = 0;
        for hash in freq_hash.iter() {
            if *hash.1 == 1 {
                to_remove_vec.push(*hash.0);
            }
        }

        for el in to_remove_vec.iter() {
            freq_hash.remove(&el);
            cleaned += 1;
        }
    	log::info!("Cleaned {} rare hashes ..", cleaned);
    }
}

pub fn increment_common_hash(buffer: &[u32], freq_hash: &mut FxHashMap<u32, u32>) -> () {

    for hash_value_entry in buffer.iter() {
//	let hash_value: u32 = hash_value_entry.hash;
        freq_hash
            .entry(*hash_value_entry)
            .and_modify(|vx| *vx += 1)
            .or_insert(1);
    }
}

    
pub fn rehash(freq_hash: &FxHashMap<u32,u32>, buffer: &[u32], mut_vec_buff: &mut Vec<u32>) -> () {
    // A simple trick to make sure too rare values don't impact the optimization too much
    
    let mut tmp_array = Vec::from(buffer);
    for (index, hash_value_entry) in buffer.iter().enumerate() {
	if index > 1 {
            match freq_hash.get(&hash_value_entry) {
	    	Some(i) => {
		    if *i == 0  {
			tmp_array[index] = 1234 as u32;
		    }
	    	}
	    	_ => {}
            }
	}
    }

    *mut_vec_buff = tmp_array;
}
