use crate::feature_buffer;
use crate::model_instance;
use rustc_hash::FxHashMap;

pub fn identify_frequent_stream(
    freq_hash: &mut FxHashMap<u32, u32>,
    fbt: &mut feature_buffer::FeatureBufferTranslator,
) {
    /// A method that in-place stores frequent hashes that are used during translation
    if freq_hash.len() % 5000000 == 0 {
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
        println!("Cleaned {} rare hashes ..", cleaned);
    }
}

pub fn rehash(freq_hash: &FxHashMap<u32, u32>, fbt: &mut feature_buffer::FeatureBufferTranslator) {
    /// A simple trick to make sure too rare values don't impact the optimization too much
    for hash_value_entry in fbt.feature_buffer.ffm_buffer.iter_mut() {
        match freq_hash.get(&hash_value_entry.hash) {
            Some(i) => {
                if *i > 1 {
                    hash_value_entry.hash = hash_value_entry.hash;
                }
            }
            _ => {
                hash_value_entry.hash = (hash_value_entry.hash / 10000) as u32;
            }
        }
    }
}
