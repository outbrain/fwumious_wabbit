use rustc_hash::FxHashMap;
use crate::model_instance;
use crate::feature_buffer;


pub fn rehash_sorted_counts (mi: &mut model_instance::ModelInstance, fbt: &mut feature_buffer::FeatureBufferTranslator){
    /// A method that in-place stores frequent hashes that are used during translation
    
    println!("Creating the hash index space based on {} listings.", mi.warmup_listing_count);

    // some groundwork to estimate how many allocations we can afford
    let embedding_slot: u32 = (mi.ffm_fields.len() as u32) * (mi.ffm_k as u32);
    let mut tmp_hash_vec: Vec::<(&u32, &u32)> = mi.freq_hash.iter().collect();
    let max_allowed_slot_size = (1 << mi.ffm_bit_precision) - 1;

    // identify frequent hashes
    tmp_hash_vec.sort_by(|a, b| b.1.cmp(&a.1));
    let mut final_hash: FxHashMap<u32, u32> = FxHashMap::default();
    let mut tmp_location_lower_bound = 0;

    // re-hash -> this is part of mi eventually (in place so we're not wasting memory)
    for hash_name in tmp_hash_vec.iter(){
	if (tmp_location_lower_bound + embedding_slot) < max_allowed_slot_size  && *hash_name.1 > 1 {
	    final_hash.insert(*hash_name.0, tmp_location_lower_bound);
	    tmp_location_lower_bound += embedding_slot;
	} else {
	    break;
	}
    }
    
    // in-place switch the internal hash + lock it in
    println!("Storing indices for {:?} most frequent hashes, tiling {:?}% of entire hash space.", final_hash.len(), (100.0 * (embedding_slot as f32 * final_hash.len() as f32) / max_allowed_slot_size as f32));
    mi.freq_hash = final_hash;
    mi.freq_hash_rehashed_already = true;
    fbt.max_freq_rehash();

}
