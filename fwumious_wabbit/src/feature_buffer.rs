use crate::model_instance;
use crate::parser;

const ONE:u32 = 1065353216;      // this is 1.0 float -> u32
const VOWPAL_FNV_PRIME:u32 = 16777619;	// vowpal magic number
const CONSTANT_NAMESPACE:usize = 128;
const CONSTANT_HASH:u32 = 11650396;


const LR_BUFFER_LEN:usize = 1024;    // this is highly unsafe...

#[derive(Clone)]
pub struct FeatureBuffer {
    pub lr_buffer: Vec<u32>,
    pub ffm_buffers: Vec<Vec<u32>>
}

pub struct FeatureBufferTranslator {
    model_instance: model_instance::ModelInstance,
    hashes_vec_in: Vec<u32>,
    hashes_vec_out: Vec<u32>,
    pub feature_buffer: FeatureBuffer,
}

impl FeatureBufferTranslator {
    pub fn new(model_instance: &model_instance::ModelInstance) -> FeatureBufferTranslator {
        let mut fb = FeatureBuffer {
            lr_buffer: Vec::new(),
            ffm_buffers: Vec::new(),
        };
        fb.lr_buffer.resize(LR_BUFFER_LEN, 0);
        let mut fbt = FeatureBufferTranslator{
                            model_instance: model_instance.clone(),
                            hashes_vec_in : Vec::with_capacity(100),
                            hashes_vec_out : Vec::with_capacity(100),
                            feature_buffer: fb,
        };
        fbt
    }
    
    pub fn print(&self) -> () {
        println!("item out {:?}", self.feature_buffer.lr_buffer);
    }
    
    
    pub fn translate_vowpal(&mut self, record_buffer: &[u32]) -> () {
        unsafe {
        let lr_buffer = &mut self.feature_buffer.lr_buffer;
        *lr_buffer.get_unchecked_mut(0) = record_buffer[1];  // copy label
        let mut output_len:usize = 1;
        let mut hashes_vec_in : &mut Vec<u32> = &mut self.hashes_vec_in;
        let mut hashes_vec_out : &mut Vec<u32> = &mut self.hashes_vec_out;
        for feature_combo_desc in &self.model_instance.feature_combo_descs {
            let feature_combo_weight_u32 = (feature_combo_desc.weight).to_bits();
            hashes_vec_in.truncate(0);
            hashes_vec_out.truncate(0);
            // we unroll first iteration of the loop and optimize
            let num_namespaces:usize = feature_combo_desc.feature_indices.len() ;
            {
                let feature_index_offset = *feature_combo_desc.feature_indices.get_unchecked(0) + 2;
                let namespace_desc = *record_buffer.get_unchecked(feature_index_offset);
                // We special case a single feature (common occurance)
                if num_namespaces == 1 {
                    if (namespace_desc & parser::IS_NOT_SINGLE_MASK) != 0 {
                        let start = ((namespace_desc >> 16) & 0x7fff) as usize;
                        let end =  (namespace_desc & 0xffff) as usize;
                        for hash_offset in start..end {
                            *lr_buffer.get_unchecked_mut(output_len) = *record_buffer.get_unchecked(hash_offset)  & self.model_instance.hash_mask;
                            *lr_buffer.get_unchecked_mut(output_len + 1) = feature_combo_weight_u32;
                            output_len += 2
                        }
                    } else {
                            *lr_buffer.get_unchecked_mut(output_len) = namespace_desc  & self.model_instance.hash_mask;
                            *lr_buffer.get_unchecked_mut(output_len + 1) = feature_combo_weight_u32;
                            output_len += 2
                    }
                    
                    continue
                }
                if (namespace_desc & parser::IS_NOT_SINGLE_MASK) != 0 {
                    let start = ((namespace_desc >> 16) & 0x7fff) as usize;
                    let end =  (namespace_desc & 0xffff) as usize;
                    for hash_offset in start..end {
                        hashes_vec_in.push(*record_buffer.get_unchecked(hash_offset));
                    }
                } else {
                    hashes_vec_in.push(namespace_desc);
                }
            }
            for feature_index in feature_combo_desc.feature_indices.get_unchecked(1 as usize .. num_namespaces) {
                let namespace_desc = *record_buffer.get_unchecked(*feature_index + 2);
                {
                    for old_hash in &(*hashes_vec_in) {
                        let half_hash = old_hash.overflowing_mul(VOWPAL_FNV_PRIME).0;
                        if (namespace_desc & parser::IS_NOT_SINGLE_MASK) != 0 {
                            let start = ((namespace_desc >> 16) & 0x7fff) as usize;
                            let end =  (namespace_desc & 0xffff) as usize;
                            for hash_offset in start..end {
                                hashes_vec_out.push(*record_buffer.get_unchecked(hash_offset) ^ half_hash);
                            }
                        } else {
                                let half_hash = old_hash.overflowing_mul(VOWPAL_FNV_PRIME).0;
                                hashes_vec_out.push(namespace_desc ^ half_hash);
                        }
                    }
                }
                hashes_vec_in.truncate(0);
                std::mem::swap(&mut hashes_vec_in, &mut hashes_vec_out);
            }
            for hash in &(*hashes_vec_in) {
                *lr_buffer.get_unchecked_mut(output_len) = *hash  & self.model_instance.hash_mask;
                *lr_buffer.get_unchecked_mut(output_len+1) = feature_combo_weight_u32;
                output_len += 2

            }
        }
        // add the constant
        if self.model_instance.add_constant_feature {
                *lr_buffer.get_unchecked_mut(output_len) = CONSTANT_HASH & self.model_instance.hash_mask;
                *lr_buffer.get_unchecked_mut(output_len+1) = ONE;
                output_len += 2
        }
        lr_buffer.set_len(output_len);


        // FFM loops have not been optimized yet
        if self.model_instance.ffm_k > 0 { 
            // currently we only support primitive features as namespaces, (from --lrqfa command)
            // this is for compatibility with vowpal
            // but in theory we could support also combo features
            let ffm_buffers = &mut self.feature_buffer.ffm_buffers;
            ffm_buffers.truncate(0);
            for ffm_field in &self.model_instance.ffm_fields {
                let mut field_hashes_buffer: Vec<u32>= Vec::with_capacity(100);
                for feature_index in ffm_field {
                    let namespace_desc = *record_buffer.get_unchecked(feature_index+2);
                    
                    if (namespace_desc & parser::IS_NOT_SINGLE_MASK) != 0 {
                        let start = ((namespace_desc >> 16) & 0x7fff) as usize;
                        let end =  (namespace_desc & 0xffff) as usize;
                        for hash_offset in start..end {
                            field_hashes_buffer.push(*record_buffer.get_unchecked(hash_offset));
                        }
                    } else {
                        field_hashes_buffer.push(namespace_desc);
                    }
    //                println!("A: {} {:?}", feature_index, field_hashes_buffer);
                }
                ffm_buffers.push(field_hashes_buffer);
            }
        }
        
        }
        
    }
}
    

mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    fn add_header(v2: Vec<u32>) -> Vec<u32> {
        let mut rr: Vec<u32> = vec![100, 1];
        rr.extend(v2);
        rr
    }
    
    fn nd(start: u32, end: u32) -> u32 {
        return (start << 16) + end;
    }


    #[test]
    fn test_constant() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = true;
        mi.feature_combo_descs.push(model_instance::FeatureComboDesc {
                                                        feature_indices: vec![0], 
                                                        weight: 1.0});
        
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![parser::NULL]); // no feature
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1, 116060, ONE]); // vw compatibility - no feature is no feature
    }
    
    
    #[test]
    fn test_single_once() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.feature_combo_descs.push(model_instance::FeatureComboDesc {
                                                        feature_indices: vec![0], 
                                                        weight: 1.0});
        
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![parser::NULL]); // no feature
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1]); // vw compatibility - no feature is no feature

        let rb = add_header(vec![0xfea]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1, 0xfea, ONE]);

        let rb = add_header(vec![parser::IS_NOT_SINGLE_MASK | nd(3,5), 0xfea, 0xfeb]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1, 0xfea, ONE, 0xfeb, ONE]);
    }

    #[test]
    fn test_single_twice() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.add_constant_feature = false;
        mi.feature_combo_descs.push(model_instance::FeatureComboDesc {
                                                        feature_indices: vec![0], 
                                                        weight: 1.0});
        mi.feature_combo_descs.push(model_instance::FeatureComboDesc {
                                                        feature_indices: vec![1], 
                                                        weight: 1.0});

        let mut fbt = FeatureBufferTranslator::new(&mi);

        let rb = add_header(vec![parser::NULL, parser::NULL]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1]);

        let rb = add_header(vec![0xfea, parser::NULL]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1, 0xfea, ONE]);

        let rb = add_header(vec![0xfea, 0xfeb]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1, 0xfea, ONE, 0xfeb, ONE]);

    }

    // for singles, vowpal and fwumious are the same
    // however for doubles theya are not
    #[test]
    fn test_double_vowpal() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.add_constant_feature = false;
        mi.feature_combo_descs.push(model_instance::FeatureComboDesc {
                                                        feature_indices: vec![0, 1], 
                                                        weight: 1.0});
        
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![parser::NULL]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1]);

        let rb = add_header(vec![123456789, parser::NULL]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1]);	// since the other feature is missing - VW compatibility says no feature is here

        let rb = add_header(vec![2988156968 & parser::MASK31, 2422381320 & parser::MASK31, parser::NULL]);
        fbt.translate_vowpal(&rb);
//        println!("out {}, out mod 2^24 {}", fbt.feature_buffer.lr_buffer[1], fbt.feature_buffer.lr_buffer[1] & ((1<<24)-1));
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1, 208368, ONE]);
        
    }
    
    #[test]
    fn test_single_with_weight_vowpal() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.add_constant_feature = false;
        mi.feature_combo_descs.push(model_instance::FeatureComboDesc {
                                                        feature_indices: vec![0], 
                                                        weight: 2.0});
        
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![0xfea]);
        fbt.translate_vowpal(&rb);
        let two = 2.0_f32.to_bits();

        assert_eq!(fbt.feature_buffer.lr_buffer, vec![1, 0xfea, two]);
    }


}

