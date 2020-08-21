use crate::model_instance;
use crate::parser;

const VOWPAL_FNV_PRIME:u32 = 16777619;	// vowpal magic number
const CONSTANT_NAMESPACE:usize = 128;
const CONSTANT_HASH:u32 = 11650396;


const LR_BUFFER_LEN:usize = 1024;    // this is highly unsafe...

#[derive(Clone, Debug, PartialEq)]
pub struct HashAndValue {
    pub hash: u32,
    pub value: f32
}

#[derive(Clone, Debug)]
pub struct FeatureBuffer {
    pub label: f32,
    pub example_importance: f32,
    pub lr_buffer: Vec<HashAndValue>,
    pub ffm_buffers: Vec<Vec<HashAndValue>>
}


#[derive(Clone)]
pub struct FeatureBufferTranslator {
    model_instance: model_instance::ModelInstance,
    // we don't want to keep allocating buffers
    hashes_vec_in: Vec<HashAndValue>,
    hashes_vec_out: Vec<HashAndValue>,
    pub feature_buffer: FeatureBuffer,
}

// A macro that takes care of decoding the individual feature - which can have two different encodings
// this simplifies a lot of the code, as it is used often
macro_rules! feature_reader {
    ( $record_buffer:ident, 
      $feature_index_offset:ident, 
      $hash_data:ident, 
      $hash_value:ident, 
      $bl:block  ) => {
        let namespace_desc = *$record_buffer.get_unchecked($feature_index_offset + parser::HEADER_LEN);
        if (namespace_desc & parser::IS_NOT_SINGLE_MASK) != 0 {
            let start = ((namespace_desc >> 16) & 0x7fff) as usize; 
            let end = (namespace_desc & 0xffff) as usize;
            for hash_offset in (start..end).step_by(2) {
                let $hash_data = *$record_buffer.get_unchecked(hash_offset);
                let $hash_value = f32::from_bits(*$record_buffer.get_unchecked(hash_offset+1));
                $bl
            }
        } else {
            let $hash_data = namespace_desc;
            let $hash_value: f32 = 1.0;
            $bl
        }
    };
}



impl FeatureBufferTranslator {
    pub fn new(model_instance: &model_instance::ModelInstance) -> FeatureBufferTranslator {
        let mut fb = FeatureBuffer {
            label: 0.0,
            example_importance: 1.0,
            lr_buffer: Vec::new(),
            ffm_buffers: Vec::new(),
        };
        fb.lr_buffer.resize(LR_BUFFER_LEN, HashAndValue {hash:0, value:0.0});
        // avoid doing any allocations in translate
        for ffm_field in &model_instance.ffm_fields {
            fb.ffm_buffers.push(Vec::new());
        }
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
        self.feature_buffer.label = record_buffer[parser::LABEL_OFFSET] as f32;  // copy label
        self.feature_buffer.example_importance = f32::from_bits(record_buffer[parser::EXAMPLE_IMPORTANCE_OFFSET]);    
        let mut output_len:usize = 0;
        let mut hashes_vec_in : &mut Vec<HashAndValue> = &mut self.hashes_vec_in;
        let mut hashes_vec_out : &mut Vec<HashAndValue> = &mut self.hashes_vec_out;
        for feature_combo_desc in &self.model_instance.feature_combo_descs {
            let feature_combo_weight = feature_combo_desc.weight;
            // we unroll first iteration of the loop and optimize
            let num_namespaces:usize = feature_combo_desc.feature_indices.len() ;
            let feature_index_offset = *feature_combo_desc.feature_indices.get_unchecked(0);
            // We special case a single feature (common occurance)
            if num_namespaces == 1 {
                feature_reader!(record_buffer, feature_index_offset, hash_data, hash_value, {
                    lr_buffer.get_unchecked_mut(output_len).hash = hash_data & self.model_instance.hash_mask;
                    lr_buffer.get_unchecked_mut(output_len).value = hash_value * feature_combo_weight;
                    output_len += 1
                });
                continue
            }
            hashes_vec_in.truncate(0);
            hashes_vec_out.truncate(0);
            feature_reader!(record_buffer, feature_index_offset, hash_data, hash_value, {
                    hashes_vec_in.push(HashAndValue {hash: hash_data, value:hash_value});
                });
            for feature_index in feature_combo_desc.feature_indices.get_unchecked(1 as usize .. num_namespaces) {
                for handv in &(*hashes_vec_in) {
                    let half_hash = handv.hash.overflowing_mul(VOWPAL_FNV_PRIME).0;
                    feature_reader!(record_buffer, feature_index, hash_data, hash_value, {
                        hashes_vec_out.push(HashAndValue{   hash: hash_data ^ half_hash,
                                                            value: handv.value * hash_value});
                    });
                }
                hashes_vec_in.truncate(0);
                std::mem::swap(&mut hashes_vec_in, &mut hashes_vec_out);
            }
            for handv in &(*hashes_vec_in) {
                lr_buffer.get_unchecked_mut(output_len).hash = handv.hash & self.model_instance.hash_mask;
                lr_buffer.get_unchecked_mut(output_len).value = handv.value * feature_combo_weight;
                output_len += 1;
            }
        }
        // add the constant
        if self.model_instance.add_constant_feature {
                lr_buffer.get_unchecked_mut(output_len).hash = CONSTANT_HASH & self.model_instance.hash_mask;
                lr_buffer.get_unchecked_mut(output_len).value = 1.0;
                output_len += 1;
        }
        lr_buffer.set_len(output_len);

        // FFM loops have not been optimized yet
        if self.model_instance.ffm_k > 0 { 
            // currently we only support primitive features as namespaces, (from --lrqfa command)
            // this is for compatibility with vowpal
            // but in theory we could support also combo features
            let ffm_buffers = &mut self.feature_buffer.ffm_buffers;
            for (field_n, ffm_field) in self.model_instance.ffm_fields.iter().enumerate() {
                let mut field_hashes_buffer = &mut ffm_buffers[field_n];
                field_hashes_buffer.truncate(0);
                for feature_index in ffm_field {
                    feature_reader!(record_buffer, feature_index, hash_data, hash_value, {
                            field_hashes_buffer.push(HashAndValue {hash: hash_data,
                                                                    value: hash_value});
                    });
                }
            }
        }
        
        }
        
    }
}
    

mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    fn add_header(v2: Vec<u32>) -> Vec<u32> {
        let mut rr: Vec<u32> = vec![100, 1, 1.0f32.to_bits()];
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
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![HashAndValue {hash:116060, value:1.0}]); // vw compatibility - no feature is no feature
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
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![]); // vw compatibility - no feature is no feature
        

        let rb = add_header(vec![0xfea]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![HashAndValue {hash:0xfea, value:1.0}]);

        let rb = add_header(vec![parser::IS_NOT_SINGLE_MASK | nd(4,8), 0xfea, 1.0f32.to_bits(), 0xfeb, 1.0f32.to_bits()]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![HashAndValue {hash:0xfea, value:1.0}, HashAndValue {hash:0xfeb, value:1.0}]);
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
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![]);

        let rb = add_header(vec![0xfea, parser::NULL]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![HashAndValue {hash:0xfea, value:1.0}]);

        let rb = add_header(vec![0xfea, 0xfeb]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![HashAndValue {hash:0xfea, value:1.0}, HashAndValue {hash:0xfeb, value:1.0}]);

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
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![]);

        let rb = add_header(vec![123456789, parser::NULL]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![]);	// since the other feature is missing - VW compatibility says no feature is here

        let rb = add_header(vec![2988156968 & parser::MASK31, 2422381320 & parser::MASK31, parser::NULL]);
        fbt.translate_vowpal(&rb);
//        println!("out {}, out mod 2^24 {}", fbt.feature_buffer.lr_buffer[1], fbt.feature_buffer.lr_buffer[1] & ((1<<24)-1));
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![HashAndValue {hash: 208368, value:1.0}]);
        
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
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![HashAndValue {hash: 0xfea, value:2.0}]);
    }
    
    #[test]
    fn test_ffm_empty() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.add_constant_feature = false;
        mi.ffm_fields.push(vec![]);   // single field, empty
        mi.ffm_k = 1;
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![0xfea]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.ffm_buffers[0], vec![]);
    }

    #[test]
    fn test_ffm_one() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.add_constant_feature = false;
        mi.ffm_fields.push(vec![0]);   // single feature in a single fields 
        mi.ffm_k = 1;
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![0xfea]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.ffm_buffers[0], vec![HashAndValue{hash: 0xfea, value: 1.0}]);
    }

    #[test]
    fn test_ffm_two_fields() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.add_constant_feature = false;
        mi.ffm_fields.push(vec![0]);   //  single namespace in a field
        mi.ffm_fields.push(vec![0,1]);   // two namespaces in a field
        mi.ffm_k = 1;
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![parser::IS_NOT_SINGLE_MASK | nd(5,9), 0xfec, 0xfea, 2.0f32.to_bits(), 0xfeb, 3.0f32.to_bits()]);
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.ffm_buffers[0], vec![ HashAndValue{hash: 0xfea, value: 2.0}, 
                                                            HashAndValue{hash: 0xfeb, value: 3.0}]);
        assert_eq!(fbt.feature_buffer.ffm_buffers[1], vec![ HashAndValue{hash: 0xfea, value: 2.0}, 
                                                            HashAndValue{hash: 0xfeb, value: 3.0}, 
                                                            HashAndValue{hash: 0xfec, value: 1.0}]);
    }

    #[test]
    fn test_example_importance() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.feature_combo_descs.push(model_instance::FeatureComboDesc {
                                                        feature_indices: vec![0], 
                                                        weight: 1.0});
        
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let mut rb = add_header(vec![parser::NULL]); // no feature
        fbt.translate_vowpal(&rb);
        assert_eq!(fbt.feature_buffer.example_importance, 1.0); // Did example importance get parsed correctly
    }

}

