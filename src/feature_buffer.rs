use crate::feature_transform_executor;
use crate::feature_transform_parser;
use crate::model_instance;
use crate::parser;
use crate::vwmap::{NamespaceFormat, NamespaceType};
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::sync::Arc;

const VOWPAL_FNV_PRIME: u32 = 16777619; // vowpal magic number
                                        //const CONSTANT_NAMESPACE:usize = 128;
const CONSTANT_HASH: u32 = 11650396;

#[derive(Clone, Debug, PartialEq)]
pub struct HashAndValue {
    pub hash: u32,
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HashAndValueAndSeq {
    pub hash: u32,
    pub value: f32,
    pub contra_field_index: u32,
}

#[derive(Clone, Debug)]
pub struct FeatureBuffer {
    pub label: f32,
    pub example_importance: f32,
    pub example_number: u64,
    pub lr_buffer: Vec<HashAndValue>,
    pub ffm_buffer: Vec<HashAndValueAndSeq>,
    pub ffm_fields_count: u32,

    // Maybe all these should be moved out and then having a composable
    pub audit_json: RefCell<Value>,
    pub audit_mode: bool,
    pub audit_aux_data: model_instance::AuditData,
    pub lr_buffer_audit: Vec<i32>, // Corresponding ids of feature combos from lr_buffer
    pub ffm_buffer_audit: Vec<u32>, // Corresponding ids of namespace indexes
}

#[derive(Clone)]
pub struct FeatureBufferTranslator {
    model_instance: model_instance::ModelInstance,
    // we don't want to keep allocating buffers
    hashes_vec_in: Vec<HashAndValue>,
    hashes_vec_out: Vec<HashAndValue>,
    pub feature_buffer: FeatureBuffer,
    pub lr_hash_mask: u32,
    pub ffm_hash_mask: u32,
    pub transform_executors: feature_transform_executor::TransformExecutors,
}

impl FeatureBuffer {
    pub fn new() -> FeatureBuffer {
        FeatureBuffer {
            label: 0.0,
            example_importance: 1.0,
            example_number: 0,
            lr_buffer: Vec::new(),
            ffm_buffer: Vec::new(),
            ffm_fields_count: 0,
            audit_json: RefCell::new(Value::Null),
            audit_mode: false,
            audit_aux_data: model_instance::default_audit_data(),
            lr_buffer_audit: Vec::new(),
            ffm_buffer_audit: Vec::new(),
        }
    }

    pub fn add_audit_json(&self, mut audit_json: Map<String, Value>) {
        let old_value = self.audit_json.replace(Value::Null);
        audit_json.insert("predcessor".to_string(), old_value);
        self.audit_json.replace(Value::Object(audit_json));
    }
    pub fn reset_audit_json(&self) {
        self.audit_json.replace(Value::Null);
    }
}

// A macro that takes care of decoding the individual feature - which can have two different encodings
// this simplifies a lot of the code, as it is used often
#[macro_export]
macro_rules! feature_reader {
    ( $record_buffer:ident,
      $transform_executors:expr,
      $namespace_descriptor:expr,
      $hash_index:ident,
      $hash_value:ident,
      $bl:block  ) => {
        if $namespace_descriptor.namespace_type == NamespaceType::Transformed {
            // This is super-unoptimized
            let executor = unsafe {
                $transform_executors
                    .executors
                    .get_unchecked($namespace_descriptor.namespace_index as usize)
            };

            // If we have a cyclic defintion (which is a bug), this will panic!
            let mut namespace_to = executor.namespace_to.borrow_mut();
            namespace_to.tmp_data.truncate(0);

            executor.function_executor.execute_function(
                $record_buffer,
                &mut namespace_to,
                &$transform_executors,
            );

            for (hash_index1, hash_value1) in &namespace_to.tmp_data {
                let $hash_index = *hash_index1;
                let $hash_value = *hash_value1;
                $bl
            }
        } else {
            let namespace_index = $namespace_descriptor.namespace_index as usize;
            let first_token = unsafe {
                *$record_buffer.get_unchecked(namespace_index + parser::HEADER_LEN as usize)
            };
            if (first_token & parser::IS_NOT_SINGLE_MASK) == 0 {
                let $hash_index = first_token;
                let $hash_value: f32 = 1.0;
                $bl
            } else {
                let start = ((first_token >> 16) & 0x3fff) as usize;
                let end = (first_token & 0xffff) as usize;
                if $namespace_descriptor.namespace_format != NamespaceFormat::F32 {
                    for hash_offset in (start..end).step_by(2) {
                        let $hash_index = unsafe { *$record_buffer.get_unchecked(hash_offset) };
                        let $hash_value = unsafe {
                            f32::from_bits(*$record_buffer.get_unchecked(hash_offset + 1))
                        };
                        $bl
                    }
                } else {
                    for hash_offset in (start..end).step_by(2) {
                        let $hash_index = unsafe { *$record_buffer.get_unchecked(hash_offset) };
                        let $hash_value: f32 = 1.0;
                        $bl
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! feature_reader_float_namespace {
    ( $record_buffer:ident,
      $namespace_descriptor:expr,
      $hash_index:ident,
      $hash_value:ident,
      $float_value:ident,
      $bl:block  ) => {
        let namespace_index = $namespace_descriptor.namespace_index as usize;
        let first_token =
            unsafe { *$record_buffer.get_unchecked(namespace_index + parser::HEADER_LEN as usize) };
        if $namespace_descriptor.namespace_format == NamespaceFormat::F32 {
            let start = ((first_token >> 16) & 0x3fff) as usize;
            let end = (first_token & 0xffff) as usize;
            for hash_offset in (start..end).step_by(2) {
                let $hash_index = unsafe { *$record_buffer.get_unchecked(hash_offset) };
                let $hash_value: f32 = 1.0;
                let $float_value =
                    unsafe { f32::from_bits(*$record_buffer.get_unchecked(hash_offset + 1)) };
                $bl
            }
        } else {
            panic!("Not a float namespace when float namespace expected");
        }
    };
}

impl FeatureBufferTranslator {
    pub fn new(mi: &model_instance::ModelInstance) -> FeatureBufferTranslator {
        // Calculate lr_hash_mask
        let lr_hash_mask = (1 << mi.bit_precision) - 1;
        // Calculate ffm_hash_mask
        let mut ffm_bits_for_dimensions = 0;
        while mi.ffm_k > (1 << (ffm_bits_for_dimensions)) {
            ffm_bits_for_dimensions += 1;
        }
        let dimensions_mask = (1 << ffm_bits_for_dimensions) - 1;
        // in ffm we will simply mask the lower bits, so we spare them for k
        let ffm_hash_mask = ((1 << mi.ffm_bit_precision) - 1) ^ dimensions_mask;

        let mut fb = FeatureBuffer::new();
        if mi.audit_mode {
            fb.audit_aux_data = mi.audit_aux_data.as_ref().unwrap().clone();
            fb.audit_mode = true;
        }

        // avoid doing any allocations in translate
        let fbt = FeatureBufferTranslator {
            model_instance: mi.clone(), // not the nicest option
            hashes_vec_in: Vec::with_capacity(100),
            hashes_vec_out: Vec::with_capacity(100),
            feature_buffer: fb,
            lr_hash_mask: lr_hash_mask,
            ffm_hash_mask: ffm_hash_mask,
            transform_executors:
                feature_transform_executor::TransformExecutors::from_namespace_transforms(
                    &mi.transform_namespaces,
                ),
        };
        fbt
    }

    pub fn print(&self) -> () {
        println!("item out {:?}", self.feature_buffer.lr_buffer);
    }

    pub fn translate(&mut self, record_buffer: &[u32], example_number: u64) -> () {
        {
            let lr_buffer = &mut self.feature_buffer.lr_buffer;
            lr_buffer.truncate(0);
            self.feature_buffer.label = record_buffer[parser::LABEL_OFFSET] as f32; // copy label
            self.feature_buffer.example_importance =
                f32::from_bits(record_buffer[parser::EXAMPLE_IMPORTANCE_OFFSET]);
            self.feature_buffer.example_number = example_number;
            let mut output_len: usize = 0;
            let mut hashes_vec_in: &mut Vec<HashAndValue> = &mut self.hashes_vec_in;
            let mut hashes_vec_out: &mut Vec<HashAndValue> = &mut self.hashes_vec_out;
            //            for feature_combo_desc in &self.model_instance.feature_combo_descs {     // We have to do this due to audit mode :(
            for (feature_combo_n, feature_combo_desc) in
                self.model_instance.feature_combo_descs.iter().enumerate()
            {
                let feature_combo_weight = feature_combo_desc.weight;
                // we unroll first iteration of the loop and optimize
                let num_namespaces: usize = feature_combo_desc.namespace_descriptors.len();
                let namespace_descriptor =
                    unsafe { *feature_combo_desc.namespace_descriptors.get_unchecked(0) };
                // We special case a single feature (common occurance)
                if num_namespaces == 1 {
                    feature_reader!(
                        record_buffer,
                        self.transform_executors,
                        namespace_descriptor,
                        hash_index,
                        hash_value,
                        {
                            lr_buffer.push(HashAndValue {
                                hash: hash_index & self.lr_hash_mask,
                                value: hash_value * feature_combo_weight,
                            });
                        }
                    );
                    if self.model_instance.audit_mode {
                        while lr_buffer.len() > self.feature_buffer.lr_buffer_audit.len() {
                            self.feature_buffer
                                .lr_buffer_audit
                                .push(feature_combo_n as i32);
                        }
                    }
                } else {
                    hashes_vec_in.truncate(0);
                    feature_reader!(
                        record_buffer,
                        self.transform_executors,
                        namespace_descriptor,
                        hash_index,
                        hash_value,
                        {
                            hashes_vec_in.push(HashAndValue {
                                hash: hash_index,
                                value: hash_value,
                            });
                        }
                    );
                    for namespace_descriptor in unsafe {
                        feature_combo_desc
                            .namespace_descriptors
                            .get_unchecked(1 as usize..num_namespaces)
                    } {
                        hashes_vec_out.truncate(0);
                        for handv in &(*hashes_vec_in) {
                            let half_hash = handv.hash.overflowing_mul(VOWPAL_FNV_PRIME).0;
                            feature_reader!(
                                record_buffer,
                                self.transform_executors,
                                *namespace_descriptor,
                                hash_index,
                                hash_value,
                                {
                                    hashes_vec_out.push(HashAndValue {
                                        hash: hash_index ^ half_hash,
                                        value: handv.value * hash_value,
                                    });
                                }
                            );
                        }
                        std::mem::swap(&mut hashes_vec_in, &mut hashes_vec_out);
                    }
                    for handv in &(*hashes_vec_in) {
                        lr_buffer.push(HashAndValue {
                            hash: handv.hash & self.lr_hash_mask,
                            value: handv.value * feature_combo_weight,
                        });
                    }
                    if self.model_instance.audit_mode {
                        while lr_buffer.len() > self.feature_buffer.lr_buffer_audit.len() {
                            self.feature_buffer
                                .lr_buffer_audit
                                .push(feature_combo_n as i32);
                        }
                    }
                }
            }
            // add the constant
            if self.model_instance.add_constant_feature {
                lr_buffer.push(HashAndValue {
                    hash: CONSTANT_HASH & self.lr_hash_mask,
                    value: 1.0,
                });
                if self.model_instance.audit_mode {
                    while lr_buffer.len() > self.feature_buffer.lr_buffer_audit.len() {
                        self.feature_buffer.lr_buffer_audit.push(-1); // -1 denotes the constant
                    }
                }
            }

            // FFM loops have not been optimized yet
            if self.model_instance.ffm_k > 0 {
                // currently we only support primitive features as namespaces, (from --lrqfa command)
                // this is for compatibility with vowpal
                // but in theory we could support also combo features
                let ffm_buffer = &mut self.feature_buffer.ffm_buffer;
                ffm_buffer.truncate(0);
                if self.model_instance.audit_mode {
                    self.feature_buffer.ffm_buffer_audit.truncate(0);
                }
                self.feature_buffer.ffm_fields_count = self.model_instance.ffm_fields.len() as u32;
                //let feature_len = self.feature_buffer.ffm_fields_count * self.model_instance.ffm_k;
                for (contra_field_index, ffm_field) in
                    self.model_instance.ffm_fields.iter().enumerate()
                {
                    for namespace_descriptor in ffm_field {
                        feature_reader!(
                            record_buffer,
                            self.transform_executors,
                            *namespace_descriptor,
                            hash_index,
                            hash_value,
                            {
                                ffm_buffer.push(HashAndValueAndSeq {
                                    hash: hash_index & self.ffm_hash_mask,
                                    value: hash_value,
                                    contra_field_index: contra_field_index as u32
                                        * self.model_instance.ffm_k as u32,
                                });
                            }
                        );
                        if self.model_instance.audit_mode {
                            while ffm_buffer.len() > self.feature_buffer.ffm_buffer_audit.len() {
                                self.feature_buffer
                                    .ffm_buffer_audit
                                    .push(namespace_descriptor.namespace_index as u32);
                            }
                        }
                    }
                }
            }
        }
    }
}

mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::parser::{IS_NOT_SINGLE_MASK, MASK31, NO_FEATURES};
    use crate::vwmap::{NamespaceDescriptor, NamespaceFormat, NamespaceType};

    fn add_header(v2: Vec<u32>) -> Vec<u32> {
        let mut rr: Vec<u32> = vec![100, 1, 1.0f32.to_bits()];
        rr.extend(v2);
        rr
    }

    fn nd(start: u32, end: u32) -> u32 {
        return (start << 16) + end;
    }

    fn ns_desc(i: u16) -> NamespaceDescriptor {
        NamespaceDescriptor {
            namespace_index: i,
            namespace_type: NamespaceType::Primitive,
            namespace_format: NamespaceFormat::Categorical,
        }
    }

    fn ns_desc_f32(i: u16) -> NamespaceDescriptor {
        NamespaceDescriptor {
            namespace_index: i,
            namespace_type: NamespaceType::Primitive,
            namespace_format: NamespaceFormat::F32,
        }
    }

    #[test]
    fn test_constant() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = true;
        mi.feature_combo_descs
            .push(model_instance::FeatureComboDesc {
                namespace_descriptors: vec![ns_desc(0)],
                weight: 1.0,
            });

        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![parser::NO_FEATURES]); // no feature
        fbt.translate(&rb, 0);
        assert_eq!(
            fbt.feature_buffer.lr_buffer,
            vec![HashAndValue {
                hash: 116060,
                value: 1.0
            }]
        ); // vw compatibility - no feature is no feature
    }

    #[test]
    fn test_single_once() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.feature_combo_descs
            .push(model_instance::FeatureComboDesc {
                namespace_descriptors: vec![ns_desc(0)],
                weight: 1.0,
            });

        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![parser::NO_FEATURES]); // no feature
        fbt.translate(&rb, 0);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![]); // vw compatibility - no feature is no feature

        let rb = add_header(vec![0xfea]);
        fbt.translate(&rb, 0);
        assert_eq!(
            fbt.feature_buffer.lr_buffer,
            vec![HashAndValue {
                hash: 0xfea,
                value: 1.0
            }]
        );

        let rb = add_header(vec![
            parser::IS_NOT_SINGLE_MASK | nd(4, 8),
            0xfea,
            1.0f32.to_bits(),
            0xfeb,
            1.0f32.to_bits(),
        ]);
        fbt.translate(&rb, 0);
        assert_eq!(
            fbt.feature_buffer.lr_buffer,
            vec![
                HashAndValue {
                    hash: 0xfea,
                    value: 1.0
                },
                HashAndValue {
                    hash: 0xfeb,
                    value: 1.0
                }
            ]
        );
    }

    #[test]
    fn test_single_twice() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.feature_combo_descs
            .push(model_instance::FeatureComboDesc {
                namespace_descriptors: vec![ns_desc(0)],
                weight: 1.0,
            });
        mi.feature_combo_descs
            .push(model_instance::FeatureComboDesc {
                namespace_descriptors: vec![ns_desc(1)],
                weight: 1.0,
            });

        let mut fbt = FeatureBufferTranslator::new(&mi);

        let rb = add_header(vec![parser::NO_FEATURES, parser::NO_FEATURES]);
        fbt.translate(&rb, 0);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![]);

        let rb = add_header(vec![0xfea, parser::NO_FEATURES]);
        fbt.translate(&rb, 0);
        assert_eq!(
            fbt.feature_buffer.lr_buffer,
            vec![HashAndValue {
                hash: 0xfea,
                value: 1.0
            }]
        );

        let rb = add_header(vec![0xfea, 0xfeb]);
        fbt.translate(&rb, 0);
        assert_eq!(
            fbt.feature_buffer.lr_buffer,
            vec![
                HashAndValue {
                    hash: 0xfea,
                    value: 1.0
                },
                HashAndValue {
                    hash: 0xfeb,
                    value: 1.0
                }
            ]
        );
    }

    // for singles, vowpal and fwumious are the same
    // however for doubles theya are not
    #[test]
    fn test_double_vowpal() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.feature_combo_descs
            .push(model_instance::FeatureComboDesc {
                namespace_descriptors: vec![ns_desc(0), ns_desc(1)],
                weight: 1.0,
            });

        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![parser::NO_FEATURES]);
        fbt.translate(&rb, 0);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![]);

        let rb = add_header(vec![123456789, parser::NO_FEATURES]);
        fbt.translate(&rb, 0);
        assert_eq!(fbt.feature_buffer.lr_buffer, vec![]); // since the other feature is missing - VW compatibility says no feature is here

        let rb = add_header(vec![
            2988156968 & parser::MASK31,
            2422381320 & parser::MASK31,
            parser::NO_FEATURES,
        ]);
        fbt.translate(&rb, 0);
        //        println!("out {}, out mod 2^24 {}", fbt.feature_buffer.lr_buffer[1], fbt.feature_buffer.lr_buffer[1] & ((1<<24)-1));
        assert_eq!(
            fbt.feature_buffer.lr_buffer,
            vec![HashAndValue {
                hash: 208368,
                value: 1.0
            }]
        );
    }

    #[test]
    fn test_single_with_weight_vowpal() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.feature_combo_descs
            .push(model_instance::FeatureComboDesc {
                namespace_descriptors: vec![ns_desc(0)],
                weight: 2.0,
            });

        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![0xfea]);
        fbt.translate(&rb, 0);
        assert_eq!(
            fbt.feature_buffer.lr_buffer,
            vec![HashAndValue {
                hash: 0xfea,
                value: 2.0
            }]
        );
    }

    #[test]
    fn test_ffm_empty() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.ffm_fields.push(vec![]); // single field, empty
        mi.ffm_k = 1;
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![0xfea]);
        fbt.translate(&rb, 0);
        assert_eq!(fbt.feature_buffer.ffm_buffer, vec![]);
    }

    #[test]
    fn test_ffm_one() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.ffm_fields.push(vec![ns_desc(0)]); // single feature in a single fields
        mi.ffm_k = 1;
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![0xfea]);
        fbt.translate(&rb, 0);
        assert_eq!(
            fbt.feature_buffer.ffm_buffer,
            vec![HashAndValueAndSeq {
                hash: 0xfea,
                value: 1.0,
                contra_field_index: 0
            }]
        );
    }

    #[test]
    fn test_ffm_two_fields() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.ffm_fields.push(vec![ns_desc(0)]); //  single namespace in a field
        mi.ffm_fields.push(vec![ns_desc(0), ns_desc(1)]); // two namespaces in a field
        mi.ffm_k = 1;
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![
            parser::IS_NOT_SINGLE_MASK | nd(5, 9),
            0xfec,
            0xfea,
            2.0f32.to_bits(),
            0xfeb,
            3.0f32.to_bits(),
        ]);
        fbt.translate(&rb, 0);
        assert_eq!(
            fbt.feature_buffer.ffm_buffer,
            vec![
                HashAndValueAndSeq {
                    hash: 0xfea,
                    value: 2.0,
                    contra_field_index: 0
                },
                HashAndValueAndSeq {
                    hash: 0xfeb,
                    value: 3.0,
                    contra_field_index: 0
                },
                HashAndValueAndSeq {
                    hash: 0xfea,
                    value: 2.0,
                    contra_field_index: 1
                },
                HashAndValueAndSeq {
                    hash: 0xfeb,
                    value: 3.0,
                    contra_field_index: 1
                },
                HashAndValueAndSeq {
                    hash: 0xfec,
                    value: 1.0,
                    contra_field_index: 1
                }
            ]
        );
    }

    #[test]
    fn test_ffm_three_fields() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.ffm_fields.push(vec![ns_desc(0)]); //  single namespace in a field        0xfea, 0xfeb
        mi.ffm_fields.push(vec![ns_desc(0), ns_desc(1)]); // two namespaces in a field	      0xfea, 0xfeb, 0xfec
        mi.ffm_fields.push(vec![ns_desc(1)]); // single namespace in a field	      0xfec
        mi.ffm_k = 1;
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![
            parser::IS_NOT_SINGLE_MASK | nd(5, 9),
            0x1,
            0xfff,
            2.0f32.to_bits(),
            0xfeb,
            3.0f32.to_bits(),
        ]);
        fbt.translate(&rb, 0);
        // Hashes get changed, because k = 3 means we'll be aligning hashes
        assert_eq!(
            fbt.feature_buffer.ffm_buffer,
            vec![
                HashAndValueAndSeq {
                    hash: 0xfff,
                    value: 2.0,
                    contra_field_index: 0
                },
                HashAndValueAndSeq {
                    hash: 0xfeb,
                    value: 3.0,
                    contra_field_index: 0
                },
                HashAndValueAndSeq {
                    hash: 0xfff,
                    value: 2.0,
                    contra_field_index: 1
                },
                HashAndValueAndSeq {
                    hash: 0xfeb,
                    value: 3.0,
                    contra_field_index: 1
                },
                HashAndValueAndSeq {
                    hash: 0x1,
                    value: 1.0,
                    contra_field_index: 1
                },
                HashAndValueAndSeq {
                    hash: 0x1,
                    value: 1.0,
                    contra_field_index: 2
                },
            ]
        );
        // Now hashes get changed, because k = 3 means we'll be aligning hashes
        mi.ffm_k = 3;
        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![
            parser::IS_NOT_SINGLE_MASK | nd(5, 9),
            0x1,
            0xfff,
            2.0f32.to_bits(),
            0xfeb,
            3.0f32.to_bits(),
        ]);
        fbt.translate(&rb, 0);
        assert_eq!(
            fbt.feature_buffer.ffm_buffer,
            vec![
                HashAndValueAndSeq {
                    hash: 0xffc,
                    value: 2.0,
                    contra_field_index: 0
                },
                HashAndValueAndSeq {
                    hash: 0xfe8,
                    value: 3.0,
                    contra_field_index: 0
                },
                HashAndValueAndSeq {
                    hash: 0xffc,
                    value: 2.0,
                    contra_field_index: 3
                },
                HashAndValueAndSeq {
                    hash: 0xfe8,
                    value: 3.0,
                    contra_field_index: 3
                },
                HashAndValueAndSeq {
                    hash: 0x0,
                    value: 1.0,
                    contra_field_index: 3
                },
                HashAndValueAndSeq {
                    hash: 0x0,
                    value: 1.0,
                    contra_field_index: 6
                },
            ]
        );
        // one more which we dont test
    }

    #[test]
    fn test_example_importance() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.feature_combo_descs
            .push(model_instance::FeatureComboDesc {
                namespace_descriptors: vec![ns_desc(0)],
                weight: 1.0,
            });

        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![parser::NO_FEATURES]); // no feature
        fbt.translate(&rb, 0);
        assert_eq!(fbt.feature_buffer.example_importance, 1.0); // Did example importance get parsed correctly
    }

    #[test]
    fn test_single_namespace_float() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.add_constant_feature = false;
        mi.feature_combo_descs
            .push(model_instance::FeatureComboDesc {
                namespace_descriptors: vec![ns_desc_f32(1)],
                weight: 1.0,
            });

        let mut fbt = FeatureBufferTranslator::new(&mi);
        let rb = add_header(vec![
            NO_FEATURES,
            nd(6, 10) | IS_NOT_SINGLE_MASK,
            NO_FEATURES,
            0xffc & MASK31,
            3.0f32.to_bits(),
            0xffa & MASK31,
            4.0f32.to_bits(),
        ]);
        fbt.translate(&rb, 0);
        assert_eq!(
            fbt.feature_buffer.lr_buffer,
            vec![
                HashAndValue {
                    hash: 0xffc,
                    value: 1.0
                },
                HashAndValue {
                    hash: 0xffa,
                    value: 1.0
                }
            ]
        );
    }
}
