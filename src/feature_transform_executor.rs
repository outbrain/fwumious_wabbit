use crate::model_instance;
use crate::parser;
use crate::vwmap;
use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;

use fasthash::murmur3;
use serde::{Serialize,Deserialize};


// this is macro, globally exported
use crate::feature_reader;
use crate::feature_transform_parser;
use crate::feature_reader_float_namespace;


#[inline(always)]
fn emit_u32(hash_data:u32, hash_value:f32, namespace_seed: u32) -> (u32, f32) {
    (murmur3::hash32_with_seed(hash_data.to_le_bytes(), namespace_seed) & parser::MASK31, hash_value)
}                                                         


pub fn transformed_feature<'a>(record_buffer: &[u32], mi: &model_instance::ModelInstance, feature_index_offset: u32) -> Vec<(u32, f32)> {
    // This is FAAAR from optimized
    let mut output:Vec<(u32, f32)> = Vec::new();
    let feature_index_offset = feature_index_offset & !feature_transform_parser::TRANSFORM_NAMESPACE_MARK; // remove transform namespace mark
    //println!("Fi {}", feature_index_offset);
    let transform_namespace = &mi.transform_namespaces.v[feature_index_offset as usize];
    if transform_namespace.function == feature_transform_parser::TransformFunction::Sqrt {
        feature_reader_float_namespace!(record_buffer, transform_namespace.from_namespace_index, hash_data, hash_value, float_value, {
            let transformed_float = float_value.sqrt();
            let transformed_int = transformed_float as u32;
            output.push(emit_u32(transformed_int, hash_value, 0));
            //println!("Input hash value {}, float value {}", hash_value, float_value);
            //println!("Sqrt: {}, sqrt_int {}", transformed_float, transformed_int);
            //println!("Output hash data {}, hash_value {}", output.last().unwrap().0, output.last().unwrap().1);
              
        });
    }

    output
}

