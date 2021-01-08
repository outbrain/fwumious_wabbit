use crate::model_instance;
use crate::parser;
use crate::vwmap;
use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;

use fasthash::murmur3;
use serde::{Serialize,Deserialize};
use dyn_clone::{clone_trait_object, DynClone};



// this is macro, globally exported
use crate::feature_reader;
use crate::feature_transform_parser;
use crate::feature_transform_parser::TransformNamespaces;
use crate::feature_reader_float_namespace;


#[derive(Clone)]
pub struct ExecutorToNamespace {
    namespace_index: u32,
    namespace_char: char,
    namespace_seed: u32,
    tmp_data: Vec<(u32, f32)>,
}

#[derive(Clone)]
pub struct ExecutorFromNamespace {
    namespace_index: u32,
    namespace_char: char,
}


impl ExecutorToNamespace {
    #[inline(always)]
    fn emit_u32(&mut self, to_data:u32, hash_value:f32) {
        let hash_data = murmur3::hash32_with_seed(to_data.to_le_bytes(), self.namespace_seed) & parser::MASK31;
        println!("AAAA {}", to_data);
        self.tmp_data.push((hash_data, hash_value));
    } 
}

#[derive(Clone)]
pub struct TransformExecutor {
    namespace_to: ExecutorToNamespace,
//    namespaces_from: Vec<ExecutorFromNamespace>, // from namespaces data resides in executor
    function_executor: Box<dyn FunctionExecutorTrait>,
}

impl TransformExecutor {
    fn from_namespace_transform(namespace_transform: feature_transform_parser::NamespaceTransform) -> Result<TransformExecutor, Box<dyn Error>> {

        let namespace_to = ExecutorToNamespace {
            namespace_index: namespace_transform.to_namespace.namespace_index,
            namespace_char: namespace_transform.to_namespace.namespace_char,
            namespace_seed: murmur3::hash32(vec![namespace_transform.to_namespace.namespace_index as u8, 255]),
            tmp_data: Vec::new(),
        };
        
        let te = TransformExecutor {
            namespace_to: namespace_to,
            function_executor: Self::get_executor(&namespace_transform.function_name, namespace_transform.from_namespaces, namespace_transform.function_parameters)?,
        };
        Ok(te)
    }

    fn get_executor(function_name: &str, namespaces_from: Vec<feature_transform_parser::Namespace>, function_params: Vec<f32>) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
    //    let mut f = FunctionFactorySqrt {};
    //    f.new_executor(namespaces_from, function_params)
        let mut executor_namespaces_from: Vec<ExecutorFromNamespace> = Vec::new();
        for namespace in &namespaces_from {
            executor_namespaces_from.push(ExecutorFromNamespace{namespace_index: namespace.namespace_index, namespace_char: namespace.namespace_char});
        }

        Ok(Box::new(FunctionSqrt {from_namespace: executor_namespaces_from[0].clone()}))

    }
}


// Some black magic from: https://stackoverflow.com/questions/30353462/how-to-clone-a-struct-storing-a-boxed-trait-object
// We need clone() because of serving. There is also an option of doing FeatureBufferTransform from scratch in each thread
pub trait FunctionExecutorTrait: DynClone + Send {
    fn execute_function(&self, record_buffer: &[u32], namespace: &mut ExecutorToNamespace);
}

clone_trait_object!(FunctionExecutorTrait);


/*
pub trait FunctionFactoryTrait {
    fn get_function_name() -> String;
    fn new_executor(&mut self, params_namespaces: Vec<ExecutorFromNamespace>, params: Vec<f32>) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>>;
}*/




fn emit_u32(hash_data:u32, hash_value:f32, namespace_seed: u32) -> (u32, f32) {
    (murmur3::hash32_with_seed(hash_data.to_le_bytes(), namespace_seed) & parser::MASK31, hash_value)
}                                                         



pub fn transformed_feature<'a>(record_buffer: &[u32], mi: &model_instance::ModelInstance, feature_index_offset: u32) -> Vec<(u32, f32)> {
    // This is FAAAR from optimized
    let mut output:Vec<(u32, f32)> = Vec::new();
    let feature_index_offset = feature_index_offset & !feature_transform_parser::TRANSFORM_NAMESPACE_MARK; // remove transform namespace mark
    //println!("Fi {}", feature_index_offset);
/*    let transform_namespace = &mi.transform_namespaces.v[feature_index_offset as usize];
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
*/
    output
}

#[derive(Clone)]
struct FunctionSqrt {
    from_namespace: ExecutorFromNamespace,
}

impl FunctionExecutorTrait for FunctionSqrt {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace) {
        feature_reader_float_namespace!(record_buffer, self.from_namespace.namespace_index, hash_data, hash_value, float_value, {
            let transformed_float = float_value.sqrt();
            let transformed_int = transformed_float as u32;
            to_namespace.emit_u32(transformed_int, hash_value);
        });
    }
}

/*struct FunctionFactorySqrt {}
impl FunctionFactoryTrait for FunctionFactorySqrt {
    fn get_function_name() -> String{
        "sqrt".to_owned()
    }
    fn new_executor(&mut self, params_namespaces: Vec<ExecutorFromNamespace>, function_params: Vec<f32>) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        Ok(Box::new(FunctionSqrt {from_namespace: params_namespaces[0].clone()}))
    }
}*/




