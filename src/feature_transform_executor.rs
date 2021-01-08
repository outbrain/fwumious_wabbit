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
use crate::feature_transform_parser::NamespaceTransforms;
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
        self.tmp_data.push((hash_data, hash_value));
    } 
}

#[derive(Clone)]
pub struct TransformExecutor {
    namespace_to: ExecutorToNamespace,
//    namespaces_from: Vec<ExecutorFromNamespace>, // from namespaces data resides in function_executor
    function_executor: Box<dyn FunctionExecutorTrait>,
}

impl TransformExecutor {
    pub fn from_namespace_transform(namespace_transform: &feature_transform_parser::NamespaceTransform) -> Result<TransformExecutor, Box<dyn Error>> {
        let namespace_to = ExecutorToNamespace {
            namespace_index: namespace_transform.to_namespace.namespace_index,
            namespace_char: namespace_transform.to_namespace.namespace_char,
            namespace_seed: murmur3::hash32(vec![namespace_transform.to_namespace.namespace_index as u8, 255]),
            tmp_data: Vec::new(),
        };
        
        let te = TransformExecutor {
            namespace_to: namespace_to,
            function_executor: Self::get_executor(&namespace_transform.function_name, 
                                                    &namespace_transform.from_namespaces, 
                                                    &namespace_transform.function_parameters)?,
        };
        Ok(te)
    }

    pub fn get_executor(function_name: &str, namespaces_from: &Vec<feature_transform_parser::Namespace>, function_params: &Vec<f32>) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        let mut executor_namespaces_from: Vec<ExecutorFromNamespace> = Vec::new();
        for namespace in namespaces_from {
            executor_namespaces_from.push(ExecutorFromNamespace{namespace_index: namespace.namespace_index, namespace_char: namespace.namespace_char});
        }

        FunctionSqrt::create_function(function_name, &executor_namespaces_from, function_params)
    }
}



#[derive(Clone)]
pub struct TransformExecutors {
    pub executors: Vec<TransformExecutor>,
}

impl TransformExecutors {
    pub fn from_namespace_transforms(namespace_transforms: &feature_transform_parser::NamespaceTransforms) -> TransformExecutors{
        let mut executors:Vec<TransformExecutor> = Vec::new();
        for transformed_namespace in &namespace_transforms.v {
            let transformed_namespace_executor = TransformExecutor::from_namespace_transform(&transformed_namespace).unwrap();
            executors.push(transformed_namespace_executor);
        }
        TransformExecutors {executors: executors}
    }

    pub fn get_transformations<'a>(&mut self, record_buffer: &[u32], feature_index_offset: u32) -> &Vec<(u32, f32)> {
        let feature_index_offset = feature_index_offset & !feature_transform_parser::TRANSFORM_NAMESPACE_MARK; // remove transform namespace mark
        //println!("Fi {}", feature_index_offset);
        let executor = &mut self.executors[feature_index_offset as usize];
        executor.namespace_to.tmp_data.truncate(0);
        executor.function_executor.execute_function(record_buffer, &mut executor.namespace_to);
        &mut executor.namespace_to.tmp_data
    }


}


// Some black magic from: https://stackoverflow.com/questions/30353462/how-to-clone-a-struct-storing-a-boxed-trait-object
// We need clone() because of serving. There is also an option of doing FeatureBufferTransform from scratch in each thread
pub trait FunctionExecutorTrait: DynClone + Send {
    fn execute_function(&self, record_buffer: &[u32], namespace: &mut ExecutorToNamespace);
    fn create_function(function_name: &str, from_namespaces: &Vec<ExecutorFromNamespace>, function_params: &Vec<f32>) ->  Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> where Self:Sized;
}

clone_trait_object!(FunctionExecutorTrait);


    

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
    
    fn create_function(function_name: &str, from_namespaces: &Vec<ExecutorFromNamespace>, function_params: &Vec<f32>) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        assert!(function_params.len() == 0);
        Ok(Box::new(Self{from_namespace: from_namespaces[0].clone()}))
    }   
}


#[derive(Clone)]
struct FunctionSqrt2 {
    from_namespace: ExecutorFromNamespace,
    greater_than: f32
}

impl FunctionExecutorTrait for FunctionSqrt2 {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace) {
        feature_reader_float_namespace!(record_buffer, self.from_namespace.namespace_index, hash_data, hash_value, float_value, {
            let mut transformed_int:u32;
            if float_value > self.greater_than {
                let transformed_float = float_value.sqrt();
                transformed_int = transformed_float as u32;
            } else {
                transformed_int =  (- (float_value as i32)) as u32;
            }
            to_namespace.emit_u32(transformed_int, hash_value);
        });
    }
    
    fn create_function(function_name: &str, from_namespaces: &Vec<ExecutorFromNamespace>, function_params: &Vec<f32>) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        assert!(function_params.len() == 1);
        Ok(Box::new(Self{from_namespace: from_namespaces[0].clone(), greater_than: function_params[0]}))
    }   
}















