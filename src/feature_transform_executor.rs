use crate::model_instance;
use crate::parser;
use crate::vwmap;
use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;

use std::cell::RefCell;

use fasthash::murmur3;
use serde::{Serialize,Deserialize};
use dyn_clone::{clone_trait_object, DynClone};



// this is macro, globally exported
use crate::feature_reader;
use crate::feature_transform_parser;
use crate::feature_transform_parser::NamespaceTransforms;
use crate::feature_reader_float_namespace;


#[derive(Clone, Copy)]
pub enum SeedNumber {
    Default = 0,
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
}

macro_rules! default_seeds {
    ($to_namespace_index: expr) => 
    {
        // These are random numbers, i threw a dice! https://xkcd.com/221/
            [      murmur3::hash32(vec![$to_namespace_index as u8, 255]),
                   murmur3::hash32(vec![3, $to_namespace_index as u8, 222, 52, 53]),
                   murmur3::hash32(vec![6, $to_namespace_index as u8, 35, 51, 245]),
                   murmur3::hash32(vec![24, $to_namespace_index as u8, 32, 0, 111]),
                   murmur3::hash32(vec![76, $to_namespace_index as u8, 23, 234, 0])
            ]    
    }
}

#[derive(Clone)]
pub struct ExecutorToNamespace {
    namespace_index: u32,
    namespace_verbose: String,
    namespace_seeds: [u32; 5],	// These are precomputed namespace seeds
    pub tmp_data: Vec<(u32, f32)>,
}

#[derive(Clone)]
pub struct ExecutorFromNamespace {
    namespace_index: u32,
    namespace_verbose: String,
    namespace_is_float: bool,
}


impl ExecutorToNamespace {

    #[inline(always)]
    fn emit_i32(&mut self, to_data:i32, hash_value:f32, seed_id: SeedNumber) {
        let hash_index = murmur3::hash32_with_seed(to_data.to_le_bytes(), self.namespace_seeds[seed_id as usize]) & parser::MASK31;
        println!("hash_index: {}", hash_index);
        self.tmp_data.push((hash_index, hash_value));
    } 

    #[inline(always)]
    fn emit_f32(&mut self, f:f32, hash_value:f32, interpolated: bool, seed_id: SeedNumber) {
        if f.is_nan() {
            println!("im here");
            self.emit_i32(f as i32, hash_value, SeedNumber::Four);
        }
        else if interpolated {
            let floor = f.floor();
            let floor_int = floor as i32;
            let part = f - floor;
            if part != 0.0 {
                self.emit_i32(floor_int + 1, hash_value * part, seed_id);
            }
            let part = 1.0 - part;
            if part != 0.0 {
                self.emit_i32(floor_int, hash_value * part, seed_id);
            }
        } else {
            self.emit_i32(f as i32, hash_value, seed_id);
        }
    } 

    #[inline(always)]
    fn emit_i32_i32(&mut self, to_data1:i32, to_data2:i32, hash_value:f32, seed_id: SeedNumber) {
        let hash_index = murmur3::hash32_with_seed(to_data1.to_le_bytes(), self.namespace_seeds[seed_id as usize]);
        let hash_index = murmur3::hash32_with_seed(to_data2.to_le_bytes(), hash_index) & parser::MASK31;
        self.tmp_data.push((hash_index, hash_value));
    } 
}

#[derive(Clone)]
pub struct TransformExecutor {
//    namespaces_from: Vec<ExecutorFromNamespace>, // from namespaces data resides in function_executor
    pub namespace_to: RefCell<ExecutorToNamespace>,
    function_executor: Box<dyn FunctionExecutorTrait>,
}

impl TransformExecutor {
    pub fn from_namespace_transform(namespace_transform: &feature_transform_parser::NamespaceTransform) -> Result<TransformExecutor, Box<dyn Error>> {
        
        let namespace_to = ExecutorToNamespace {
            namespace_index: namespace_transform.to_namespace.namespace_index,
            namespace_verbose: namespace_transform.to_namespace.namespace_verbose.to_owned(),
            // These are random numbers, i threw a dice!
            namespace_seeds: default_seeds!(namespace_transform.to_namespace.namespace_index),
            tmp_data: Vec::new(),
        };
        
        let te = TransformExecutor {
            namespace_to: RefCell::new(namespace_to),
            function_executor: Self::create_executor(&namespace_transform.function_name, 
                                                    &namespace_transform.from_namespaces, 
                                                    &namespace_transform.function_parameters)?,
        };
        Ok(te)
    }

    pub fn create_executor(function_name: &str, namespaces_from: &Vec<feature_transform_parser::Namespace>, function_params: &Vec<f32>) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        let mut executor_namespaces_from: Vec<ExecutorFromNamespace> = Vec::new();
        for namespace in namespaces_from {
            executor_namespaces_from.push(ExecutorFromNamespace{namespace_index: namespace.namespace_index, 
                                                                namespace_verbose: namespace.namespace_verbose.to_owned(),
                                                                namespace_is_float: namespace.namespace_is_float});
       }
        if function_name == "BinnerMinSqrt" {
            TransformerBinner::create_function(&(|x| x.sqrt()), function_name, &executor_namespaces_from, function_params, false)
        } else if function_name == "BinnerMinLn" {
            TransformerBinner::create_function(&(|x| x.ln()), function_name, &executor_namespaces_from, function_params, false)
        } else if function_name == "BinnerMinLog20" {
            TransformerBinner::create_function(&(|x| x.log(2.0)), function_name, &executor_namespaces_from, function_params, false)
        } else if function_name == "BinnerMinLog15" {
            TransformerBinner::create_function(&(|x| x.log(1.5)), function_name, &executor_namespaces_from, function_params, false)
        } else if function_name == "BinnerMinInterpolatedSqrt" {
            TransformerBinner::create_function(&(|x| x.sqrt()), function_name, &executor_namespaces_from, function_params, true)
        } else if function_name == "BinnerMinInterpolatedLn" {
            TransformerBinner::create_function(&(|x| x.ln()), function_name, &executor_namespaces_from, function_params, true)
        } else if function_name == "BinnerMinInterpolatedLog20" {
            TransformerBinner::create_function(&(|x| x.log(2.0)), function_name, &executor_namespaces_from, function_params, true)
        } else if function_name == "BinnerMinInterpolatedLog15" {
            TransformerBinner::create_function(&(|x| x.log(1.5)), function_name, &executor_namespaces_from, function_params, true)
        } else if function_name == "BinnerLogRatio" {
            TransformerLogRatioBinner::create_function(function_name, &executor_namespaces_from, function_params, false)
        } else if function_name == "BinnerInterpolatedLogRatio" {
            TransformerLogRatioBinner::create_function(function_name, &executor_namespaces_from, function_params, true)
        } else if function_name == "MultiplyWeight" {
            TransformerMultiplyWeight::create_function(function_name, &executor_namespaces_from, function_params)
        } else {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown transformer function: {}", function_name))));            
        
        }
    }
}



#[derive(Clone)]
pub struct TransformExecutors {
    pub executors: Vec<TransformExecutor>,

}

impl TransformExecutors {
    pub fn from_namespace_transforms(namespace_transforms: &feature_transform_parser::NamespaceTransforms) -> TransformExecutors{
        let mut executors:Vec<TransformExecutor> = Vec::new();
        let mut namespaces_to: Vec<ExecutorToNamespace> = Vec::new();
        for transformed_namespace in &namespace_transforms.v {
            let transformed_namespace_executor = TransformExecutor::from_namespace_transform(&transformed_namespace).unwrap();
            executors.push(transformed_namespace_executor);
//            namespaces_to.push(tmp_data: Vec::new());

        }
        TransformExecutors {executors: executors}
    }

    #[inline(always)]
    pub fn get_transformations<'a>(&self, record_buffer: &[u32], feature_index_offset: u32) -> u32  {
        let executor_index = feature_index_offset & !feature_transform_parser::TRANSFORM_NAMESPACE_MARK; // remove transform namespace mark
        let executor = &self.executors[executor_index as usize];
        
        // If we have a cyclic defintion (which is a bug), this will panic!
        let mut namespace_to = executor.namespace_to.borrow_mut();
        namespace_to.tmp_data.truncate(0);
        
        executor.function_executor.execute_function(record_buffer, &mut namespace_to, &self);
        executor_index
    }


}


// Some black magic from: https://stackoverflow.com/questions/30353462/how-to-clone-a-struct-storing-a-boxed-trait-object
// We need clone() because of serving. There is also an option of doing FeatureBufferTransform from scratch in each thread
pub trait FunctionExecutorTrait: DynClone + Send {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace, transform_executors: &TransformExecutors);
}
clone_trait_object!(FunctionExecutorTrait);


// BASIC EXAMPLE    
// Basic example of a "full blown" simple FunctionExecutorTrait
#[derive(Clone)]
struct FunctionExampleSqrt {
    from_namespace: ExecutorFromNamespace,
}

impl FunctionExecutorTrait for FunctionExampleSqrt {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace, transform_executors: &TransformExecutors) {
        feature_reader_float_namespace!(record_buffer, self.from_namespace.namespace_index, hash_index, hash_value, float_value, {
            let transformed_float = float_value.sqrt();
            let transformed_int = transformed_float as i32;
            to_namespace.emit_i32(transformed_int, hash_value, SeedNumber::Default);
        });
    }
}

impl FunctionExampleSqrt {
    fn create_function(function_name: &str, from_namespaces: &Vec<ExecutorFromNamespace>, function_params: &Vec<f32>) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        // For simplicity of example, we just assert instead of full error reporting
        assert!(function_params.len() == 0);
        assert!(from_namespaces.len() == 1);
        assert!(from_namespaces[0].namespace_is_float == true);
        Ok(Box::new(Self{from_namespace: from_namespaces[0].clone()}))
    }   
}

// -------------------------------------------------------------------
// TransformerBinner - A basic binner
// It can take any function as a binning function f32 -> f32. Then output is rounded to integer 
// However if output is smaller than floating parameter (greater_than), then output is custom encoded with that value
// Example of use: you want to bin number of pageviews per user, so you generally want to do sqrt on it, but not do binning when pageviews <= 10
// In that case you would call BinnerMinSqrt(A)(10.0)

// What does interpolated mean?
// Example: BinnerMinSqrt(X)(10.0)
// let's assume X is 150. sqrt(150) = 12.247
// You now want two values emitted - 12 at value 0.247 and 13 at value (1-0.247)


#[derive(Clone)]
struct TransformerBinner {
    from_namespace: ExecutorFromNamespace,
    greater_than: f32,
    interpolated: bool,
    function_pointer: &'static (dyn Fn(f32) -> f32 +'static + Sync), 
}

impl FunctionExecutorTrait for TransformerBinner {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace, transform_executors: &TransformExecutors) {
        feature_reader_float_namespace!(record_buffer, self.from_namespace.namespace_index, hash_index, hash_value, float_value, {
            if float_value <= self.greater_than {
                to_namespace.emit_i32(float_value as i32, hash_value, SeedNumber::Default);
            } else {
                let transformed_float = (self.function_pointer)(float_value);
                to_namespace.emit_f32(transformed_float, hash_value, self.interpolated, SeedNumber::One);
            }
        });
    }
}


impl TransformerBinner {
    fn create_function(function_pointer: &'static (dyn Fn(f32) -> f32 +'static + Sync), 
                        function_name: &str, 
                        from_namespaces: &Vec<ExecutorFromNamespace>, 
                        function_params: &Vec<f32>,
                        interpolated: bool,
                        ) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        if function_params.len() != 1 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes exactly one float argument, example {}(A)(2.0)", function_name, function_name))));            
        }
        if from_namespaces.len() != 1 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes exactly one namespace argument, example {}(A)(2.0)", function_name, function_name))));            
        }
        for namespace in from_namespaces.iter() {
            if !namespace.namespace_is_float {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("All namespaces of function {} have to be flaot: From namespace ({}) has to be defined as --float_namespaces", function_name, namespace.namespace_verbose))));
            }
        }

        Ok(Box::new(Self{from_namespace: from_namespaces[0].clone(), 
                        greater_than: function_params[0],
                        function_pointer: function_pointer,
                        interpolated: interpolated,
                        }))
    }
}   


// LogRatio Binner
//
//
//
//

#[derive(Clone)]
struct TransformerLogRatioBinner {
    from_namespace1: ExecutorFromNamespace,
    from_namespace2: ExecutorFromNamespace,
    greater_than: f32,
    interpolated: bool,
    resolution: f32,
}

impl FunctionExecutorTrait for TransformerLogRatioBinner {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace, transform_executors: &TransformExecutors) {
        feature_reader_float_namespace!(record_buffer, self.from_namespace1.namespace_index, hash_index1, hash_value1, float_value1, {
            feature_reader_float_namespace!(record_buffer, self.from_namespace2.namespace_index, hash_index2, hash_value2, float_value2, {

                let joint_value = hash_value1 * hash_value2;
                let val1 = float_value1;
                let val2 = float_value2;
                println!("{}, {}", val1, val2);
                if val2 + val1 < self.greater_than {
                    to_namespace.emit_i32_i32(val1 as i32, val2 as i32, joint_value, SeedNumber::One);    
                } else if val1 == 0.0 {
                    to_namespace.emit_f32(val2.sqrt(), joint_value, self.interpolated, SeedNumber::Two);    
                } else if val2 == 0.0 {
                    to_namespace.emit_i32(val1 as i32, joint_value, SeedNumber::Three);
                } else {
                    let o = (val1/val2).ln()*self.resolution;
                    println!("{}", o);
                    to_namespace.emit_f32(o, joint_value, self.interpolated, SeedNumber::Default);
                }
            });
        });
    }
}

impl TransformerLogRatioBinner {
    fn create_function(
                        function_name: &str, 
                        from_namespaces: &Vec<ExecutorFromNamespace>, 
                        function_params: &Vec<f32>,
                        interpolated: bool,
                        ) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        if function_params.len() != 2 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes exactly one float argument, example {}(A)(2.0)", function_name, function_name))));            
        }
        if from_namespaces.len() != 2 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes exactly two namespace arguments, example {}(A,B)(2.0)", function_name, function_name))));            
        }
        for namespace in from_namespaces.iter() {
            if !namespace.namespace_is_float {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("All namespaces of function {} have to be flaot: From namespace ({}) has to be defined as --float_namespaces", function_name, namespace.namespace_verbose))));
            }
        }

        Ok(Box::new(Self{from_namespace1: from_namespaces[0].clone(), 
                        from_namespace2: from_namespaces[1].clone(), 
                        greater_than: function_params[0],
                        resolution: function_params[1],
                        interpolated: interpolated,
                        }))
    }
}   



// Value multiplier transformer
// -------------------------------------------------------------------
// TransformerMultiplyWeight - A basic weight multiplier transformer
// Example of use: if you want to multiply whole namespace with certain factor and thus increase its learning rate (let's say 2.0)
// In that case you would call MutliplyWeight(document_id)(2.0)
// Important - document_id does not need to be float and isnt really changed 


#[derive(Clone)]
struct TransformerMultiplyWeight {
    from_namespace: ExecutorFromNamespace,
    multiplier: f32,
}

impl FunctionExecutorTrait for TransformerMultiplyWeight {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace, transform_executors: &TransformExecutors) {
        feature_reader!(record_buffer, transform_executors, self.from_namespace.namespace_index, hash_index, hash_value, {
            to_namespace.emit_i32(hash_index as i32, hash_value * self.multiplier, SeedNumber::Default);
        });
    }
}


impl TransformerMultiplyWeight {
    fn create_function( function_name: &str, 
                        from_namespaces: &Vec<ExecutorFromNamespace>, 
                        function_params: &Vec<f32>,
                        ) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        if function_params.len() != 1 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes exactly one float argument, example {}(A)(2.0)", function_name, function_name))));            
        }
        if from_namespaces.len() != 1 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes exactly one namespace argument, example {}(A)(2.0)", function_name, function_name))));            
        }
        Ok(Box::new(Self{from_namespace: from_namespaces[0].clone(),
                        multiplier: function_params[0], 
                        }))
    }
}   









mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::parser::{IS_NOT_SINGLE_MASK, IS_FLOAT_NAMESPACE_MASK, MASK31};

    fn add_header(v2: Vec<u32>) -> Vec<u32> {
        let mut rr: Vec<u32> = vec![100, 1, 1.0f32.to_bits()];
        rr.extend(v2);
        rr
    }
    
    fn nd(start: u32, end: u32) -> u32 {
        return (start << 16) + end;
    }

    #[test]
    fn test_transformerbinner_fail() {
        // this fails because input namespace is not float namespace    
        let from_namespace = ExecutorFromNamespace {
            namespace_index: 0,
            namespace_verbose: "a".to_string(),
            namespace_is_float: false,
        };
        let to_namespace_index = 1;
                            
        let to_namespace_empty = ExecutorToNamespace {
            namespace_index: to_namespace_index,
            namespace_verbose: "b".to_string(),
            namespace_seeds: default_seeds!(to_namespace_index),	// These are precomputed namespace seeds
            tmp_data: Vec::new(),
        };
        
        let result = TransformerBinner::create_function(&(|x| x.sqrt()), "Blah", &vec![from_namespace], &vec![40.], false);
        assert!(result.is_err());

    }

    #[test]
    fn test_transformerbinner() {
        
        let from_namespace = ExecutorFromNamespace {
            namespace_index: 0,
            namespace_verbose: "a".to_string(),
            namespace_is_float: true,
        };
        let to_namespace_index = 1;
                            
        let to_namespace_empty = ExecutorToNamespace {
            namespace_index: to_namespace_index,
            namespace_verbose: "b".to_string(),
            namespace_seeds: default_seeds!(to_namespace_index),	// These are precomputed namespace seeds
            tmp_data: Vec::new(),
        };
        
        let transformer = TransformerBinner::create_function(&(|x| x.sqrt()), "Blah", &vec![from_namespace], &vec![40.], false).unwrap();
        let record_buffer = [7,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(4, 7) | IS_NOT_SINGLE_MASK | IS_FLOAT_NAMESPACE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            2.0f32.to_bits(),       // Feature value of the feature
                            3.0f32.to_bits()];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used

        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32(3, 2.0f32, SeedNumber::Default);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);
        
        
        // Now let's try with value> 40.0
        let record_buffer = [7,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(4, 7) | IS_NOT_SINGLE_MASK | IS_FLOAT_NAMESPACE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            2.0f32.to_bits(),       // Feature value of the feature
                            300.0f32.to_bits()];       // Float feature value

        let mut to_namespace = to_namespace_empty.clone();
        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32((300.0_f32).sqrt() as i32, 2.0f32, SeedNumber::One);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);        
    }

    #[test]
    fn test_transformerlogratiobinner() {
        
        let from_namespace_1 = ExecutorFromNamespace {
            namespace_index: 0,
            namespace_verbose: "a".to_string(),
            namespace_is_float: true,
        };

        let from_namespace_2 = ExecutorFromNamespace {
            namespace_index: 1,
            namespace_verbose: "c".to_string(),
            namespace_is_float: true,
        };
        let to_namespace_index = 1;
                            
        let to_namespace_empty = ExecutorToNamespace {
            namespace_index: to_namespace_index,
            namespace_verbose: "b".to_string(),
            namespace_seeds: default_seeds!(to_namespace_index),	// These are precomputed namespace seeds
            tmp_data: Vec::new(),
        };
        
        let transformer = TransformerLogRatioBinner::create_function("Blah", &vec![from_namespace_1, from_namespace_2], &vec![40., 10.0], false).unwrap();
        let record_buffer = [11,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(5, 8) | IS_NOT_SINGLE_MASK | IS_FLOAT_NAMESPACE_MASK, 
                            nd(8, 11) | IS_NOT_SINGLE_MASK | IS_FLOAT_NAMESPACE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            2.0f32.to_bits(),       // Feature value of the feature
                            3.0f32.to_bits(),
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            3.0f32.to_bits(),       // Feature value of the feature
                            7.0f32.to_bits(),
                            
                            ];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used
        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32_i32(3 as i32, 7 as i32, 6.0, SeedNumber::One);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);
        
        
        // Now let's have 30.0/60.0

        let record_buffer = [11,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(5, 8) | IS_NOT_SINGLE_MASK | IS_FLOAT_NAMESPACE_MASK, 
                            nd(8, 11) | IS_NOT_SINGLE_MASK | IS_FLOAT_NAMESPACE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            2.0f32.to_bits(),       // Feature value of the feature
                            30.0f32.to_bits(),
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            3.0f32.to_bits(),       // Feature value of the feature
                            60.0f32.to_bits(),
                            
                            ];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used
        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_f32((30.0/60.0_f32).ln() * 10.0, 6.0, false, SeedNumber::Default);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);

    }

    #[test]
    fn test_interpolation() {
        let to_namespace_empty = ExecutorToNamespace {
                namespace_index: 1,
                namespace_verbose: "b".to_string(),
                namespace_seeds: default_seeds!(1),	// These are precomputed namespace seeds
                tmp_data: Vec::new(),
            };
        let mut to_namespace = to_namespace_empty.clone();
        to_namespace.emit_f32(5.4, 20.0, true, SeedNumber::Default);
        let to_data_1:i32 = 6;
        let to_data_1_value = 20.0 * (5.4 - 5.0);
        let hash_index_1 = murmur3::hash32_with_seed(to_data_1.to_le_bytes(), to_namespace.namespace_seeds[SeedNumber::Default as usize]) & parser::MASK31;
        let to_data_2:i32 = 5;
        let to_data_2_value = 20.0 * (6.0 - 5.4);
        let hash_index_2 = murmur3::hash32_with_seed(to_data_2.to_le_bytes(), to_namespace.namespace_seeds[SeedNumber::Default as usize]) & parser::MASK31;
        assert_eq!(to_namespace.tmp_data, vec![(hash_index_1, to_data_1_value), (hash_index_2, to_data_2_value)]);            
    } 
    
    #[test]
    fn test_transformerweightmutliplier() {
        
        let from_namespace_float = ExecutorFromNamespace {
            namespace_index: 0,
            namespace_verbose: "a".to_string(),
            namespace_is_float: true,
        };
        let to_namespace_index = 1;
                            
        let to_namespace_empty = ExecutorToNamespace {
            namespace_index: to_namespace_index,
            namespace_verbose: "b".to_string(),
            namespace_seeds: default_seeds!(to_namespace_index),	// These are precomputed namespace seeds
            tmp_data: Vec::new(),
        };
        
        let transformer = TransformerMultiplyWeight::create_function("Blah", &vec![from_namespace_float], &vec![40.]).unwrap();
        let record_buffer = [7,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(4, 7) | IS_NOT_SINGLE_MASK | IS_FLOAT_NAMESPACE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            2.0f32.to_bits(),       // Feature value of the feature
                            3.0f32.to_bits()];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used

        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32((1775699190 & MASK31) as i32, 2.0f32 * 40., SeedNumber::Default);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);
        
        // But weightmultiplier can take non-float namespaces
        let from_namespace_nonfloat = ExecutorFromNamespace {
            namespace_index: 0,
            namespace_verbose: "a".to_string(),
            namespace_is_float: false,
        };

        let transformer = TransformerMultiplyWeight::create_function("Blah", &vec![from_namespace_nonfloat], &vec![40.]).unwrap();
        let record_buffer = [7,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(4, 6) | IS_NOT_SINGLE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            2.0f32.to_bits()];       // Feature value of the feature
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used

        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32((1775699190 & MASK31) as i32, 2.0f32 * 40., SeedNumber::Default);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);
        
        
        
    }



















}

