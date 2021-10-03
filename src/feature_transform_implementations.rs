
use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;


use crate::parser;
use crate::feature_reader;
use crate::feature_reader_float_namespace;

use crate::feature_transform_executor::{SeedNumber, ExecutorFromNamespace, ExecutorToNamespace, FunctionExecutorTrait, TransformExecutors};
use crate::feature_transform_parser;
use crate::vwmap::{NamespaceType, NamespaceFormat, NamespaceDescriptor};



// Basic example of a "full blown" simple FunctionExecutorTrait
#[derive(Clone)]
struct FunctionExampleSqrt {
    from_namespace: ExecutorFromNamespace,
}

impl FunctionExecutorTrait for FunctionExampleSqrt {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace, transform_executors: &TransformExecutors) {
        feature_reader_float_namespace!(record_buffer, self.from_namespace.namespace_descriptor, hash_index, hash_value, float_value, {
            let transformed_float = float_value.sqrt();
            let transformed_int = transformed_float as i32;
            to_namespace.emit_i32::<{SeedNumber::Default as usize}>(transformed_int, hash_value);
        });
    }
}

impl FunctionExampleSqrt {
    fn create_function(function_name: &str, from_namespaces: &Vec<feature_transform_parser::Namespace>, function_params: &Vec<f32>) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        // For simplicity of example, we just assert instead of full error reporting
        assert!(function_params.len() == 0);
        assert!(from_namespaces.len() == 1);
        assert!(from_namespaces[0].namespace_descriptor.namespace_type == NamespaceType::Primitive);
        assert!(from_namespaces[0].namespace_descriptor.namespace_format == NamespaceFormat::F32);
        Ok(Box::new(Self{from_namespace: ExecutorFromNamespace{namespace_descriptor: from_namespaces[0].namespace_descriptor}}))
    }   
}

// -------------------------------------------------------------------
// TransformerBinner - A basic binner
// It can take any function as a binning function f32 -> f32. Then output is rounded to integer 

// What does greater_than do? 
// If output is smaller than the first floating parameter (greater_than), then output is rounded to integer
// If the output is larger than the first floating point parameter (greater_than) then we first substract greater_than from the input and apply transform function
// Example of use: you want to bin number of pageviews per user, so you generally want to do sqrt on it, but only do floor binning when pageviews <= 10
// In that case you would call BinnerMinSqrt(A)(10.0, 1.0)

// What does resolution mean?
// Example: BinnerSqrt(X)(10.0, 2.0)
// let's assume X is 150. sqrt(150) * 1.0 = 12.247 
// Since reslution is 2.0, we will first mutiply 12.247 by 2 and get to 24.5. We then round that to integer = 24

// What does interpolated mean?
// Example: BinnerSqrt(X)(10.0, 1.0)
// let's assume X is 150. sqrt(150) * 1.0 = 12.247 
// You now want two values emitted - 12 at value 0.247 and 13 at value (1-0.247)




#[derive(Clone)]
pub struct TransformerBinner {
    from_namespace: ExecutorFromNamespace,
    greater_than: f32,
    resolution: f32,
    interpolated: bool,
    function_pointer: &'static (dyn Fn(f32, f32) -> f32 +'static + Sync), 
}

impl FunctionExecutorTrait for TransformerBinner {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace, transform_executors: &TransformExecutors) {
        feature_reader_float_namespace!(record_buffer, self.from_namespace.namespace_descriptor, hash_index, hash_value, float_value, {
            if float_value < self.greater_than {
                to_namespace.emit_i32::<{SeedNumber::Default as usize}>(float_value as i32, hash_value);
            } else {
                let transformed_float = (self.function_pointer)(float_value - self.greater_than, self.resolution);
                to_namespace.emit_f32::<{SeedNumber::One as usize}>(transformed_float, hash_value, self.interpolated);
            }
        });
    }
}


impl TransformerBinner {
    pub fn create_function(function_pointer: &'static (dyn Fn(f32, f32) -> f32 +'static + Sync), 
                        function_name: &str, 
                        from_namespaces: &Vec<feature_transform_parser::Namespace>, 
                        function_params: &Vec<f32>,
                        interpolated: bool,
                        ) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {

        if function_params.len() > 2 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes up to two float arguments, example {}(A)(2.0, 3.5). Both are optional.\nFirst parameter is the minimum parameter to apply function at (default: -MAX), second parameter is resolution (default: 1.0))", function_name, function_name))));
        }
        
        let greater_than = match function_params.get(0) {
            Some(&greater_than) => greater_than, 
            None => 0.0
        };
        if greater_than < 0.0 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} parameter greater_than cannot be negative (passed : {}))", function_name, greater_than))));
        }

        let resolution = match function_params.get(1) {
            Some(&resolution) => resolution, 
            None => 1.0
        };

//        println!("Greater than : {}, resolution: {}", greater_than, resolution);
        
        if from_namespaces.len() != 1 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes exactly one namespace argument, example {}(A)(2.0)", function_name, function_name))));
        }

        for namespace in from_namespaces.iter() {
            if namespace.namespace_descriptor.namespace_format != NamespaceFormat::F32 {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("All namespaces of function {} have to be of type f32: From namespace ({}) should be typed in vw_namespace_map.csv", function_name, namespace.namespace_verbose))));
            }
        }

        Ok(Box::new(Self{from_namespace: ExecutorFromNamespace{namespace_descriptor: from_namespaces[0].namespace_descriptor}, 
                        resolution: resolution,
                        greater_than: greater_than,
                        function_pointer: function_pointer,
                        interpolated: interpolated,
                        }))
    }
}   


// -------------------------------------------------------------------
// LogRatio Binner
//
//
//
//

#[derive(Clone)]
pub struct TransformerLogRatioBinner {
    from_namespace1: ExecutorFromNamespace,
    from_namespace2: ExecutorFromNamespace,
    greater_than: f32,
    interpolated: bool,
    resolution: f32,
}

impl FunctionExecutorTrait for TransformerLogRatioBinner {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace, transform_executors: &TransformExecutors) {
        feature_reader_float_namespace!(record_buffer, self.from_namespace1.namespace_descriptor, hash_index1, hash_value1, float_value1, {
            feature_reader_float_namespace!(record_buffer, self.from_namespace2.namespace_descriptor, hash_index2, hash_value2, float_value2, {

                let joint_value = hash_value1 * hash_value2;
                let val1 = float_value1;
                let val2 = float_value2;
                if val2 + val1 < self.greater_than {
                    to_namespace.emit_i32_i32::<{SeedNumber::One as usize}>(val1 as i32, val2 as i32, joint_value);    
                } else if val1 == 0.0 {
                    // val2 has to be greater or equal to self.greater_than (if it wasn't we'd take the first if branch
                    to_namespace.emit_f32::<{SeedNumber::Two as usize}>((val2 - self.greater_than).ln(), joint_value, self.interpolated);    
                } else if val2 == 0.0 {
                    // val1 has to be greater or equal to self.greater_than (if it wasn't we'd take the first if branch
                    to_namespace.emit_f32::<{SeedNumber::Three as usize}>((val1 - self.greater_than).ln(), joint_value, self.interpolated);    
                } else {
                    let o = (val1/val2).ln()*self.resolution;
                    to_namespace.emit_f32::<{SeedNumber::Default as usize}>(o, joint_value, self.interpolated);
                }
            });
        });
    }
}

impl TransformerLogRatioBinner {
    pub fn create_function(
                        function_name: &str, 
                        from_namespaces: &Vec<feature_transform_parser::Namespace>, 
                        function_params: &Vec<f32>,
                        interpolated: bool,
                        ) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {

        if function_params.len() > 2 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes up to two float arguments, example {}(A)(2.0, 3.5). Both are optional.\nFirst parameter is the minimum parameter to apply function at (default: -MAX), second parameter is resolution (default: 1.0))", function_name, function_name))));
        }
        
        let greater_than = match function_params.get(0) {
            Some(&greater_than) => greater_than, 
            None => 0.0
        };
        if greater_than < 0.0 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} parameter greater_than cannot be negative (passed : {}))", function_name, greater_than))));
        }

        let resolution = match function_params.get(1) {
            Some(&resolution) => resolution, 
            None => 1.0
        };

        if from_namespaces.len() != 2 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes exactly two namespace arguments, example {}(A,B)(2.0)", function_name, function_name))));            
        }
        for namespace in from_namespaces.iter() {
            if namespace.namespace_descriptor.namespace_format != NamespaceFormat::F32 {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("All namespaces of function {} have to be of type f32: From namespace ({}) should be typed in vw_namespace_map.csv", function_name, namespace.namespace_verbose))));
            }
        }

        Ok(Box::new(Self{from_namespace1: ExecutorFromNamespace{namespace_descriptor: from_namespaces[0].namespace_descriptor}, 
                        from_namespace2: ExecutorFromNamespace{namespace_descriptor: from_namespaces[1].namespace_descriptor}, 
                        resolution: resolution,
                        greater_than: greater_than,
                        interpolated: interpolated,
                        }))
    }
}   



// Value multiplier transformer
// -------------------------------------------------------------------
// TransformerWeight - A basic weight multiplier transformer
// Example of use: if you want to multiply whole namespace with certain factor and thus increase its learning rate (let's say 2.0)
// In that case you would call MutliplyWeight(document_id)(2.0)
// Important - document_id does not need to be float and isnt really changed 


#[derive(Clone)]
pub struct TransformerWeight {
    from_namespace: ExecutorFromNamespace,
    multiplier: f32,
}

impl FunctionExecutorTrait for TransformerWeight {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace, transform_executors: &TransformExecutors) {
        feature_reader!(record_buffer, transform_executors, self.from_namespace.namespace_descriptor, hash_index, hash_value, {
            to_namespace.emit_i32::<{SeedNumber::Default as usize}>(hash_index as i32, hash_value * self.multiplier);
        });
    }
}


impl TransformerWeight {
    pub fn create_function( function_name: &str, 
                        from_namespaces: &Vec<feature_transform_parser::Namespace>, 
                        function_params: &Vec<f32>,
                        ) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        if function_params.len() != 1 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes exactly one float argument, example {}(A)(2.0)", function_name, function_name))));            
        }
        if from_namespaces.len() != 1 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes exactly one namespace argument, example {}(A)(2.0)", function_name, function_name))));            
        }
        // We do not check if input namespace is float, Weight does not require float namespace as input
        
        Ok(Box::new(Self{from_namespace: ExecutorFromNamespace{namespace_descriptor: from_namespaces[0].namespace_descriptor},
                        multiplier: function_params[0], 
                        }))
    }
}   



// Combine Binner
// Supporting max 5 input namespaces. Because 5 ought to be enough for everybody!
// There is an issue that compilation time here is immense
//

#[derive(Clone)]
pub struct TransformerCombine {
    from_namespaces: [ExecutorFromNamespace; 4],
    n_namespaces: u8,
}

impl FunctionExecutorTrait for TransformerCombine {
    fn execute_function(&self, record_buffer: &[u32], to_namespace: &mut ExecutorToNamespace, transform_executors: &TransformExecutors) {
        // Sure this could have been written with either using:
        //   - Stack machine: I didn't want to introduce another dynamic layer
        //   - Automatic code generation: Didn't have time to learn macros that well
        // So we are left with good old "spaghetti technique"
        match self.n_namespaces {
            2 =>    feature_reader!(record_buffer, transform_executors, self.from_namespaces[0].namespace_descriptor, hash_index0, hash_value0, {
                        feature_reader!(record_buffer, transform_executors, self.from_namespaces[1].namespace_descriptor, hash_index1, hash_value1, {
                            to_namespace.emit_i32::<{SeedNumber::Default as usize}>((hash_index0 ^ hash_index1) as i32, hash_value0 * hash_value1);
                        });
                    }),
            3 =>    feature_reader!(record_buffer, transform_executors, self.from_namespaces[0].namespace_descriptor, hash_index0, hash_value0, {
                        feature_reader!(record_buffer, transform_executors, self.from_namespaces[1].namespace_descriptor, hash_index1, hash_value1, {
                            feature_reader!(record_buffer, transform_executors, self.from_namespaces[2].namespace_descriptor, hash_index2, hash_value2, {
                                to_namespace.emit_i32::<{SeedNumber::Default as usize}>((hash_index0 ^ hash_index1 ^ hash_index2) as i32, hash_value0 * hash_value1 * hash_value2);
                            });
                        });
                    }),
            4 =>    feature_reader!(record_buffer, transform_executors, self.from_namespaces[0].namespace_descriptor, hash_index0, hash_value0, {
                        feature_reader!(record_buffer, transform_executors, self.from_namespaces[1].namespace_descriptor, hash_index1, hash_value1, {
                            feature_reader!(record_buffer, transform_executors, self.from_namespaces[2].namespace_descriptor, hash_index2, hash_value2, {
                                feature_reader!(record_buffer, transform_executors, self.from_namespaces[3].namespace_descriptor, hash_index3, hash_value3, {
                                    to_namespace.emit_i32::<{SeedNumber::Default as usize}>((hash_index0 ^ hash_index1 ^ hash_index2 ^ hash_index3) as i32, 
                                                            hash_value0 * hash_value1 * hash_value2 * hash_value3);
                                });
                            });
                        });
                    }),
/* Disabled since we have compilation time issues */
/*            5 =>    feature_reader!(record_buffer, transform_executors, self.from_namespaces[0].namespace_descriptor, hash_index0, hash_value0, {
                        feature_reader!(record_buffer, transform_executors, self.from_namespaces[1].namespace_descriptor, hash_index1, hash_value1, {
                            feature_reader!(record_buffer, transform_executors, self.from_namespaces[2].namespace_descriptor, hash_index2, hash_value2, {
                                feature_reader!(record_buffer, transform_executors, self.from_namespaces[3].namespace_descriptor, hash_index3, hash_value3, {
                                    feature_reader!(record_buffer, transform_executors, self.from_namespaces[4].namespace_descriptor, hash_index4, hash_value4, {
                                        to_namespace.emit_i32::<{SeedNumber::Default as usize}>((hash_index0 ^ hash_index1 ^ hash_index2 ^ hash_index3 ^ hash_index4) as i32, 
                                                                hash_value0 * hash_value1 * hash_value2 * hash_value3 * hash_value4);
                                    });                                                                
                                });
                            });
                        });
                    }),*/
            _ => {
                panic!("Impossible number of from_namespaces in function TransformCombine - this should have been caught at parsing stage")
            } 
                    
        }
    }
}

impl TransformerCombine {
    pub fn create_function(
                        function_name: &str, 
                        from_namespaces: &Vec<feature_transform_parser::Namespace>, 
                        function_params: &Vec<f32>,
                        ) -> Result<Box<dyn FunctionExecutorTrait>, Box<dyn Error>> {
        if function_params.len() != 0 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes no float arguments {}(A)()", function_name, function_name))));
        }
        if from_namespaces.len() < 2 || from_namespaces.len() > 4 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Function {} takes between 2 and 4 namespace arguments, example {}(A,B)()", function_name, function_name))));
        }
        // We do not need to check if the input namespace is float, Combine does not require float namespace as input        

        // We use fixed arrays, so we need to fill the array with defaults first
        let c = ExecutorFromNamespace{namespace_descriptor: NamespaceDescriptor {namespace_index: 0, 
                                                                                namespace_type: NamespaceType::Primitive, 
                                                                                namespace_format: NamespaceFormat::Categorical}, 
                                    };
        let mut executor_from_namespaces: [ExecutorFromNamespace;4] = [c.clone(),c.clone(),c.clone(),c.clone()];
        for (x, namespace) in from_namespaces.iter().enumerate() {
            executor_from_namespaces[x].namespace_descriptor = namespace.namespace_descriptor;
        }

        Ok(Box::new(Self{from_namespaces: executor_from_namespaces,
                        n_namespaces: from_namespaces.len() as u8, 
                        }))
    }
}   








mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::parser::{IS_NOT_SINGLE_MASK, MASK31};
    use crate::feature_transform_executor::default_seeds;

    fn add_header(v2: Vec<u32>) -> Vec<u32> {
        let mut rr: Vec<u32> = vec![100, 1, 1.0f32.to_bits()];
        rr.extend(v2);
        rr
    }
    
    fn nd(start: u32, end: u32) -> u32 {
        return (start << 16) + end;
    }
    
    fn ns_desc(i: u16) -> NamespaceDescriptor {
        NamespaceDescriptor {namespace_index: i, 
                             namespace_type: NamespaceType::Primitive,
                             namespace_format: NamespaceFormat::Categorical,
                             }
    }

    fn ns_desc_f32(i: u16) -> NamespaceDescriptor {
        NamespaceDescriptor {namespace_index: i, 
                             namespace_type: NamespaceType::Primitive,
                             namespace_format: NamespaceFormat::F32,
                             }

    }


    #[test]
    fn test_transformerbinner_fail() {
        // this fails because input namespace is not float namespace    
        let from_namespace = feature_transform_parser::Namespace {
            namespace_verbose: "a".to_string(),
            namespace_descriptor: ns_desc(0),
        };
        
        let to_namespace_empty = ExecutorToNamespace {
            namespace_descriptor: ns_desc(1),
            namespace_seeds: default_seeds(1),	// These are precomputed namespace seeds
            tmp_data: Vec::new(),
        };
        
        let result = TransformerBinner::create_function(&(|x, y| x.sqrt() * y), "Blah", &vec![from_namespace], &vec![40., 1.4], false);
        assert!(result.is_err());

    }

    #[test]
    fn test_transformerbinner() {
        
        let from_namespace = feature_transform_parser::Namespace {
            namespace_descriptor: ns_desc_f32(0),
            namespace_verbose: "a".to_string(),
        };
        let to_namespace_index = 1;
                            
        let to_namespace_empty = ExecutorToNamespace {
            namespace_descriptor: ns_desc(to_namespace_index),
            namespace_seeds: default_seeds(to_namespace_index as u32),	// These are precomputed namespace seeds
            tmp_data: Vec::new(),
        };
        
        let transformer = TransformerBinner::create_function(&(|x, y| x.sqrt() * y), "Blah", &vec![from_namespace], &vec![40.0, 1.], false).unwrap();
        let record_buffer = [6,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(4, 6) | IS_NOT_SINGLE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            3.0f32.to_bits()];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used

        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32::<{SeedNumber::Default as usize}>(3, 1.0f32);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);
        
        
        // Now let's try with value> 40.0
        let record_buffer = [6,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(4, 6) | IS_NOT_SINGLE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            300.0f32.to_bits()];       // Float feature value

        let mut to_namespace = to_namespace_empty.clone();
        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32::<{SeedNumber::One as usize}>((300.0_f32 - 40.0_f32).sqrt() as i32, 1.0f32);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);        
    }

    #[test]
    fn test_transformerlogratiobinner() {
        
        let from_namespace_1 = feature_transform_parser::Namespace {
            namespace_descriptor: ns_desc_f32(0),
            namespace_verbose: "a".to_string(),
        };

        let from_namespace_2 = feature_transform_parser::Namespace {
            namespace_descriptor: ns_desc_f32(1),
            namespace_verbose: "c".to_string(),
        };
        
        let to_namespace_index = 1;                           
        let to_namespace_empty = ExecutorToNamespace {
            namespace_descriptor: ns_desc(to_namespace_index),
            namespace_seeds: default_seeds(to_namespace_index as u32),	// These are precomputed namespace seeds
            tmp_data: Vec::new(),
        };
        
        let transformer = TransformerLogRatioBinner::create_function("Blah", &vec![from_namespace_1, from_namespace_2], &vec![40.0, 10.], false).unwrap();
        let record_buffer = [9,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(5, 7) | IS_NOT_SINGLE_MASK, 
                            nd(7, 9) | IS_NOT_SINGLE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            3.0f32.to_bits(),
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            7.0f32.to_bits(),
                            ];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used
        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32_i32::<{SeedNumber::One as usize}>(3 as i32, 7 as i32, 1.0);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);
        
        

        // Now let's have 30.0/60.0
        let record_buffer = [9,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(5, 7) | IS_NOT_SINGLE_MASK, 
                            nd(7, 9) | IS_NOT_SINGLE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            30.0f32.to_bits(),
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            60.0f32.to_bits(),
                            
                            ];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used
        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_f32::<{SeedNumber::Default as usize}>((30.0/60.0_f32).ln() * 10.0, 1.0, false);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);


        // Now let's have 30.0/0.0
        let record_buffer = [9,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(5, 7) | IS_NOT_SINGLE_MASK, 
                            nd(7, 9) | IS_NOT_SINGLE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            30.0f32.to_bits(),
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            0.0f32.to_bits(),
                            
                            ];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used
        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32_i32::<{SeedNumber::One as usize}>(30, 0, 1.0);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);

        // Now let's have 0.0/50.0
        let record_buffer = [9,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(5, 7) | IS_NOT_SINGLE_MASK, 
                            nd(7, 9) | IS_NOT_SINGLE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            0.0f32.to_bits(),
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            50.0f32.to_bits(),
                            
                            ];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used
        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_f32::<{SeedNumber::Two as usize}>((50_f32 - 40_f32).ln(), 1.0, false);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);



        // Now let's have 50.0/0.0
        let record_buffer = [9,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(5, 7) | IS_NOT_SINGLE_MASK, 
                            nd(7, 9) | IS_NOT_SINGLE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            50.0f32.to_bits(),
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            0.0f32.to_bits(),
                            
                            ];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used
        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_f32::<{SeedNumber::Three as usize}>((50_f32 - 40_f32).ln(), 1.0, false);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);



    }

    
    #[test]
    fn test_transformerweightmutliplier() {
        
        let from_namespace_float = feature_transform_parser::Namespace {
            namespace_descriptor: ns_desc_f32(0),
            namespace_verbose: "a".to_string(),
        };
        let to_namespace_index = 1;
                            
        let to_namespace_empty = ExecutorToNamespace {
            namespace_descriptor: ns_desc(to_namespace_index),
            namespace_seeds: default_seeds(to_namespace_index as u32),	// These are precomputed namespace seeds
            tmp_data: Vec::new(),
        };
        
        let transformer = TransformerWeight::create_function("Blah", &vec![from_namespace_float], &vec![40.]).unwrap();
        let record_buffer = [6,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(4, 6) | IS_NOT_SINGLE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            3.0f32.to_bits()];       // Float feature value
 
        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used

        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32::<{SeedNumber::Default as usize}>((1775699190 & MASK31) as i32, 1.0f32 * 40.);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);
        
        // But weightmultiplier can take non-float namespaces
        let from_namespace_nonfloat = feature_transform_parser::Namespace {
            namespace_descriptor: ns_desc(0),
            namespace_verbose: "a".to_string(),
        };

        let transformer = TransformerWeight::create_function("Blah", &vec![from_namespace_nonfloat], &vec![40.]).unwrap();
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
        to_namespace_comparison.emit_i32::<{SeedNumber::Default as usize}>((1775699190 & MASK31) as i32, 2.0f32 * 40.);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);
        
        
        
    }


    #[test]
    fn test_transformercombine() {
        
        let from_namespace_1 = feature_transform_parser::Namespace {
            namespace_descriptor: ns_desc_f32(0),
            namespace_verbose: "a".to_string(),
        };


        let from_namespace_2 = feature_transform_parser::Namespace {
            namespace_descriptor: ns_desc(1),
            namespace_verbose: "b".to_string(),
        };

        let to_namespace_index = 2;
                            
        let to_namespace_empty = ExecutorToNamespace {
            namespace_descriptor: ns_desc(to_namespace_index),
            namespace_seeds: default_seeds(to_namespace_index as u32),	// These are precomputed namespace seeds
            tmp_data: Vec::new(),
        };
        
        let transformer = TransformerCombine::create_function("Blah", &vec![from_namespace_1, from_namespace_2], &vec![]).unwrap();

        let record_buffer = [9,	// length 
                            0,	// label
                            (1.0_f32).to_bits(), // Example weight 
                            nd(5, 7) | IS_NOT_SINGLE_MASK, 
                            nd(7, 9) | IS_NOT_SINGLE_MASK, 
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            3.0f32.to_bits(),	    // Float value of the feature
                            // Feature triple
                            1775699190 & MASK31,    // Hash location 
                            3.0f32.to_bits(),       // Weight of the feature
                            ];      

        let mut to_namespace = to_namespace_empty.clone();
        let mut transform_executors = TransformExecutors {executors: vec![]}; // not used

        transformer.execute_function(&record_buffer, &mut to_namespace, &mut transform_executors);

        // Couldn't get mocking to work, so instead of intercepting call to emit_i32, we just repeat it and see if the results match
        let mut to_namespace_comparison = to_namespace_empty.clone();
        to_namespace_comparison.emit_i32::<{SeedNumber::Default as usize}>((1775699190 ^ 1775699190) as i32, 3.0f32);
        assert_eq!(to_namespace.tmp_data, to_namespace_comparison.tmp_data);
    }


}

