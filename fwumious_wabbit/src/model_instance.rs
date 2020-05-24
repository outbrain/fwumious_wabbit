use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;

use std::io::Read;
use std::fs::File;
use serde::{Serialize,Deserialize};//, Deserialize};
use clap::Arg;
use serde_json::{Value,from_str};

use crate::vwmap;


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FeatureComboDesc {
    pub feature_indices: Vec<usize>,
    pub weight:f32,
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelInstance {
    pub learning_rate: f32,    
    pub minimum_learning_rate: f32,
    pub power_t: f32,
    pub bit_precision: u8,
    pub hash_mask: u32,
    pub add_constant_feature: bool,
    pub feature_combo_descs: Vec<FeatureComboDesc>,
    pub ffm_fields: Vec<Vec<usize>>, // we'll make first version fully compatible with VW's --lrmfa
    pub ffm_k: u32,
    pub ffm_bit_precision: u32,
}


impl ModelInstance {
    pub fn new_empty() -> Result<ModelInstance, Box<dyn Error>> {
        let mut mi = ModelInstance {
            learning_rate: 0.5, // vw default
            minimum_learning_rate: 0.0, 
            bit_precision: 18,      // vw default
            hash_mask: (1 << 18) -1,
            power_t: 0.5,
            add_constant_feature: true,
            feature_combo_descs: Vec::new(),
            ffm_fields: Vec::new(),
            ffm_k: 0,
            ffm_bit_precision: 18,
        };
        Ok(mi)
    }
    
    pub fn new_from_cmdline<'a>(cl: &clap::ArgMatches<'a>, vw: &vwmap::VwNamespaceMap) -> Result<ModelInstance, Box<dyn Error>> {
        let mut mi = ModelInstance::new_empty()?;
        let mut add_namespaces_combo = |namespaces_str: String| -> Result<(), Box<dyn Error>> {
            //let mut feature_vec: Vec<usize> = Vec::new();
            let mut feature_combo_desc = FeatureComboDesc {
                                        feature_indices: Vec::new(),
                                        weight: 1.0
                                        };
            for char in namespaces_str.chars() {
               // create an list of indexes dfrom list of namespace chars
               let index = match vw.map_char_to_index.get(&char) {
                   Some(index) => *index,
                   None => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown namespace char in command line: {}", char))))
               };
               feature_combo_desc.feature_indices.push(index);
               // now we handle calculating total correct weight for combo feature
               let feature_name:&String = vw.map_char_to_name.get(&char).unwrap();
               let ss:Vec<&str> = feature_name.split(":").collect();
               match ss.len() {
                   1 => continue,
                   2 => {
                       let weight:f32 = ss[1].parse()?;
                       feature_combo_desc.weight *= weight;
                   },
                   _ => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Feature name definition has multiple weights separated by colon: {}", feature_name))))
               }
            }
            mi.feature_combo_descs.push(feature_combo_desc);
            Ok(())
        };

        if let Some(in_v) = cl.values_of("keep") {
            for namespaces_str in in_v {
                if namespaces_str.len() != 1 {
                    return Err(Box::new(IOError::new(ErrorKind::Other, format!("--keep can only have single letter as a namespace parameter: {}", namespaces_str))))
                }
                add_namespaces_combo(namespaces_str.to_string())?;
            }
        }
        
        if let Some(in_v) = cl.values_of("interactions") {
            for namespaces_str in in_v {                
//                println!("An input keep parameter: {}", namespaces_str);
                if namespaces_str.len() <= 1 {
                    return Err(Box::new(IOError::new(ErrorKind::Other, format!("--interactions needs two or more namespaces: \"{}\"", namespaces_str))))
                }
                add_namespaces_combo(namespaces_str.to_string())?;
            }
        }

        if let Some(in_v) = cl.value_of("lrqfa") {
            let vsplit: Vec<&str> = in_v.split("-").collect(); // We use - as a delimiter instead of first numbers as vowpal does it
            if vsplit.len() != 2 {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("--lrqfa takes namespaces-k, like ABC-12, your string was: \"{}\"", in_v))))
            }
            let namespaces_str = vsplit[0];
            let k_str = vsplit[1];
            for char in namespaces_str.chars() {
                // create an list of indexes dfrom list of namespace chars
                let index = match vw.map_char_to_index.get(&char) {
                    Some(index) => *index,
                    None => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown namespace char in command line: {}", char))))
                };
                mi.ffm_fields.push(vec![index]);
            }
            mi.ffm_k = k_str.parse().expect("Number expected");
        }

        if let Some(val) = cl.value_of("ffm_k") {
            mi.ffm_k = val.parse()?;
        }

        if let Some(in_v) = cl.values_of("ffm_field") {
            for namespaces_str in in_v {          
                let mut field: Vec<usize>= Vec::new();
                for char in namespaces_str.chars() {
                    //println!("K: {}", char);
                    let index = match vw.map_char_to_index.get(&char) {
                        Some(index) => *index,
                        None => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown namespace char in command line: {}", char))))
                    };
                    field.push(index);
                }
                mi.ffm_fields.push(field);
            }
        }
        if let Some(val) = cl.value_of("ffm_bit_precision") {
            mi.ffm_bit_precision = val.parse()?;
            println!("FFM num weight bits = {}", mi.ffm_bit_precision); // vwcompat
        }

        if let Some(val) = cl.value_of("bit_precision") {
            mi.bit_precision = val.parse()?;
            println!("Num weight bits = {}", mi.bit_precision); // vwcompat
            mi.hash_mask = (1 << mi.bit_precision) -1;
        }
        if let Some(val) = cl.value_of("learning_rate") {
            mi.learning_rate = val.parse()?;
        }

        if let Some(val) = cl.value_of("minimum_learning_rate") {
            mi.minimum_learning_rate = val.parse()?;
        }

        if let Some(val) = cl.value_of("power_t") {
            mi.power_t = val.parse()?;
        }
        if let Some(val) = cl.value_of("link") {
            if val != "logistic" {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("--link only supports 'logistic'"))))
            }            
        }
        if let Some(val) = cl.value_of("loss_function") {
            if val != "logistic" {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("--loss_function only supports 'logistic'"))))
            }            
        }
        if let Some(val) = cl.value_of("l2") {
            let v2:f32 = val.parse()?;
            if v2.abs() > 0.00000001 {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("--l2 can only be 0.0"))))
            }
        }
        
        
        if cl.is_present("noconstant") {
            mi.add_constant_feature = false;
//               return Err(Box::new(IOError::new(ErrorKind::Other, format!("You must use --noconstant"))))
        }

        // We currently only support SGD + adaptive, which means both options have to be specified
        if !cl.is_present("sgd") {
               return Err(Box::new(IOError::new(ErrorKind::Other, format!("You must use --sgd"))))
        }
        if !cl.is_present("adaptive") {
               return Err(Box::new(IOError::new(ErrorKind::Other, format!("You must use --adaptive"))))
        }

        
        // Vowpal supports a mode with "prehashed" features, where numeric strings are treated as
        // numeric precomputed hashes. This is even default option.
        // It is generally a bad idea except if you strings really are precomputed hashes... 
        if !cl.is_present("hash") {
               return Err(Box::new(IOError::new(ErrorKind::Other, format!("You have to use --hash all "))))
        } else
        if let Some(val) = cl.value_of("hash") {
            if val != "all" {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("Only --hash all supported"))))
            }            
        }
        Ok(mi)
    }


    pub fn new_from_jsonfile(input_filename: &str, vw: &vwmap::VwNamespaceMap) -> Result<ModelInstance, Box<dyn Error>> {
        let mut mi = ModelInstance::new_empty()?;
        let mut input = File::open(input_filename)?;
        let mut contents = String::new();
        input.read_to_string(&mut contents)?;
        let j: Value = serde_json::from_str(&contents)?;
        let descj = &j["desc"];
        mi.learning_rate = descj["learning_rate"].as_f64().unwrap() as f32;
        mi.bit_precision = descj["bit_precision"].as_u64().unwrap() as u8;
        let features = descj["features"].as_array().unwrap();
        for feature in features {
            let mut feature_combo_desc = FeatureComboDesc {
                                feature_indices: Vec::new(),
                                weight: 1.0,
                                };

//            let mut feature_vec: Vec<usize> = Vec::new();
            let fname = feature.as_str().unwrap();
            let primitive_features = fname.split(",");
            for primitive_feature_name in primitive_features {
            //println!("F: {:?}", primitive_feature_name);
                let index = match vw.map_name_to_index.get(primitive_feature_name) {
                    Some(index) => *index,
                    None => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown feature name in model json: {}", primitive_feature_name))))
                };
                feature_combo_desc.feature_indices.push(index);
            }
            mi.feature_combo_descs.push(feature_combo_desc);
//            mi.feature_combos.push(feature_vec);
        }

        Ok(mi)
    }
    
    
    
    
}