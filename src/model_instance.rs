use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;

use std::io::Read;
use std::io;
use std::fs;
use std::fs::File;
use serde::{Serialize,Deserialize};//, Deserialize};
use serde_json::{Value};

use crate::vwmap;
use crate::consts;
use crate::feature_transform_parser;
use crate::regressor;
use crate::vwmap::{NamespaceDescriptor, NamespaceType};


#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct FeatureComboDesc {
    pub namespace_descriptors: Vec<vwmap::NamespaceDescriptor>,
    pub weight:f32,
}


#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Copy)]
pub enum Optimizer {
    SGD = 1,
    Adagrad = 2,
}

pub type FieldDesc = Vec<vwmap::NamespaceDescriptor>;


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelInstance {
    pub learning_rate: f32,
    #[serde(default = "default_f32_zero")]
    pub minimum_learning_rate: f32,
    pub power_t: f32,
    pub bit_precision: u8,
    pub hash_mask: u32,		// DEPRECATED, UNUSED -- this is recalculated in feature_buffer.rs
    pub add_constant_feature: bool,
    pub feature_combo_descs: Vec<FeatureComboDesc>,
    pub ffm_fields: Vec<FieldDesc>,
    pub ffm_interactions: Vec<(u32, u32, f32)>, // triplet of interactions - first field, second field, strength[A
    #[serde(default = "default_u32_zero")]
    pub ffm_k: u32,
    #[serde(default = "default_u32_zero")]
    pub ffm_bit_precision: u32,
    #[serde(default = "default_bool_false")]
    pub ffm_separate_vectors: bool, // DEPRECATED, UNUSED
    #[serde(default = "default_bool_false")]
    pub fastmath: bool,
    #[serde(default = "default_bool_false")]
    pub ffm_interaction_matrix: bool,

    #[serde(default = "default_f32_zero")]
    pub ffm_k_threshold: f32,
    #[serde(default = "default_f32_zero")]
    pub ffm_init_center: f32,
    #[serde(default = "default_f32_zero")]
    pub ffm_init_width: f32,
    #[serde(default = "default_f32_zero")]
    pub ffm_init_zero_band: f32,	// from 0.0 to 1.0, percentage of ffm_init_width
    #[serde(default = "default_f32_zero")]
    pub ffm_init_acc_gradient: f32,
    #[serde(default = "default_f32_zero")]
    pub init_acc_gradient: f32,
    // these are only used for learning, so it doesnt matter they got set to zero as default
    #[serde(default = "default_f32_zero")]
    pub ffm_learning_rate: f32,
    #[serde(default = "default_f32_zero")]
    pub ffm_power_t: f32,

    #[serde(default = "default_optimizer_adagrad")]
    pub optimizer: Optimizer,

    pub transform_namespaces: feature_transform_parser::NamespaceTransforms,

}

fn default_u32_zero() -> u32{0}
fn default_f32_zero() -> f32{0.0}
fn default_bool_false() -> bool{false}
fn default_optimizer_adagrad() -> Optimizer{Optimizer::Adagrad}

impl ModelInstance {
    pub fn new_empty() -> Result<ModelInstance, Box<dyn Error>> {
        let mi = ModelInstance {
            learning_rate: 0.5, // vw default
            ffm_learning_rate: 0.5, // vw default
            minimum_learning_rate: 0.0,
            bit_precision: 18,      // vw default
            hash_mask: 0, // DEPRECATED, UNUSED
            power_t: 0.5,
            ffm_power_t: 0.5,
            add_constant_feature: true,
            feature_combo_descs: Vec::new(),
            ffm_fields: Vec::new(),
            ffm_interactions: Vec::new(),
            ffm_k: 0,
            ffm_bit_precision: 18,
            ffm_separate_vectors: false, // DEPRECATED, UNUSED
            fastmath: true,
            ffm_k_threshold: 0.0,
            ffm_init_center: 0.0,
            ffm_init_width: 0.0,
            ffm_init_zero_band: 0.0,
            ffm_init_acc_gradient: 0.0,
            ffm_interaction_matrix: false,
            init_acc_gradient: 1.0,

            optimizer: Optimizer::SGD,
            transform_namespaces: feature_transform_parser::NamespaceTransforms::new(),
        };
        Ok(mi)
    }


    pub fn create_feature_combo_desc(&self, vw: &vwmap::VwNamespaceMap, s: &str) -> Result<FeatureComboDesc, Box<dyn Error>> {

        let vsplit: Vec<&str> = s.split(":").collect(); // We use : as a delimiter for weight
        let mut combo_weight: f32 = 1.0;
        if vsplit.len() > 2 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("only one value parameter allowed (denoted with \":\"): \"{:?}\"", s))))
        }
        if vsplit.len() == 2 {
            let weight_str = vsplit[1];
            combo_weight = weight_str.parse()?;
        }

        let namespaces_str = vsplit[0];
        let mut namespace_descriptors: Vec<vwmap::NamespaceDescriptor> = Vec::new();
        for char in namespaces_str.chars() {
           let namespace_descriptor = feature_transform_parser::get_namespace_descriptor(&self.transform_namespaces, vw, char)?;
           namespace_descriptors.push(namespace_descriptor);
        }
        Ok(FeatureComboDesc {
                              namespace_descriptors: namespace_descriptors,
                              weight: combo_weight
                            })
    }

    fn create_feature_combo_desc_from_verbose(&self, vw: &vwmap::VwNamespaceMap, s: &str) -> Result<FeatureComboDesc, Box<dyn Error>> {
        let vsplit: Vec<&str> = s.split(":").collect(); // We use : as a delimiter for weight
        let mut combo_weight: f32 = 1.0;

        if vsplit.len() == 2 {
            let weight_str = vsplit[1];
            combo_weight = match weight_str.parse() {
               Ok(x) => x,
               Err(y) => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Could not parse the value of a feature combination: {}", weight_str))))
            }
        } else if vsplit.len() > 2 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Verbose features cannot have \":\" as part of their names: \"{:?}\"", s))))
        }

        let namespaces_verbose: Vec<&str> = vsplit[0].split(",").collect();  // verbose names are separated by comma
        let mut namespace_descriptors: Vec<vwmap::NamespaceDescriptor> = Vec::new();
        for namespace_verbose in namespaces_verbose {
           let namespace_descriptor = feature_transform_parser::get_namespace_descriptor_verbose(&self.transform_namespaces, vw, namespace_verbose)?;
           namespace_descriptors.push(namespace_descriptor);
        }
        Ok(FeatureComboDesc {
                              namespace_descriptors: namespace_descriptors,
                              weight: combo_weight
                            })
    }

    fn create_field_desc_from_verbose(&self, vw: &vwmap::VwNamespaceMap, s: &str) -> Result<FieldDesc, Box<dyn Error>> {
        let vsplit: Vec<&str> = s.split(":").collect(); // We use : as a delimiter for weight
        if vsplit.len() > 1 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Fields currently do not support passing a value via : {:?}", s))))
        }
        let namespaces_verbose: Vec<&str> = s.split(",").collect();  // verbose names are separated by comma
        let mut field: FieldDesc = Vec::new();
        for namespace_verbose in namespaces_verbose {
            let namespace_descriptor = feature_transform_parser::get_namespace_descriptor_verbose(&self.transform_namespaces, vw, namespace_verbose)?;
            field.push(namespace_descriptor);
        }
        Ok(field)
    }


    pub fn parse_ffm_interaction_mask(&self, s: &str, num_fields: i32) -> Result<(u32, u32, f32), Box<dyn Error>> {
        let vsplit: Vec<&str> = s.split(":").collect(); // We use : as a delimiter for splitting first and second field id
        if vsplit.len() != 3 {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Field interaction mask can be field_id:field_id or field_id1:field_id2:value, cannot parse: {}", s))))
        }
        let field_id_1: u32 = match vsplit[0].parse() {
           Ok(x) => x,
           Err(y) => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Could not parse the id of a field 1: {}", vsplit[0]))))
        };
        let field_id_2: u32 = match vsplit[1].parse() {
           Ok(x) => x,
           Err(y) => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Could not parse the id of a field 2: {}", vsplit[1]))))
        };

        let weight_str = vsplit[2];
        let mask:f32 = match weight_str.parse() {
           Ok(x) => x,
           Err(y) => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Could not parse the value of a field interaction: {}", weight_str))))
        };


        if field_id_1 as i32 >= num_fields {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Field interaction mask field_id_1 can only be between 0 and {} (number of fields), found: {}", num_fields - 1, field_id_1))))
        }

        if field_id_2 as i32 >= num_fields {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Field interaction mask field_id_2 can only be between 0 and {} (number of fields), found: {}", num_fields - 1, field_id_2))))
        }


        Ok((field_id_1, field_id_2, mask))
    }


    pub fn new_from_cmdline<'a>(cl: &clap::ArgMatches<'a>, vw: &vwmap::VwNamespaceMap) -> Result<ModelInstance, Box<dyn Error>> {
        let mut mi = ModelInstance::new_empty()?;

        let vwcompat: bool = cl.is_present("vwcompat");

        if vwcompat {
            mi.fastmath = false;

            mi.init_acc_gradient = 0.0;

            if !cl.is_present("keep") {
                return Err(Box::new(IOError::new(ErrorKind::Other, "--vwcompat requires at least one --keep parameter, we do not implicitly take all features available")))
            }

            // Vowpal supports a mode with "prehashed" features, where numeric strings are treated as
            // numeric precomputed hashes. This is even default option.
            // It is generally a bad idea except if you strings really are precomputed hashes...
            if !cl.is_present("hash") {
                   return Err(Box::new(IOError::new(ErrorKind::Other, format!("--vwcompat requires use of --hash all"))))
            } else

            if let Some(val) = cl.value_of("hash") {
                if val != "all" {
                    return Err(Box::new(IOError::new(ErrorKind::Other, format!("--vwcompat requires use of --hash all"))))
                }
            }

            // --sgd will turn off adaptive, invariant and normalization in vowpal. You can turn adaptive back on in vw and fw with --adaptive
            if !cl.is_present("sgd") {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("--vwcompat requires use of --sgd"))))
            }


        }

        // we first need transform namespaces, before processing keep or interactions

        if let Some(in_v) = cl.values_of("transform") {
            let mut namespace_parser = feature_transform_parser::NamespaceTransformsParser::new();
            for value_str in in_v {
                namespace_parser.add_transform_namespace(vw, value_str)?;
            }
            mi.transform_namespaces = namespace_parser.resolve(vw)?;
        }

        if let Some(in_v) = cl.values_of("keep") {
            for value_str in in_v {
                mi.feature_combo_descs.push(mi.create_feature_combo_desc(vw, value_str)?);
            }
        }

        if let Some(in_v) = cl.values_of("interactions") {
            for value_str in in_v {
                mi.feature_combo_descs.push(mi.create_feature_combo_desc(vw, value_str)?);
            }
        }

        if let Some(in_v) = cl.values_of("linear") {
            for value_str in in_v {
                mi.feature_combo_descs.push(mi.create_feature_combo_desc_from_verbose(vw, value_str)?);
            }
        }

        if let Some(val) = cl.value_of("ffm_k") {
            mi.ffm_k = val.parse()?;
            if mi.ffm_k > consts::FFM_MAX_K as u32{
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("Maximum ffm_k is: {}, passed: {}", consts::FFM_MAX_K, mi.ffm_k))))
            }
        }

        if let Some(val) = cl.value_of("ffm_init_center") {
            mi.ffm_init_center = val.parse()?;
        }

        if let Some(val) = cl.value_of("ffm_init_width") {
            mi.ffm_init_width = val.parse()?;
        }

        if let Some(val) = cl.value_of("init_acc_gradient") {
            if vwcompat {
                return Err(Box::new(IOError::new(ErrorKind::Other, "Initial accumulated gradient is not supported in --vwcompat mode")))
            }
            mi.init_acc_gradient = val.parse()?;
        }

        if let Some(val) = cl.value_of("ffm_init_acc_gradient") {
            mi.ffm_init_acc_gradient = val.parse()?;
        } else {
            mi.ffm_init_acc_gradient = mi.init_acc_gradient;
        }

        if let Some(in_v) = cl.values_of("ffm_field") {
            for namespaces_str in in_v {
                let mut field: Vec<vwmap::NamespaceDescriptor>= Vec::new();
                for char in namespaces_str.chars() {
                    //println!("K: {}", char);
                    let namespace_descriptor = feature_transform_parser::get_namespace_descriptor(&mi.transform_namespaces, vw, char)?;
                    field.push(namespace_descriptor);
                }
                mi.ffm_fields.push(field);
            }
        }

        if let Some(in_v) = cl.values_of("ffm_field_verbose") {
            for value_str in in_v {
                mi.ffm_fields.push(mi.create_field_desc_from_verbose(vw, value_str)?);
            }
        }

        mi.ffm_interaction_matrix = cl.is_present("ffm_interaction_matrix");

        if let Some(in_v) = cl.values_of("ffm_interaction") {
            if !mi.ffm_interaction_matrix {
                return Err(Box::new(IOError::new(ErrorKind::Other, "You need to turn on --ffm_interaction_matrix")))
            }
            for value_str in in_v {
                mi.ffm_interactions.push(mi.parse_ffm_interaction_mask(value_str, mi.ffm_fields.len() as i32)?);
            }
        }


        if let Some(val) = cl.value_of("ffm_bit_precision") {
            mi.ffm_bit_precision = val.parse()?;
            println!("FFM num weight bits = {}", mi.ffm_bit_precision); // vwcompat
        }

        if let Some(val) = cl.value_of("bit_precision") {
            mi.bit_precision = val.parse()?;
            println!("Num weight bits = {}", mi.bit_precision); // vwcompat
        }

        if let Some(val) = cl.value_of("learning_rate") {
            mi.learning_rate = val.parse()?;
        }
        if let Some(val) = cl.value_of("ffm_learning_rate") {
            mi.ffm_learning_rate = val.parse()?;
        } else {
            mi.ffm_learning_rate = mi.learning_rate;
        }





        if let Some(val) = cl.value_of("minimum_learning_rate") {
            mi.minimum_learning_rate = val.parse()?;
        }

        if let Some(val) = cl.value_of("power_t") {
            mi.power_t = val.parse()?;
        }
        if let Some(val) = cl.value_of("ffm_power_t") {
            mi.ffm_power_t = val.parse()?;
        } else {
            mi.ffm_power_t = mi.power_t;
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
        }

        // We currently only support SGD + adaptive, which means both options have to be specified
        if cl.is_present("sgd") {
            mi.optimizer = Optimizer::SGD;
        }

        if cl.is_present("adaptive") {
            mi.optimizer = Optimizer::Adagrad;
        }

        Ok(mi)
    }


	pub fn update_hyperparameters_from_cmd<'a>(cmd_arguments: &clap::ArgMatches<'a>, mi: &mut ModelInstance) -> Result<(), Box<dyn Error>> {
		/*! A method that enables updating hyperparameters of an existing (pre-loaded) model.
		Currently limited to the most commonly used hyperparameters: ffm_learning_rate, ffm_power_t, power_t, learning_rate. */

		println!("Replacing initial regressor's hyperparameters from the command line ..");
		let mut replacement_hyperparam_ids: Vec<(String, String)> = vec![];

		// Handle learning rates
		if cmd_arguments.is_present("learning_rate") {

			if let Some(val) = cmd_arguments.value_of("learning_rate") {
				let hvalue = val.parse::<f32>()?;
				mi.learning_rate = hvalue;
				replacement_hyperparam_ids.push(("learning_rate".to_string(), hvalue.to_string()));
			}
		}

		if cmd_arguments.is_present("ffm_learning_rate") {

			if let Some(val) = cmd_arguments.value_of("ffm_learning_rate") {
				let hvalue = val.parse::<f32>()?;
				mi.ffm_learning_rate = hvalue;
				replacement_hyperparam_ids.push(("ffm_learning_rate".to_string(), hvalue.to_string()));
			}
		}

		// Handle power of t
		if cmd_arguments.is_present("power_t") {

			if let Some(val) = cmd_arguments.value_of("power_t") {
				let hvalue = val.parse::<f32>()?;
				mi.power_t = hvalue;
				replacement_hyperparam_ids.push(("power_t".to_string(), hvalue.to_string()));
			}
		}

		if cmd_arguments.is_present("ffm_power_t") {
			if let Some(val) = cmd_arguments.value_of("ffm_power_t") {
				let hvalue = val.parse::<f32>()?;
				mi.ffm_power_t = hvalue;
				replacement_hyperparam_ids.push(("ffm_power_t".to_string(), hvalue.to_string()));
			}
		}

		for (hyper_name, hyper_value) in replacement_hyperparam_ids.into_iter() {
			println!("Warning! Updated hyperparameter {} to value {}", hyper_name, hyper_value);
		}

		Ok(())

	}
}


#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
	use clap::{App, Arg,  AppSettings};

    fn ns_desc(i: u16) -> NamespaceDescriptor {
        NamespaceDescriptor {namespace_index: i,
                             namespace_type: vwmap::NamespaceType::Primitive,
                             namespace_format: vwmap::NamespaceFormat::Categorical}
    }


    #[test]
    fn test_interaction_parsing() {
        let vw_map_string = r#"
A,featureA
B,featureB
C,featureC
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mi = ModelInstance::new_empty().unwrap();

        let result = mi.create_feature_combo_desc(&vw, "A").unwrap();
        assert_eq!(result, FeatureComboDesc {
                                namespace_descriptors: vec![ns_desc(0)],
                                weight: 1.0
                                });

        let result = mi.create_feature_combo_desc(&vw, "BA:1.5").unwrap();
        assert_eq!(result, FeatureComboDesc {
                                namespace_descriptors: vec![ns_desc(1), ns_desc(0)],
                                weight: 1.5
                                });

    }

    #[test]
    fn test_weight_parsing() {
        let vw_map_string = r#"
A,featureA:2
B,featureB:3
"#;
        // The main point is that weight in feature names from vw_map_str is ignored
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mi = ModelInstance::new_empty().unwrap();
        let result = mi.create_feature_combo_desc(&vw, "BA:1.5").unwrap();
        assert_eq!(result, FeatureComboDesc {
                                namespace_descriptors: vec![ns_desc(1), ns_desc(0)],
                                weight: 1.5
                                });

    }

    #[test]
    fn test_feature_combo_verbose_parsing() {
        let vw_map_string = r#"
A,featureA
B,featureB
C,featureC
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mi = ModelInstance::new_empty().unwrap();
        let result = mi.create_feature_combo_desc_from_verbose(&vw, "featureA").unwrap();
        assert_eq!(result, FeatureComboDesc {
                                namespace_descriptors: vec![ns_desc(0)],

                                weight: 1.0
                                });

        let result = mi.create_feature_combo_desc_from_verbose(&vw, "featureB,featureA:1.5").unwrap();
        assert_eq!(result, FeatureComboDesc {
                                namespace_descriptors: vec![ns_desc(1), ns_desc(0)],
                                weight: 1.5
                                });

        let result = mi.create_feature_combo_desc_from_verbose(&vw, "featureB:1.5,featureA");
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Could not parse the value of a feature combination: 1.5,featureA\" })");

    }

    #[test]
    fn test_field_verbose_parsing() {
        let vw_map_string = r#"
A,featureA
B,featureB
C,featureC
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mi = ModelInstance::new_empty().unwrap();

        let result = mi.create_field_desc_from_verbose(&vw, "featureA").unwrap();
        assert_eq!(result, vec![ns_desc(0)]);

        let result = mi.create_field_desc_from_verbose(&vw, "featureA,featureC").unwrap();
        assert_eq!(result, vec![ns_desc(0), ns_desc(2)]);


        let result = mi.create_field_desc_from_verbose(&vw, "featureA,featureC:3");
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Fields currently do not support passing a value via : \\\"featureA,featureC:3\\\"\" })");

    }




    #[test]
    fn test_ffm_interaction_parsing() {
        let mi = ModelInstance::new_empty().unwrap();

        let result = mi.parse_ffm_interaction_mask("0:1:0", 2).unwrap();
        assert_eq!(result, (0,1,0.0_f32));

        let result = mi.parse_ffm_interaction_mask("0:1:0.5", 2).unwrap();
        assert_eq!(result, (0,1,0.5_f32));

        let result = mi.parse_ffm_interaction_mask("0:1:1", 2).unwrap();
        assert_eq!(result, (0,1,1.0_f32));

        let result = mi.parse_ffm_interaction_mask("0:1", 2);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Field interaction mask can be field_id:field_id or field_id1:field_id2:value, cannot parse: 0:1\" })");

        let result = mi.parse_ffm_interaction_mask("0:1:abc", 2);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Could not parse the value of a field interaction: abc\" })");

        let result = mi.parse_ffm_interaction_mask("1.0:1:1", 2);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Could not parse the id of a field 1: 1.0\" })");

        let result = mi.parse_ffm_interaction_mask("0:10:0", 2);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Field interaction mask field_id_2 can only be between 0 and 1 (number of fields), found: 10\" })");


    }






}