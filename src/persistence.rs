use std::{mem,slice};

use std::str;
use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;

use std::io::{Read, Write};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Take;
use std::fs::File;
use std::io;
use std::fs;

use crate::model_instance;
use crate::regressor;
use crate::vwmap;
use crate::learning_rate;
use crate::regressor::Regressor;
use learning_rate::LearningRateTrait;
use regressor::RegressorTrait;

const REGRESSOR_HEADER_MAGIC_STRING: &[u8; 4] = b"FWRE";    // Fwumious Wabbit REgressor
const REGRESSOR_HEADER_VERSION:u32 = 4;

impl model_instance::ModelInstance {
    pub fn save_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        let serialized = serde_json::to_vec_pretty(&self)?;
        output_bufwriter.write_u64::<LittleEndian>(serialized.len() as u64)?;
        output_bufwriter.write(&serialized)?;
        Ok(())
    }
    pub fn new_from_buf(input_bufreader: &mut dyn io::Read) -> Result<model_instance::ModelInstance, Box<dyn Error>> {
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let mi:model_instance::ModelInstance = serde_json::from_reader(input_bufreader.take(len as u64))?;
        Ok(mi)
    }
}

impl vwmap::VwNamespaceMap {
    pub fn save_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        let serialized = serde_json::to_vec_pretty(&self.vw_source)?;
        output_bufwriter.write_u64::<LittleEndian>(serialized.len() as u64)?;
        output_bufwriter.write(&serialized)?;
        Ok(())
    }

    pub fn new_from_buf(input_bufreader: &mut dyn io::Read) -> Result<vwmap::VwNamespaceMap, Box<dyn Error>> {
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let vw_source:vwmap::VwNamespaceMapSource = serde_json::from_reader(input_bufreader.take(len as u64))?;
        let vw = vwmap::VwNamespaceMap::new_from_source(vw_source)?;
        Ok(vw)
    }
}



pub fn save_regressor_to_filename(
                        filename: &str, 
                        mi: &model_instance::ModelInstance,
                        vwmap: &vwmap::VwNamespaceMap,
                        re: Box<dyn regressor::RegressorTrait>,
                        ) -> Result<(), Box<dyn Error>> {
        let mut output_bufwriter = &mut io::BufWriter::new(fs::File::create(filename).unwrap());
        write_regressor_header(output_bufwriter)?;
        vwmap.save_to_buf(output_bufwriter)?;
        mi.save_to_buf(output_bufwriter)?;
        re.write_weights_to_buf(output_bufwriter)?;
        Ok(())
    }

fn write_regressor_header(output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
    // we will write magic string FWFW
    // And then 32 bit unsigned version of the regressor
    output_bufwriter.write(REGRESSOR_HEADER_MAGIC_STRING)?;
    output_bufwriter.write_u32::<LittleEndian>(REGRESSOR_HEADER_VERSION)?;
    Ok(())
}

pub fn new_regressor_from_filename(
                    filename: &str, 
                    ) -> Result<(model_instance::ModelInstance,
                                 vwmap::VwNamespaceMap,
                                 Box<dyn regressor::RegressorTrait>), Box<dyn Error>> {
    let mut input_bufreader = &mut io::BufReader::new(fs::File::open(filename).unwrap());
    verify_header(input_bufreader).expect("Regressor header error");    
    let vw = vwmap::VwNamespaceMap::new_from_buf(input_bufreader).expect("Loading vwmap from regressor failed");
    let mi = model_instance::ModelInstance::new_from_buf(input_bufreader).expect("Loading model instance from regressor failed");
    let mut re = regressor::get_regressor(&mi);
    re.overwrite_weights_from_buf(&mut input_bufreader)?;
    Ok((mi, vw, re))
}

fn verify_header(input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
    let mut magic_string: [u8; 4] = [0;4];
    input_bufreader.read(&mut magic_string)?;
    if &magic_string != REGRESSOR_HEADER_MAGIC_STRING {
        return Err("Cache header does not begin with magic bytes FWFW")?;
    }
    
    let version = input_bufreader.read_u32::<LittleEndian>()?;
    if REGRESSOR_HEADER_VERSION != version {
        return Err(format!("Cache file version of this binary: {}, version of the cache file: {}", REGRESSOR_HEADER_VERSION, version))?;
    }
    Ok(())
}        




#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::feature_buffer;
    use crate::feature_buffer::HashAndValue;
    use crate::feature_buffer::HashAndValueAndSeq;
    use std::sync::Arc;
    use regressor::Regressor;

    use tempfile::{tempdir, NamedTempFile};
    #[test]
    fn save_empty_model() {
        let vw_map_string = r#"
A,featureA
B,featureB
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;        
        let mut rr = regressor::Regressor::<learning_rate::LearningRateAdagradFlex>::new(&mi);
        let dir = tempfile::tempdir().unwrap();
        let regressor_filepath = dir.path().join("test_regressor.fw");
        save_regressor_to_filename(regressor_filepath.to_str().unwrap(), &mi, &vw, Box::new(rr)).unwrap();
    }    

    fn lr_vec(v:Vec<feature_buffer::HashAndValue>) -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
                    label: 0.0,
                    example_importance: 1.0,
                    lr_buffer: v,
                    ffm_buffer: Vec::new(),
                    ffm_fields_count: 0,
        }
    }

    #[test]
    fn save_load_and_test_mode_lr() {
        let vw_map_string = r#"
A,featureA
B,featureB
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.bit_precision = 18;        
        let mut rr = regressor::Regressor::<learning_rate::LearningRateAdagradFlex>::new(&mi);
        let mut p: f32;
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), true, 0);
        assert_eq!(p, 0.45016602);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), false, 0);
        assert_eq!(p, 0.41731137);
        let dir = tempdir().unwrap();
        let regressor_filepath = dir.path().join("test_regressor.fw");
        save_regressor_to_filename(regressor_filepath.to_str().unwrap(), &mi, &vw, Box::new(rr)).unwrap();

        // Now let's load the saved regressor
        let (mi2, vw2, mut re2) = new_regressor_from_filename(regressor_filepath.to_str().unwrap()).unwrap();

        // predict with the same feature vector
        p = re2.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), false, 0);
        assert_eq!(p, 0.41731137);

        let re_fixed = Arc::new(re2.get_fixed_regressor());
        p = re_fixed.predict(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), 0);
        assert_eq!(p, 0.41731137);
    }    

    fn ffm_fixed_init<T:LearningRateTrait>(mut rg: &mut Regressor<T>) -> () {
        for i in rg.ffm_weights_offset as usize..rg.weights.len() {
            rg.weights[i].weight = 1.0;
            rg.weights[i].acc_grad = 1.0;
        }
    }


    fn ffm_vec(v:Vec<feature_buffer::HashAndValueAndSeq>, ffm_fields_count:u32) -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
                    label: 0.0,
                    example_importance: 1.0,
                    lr_buffer: Vec::new(),
                    ffm_buffer: v,
                    ffm_fields_count: ffm_fields_count,
        }
    }


    #[test]
    fn save_load_and_test_mode_ffm() {
        let vw_map_string = r#"
A,featureA
B,featureB
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_power_t = 0.0;
        mi.ffm_learning_rate = 0.1;
        mi.ffm_fields = vec![vec![],vec![]]; 
        let mut rr = regressor::Regressor::<learning_rate::LearningRateAdagradFlex>::new(&mi);
        let mut p: f32;

        ffm_fixed_init(&mut rr);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: 1}
                                  ], 2);
        p = rr.learn(&ffm_buf, true, 0);
        assert_eq!(p, 0.880797); 
        p = rr.learn(&ffm_buf, false, 0);
        assert_eq!(p, 0.79534113);


        let dir = tempdir().unwrap();
        let regressor_filepath = dir.path().join("test_regressor2.fw");
        save_regressor_to_filename(regressor_filepath.to_str().unwrap(), &mi, &vw, Box::new(rr)).unwrap();

        // Now let's load the saved regressor
        let (mi2, vw2, mut re2) = new_regressor_from_filename(regressor_filepath.to_str().unwrap()).unwrap();

        // predict with the same feature vector
        p = re2.learn(&ffm_buf, false, 0);
        assert_eq!(p, 0.79534113);

        let re_fixed = Arc::new(re2.get_fixed_regressor());
        p = re_fixed.predict(&ffm_buf, 0);
        assert_eq!(p, 0.79534113);

    }    

}
