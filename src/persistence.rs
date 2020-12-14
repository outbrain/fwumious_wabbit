use std::str;
use std::error::Error;

use std::io::{Read};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::File;
use std::io;
use std::fs;

use crate::model_instance;
use crate::regressor;
use crate::vwmap;
use crate::optimizer;
use optimizer::OptimizerTrait;
use regressor::Regressor;

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
                        re: regressor::Regressor,
                        ) -> Result<(), Box<dyn Error>> {
        let output_bufwriter = &mut io::BufWriter::new(fs::File::create(filename).expect(format!("Cannot open {} to save regressor to", filename).as_str()));
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

fn load_regressor_without_weights(input_bufreader: &mut io::BufReader::<File>) 
                        -> Result<(model_instance::ModelInstance,
                                   vwmap::VwNamespaceMap,
                                   regressor::Regressor,
                                 ), Box<dyn Error>> {
    verify_header(input_bufreader).expect("Regressor header error");    
    let vw = vwmap::VwNamespaceMap::new_from_buf(input_bufreader).expect("Loading vwmap from regressor failed");
    let mi = model_instance::ModelInstance::new_from_buf(input_bufreader).expect("Loading model instance from regressor failed");
    let re = regressor::get_regressor_without_weights(&mi);
    Ok((mi, vw, re))
}


pub fn new_regressor_from_filename(filename: &str, immutable: bool) 
                        -> Result<(model_instance::ModelInstance,
                                   vwmap::VwNamespaceMap,
                                   regressor::Regressor), 
                                  Box<dyn Error>> {
    let mut input_bufreader = io::BufReader::new(fs::File::open(filename).unwrap());
    let (mi, vw, mut re) = load_regressor_without_weights(&mut input_bufreader)?;
    if !immutable {
        re.allocate_and_init_weights(&mi);
        re.overwrite_weights_from_buf(&mut input_bufreader)?;
        Ok((mi, vw, re))
    } else {
        let immutable_re = re.immutable_regressor_from_buf(&mi, &mut input_bufreader)?;
        Ok((mi, vw, immutable_re))
    }
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

    use regressor::Regressor;
    use regressor::BlockTrait;
    use crate::block_ffm::BlockFFM;
    
    use tempfile::{tempdir};
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
        mi.optimizer = model_instance::Optimizer::Adagrad;
        mi.fastmath = false;
        let rr = regressor::get_regressor(&mi);
        let dir = tempfile::tempdir().unwrap();
        let regressor_filepath = dir.path().join("test_regressor.fw");
        save_regressor_to_filename(regressor_filepath.to_str().unwrap(), &mi, &vw, rr).unwrap();
    }    

    fn lr_vec(v:Vec<feature_buffer::HashAndValue>) -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
                    label: 0.0,
                    example_importance: 1.0,
                    example_number: 0,
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
        mi.optimizer = model_instance::Optimizer::Adagrad;
        mi.fastmath = false;
        mi.init_acc_gradient = 0.0;
        let mut re = regressor::get_regressor(&mi);

        let fbuf = &lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]);
        assert_eq!(re.learn(fbuf, true), 0.5);
        assert_eq!(re.learn(fbuf, true), 0.45016602);
        assert_eq!(re.learn(fbuf, false), 0.41731137);

        let CONST_RESULT = 0.41731137;
        assert_eq!(re.learn(fbuf, false), CONST_RESULT);

        // Now we test conversion to fixed regressor 
        {
            let re_fixed = re.immutable_regressor(&mi).unwrap();
            // predict with the same feature vector
            assert_eq!(re_fixed.predict(&fbuf), CONST_RESULT);
        }
        // Now we test saving and loading a) regular regressor, b) fixed regressor
        {
            let dir = tempdir().unwrap();
            let regressor_filepath = dir.path().join("test_regressor2.fw");
            save_regressor_to_filename(regressor_filepath.to_str().unwrap(), &mi, &vw, re).unwrap();

            // a) load as regular regressor
            let (_mi2, _vw2, mut re2) = new_regressor_from_filename(regressor_filepath.to_str().unwrap(), false).unwrap();
            assert_eq!(re2.learn(fbuf, false), CONST_RESULT);
            assert_eq!(re2.predict(fbuf), CONST_RESULT);

            // a) load as regular regressor, immutable
            let (_mi2, _vw2, mut re2) = new_regressor_from_filename(regressor_filepath.to_str().unwrap(), true).unwrap();
            assert_eq!(re2.learn(fbuf, false), CONST_RESULT);
            assert_eq!(re2.predict(fbuf), CONST_RESULT);

        }

    }    

    fn ffm_fixed_init(rg: &mut Regressor) -> () {
        // This is a bit of black magic - we "know" that FFM is at index 1 and we downcast...
        let block_ffm = &mut rg.blocks_boxes[1];
        let mut block_ffm = block_ffm.as_any().downcast_mut::<BlockFFM<optimizer::OptimizerAdagradFlex>>().unwrap();

        // TODO: this is not future compatible
        for i in 0..block_ffm.get_serialized_len() {// it only happens that this matches number of weights
            block_ffm.testing_set_weights(0, 0, i, &[1.0f32]).unwrap();
        }
    }


    fn ffm_vec(v:Vec<feature_buffer::HashAndValueAndSeq>, ffm_fields_count:u32) -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
                    label: 0.0,
                    example_importance: 1.0,
                    example_number: 0,
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
        mi.optimizer = model_instance::Optimizer::Adagrad;
        mi.fastmath = false;
        let mut re = regressor::Regressor::new::<optimizer::OptimizerAdagradFlex>(&mi);
        let mut p: f32;

        ffm_fixed_init(&mut re);
        let fbuf = &ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:3 * 1000, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: 1}
                                  ], 2);
        p = re.learn(fbuf, true);
        assert_eq!(p, 0.9933072); 
        let CONST_RESULT = 0.9395168;
        p = re.learn(fbuf, false);
        assert_eq!(p, CONST_RESULT);
        p = re.predict(fbuf);
        assert_eq!(p, CONST_RESULT);

        // Now we test conversion to fixed regressor 
        {
            let re_fixed = re.immutable_regressor(&mi).unwrap();
            // predict with the same feature vector
            assert_eq!(re_fixed.predict(&fbuf), CONST_RESULT);
        }
        // Now we test saving and loading a) regular regressor, b) fixed regressor
        {
            let dir = tempdir().unwrap();
            let regressor_filepath = dir.path().join("test_regressor2.fw");
            save_regressor_to_filename(regressor_filepath.to_str().unwrap(), &mi, &vw, re).unwrap();

            // a) load as regular regressor
            let (_mi2, _vw2, mut re2) = new_regressor_from_filename(regressor_filepath.to_str().unwrap(), false).unwrap();
            assert_eq!(re2.get_name(), "Regressor with optimizer \"AdagradFlex\"");
            assert_eq!(re2.learn(fbuf, false), CONST_RESULT);
            assert_eq!(re2.predict(fbuf), CONST_RESULT);

            // b) load as regular regressor, immutable
            let (_mi2, _vw2, mut re2) = new_regressor_from_filename(regressor_filepath.to_str().unwrap(), true).unwrap();
            assert_eq!(re2.get_name(), "Regressor with optimizer \"SGD\"");
            assert_eq!(re2.learn(fbuf, false), CONST_RESULT);
            assert_eq!(re2.predict(fbuf), CONST_RESULT);

        }
        

    }    

}
