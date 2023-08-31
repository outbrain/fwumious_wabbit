#![allow(dead_code,unused_imports)]

use std::error::Error;
use std::str;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs;
use std::fs::File;
use std::io;
use std::io::Read;

use crate::model_instance;
use crate::optimizer;
use crate::regressor;
use crate::vwmap;

use crate::multithread_helpers::BoxedRegressorTrait;
use crate::regressor::Regressor;

const REGRESSOR_HEADER_MAGIC_STRING: &[u8; 4] = b"FWRE"; // Fwumious Wabbit REgressor
const REGRESSOR_HEADER_VERSION: u32 = 6; // Change to 5: introduce namespace descriptors which changes regressor

impl model_instance::ModelInstance {
    pub fn save_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        let serialized = serde_json::to_vec_pretty(&self)?;
        output_bufwriter.write_u64::<LittleEndian>(serialized.len() as u64)?;
        output_bufwriter.write_all(&serialized)?;
        Ok(())
    }
    pub fn new_from_buf(
        input_bufreader: &mut dyn io::Read,
    ) -> Result<model_instance::ModelInstance, Box<dyn Error>> {
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let mi: model_instance::ModelInstance = serde_json::from_reader(input_bufreader.take(len))?;
        Ok(mi)
    }
}

impl vwmap::VwNamespaceMap {
    pub fn save_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        let serialized = serde_json::to_vec_pretty(&self.vw_source)?;
        output_bufwriter.write_u64::<LittleEndian>(serialized.len() as u64)?;
        output_bufwriter.write_all(&serialized)?;
        Ok(())
    }

    pub fn new_from_buf(
        input_bufreader: &mut dyn io::Read,
    ) -> Result<vwmap::VwNamespaceMap, Box<dyn Error>> {
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let vw_source: vwmap::VwNamespaceMapSource =
            serde_json::from_reader(input_bufreader.take(len))?;
        let vw = vwmap::VwNamespaceMap::new_from_source(vw_source)?;
        Ok(vw)
    }
}

pub fn save_sharable_regressor_to_filename(
    filename: &str,
    mi: &model_instance::ModelInstance,
    vwmap: &vwmap::VwNamespaceMap,
    re: BoxedRegressorTrait,
) -> Result<(), Box<dyn Error>> {
    let output_bufwriter = &mut io::BufWriter::new(
        fs::File::create(filename)
            .unwrap_or_else(|_| panic!("Cannot open {} to save regressor to", filename)),
    );
    write_regressor_header(output_bufwriter)?;
    vwmap.save_to_buf(output_bufwriter)?;
    mi.save_to_buf(output_bufwriter)?;
    re.write_weights_to_buf(output_bufwriter)?;
    Ok(())
}

pub fn save_regressor_to_filename(
    filename: &str,
    mi: &model_instance::ModelInstance,
    vwmap: &vwmap::VwNamespaceMap,
    re: Regressor,
) -> Result<(), Box<dyn Error>> {
    let output_bufwriter = &mut io::BufWriter::new(
        fs::File::create(filename)
            .unwrap_or_else(|_| panic!("Cannot open {} to save regressor to", filename)),
    );
    write_regressor_header(output_bufwriter)?;
    vwmap.save_to_buf(output_bufwriter)?;
    mi.save_to_buf(output_bufwriter)?;
    re.write_weights_to_buf(output_bufwriter)?;
    Ok(())
}

fn write_regressor_header(output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
    // we will write magic string FWFW
    // And then 32 bit unsigned version of the regressor
    output_bufwriter.write_all(REGRESSOR_HEADER_MAGIC_STRING)?;
    output_bufwriter.write_u32::<LittleEndian>(REGRESSOR_HEADER_VERSION)?;
    Ok(())
}

fn load_regressor_without_weights(
    input_bufreader: &mut io::BufReader<File>,
    cmd_arguments: Option<&clap::ArgMatches>,
) -> Result<
    (
        model_instance::ModelInstance,
        vwmap::VwNamespaceMap,
        regressor::Regressor,
    ),
    Box<dyn Error>,
> {
    verify_header(input_bufreader).expect("Regressor header error");
    let vw = vwmap::VwNamespaceMap::new_from_buf(input_bufreader)
        .expect("Loading vwmap from regressor failed");

    let mut mi = model_instance::ModelInstance::new_from_buf(input_bufreader)
        .expect("Loading model instance from regressor failed");

    if let Some(cmd_args) = cmd_arguments {
        model_instance::ModelInstance::update_hyperparameters_from_cmd(cmd_args, &mut mi)?;
    }

    let mi = mi;
    let re = regressor::get_regressor_without_weights(&mi);

    Ok((mi, vw, re))
}

pub fn new_regressor_from_filename(
    filename: &str,
    immutable: bool,
    cmd_arguments: Option<&clap::ArgMatches>,
) -> Result<
    (
        model_instance::ModelInstance,
        vwmap::VwNamespaceMap,
        regressor::Regressor,
    ),
    Box<dyn Error>,
> {
    let mut input_bufreader = io::BufReader::new(fs::File::open(filename).unwrap());
    let (mut mi, vw, mut re) = load_regressor_without_weights(&mut input_bufreader, cmd_arguments)?;
    if !immutable {
        re.allocate_and_init_weights(&mi);
        re.overwrite_weights_from_buf(&mut input_bufreader)?;
        Ok((mi, vw, re))
    } else {
        mi.optimizer = model_instance::Optimizer::SGD;
        let mut immutable_re = re.immutable_regressor_without_weights(&mi)?;
        immutable_re.allocate_and_init_weights(&mi);
        re.into_immutable_regressor_from_buf(&mut immutable_re, &mut input_bufreader)?;
        Ok((mi, vw, immutable_re))
    }
}

pub fn hogwild_load(re: &mut regressor::Regressor, filename: &str) -> Result<(), Box<dyn Error>> {
    let mut input_bufreader = io::BufReader::new(fs::File::open(filename)?);
    let (_, _, mut re_hw) = load_regressor_without_weights(&mut input_bufreader, None)?;
    // TODO: Here we should do safety comparison that the regressor is really the same;
    if !re.immutable {
        re.overwrite_weights_from_buf(&mut input_bufreader)?;
    } else {
        re_hw.into_immutable_regressor_from_buf(re, &mut input_bufreader)?;
    }
    Ok(())
}

fn verify_header(input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
    let mut magic_string: [u8; 4] = [0; 4];
    input_bufreader.read(&mut magic_string)?;
    if &magic_string != REGRESSOR_HEADER_MAGIC_STRING {
        return Err("Cache header does not begin with magic bytes FWFW")?;
    }

    let version = input_bufreader.read_u32::<LittleEndian>()?;
    if REGRESSOR_HEADER_VERSION != version {
        return Err(format!(
            "Cache file version of this binary: {}, version of the cache file: {}",
            REGRESSOR_HEADER_VERSION, version
        ))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::assert_epsilon;
    use crate::block_ffm;
    use crate::feature_buffer;
    use crate::feature_buffer::HashAndValue;
    use crate::feature_buffer::HashAndValueAndSeq;
    use crate::model_instance::Optimizer;
    use regressor::BlockTrait;
    use regressor::Regressor;

    use crate::optimizer::OptimizerTrait;
    use tempfile::tempdir;

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
        mi.optimizer = model_instance::Optimizer::AdagradFlex;
        let rr = regressor::get_regressor_with_weights(&mi);
        let dir = tempfile::tempdir().unwrap();
        let regressor_filepath = dir.path().join("test_regressor.fw");
        save_regressor_to_filename(regressor_filepath.to_str().unwrap(), &mi, &vw, rr).unwrap();
    }

    fn lr_vec(v: Vec<feature_buffer::HashAndValue>) -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
            label: 0.0,
            example_importance: 1.0,
            example_number: 0,
            lr_buffer: v,
            ffm_buffer: Vec::new(),
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
        mi.optimizer = model_instance::Optimizer::AdagradFlex;
        mi.init_acc_gradient = 0.0;
        let mut re = regressor::Regressor::new(&mi);
        let mut pb = re.new_portbuffer();

        let fbuf = &lr_vec(vec![
            HashAndValue {
                hash: 1,
                value: 1.0,
                combo_index: 0,
            },
            HashAndValue {
                hash: 2,
                value: 1.0,
                combo_index: 0,
            },
        ]);
        assert_eq!(re.learn(fbuf, &mut pb, true), 0.5);
        assert_eq!(re.learn(fbuf, &mut pb, true), 0.45016602);
        assert_eq!(re.learn(fbuf, &mut pb, false), 0.41731137);

        let expected_result = 0.41731137;
        assert_eq!(re.learn(fbuf, &mut pb, false), expected_result);

        // Now we test conversion to fixed regressor
        {
            mi.optimizer = model_instance::Optimizer::SGD;
            let re_fixed = re.immutable_regressor(&mi).unwrap();
            // predict with the same feature vector
            assert_eq!(re_fixed.predict(fbuf, &mut pb), expected_result);
            mi.optimizer = model_instance::Optimizer::AdagradFlex;
        }
        // Now we test saving and loading a) regular regressor, b) fixed regressor
        {
            let dir = tempdir().unwrap();
            let regressor_filepath = dir.path().join("test_regressor2.fw");
            save_regressor_to_filename(regressor_filepath.to_str().unwrap(), &mi, &vw, re).unwrap();

            // a) load as regular regressor
            let (_mi2, _vw2, mut re2) =
                new_regressor_from_filename(regressor_filepath.to_str().unwrap(), false, None)
                    .unwrap();
            assert_eq!(re2.learn(fbuf, &mut pb, false), expected_result);
            assert_eq!(re2.predict(fbuf, &mut pb), expected_result);

            // a) load as regular regressor, immutable
            let (_mi2, _vw2, mut re2) =
                new_regressor_from_filename(regressor_filepath.to_str().unwrap(), true, None)
                    .unwrap();
            assert_eq!(re2.learn(fbuf, &mut pb, false), expected_result);
            assert_eq!(re2.predict(fbuf, &mut pb), expected_result);
        }
    }

    fn ffm_fixed_init(rg: &mut Regressor) {
        // This is a bit of black magic - we "know" that FFM is at index 1 and we downcast...
        let block_ffm = &mut rg.blocks_boxes[1];
        let block_ffm = block_ffm
            .as_any()
            .downcast_mut::<block_ffm::BlockFFM<optimizer::OptimizerAdagradFlex>>()
            .unwrap();

        for i in 0..block_ffm.get_serialized_len() {
            // it only happens that this matches number of weights
            block_ffm.weights[i] = 1.0;
            block_ffm.optimizer[i].optimizer_data = block_ffm.optimizer_ffm.initial_data();
        }
    }

    fn ffm_vec(v: Vec<feature_buffer::HashAndValueAndSeq>) -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
            label: 0.0,
            example_importance: 1.0,
            example_number: 0,
            lr_buffer: Vec::new(),
            ffm_buffer: v,
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
        mi.ffm_fields = vec![vec![], vec![]];
        mi.optimizer = Optimizer::AdagradFlex;
        let mut re = regressor::Regressor::new(&mi);
        let mut pb = re.new_portbuffer();

        let mut p: f32;

        ffm_fixed_init(&mut re);
        let fbuf = &ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 3 * 1000,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 2.0,
                contra_field_index: 1,
            },
        ]);
        pb.reset();
        p = re.learn(fbuf, &mut pb, true);
        assert_eq!(p, 0.9933072);
        let expected_result = 0.9395168;
        p = re.learn(fbuf, &mut pb, false);
        assert_epsilon!(p, expected_result);
        p = re.predict(fbuf, &mut pb);
        assert_epsilon!(p, expected_result);

        // Now we test conversion to fixed regressor
        {
            mi.optimizer = Optimizer::SGD;
            let re_fixed = re.immutable_regressor(&mi).unwrap();
            // predict with the same feature vector
            mi.optimizer = Optimizer::AdagradFlex;
            assert_epsilon!(re_fixed.predict(fbuf, &mut pb), expected_result);
        }
        // Now we test saving and loading a) regular regressor, b) fixed regressor
        {
            let dir = tempdir().unwrap();
            let regressor_filepath = dir.path().join("test_regressor2.fw");
            save_regressor_to_filename(regressor_filepath.to_str().unwrap(), &mi, &vw, re).unwrap();

            // a) load as regular regressor
            let (_mi2, _vw2, mut re2) =
                new_regressor_from_filename(regressor_filepath.to_str().unwrap(), false, None)
                    .unwrap();
            assert_eq!(re2.get_name(), "Regressor with optimizer \"AdagradFlex\"");
            assert_epsilon!(re2.learn(fbuf, &mut pb, false), expected_result);
            assert_epsilon!(re2.predict(fbuf, &mut pb), expected_result);

            // b) load as regular regressor, immutable
            let (_mi2, _vw2, mut re2) =
                new_regressor_from_filename(regressor_filepath.to_str().unwrap(), true, None)
                    .unwrap();
            assert_eq!(re2.get_name(), "Regressor with optimizer \"SGD\"");
            assert_epsilon!(re2.learn(fbuf, &mut pb, false), expected_result);
            assert_epsilon!(re2.predict(fbuf, &mut pb), expected_result);
        }
    }

    fn lr_and_ffm_vec(
        v1: Vec<feature_buffer::HashAndValue>,
        v2: Vec<feature_buffer::HashAndValueAndSeq>,
    ) -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
            label: 0.0,
            example_importance: 1.0,
            example_number: 0,
            lr_buffer: v1,
            ffm_buffer: v2,
        }
    }

    #[test]
    fn test_hogwild_load() {
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
        mi.ffm_fields = vec![vec![], vec![]];
        mi.optimizer = Optimizer::AdagradFlex;

        let mut re_1 = regressor::Regressor::new(&mi);
        let mut re_2 = regressor::Regressor::new(&mi);
        let mut pb_1 = re_1.new_portbuffer();
        let mut pb_2 = re_2.new_portbuffer();
        let mut p: f32;

        ffm_fixed_init(&mut re_1);
        ffm_fixed_init(&mut re_2);
        let fbuf_1 = &lr_and_ffm_vec(
            vec![
                HashAndValue {
                    hash: 52,
                    value: 0.5,
                    combo_index: 0,
                },
                HashAndValue {
                    hash: 2,
                    value: 1.0,
                    combo_index: 0,
                },
            ],
            vec![
                HashAndValueAndSeq {
                    hash: 1,
                    value: 0.5,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 3 * 1000,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 101,
                    value: 2.0,
                    contra_field_index: 1,
                },
            ],
        );
        let fbuf_2 = &lr_and_ffm_vec(
            vec![
                HashAndValue {
                    hash: 1,
                    value: 1.0,
                    combo_index: 0,
                },
                HashAndValue {
                    hash: 2,
                    value: 1.0,
                    combo_index: 0,
                },
            ],
            vec![
                HashAndValueAndSeq {
                    hash: 1,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 3 * 1000,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 100,
                    value: 2.0,
                    contra_field_index: 1,
                },
            ],
        );

        p = re_1.learn(fbuf_1, &mut pb_1, true);
        assert_eq!(p, 0.97068775);
        let expected_result_1_on_1 = 0.8922257;
        p = re_1.learn(fbuf_1, &mut pb_1, false);
        assert_eq!(p, expected_result_1_on_1);
        p = re_1.predict(fbuf_1, &mut pb_1);
        assert_eq!(p, expected_result_1_on_1);

        p = re_2.learn(fbuf_2, &mut pb_2, true);
        assert_eq!(p, 0.9933072);
        let expected_result_2_on_2 = 0.92719215;
        p = re_2.learn(fbuf_2, &mut pb_2, false);
        assert_eq!(p, expected_result_2_on_2);
        p = re_2.predict(fbuf_2, &mut pb_2);
        assert_eq!(p, expected_result_2_on_2);

        p = re_2.learn(fbuf_1, &mut pb_2, false);
        assert_eq!(p, 0.93763095);
        let expected_result_1_on_2 = 0.93763095;
        p = re_2.learn(fbuf_1, &mut pb_2, false);
        assert_eq!(p, expected_result_1_on_2);
        p = re_2.predict(fbuf_1, &mut pb_1);
        assert_eq!(p, expected_result_1_on_2);

        p = re_1.learn(fbuf_2, &mut pb_1, false);
        assert_eq!(p, 0.98559695);
        let expected_result_2_on_1 = 0.98559695;
        p = re_1.learn(fbuf_2, &mut pb_1, false);
        assert_eq!(p, expected_result_2_on_1);
        p = re_1.predict(fbuf_2, &mut pb_2);
        assert_eq!(p, expected_result_2_on_1);

        // Now we test saving and loading a) regular regressor, b) immutable regressor
        // FYI ... this confusing tests have actually caught bugs in the code, they are hard to maintain, but important
        {
            let dir = tempdir().unwrap();
            let regressor_filepath_1 = dir
                .path()
                .join("test_regressor1.fw")
                .to_str()
                .unwrap()
                .to_owned();
            save_regressor_to_filename(&regressor_filepath_1, &mi, &vw, re_1).unwrap();
            let regressor_filepath_2 = dir
                .path()
                .join("test_regressor2.fw")
                .to_str()
                .unwrap()
                .to_owned();
            save_regressor_to_filename(&regressor_filepath_2, &mi, &vw, re_2).unwrap();

            // The mutable path
            let (_mi1, _vw1, mut new_re_1) =
                new_regressor_from_filename(&regressor_filepath_1, false, None).unwrap();
            assert_eq!(
                new_re_1.get_name(),
                "Regressor with optimizer \"AdagradFlex\""
            );
            assert_eq!(
                new_re_1.learn(fbuf_1, &mut pb_1, false),
                expected_result_1_on_1
            );
            assert_eq!(new_re_1.predict(fbuf_1, &mut pb_1), expected_result_1_on_1);
            assert_eq!(
                new_re_1.learn(fbuf_2, &mut pb_1, false),
                expected_result_2_on_1
            );
            assert_eq!(new_re_1.predict(fbuf_2, &mut pb_2), expected_result_2_on_1);
            hogwild_load(&mut new_re_1, &regressor_filepath_2).unwrap();
            assert_eq!(
                new_re_1.learn(fbuf_2, &mut pb_1, false),
                expected_result_2_on_2
            );
            assert_eq!(new_re_1.predict(fbuf_2, &mut pb_2), expected_result_2_on_2);
            hogwild_load(&mut new_re_1, &regressor_filepath_1).unwrap();
            assert_eq!(
                new_re_1.learn(fbuf_1, &mut pb_1, false),
                expected_result_1_on_1
            );
            assert_eq!(new_re_1.predict(fbuf_1, &mut pb_1), expected_result_1_on_1);
            assert_eq!(
                new_re_1.learn(fbuf_2, &mut pb_1, false),
                expected_result_2_on_1
            );
            assert_eq!(new_re_1.predict(fbuf_2, &mut pb_2), expected_result_2_on_1);

            // The immutable path
            let (_mi1, _vw1, mut new_re_1) =
                new_regressor_from_filename(&regressor_filepath_1, true, None).unwrap();
            assert_eq!(new_re_1.get_name(), "Regressor with optimizer \"SGD\"");
            assert_eq!(
                new_re_1.learn(fbuf_1, &mut pb_1, false),
                expected_result_1_on_1
            );
            assert_eq!(new_re_1.predict(fbuf_1, &mut pb_1), expected_result_1_on_1);
            assert_eq!(
                new_re_1.learn(fbuf_2, &mut pb_1, false),
                expected_result_2_on_1
            );
            assert_eq!(new_re_1.predict(fbuf_2, &mut pb_2), expected_result_2_on_1);
            hogwild_load(&mut new_re_1, &regressor_filepath_2).unwrap();
            assert_eq!(
                new_re_1.learn(fbuf_2, &mut pb_1, false),
                expected_result_2_on_2
            );
            assert_eq!(new_re_1.predict(fbuf_2, &mut pb_2), expected_result_2_on_2);
            hogwild_load(&mut new_re_1, &regressor_filepath_1).unwrap();
            assert_eq!(
                new_re_1.learn(fbuf_1, &mut pb_1, false),
                expected_result_1_on_1
            );
            assert_eq!(new_re_1.predict(fbuf_1, &mut pb_1), expected_result_1_on_1);
            assert_eq!(
                new_re_1.learn(fbuf_2, &mut pb_1, false),
                expected_result_2_on_1
            );
            assert_eq!(new_re_1.predict(fbuf_2, &mut pb_2), expected_result_2_on_1);
        }
    }
}
