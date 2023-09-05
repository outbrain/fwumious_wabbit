#![allow(dead_code,unused_imports)]

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rustc_hash::FxHashSet;
use std::any::Any;
use std::error::Error;
use std::io;
use std::io::Cursor;

use crate::block_ffm;
use crate::block_helpers;
use crate::block_loss_functions;
use crate::block_lr;
use crate::block_misc;
use crate::block_neural;
use crate::block_neural::InitType;
use crate::block_normalize;
use crate::block_relu;
use crate::feature_buffer;
use crate::feature_buffer::HashAndValueAndSeq;
use crate::graph;
use crate::model_instance;
use crate::optimizer;
use crate::port_buffer;

pub const FFM_CONTRA_BUF_LEN: usize = 16384;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FFMFeature {
    pub index: u32,
    pub contra_field_index: u32,
}

impl From<&HashAndValueAndSeq> for FFMFeature {
    fn from(value: &HashAndValueAndSeq) -> Self {
        FFMFeature {
            index: value.hash,
            contra_field_index: value.contra_field_index,
        }
    }
}

pub enum BlockCache {
    FFM {
        contra_fields: [f32; FFM_CONTRA_BUF_LEN],
        features_present: FxHashSet<FFMFeature>,
        ffm: Vec<f32>,
    },
    LR {
        combo_indexes: Vec<bool>,
        lr: Vec<f32>,
    },
}

pub trait BlockTrait {
    fn as_any(&mut self) -> &mut dyn Any; // This enables downcasting
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    );

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    );

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        caches: &[BlockCache],
    );

    fn prepare_forward_cache(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        caches: &mut [BlockCache],
    ) {
        block_helpers::prepare_forward_cache(further_blocks, fb, caches);
    }

    fn create_forward_cache(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        caches: &mut Vec<BlockCache>,
    ) {
        block_helpers::create_forward_cache(further_blocks, caches);
    }

    fn allocate_and_init_weights(&mut self, _mi: &model_instance::ModelInstance) {}

    fn get_serialized_len(&self) -> usize {
        0
    }

    fn write_weights_to_buf(
        &self,
        _output_bufwriter: &mut dyn io::Write,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn read_weights_from_buf(
        &mut self,
        _input_bufreader: &mut dyn io::Read,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn get_num_output_values(&self, _output: graph::OutputSlot) -> usize;

    fn get_num_output_slots(&self) -> usize {
        1
    }

    fn get_input_offset(&mut self, _input: graph::InputSlot) -> Result<usize, Box<dyn Error>> {
        Err("get_input_offset() is only supported by CopyBlock".to_string())?
    }

    fn set_input_offset(&mut self, _input: graph::InputSlot, _offset: usize) {}
    fn set_output_offset(&mut self, _output: graph::OutputSlot, _offset: usize) {}

    fn get_block_type(&self) -> graph::BlockType {
        graph::BlockType::Regular
    }

    fn read_weights_from_buf_into_forward_only(
        &self,
        _input_bufreader: &mut dyn io::Read,
        _forward: &mut Box<dyn BlockTrait>,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}

pub struct Regressor {
    pub regressor_name: String,
    pub blocks_boxes: Vec<Box<dyn BlockTrait>>,
    pub tape_len: usize,
    pub immutable: bool,
}

pub fn get_regressor_without_weights(mi: &model_instance::ModelInstance) -> Regressor {
    Regressor::new_without_weights(mi)
}

pub fn get_regressor_with_weights(mi: &model_instance::ModelInstance) -> Regressor {
    let mut re = get_regressor_without_weights(mi);
    re.allocate_and_init_weights(mi);
    re
}

#[derive(PartialEq)]
enum NNActivation {
    None,
    Relu,
}

#[derive(PartialEq)]
enum NNLayerNorm {
    None,
    BeforeRelu,
    AfterRelu,
}

impl Regressor {
    pub fn new_without_weights(mi: &model_instance::ModelInstance) -> Regressor {
        let mut rg = Regressor {
            blocks_boxes: Vec::new(),
            regressor_name: format!("Regressor with optimizer \"{:?}\"", mi.optimizer),
            immutable: false,
            tape_len: usize::MAX,
        };

        let mut bg = graph::BlockGraph::new();
        // A bit more elaborate than necessary. Let's really make it clear what's happening
        let mut output = block_lr::new_lr_block(&mut bg, mi).unwrap();

        if mi.ffm_k > 0 {
            let block_ffm = block_ffm::new_ffm_block(&mut bg, mi).unwrap();
            let triangle_ffm = block_misc::new_triangle_block(&mut bg, block_ffm).unwrap();
            output = block_misc::new_join_block(&mut bg, vec![output, triangle_ffm]).unwrap();
        }

        if !mi.nn_config.layers.is_empty() {
            let mut join_block: Option<graph::BlockPtrOutput> = None;
            if mi.nn_config.topology == "one" {
                let (a1, a2) = block_misc::new_copy_block_2(&mut bg, output).unwrap();
                output = a1;
                join_block = Some(a2);
            } else if mi.nn_config.topology == "two" {
                // do not copy out the
            } else if mi.nn_config.topology == "four" {
                let (a1, a2) = block_misc::new_copy_block_2(&mut bg, output).unwrap();
                output = a1;
                join_block = Some(a2);
                output = block_normalize::new_normalize_layer_block(&mut bg, mi, output).unwrap();
            } else if mi.nn_config.topology == "five" {
                let (a1, a2) = block_misc::new_copy_block_2(&mut bg, output).unwrap();
                output = a1;
                join_block = Some(a2);
                output = block_normalize::new_stop_block(&mut bg, mi, output).unwrap();
            } else {
                Err(format!(
                    "unknown nn topology: \"{}\"",
                    mi.nn_config.topology
                ))
                .unwrap()
            }

            for (layer_num, layer) in mi.nn_config.layers.iter().enumerate() {
                let mut layer = layer.clone();
                let activation_str: String = layer
                    .remove("activation")
                    .unwrap_or("none".to_string())
                    .to_string();
                let layernorm_str: String = layer
                    .remove("layernorm")
                    .unwrap_or("none".to_string())
                    .to_string();
                let width: usize = layer
                    .remove("width")
                    .unwrap_or("20".to_string())
                    .parse()
                    .unwrap();
                let maxnorm: f32 = layer
                    .remove("maxnorm")
                    .unwrap_or("0.0".to_string())
                    .parse()
                    .unwrap();
                let dropout: f32 = layer
                    .remove("dropout")
                    .unwrap_or("0.0".to_string())
                    .parse()
                    .unwrap();

                let init_type_str: String =
                    layer.remove("init").unwrap_or("hu".to_string()).to_string();

                if !layer.is_empty() {
                    panic!(
                        "Unknown --nn parameter for layer number {} : {:?}",
                        layer_num, layer
                    );
                }

                let activation = match &*activation_str {
                    "none" => NNActivation::None,
                    "relu" => NNActivation::Relu,
                    _ => Err(format!(
                        "unknown nn activation type: \"{}\"",
                        activation_str
                    ))
                    .unwrap(),
                };

                let layernorm = match &*layernorm_str {
                    "none" => NNLayerNorm::None,
                    "before" => NNLayerNorm::BeforeRelu,
                    "after" => NNLayerNorm::AfterRelu,
                    _ => Err(format!("unknown nn layer norm: \"{}\"", layernorm_str)).unwrap(),
                };

                let init_type = match &*init_type_str {
                    "xavier" => InitType::Xavier,
                    "hu" => InitType::Hu,
                    "one" => InitType::One,
                    "zero" => InitType::Zero,
                    _ => Err(format!(
                        "unknown nn initialization type: \"{}\"",
                        init_type_str
                    ))
                    .unwrap(),
                };
                let neuron_type = block_neural::NeuronType::WeightedSum;
                output = block_neural::new_neuronlayer_block(
                    &mut bg,
                    mi,
                    output,
                    neuron_type,
                    width,
                    init_type,
                    dropout, // dropout
                    maxnorm, // max norm
                    false,
                )
                .unwrap();

                if layernorm == NNLayerNorm::BeforeRelu {
                    output =
                        block_normalize::new_normalize_layer_block(&mut bg, mi, output).unwrap();
                }
                if activation == NNActivation::Relu {
                    output = block_relu::new_relu_block(&mut bg, mi, output).unwrap();
                }
                if layernorm == NNLayerNorm::AfterRelu {
                    output =
                        block_normalize::new_normalize_layer_block(&mut bg, mi, output).unwrap();
                }
            }
            // If we have split
            if join_block.is_some() {
                output =
                    block_misc::new_join_block(&mut bg, vec![output, join_block.unwrap()]).unwrap();
            }
            output = block_neural::new_neuron_block(
                &mut bg,
                mi,
                output,
                block_neural::NeuronType::WeightedSum,
                block_neural::InitType::One,
            )
            .unwrap();
        }

        // now sigmoid has a single input
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, output, true).unwrap();
        bg.finalize();
        rg.tape_len = bg.get_tape_size();

        rg.blocks_boxes = bg.take_blocks();

        rg
    }

    pub fn allocate_and_init_weights_(&mut self, mi: &model_instance::ModelInstance) {
        for rr in &mut self.blocks_boxes {
            rr.allocate_and_init_weights(mi);
        }
    }

    pub fn new(mi: &model_instance::ModelInstance) -> Regressor {
        let mut rg = Regressor::new_without_weights(mi);
        rg.allocate_and_init_weights(mi);
        rg
    }

    pub fn get_name(&self) -> String {
        self.regressor_name.to_owned()
    }

    pub fn new_portbuffer(&self) -> port_buffer::PortBuffer {
        port_buffer::PortBuffer::new(self.tape_len)
    }

    pub fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.allocate_and_init_weights_(mi);
    }

    pub fn learn(
        &mut self,
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) -> f32 {
        if update && self.immutable {
            // Important to know: learn() functions in blocks aren't guaranteed to be thread-safe
            panic!("This regressor is immutable, you cannot call learn() with update = true");
        }
        let update: bool = update && (fb.example_importance != 0.0);
        if !update {
            // Fast-path for no-update case
            return self.predict(fb, pb);
        }

        pb.reset(); // empty the tape
        let further_blocks = &mut self.blocks_boxes[..];
        block_helpers::forward_backward(further_blocks, fb, pb, update);

        assert_eq!(pb.observations.len(), 1);

        pb.observations.pop().unwrap()
    }

    pub fn predict(
        &self,
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) -> f32 {
        // TODO: we should find a way of not using unsafe
        pb.reset(); // empty the tape

        let further_blocks = &self.blocks_boxes[..];
        block_helpers::forward(further_blocks, fb, pb);

        assert_eq!(pb.observations.len(), 1);

        pb.observations.pop().unwrap()
    }

    pub fn predict_with_cache(
        &self,
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        caches: &[BlockCache],
    ) -> f32 {
        pb.reset(); // empty the tape

        let further_blocks = &self.blocks_boxes[..];
        block_helpers::forward_with_cache(further_blocks, fb, pb, caches);

        assert_eq!(pb.observations.len(), 1);
        pb.observations.pop().unwrap()
    }

    pub fn setup_cache(
        &mut self,
        fb: &feature_buffer::FeatureBuffer,
        caches: &mut Vec<BlockCache>,
        should_create: bool,
    ) {
        let further_blocks = self.blocks_boxes.as_mut_slice();
        if should_create {
            block_helpers::create_forward_cache(further_blocks, caches);
        }
        block_helpers::prepare_forward_cache(further_blocks, fb, caches.as_mut_slice());
    }

    // Yeah, this is weird. I just didn't want to break the format compatibility at this point
    pub fn write_weights_to_buf(
        &self,
        output_bufwriter: &mut dyn io::Write,
    ) -> Result<(), Box<dyn Error>> {
        let length = self
            .blocks_boxes
            .iter()
            .map(|block| block.get_serialized_len())
            .sum::<usize>() as u64;
        output_bufwriter.write_u64::<LittleEndian>(length)?;

        for v in &self.blocks_boxes {
            v.write_weights_to_buf(output_bufwriter)?;
        }
        Ok(())
    }

    pub fn overwrite_weights_from_buf(
        &mut self,
        input_bufreader: &mut dyn io::Read,
    ) -> Result<(), Box<dyn Error>> {
        // This is a bit weird format
        // You would expect each block to have its own sig
        // We'll break compatibility in next release or something similar
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let expected_length = self
            .blocks_boxes
            .iter()
            .map(|block| block.get_serialized_len())
            .sum::<usize>() as u64;
        if len != expected_length {
            return Err(format!(
                "Lenghts of weights array in regressor file differ: got {}, expected {}",
                len, expected_length
            ))?;
        }
        for v in &mut self.blocks_boxes {
            v.read_weights_from_buf(input_bufreader)?;
        }

        Ok(())
    }

    pub fn immutable_regressor_without_weights(
        &mut self,
        mi: &model_instance::ModelInstance,
    ) -> Result<Regressor, Box<dyn Error>> {
        // make sure we are creating immutable regressor from SGD mi
        assert_eq!(mi.optimizer, model_instance::Optimizer::SGD);

        let mut rg = Regressor::new_without_weights(mi);
        rg.immutable = true;
        Ok(rg)
    }

    pub fn into_immutable_regressor_from_buf(
        &mut self,
        rg: &mut Regressor,
        input_bufreader: &mut dyn io::Read,
    ) -> Result<(), Box<dyn Error>> {
        // TODO Ideally we would make a copy, not based on model_instance. but this is easier at the moment

        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let expected_length = self
            .blocks_boxes
            .iter()
            .map(|bb| bb.get_serialized_len())
            .sum::<usize>() as u64;
        if len != expected_length {
            return Err(format!(
                "Lenghts of weights array in regressor file differ: got {}, expected {}",
                len, expected_length
            ))?;
        }
        for (i, v) in &mut self.blocks_boxes.iter().enumerate() {
            v.read_weights_from_buf_into_forward_only(input_bufreader, &mut rg.blocks_boxes[i])?;
        }

        Ok(())
    }

    // Create immutable regressor from current regressor
    pub fn immutable_regressor(
        &mut self,
        mi: &model_instance::ModelInstance,
    ) -> Result<Regressor, Box<dyn Error>> {
        // Only to be used by unit tests
        // make sure we are creating immutable regressor from SGD mi
        assert_eq!(mi.optimizer, model_instance::Optimizer::SGD);
        let mut rg = self.immutable_regressor_without_weights(mi)?;
        rg.allocate_and_init_weights(mi);

        let mut tmp_vec: Vec<u8> = Vec::new();
        for (i, v) in &mut self.blocks_boxes.iter().enumerate() {
            let mut cursor = Cursor::new(&mut tmp_vec);
            v.write_weights_to_buf(&mut cursor)?;
            cursor.set_position(0);
            v.read_weights_from_buf_into_forward_only(&mut cursor, &mut rg.blocks_boxes[i])?;
        }
        Ok(rg)
    }
}

mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::feature_buffer::HashAndValue;

    /* LR TESTS */
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
    fn test_learning_turned_off() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.optimizer = model_instance::Optimizer::AdagradLUT;
        let mut re = Regressor::new(&mi);
        let mut pb = re.new_portbuffer();
        // Empty model: no matter how many features, prediction is 0.5
        assert_eq!(re.learn(&lr_vec(vec![]), &mut pb, false), 0.5);
        assert_eq!(
            re.learn(
                &lr_vec(vec![HashAndValue {
                    hash: 1,
                    value: 1.0,
                    combo_index: 0,
                }]),
                &mut pb,
                false
            ),
            0.5
        );
        assert_eq!(
            re.learn(
                &lr_vec(vec![
                    HashAndValue {
                        hash: 1,
                        value: 1.0,
                        combo_index: 0,
                    },
                    HashAndValue {
                        hash: 2,
                        value: 1.0,
                        combo_index: 0,
                    }
                ]),
                &mut pb,
                false
            ),
            0.5
        );
    }

    #[test]
    fn test_power_t_zero() {
        // When power_t is zero, then all optimizers behave exactly like SGD
        // So we want to test all three
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;

        let vec_in = &lr_vec(vec![HashAndValue {
            hash: 1,
            value: 1.0,
            combo_index: 0,
        }]);

        // Here learning rate mechanism does not affect the observations, so let's verify three different ones
        mi.optimizer = model_instance::Optimizer::AdagradFlex;

        let mut regressors: Vec<Box<Regressor>> = vec![
            //Box::new(Regressor::<optimizer::OptimizerAdagradLUT>::new(&mi)),
            Box::new(Regressor::new(&mi)),
            //Box::new(Regressor::<optimizer::OptimizerSGD>::new(&mi))
        ];

        let mut pb = regressors[0].new_portbuffer();

        for re in &mut regressors {
            assert_eq!(re.learn(vec_in, &mut pb, true), 0.5);
            assert_eq!(re.learn(vec_in, &mut pb, true), 0.48750263);
            assert_eq!(re.learn(vec_in, &mut pb, true), 0.47533244);
        }
    }

    #[test]
    fn test_double_same_feature() {
        // this is a tricky test - what happens on collision
        // depending on the order of math, observations are different
        // so this is here, to make sure the math is always the same
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.optimizer = model_instance::Optimizer::AdagradLUT;

        let mut re = Regressor::new(&mi);
        let mut pb = re.new_portbuffer();
        let vec_in = &lr_vec(vec![
            HashAndValue {
                hash: 1,
                value: 1.0,
                combo_index: 0,
            },
            HashAndValue {
                hash: 1,
                value: 2.0,
                combo_index: 0,
            },
        ]);

        assert_eq!(re.learn(vec_in, &mut pb, true), 0.5);
        assert_eq!(re.learn(vec_in, &mut pb, true), 0.38936076);
        assert_eq!(re.learn(vec_in, &mut pb, true), 0.30993468);
    }

    #[test]
    fn test_power_t_half__() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.init_acc_gradient = 0.0;
        mi.optimizer = model_instance::Optimizer::AdagradFlex;
        let mut re = Regressor::new(&mi);
        let mut pb = re.new_portbuffer();

        assert_eq!(
            re.learn(
                &lr_vec(vec![HashAndValue {
                    hash: 1,
                    value: 1.0,
                    combo_index: 0
                }]),
                &mut pb,
                true
            ),
            0.5
        );
        assert_eq!(
            re.learn(
                &lr_vec(vec![HashAndValue {
                    hash: 1,
                    value: 1.0,
                    combo_index: 0
                }]),
                &mut pb,
                true
            ),
            0.4750208
        );
        assert_eq!(
            re.learn(
                &lr_vec(vec![HashAndValue {
                    hash: 1,
                    value: 1.0,
                    combo_index: 0
                }]),
                &mut pb,
                true
            ),
            0.45788094
        );
    }

    #[test]
    fn test_power_t_half_fastmath() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.fastmath = true;
        mi.optimizer = model_instance::Optimizer::AdagradLUT;
        mi.init_acc_gradient = 0.0;

        let mut re = get_regressor_with_weights(&mi);
        let mut pb = re.new_portbuffer();
        let mut p: f32;

        p = re.learn(
            &lr_vec(vec![HashAndValue {
                hash: 1,
                value: 1.0,
                combo_index: 0,
            }]),
            &mut pb,
            true,
        );
        assert_eq!(p, 0.5);
        p = re.learn(
            &lr_vec(vec![HashAndValue {
                hash: 1,
                value: 1.0,
                combo_index: 0,
            }]),
            &mut pb,
            true,
        );
        if optimizer::FASTMATH_LR_LUT_BITS == 12 {
            assert_eq!(p, 0.47539312);
        } else if optimizer::FASTMATH_LR_LUT_BITS == 11 {
            assert_eq!(p, 0.475734);
        } else {
            assert!(
                false,
                "Exact value for the test is missing, please edit the test"
            );
        }
    }

    #[test]
    fn test_power_t_half_two_features() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.bit_precision = 18;
        mi.init_acc_gradient = 0.0;
        mi.optimizer = model_instance::Optimizer::AdagradFlex;

        let mut re = Regressor::new(&mi);
        let mut pb = re.new_portbuffer();
        // Here we take twice two features and then once just one
        assert_eq!(
            re.learn(
                &lr_vec(vec![
                    HashAndValue {
                        hash: 1,
                        value: 1.0,
                        combo_index: 0
                    },
                    HashAndValue {
                        hash: 2,
                        value: 1.0,
                        combo_index: 0
                    }
                ]),
                &mut pb,
                true
            ),
            0.5
        );
        assert_eq!(
            re.learn(
                &lr_vec(vec![
                    HashAndValue {
                        hash: 1,
                        value: 1.0,
                        combo_index: 0
                    },
                    HashAndValue {
                        hash: 2,
                        value: 1.0,
                        combo_index: 0
                    }
                ]),
                &mut pb,
                true
            ),
            0.45016602
        );
        assert_eq!(
            re.learn(
                &lr_vec(vec![HashAndValue {
                    hash: 1,
                    value: 1.0,
                    combo_index: 0
                }]),
                &mut pb,
                true
            ),
            0.45836908
        );
    }

    #[test]
    fn test_non_one_weight() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        mi.optimizer = model_instance::Optimizer::AdagradLUT;

        let mut re = Regressor::new(&mi);
        let mut pb = re.new_portbuffer();

        assert_eq!(
            re.learn(
                &lr_vec(vec![HashAndValue {
                    hash: 1,
                    value: 2.0,
                    combo_index: 0
                }]),
                &mut pb,
                true
            ),
            0.5
        );
        assert_eq!(
            re.learn(
                &lr_vec(vec![HashAndValue {
                    hash: 1,
                    value: 2.0,
                    combo_index: 0
                }]),
                &mut pb,
                true
            ),
            0.45016602
        );
        assert_eq!(
            re.learn(
                &lr_vec(vec![HashAndValue {
                    hash: 1,
                    value: 2.0,
                    combo_index: 0
                }]),
                &mut pb,
                true
            ),
            0.40611085
        );
    }

    #[test]
    fn test_example_importance() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        mi.optimizer = model_instance::Optimizer::AdagradLUT;
        mi.fastmath = true;

        let mut re = Regressor::new(&mi);
        let mut pb = re.new_portbuffer();

        let mut fb_instance = lr_vec(vec![HashAndValue {
            hash: 1,
            value: 1.0,
            combo_index: 0,
        }]);
        fb_instance.example_importance = 0.5;
        assert_eq!(re.learn(&fb_instance, &mut pb, true), 0.5);
        assert_eq!(re.learn(&fb_instance, &mut pb, true), 0.49375027);
        assert_eq!(re.learn(&fb_instance, &mut pb, true), 0.4875807);
    }
}
