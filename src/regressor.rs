//use std::mem::{self};
use std::any::Any;
use std::mem;
use std::slice;
use std::sync::Arc;
use core::arch::x86_64::*;
use merand48::*;
use std::io;
use std::io::Cursor;
use std::error::Error;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::cmp::min;

use crate::model_instance;
use crate::feature_buffer;
use crate::port_buffer;
use crate::optimizer;
use optimizer::OptimizerTrait;
use crate::block_ffm;
use crate::block_lr;
use crate::block_loss_functions;
use crate::block_neural;
use crate::block_relu;
use crate::block_misc;
use crate::graph;
use crate::block_neural::{InitType};
use crate::block_helpers;

pub trait BlockTrait {
    fn as_any(&mut self) -> &mut dyn Any; // This enables downcasting
    fn forward_backward(&mut self, 
                         further_blocks: &mut [Box<dyn BlockTrait>], 
                         fb: &feature_buffer::FeatureBuffer,
                         pb: &mut port_buffer::PortBuffer, 
                         update:bool);

    fn forward(		&self, 
                         further_blocks: &[Box<dyn BlockTrait>], 
                         fb: &feature_buffer::FeatureBuffer,
                         pb: &mut port_buffer::PortBuffer, );

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {}
    fn get_serialized_len(&self) -> usize {0}
    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {Ok(())}
    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {Ok(())}
    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize;
    fn get_num_output_slots(&self) -> usize;
    fn get_input_offset(&mut self, input: graph::InputSlot) -> Result<usize, Box<dyn Error>> {Err(format!("get_input_offset() is only supported by CopyBlock"))?}
    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {}
    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {}
    fn get_block_type(&self) -> graph::BlockType {graph::BlockType::Regular}  

    fn read_weights_from_buf_into_forward_only(&self, input_bufreader: &mut dyn io::Read, forward: &mut Box<dyn BlockTrait>) -> Result<(), Box<dyn Error>> {Ok(())}

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>> {Ok(())}
}


pub struct Regressor {
    pub regressor_name: String,
    pub blocks_boxes: Vec<Box<dyn BlockTrait>>,
    pub tape_len: usize,
    pub immutable: bool,
}


pub fn get_regressor_without_weights(mi: &model_instance::ModelInstance) -> Regressor {
        Regressor::new_without_weights(&mi)
}

pub fn get_regressor_with_weights(mi: &model_instance::ModelInstance) -> Regressor {
    let mut re = get_regressor_without_weights(mi);
    re.allocate_and_init_weights(mi);
    re
}

#[derive(PartialEq)]
enum NNActivation {
    None,
    Relu
}

impl Regressor  {
    pub fn new_without_weights(mi: &model_instance::ModelInstance) -> Regressor {

        let mut rg = Regressor{
            blocks_boxes: Vec::new(),
            regressor_name: format!("Regressor with optimizer \"{:?}\"", mi.optimizer),
            immutable: false,
            tape_len: usize::MAX,
        };

        let mut bg = graph::BlockGraph::new();
        // A bit more elaborate than necessary. Let's really make it clear what's happening
        let mut output = block_lr::new_lr_block(&mut bg, mi).unwrap();

        if mi.ffm_k > 0 {
            let mut block_ffm = block_ffm::new_ffm_block(&mut bg, mi).unwrap();
            let mut triangle_ffm = block_misc::new_triangle_block(&mut bg, block_ffm).unwrap();
            output = block_misc::new_join_block(&mut bg, vec![output, triangle_ffm]).unwrap();
            //output = block_misc::new_join_block(&mut bg, vec![output, block_ffm]).unwrap();
        }


        if mi.nn_config.layers.len() > 0 {
            let mut join_block : Option<graph::BlockPtrOutput> = None;
            if mi.nn_config.topology == "one" {
                let mut outputs = block_misc::new_copy_block(&mut bg, output, 2).unwrap();
                join_block = Some(outputs.pop().unwrap());
                output = outputs.pop().unwrap();
            } else if mi.nn_config.topology == "two" {
                // do not copy out the 
            } else if mi.nn_config.topology == "three" {
                join_block = Some(output);
                let mut lr_block = block_lr::new_lr_block(&mut bg, mi).unwrap();
                let mut block_ffm = block_ffm::new_ffm_block(&mut bg, mi).unwrap();
                let mut triangle_ffm = block_misc::new_triangle_block(&mut bg, block_ffm).unwrap();
                output = block_misc::new_join_block(&mut bg, vec![lr_block, triangle_ffm]).unwrap();
                println!("topology 3: we have entirely separate ebeddings for nn");
            } else {
                Err(format!("unknown nn topology: \"{}\"", mi.nn_config.topology)).unwrap()
            }
            


            for (layer_num, layer) in mi.nn_config.layers.iter().enumerate() {
                let mut layer = layer.clone();
                let activation_str: String = layer.remove("activation").unwrap_or("none".to_string()).to_string();
                let width: usize = layer.remove("width").unwrap_or("20".to_string()).parse().unwrap();
                let maxnorm: f32 = layer.remove("maxnorm").unwrap_or("0.0".to_string()).parse().unwrap();
                let dropout: f32 = layer.remove("dropout").unwrap_or("0.0".to_string()).parse().unwrap();
                let init_type_str: String = layer.remove("init").unwrap_or("hu".to_string()).to_string();
                
                if layer.len() > 0 {
                    panic!("Unknown --nn parameter for layer number {} : {:?}", layer_num, layer); 
                }
                
                let activation = match &*activation_str {
                    "none" => NNActivation::None,
                    "relu" => NNActivation::Relu,
                    _ => Err(format!("unknown nn activation type: \"{}\"", activation_str)).unwrap()
                };
                
                let init_type = match &*init_type_str {
                    "xavier" => InitType::Xavier,
                    "hu" => InitType::Hu,
                    "one" => InitType::One,
                    "zero" => InitType::Zero,
                    _ => Err(format!("unknown nn initialization type: \"{}\"", init_type_str)).unwrap()
                };
                let neuron_type = block_neural::NeuronType::WeightedSum;
                println!("Neuron layer: width: {}, neuron type: {:?}, dropout: {}, maxnorm: {}, init_type: {:?}",
                                        width, neuron_type, dropout, maxnorm, init_type);
                output =  block_neural::new_neuronlayer_block(&mut bg, 
                                            &mi, 
                                            output,
                                            neuron_type, 
                                            width,
                                            init_type,
                                            dropout, // dropout
                                            maxnorm, // max norm
                                            ).unwrap();

                if activation == NNActivation::Relu {
                    output = block_relu::new_relu_block(&mut bg, &mi, output).unwrap();
                    println!("Relu layer");
                }


            }
            // If we have split
            if join_block.is_some() {
                output = block_misc::new_join_block(&mut bg, vec![output, join_block.unwrap()]).unwrap();
            }
            output = block_neural::new_neuron_block(&mut bg, &mi, output, block_neural::NeuronType::WeightedSum, block_neural::InitType::One).unwrap();
        }
         

        // now sigmoid has a single input
//        println!("INPUTS : {}", inputs);
        let lossf = block_loss_functions::new_logloss_block(&mut bg, output, true).unwrap();
        bg.finalize();
//        bg.println();
        rg.tape_len = bg.get_tape_size();
        
        rg.blocks_boxes = bg.take_blocks();
        /*for (i, block) in bg.blocks.into_iter().enumerate() {
            rg.blocks_boxes.push(block);
        }*/
        
        rg
    }
    
    pub fn allocate_and_init_weights_(&mut self, mi: &model_instance::ModelInstance) {
        for rr in &mut self.blocks_boxes {
            rr.allocate_and_init_weights(mi);
        }
    }
    

    pub fn new(mi: &model_instance::ModelInstance) -> Regressor 
    {
        let mut rg = Regressor::new_without_weights(mi);
        rg.allocate_and_init_weights(mi);
        rg
    }

    pub fn get_name(&self) -> String {
        self.regressor_name.to_owned()    
    }


    pub fn new_portbuffer(&self) -> port_buffer::PortBuffer
    {
        port_buffer::PortBuffer::new(self.tape_len)
    }

    pub fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.allocate_and_init_weights_(mi);
    }

    pub fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, pb: &mut port_buffer::PortBuffer, update: bool) -> f32 {
        if update && self.immutable {
            // Important to know: learn() functions in blocks aren't guaranteed to be thread-safe
            panic!("This regressor is immutable, you cannot call learn() with update = true");
        }
        let update:bool = update && (fb.example_importance != 0.0);
        if !update { // Fast-path for no-update case
            return self.predict(fb, pb);
        }

        pb.reset(); // empty the tape
        let further_blocks = &mut self.blocks_boxes[..];
        block_helpers::forward_backward(further_blocks, fb, pb, update);

        assert_eq!(pb.observations.len(), 1);
        let prediction_probability = pb.observations.pop().unwrap();
    
        return prediction_probability
    }
    
    pub fn predict(&self, fb: &feature_buffer::FeatureBuffer, pb: &mut port_buffer::PortBuffer) -> f32 {
        // TODO: we should find a way of not using unsafe
        pb.reset(); // empty the tape

        let further_blocks = &self.blocks_boxes[..];
        block_helpers::forward(further_blocks, fb, pb);

        assert_eq!(pb.observations.len(), 1);
        let prediction_probability = pb.observations.pop().unwrap();

        return prediction_probability
    }
    
    // Yeah, this is weird. I just didn't want to break the format compatibility at this point
    pub fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        let length = self.blocks_boxes.iter().map(|block| block.get_serialized_len()).sum::<usize>() as u64;
        output_bufwriter.write_u64::<LittleEndian>(length as u64)?;

        for v in &self.blocks_boxes {
            v.write_weights_to_buf(output_bufwriter)?;
        }
        Ok(())
    }
    

    pub fn overwrite_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        // This is a bit weird format
        // You would expect each block to have its own sig
        // We'll break compatibility in next release or something similar
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let expected_length = self.blocks_boxes.iter().map(|block| block.get_serialized_len()).sum::<usize>() as u64;
        if len != expected_length {
            return Err(format!("Lenghts of weights array in regressor file differ: got {}, expected {}", len, expected_length))?;
        }
        for v in &mut self.blocks_boxes {
            v.read_weights_from_buf(input_bufreader)?;
        }

        Ok(())
    }


    pub fn immutable_regressor_without_weights(&mut self, mi: &model_instance::ModelInstance)  -> Result<Regressor, Box<dyn Error>> {
        // make sure we are creating immutable regressor from SGD mi
        assert!(mi.optimizer == model_instance::Optimizer::SGD);       

        let mut rg = Regressor::new_without_weights(&mi);
        rg.immutable = true;
        Ok(rg)        
    }

    
    pub fn into_immutable_regressor_from_buf(&mut self, rg: &mut Regressor, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        // TODO Ideally we would make a copy, not based on model_instance. but this is easier at the moment
    
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let expected_length = self.blocks_boxes.iter().map(|bb| bb.get_serialized_len()).sum::<usize>() as u64;
        if len != expected_length {
            return Err(format!("Lenghts of weights array in regressor file differ: got {}, expected {}", len, expected_length))?;
        }
        for (i, v) in &mut self.blocks_boxes.iter().enumerate() {
            v.read_weights_from_buf_into_forward_only(input_bufreader, &mut rg.blocks_boxes[i])?;
        }

        Ok(())
    }

    // Create immutable regressor from current regressor
    pub fn immutable_regressor(&mut self, mi: &model_instance::ModelInstance) -> Result<Regressor, Box<dyn Error>> {
        // Only to be used by unit tests 
        // make sure we are creating immutable regressor from SGD mi
        assert!(mi.optimizer == model_instance::Optimizer::SGD);       
        let mut rg = self.immutable_regressor_without_weights(&mi)?;
        rg.allocate_and_init_weights(&mi);

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
    fn test_learning_turned_off() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap(); 
        mi.optimizer = model_instance::Optimizer::AdagradLUT;
        let mut re = Regressor::new(&mi);
        let mut pb = re.new_portbuffer();
        // Empty model: no matter how many features, prediction is 0.5
        assert_eq!(re.learn(&lr_vec(vec![]), &mut pb, false), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0, combo_index: 0,}]), &mut pb, false), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0, combo_index: 0,}, HashAndValue{hash:2, value: 1.0, combo_index: 0,}]), &mut pb, false), 0.5);
    }

    #[test]
    fn test_power_t_zero() {
        // When power_t is zero, then all optimizers behave exactly like SGD
        // So we want to test all three   
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        
        let vec_in = &lr_vec(vec![HashAndValue{hash: 1, value: 1.0, combo_index: 0,}]);
        
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
        let vec_in = &lr_vec(vec![HashAndValue{hash: 1, value: 1.0, combo_index: 0}, HashAndValue{hash: 1, value: 2.0, combo_index: 0,}]);

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
        
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0, combo_index: 0}]), &mut pb, true), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0, combo_index: 0}]), &mut pb, true), 0.4750208);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0, combo_index: 0}]), &mut pb, true), 0.45788094);
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
        
        p = re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0, combo_index: 0}]), &mut pb, true);
        assert_eq!(p, 0.5);
        p = re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0, combo_index: 0}]), &mut pb, true);
        if optimizer::FASTMATH_LR_LUT_BITS == 12 { 
            assert_eq!(p, 0.47539312);
        } else if optimizer::FASTMATH_LR_LUT_BITS == 11 { 
            assert_eq!(p, 0.475734);
        } else {
            assert!(false, "Exact value for the test is missing, please edit the test");
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
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0, combo_index: 0}, HashAndValue{hash:2, value: 1.0, combo_index: 0}]), &mut pb, true), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0, combo_index: 0}, HashAndValue{hash:2, value: 1.0, combo_index: 0}]), &mut pb, true), 0.45016602);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0, combo_index: 0}]), &mut pb,  true), 0.45836908);
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
        
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0, combo_index: 0}]), &mut pb, true), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0, combo_index: 0}]), &mut pb, true), 0.45016602);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0, combo_index: 0}]), &mut pb, true), 0.40611085);
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
        
        let mut fb_instance = lr_vec(vec![HashAndValue{hash: 1, value: 1.0, combo_index: 0}]);
        fb_instance.example_importance = 0.5;
        assert_eq!(re.learn(&fb_instance, &mut pb, true), 0.5);
        assert_eq!(re.learn(&fb_instance, &mut pb, true), 0.49375027);
        assert_eq!(re.learn(&fb_instance, &mut pb, true), 0.4875807);
    }

}

