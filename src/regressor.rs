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
use crate::block_neuron;



pub trait BlockTrait {
    fn as_any(&mut self) -> &mut dyn Any; // This enables downcasting
    fn forward_backward(&mut self, 
                         further_blocks: &mut [Box<dyn BlockTrait>], 
                         fb: &feature_buffer::FeatureBuffer,
                         pb: &mut port_buffer::PortBuffer, 
                         update:bool);

    fn forward(&self, 
                         further_blocks: &[Box<dyn BlockTrait>], 
                         wsum: f32, 
                         fb: &feature_buffer::FeatureBuffer) -> f32;

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance);
    fn get_serialized_len(&self) -> usize;
    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>>;
    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>>;
    fn get_num_outputs(&self) -> u32;
    fn set_num_inputs(&mut self, num_inputs: u32);
    fn set_input_tape_index(&mut self, input_tape_index: i32);
    fn set_output_tape_index(&mut self, output_tape_index: i32);
    fn get_output_tape_index(&self) -> i32;

    fn read_weights_from_buf_into_forward_only(&self, input_bufreader: &mut dyn io::Read, forward: &mut Box<dyn BlockTrait>) -> Result<(), Box<dyn Error>>;

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>>;
}


pub struct Regressor {
    pub regressor_name: String,
    pub blocks_boxes: Vec<Box<dyn BlockTrait>>,
    pub immutable: bool,
    pub result_tape_index: i32,
}


pub fn get_regressor_without_weights(mi: &model_instance::ModelInstance) -> Regressor {
        Regressor::new_without_weights(&mi)
}

pub fn get_regressor_with_weights(mi: &model_instance::ModelInstance) -> Regressor {
    let mut re = get_regressor_without_weights(mi);
    re.allocate_and_init_weights(mi);
    re
}

impl Regressor  {
    pub fn new_without_weights(mi: &model_instance::ModelInstance) -> Regressor {

        let mut rg = Regressor{
            blocks_boxes: Vec::new(),
            regressor_name: format!("Regressor with optimizer \"{:?}\"", mi.optimizer),
            immutable: false,
            result_tape_index: -1,
        };

        let mut inputs = 0;
        // A bit more elaborate than necessary. Let's really make it clear what's happening
        let mut reg_lr = block_lr::new_without_weights(mi).unwrap();
        inputs += reg_lr.get_num_outputs();
        reg_lr.set_output_tape_index(0);
        rg.blocks_boxes.push(reg_lr);

        if mi.ffm_k > 0 {
            let mut reg_ffm = block_ffm::new_without_weights(mi).unwrap();
            inputs += reg_ffm.get_num_outputs();
            reg_ffm.set_output_tape_index(0);
            rg.blocks_boxes.push(reg_ffm);
        }
         
        // If we put sum function here, it has to be neutral
        let mut reg = block_neuron::new_without_weights(mi, block_neuron::NeuronType::Sum).unwrap();
        reg.set_num_inputs(inputs);
        inputs = reg.get_num_outputs();
        reg.set_input_tape_index(0);
        reg.set_output_tape_index(1);
        rg.blocks_boxes.push(reg);
                    
        // now sigmoid has a single input
        let mut reg_sigmoid = block_loss_functions::new_without_weights(mi).unwrap();
        reg_sigmoid.set_num_inputs(inputs);
        reg_sigmoid.set_input_tape_index(1);
        reg_sigmoid.set_output_tape_index(2);
        rg.blocks_boxes.push(reg_sigmoid);
        rg.result_tape_index = 2;

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


    pub fn new_portbuffer(&self, mi: &model_instance::ModelInstance) -> port_buffer::PortBuffer
    {
        port_buffer::PortBuffer::new(mi)
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
/*        if !update { // Fast-path for no-update case
            return self.predict(fb);
        }
*/
        let blocks_list = &mut self.blocks_boxes[..];
        let (current, further_blocks) = &mut blocks_list.split_at_mut(1);
        pb.reset(); // empty the tape

        current[0].forward_backward(further_blocks, fb, pb, update);
        let prediction_probability = pb.tapes[self.result_tape_index as usize].pop().unwrap();
    
        return prediction_probability
    }
    
    pub fn predict(&self, fb: &feature_buffer::FeatureBuffer) -> f32 {
        // TODO: we should find a way of not using unsafe
        let blocks_list = &self.blocks_boxes[..];
        let (current, further_blocks) = blocks_list.split_at(1);
        let prediction_probability = current[0].forward(further_blocks, 0.0, fb);
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
        let mut pb = re.new_portbuffer(&mi);
        // Empty model: no matter how many features, prediction is 0.5
        assert_eq!(re.learn(&lr_vec(vec![]), &mut pb, false), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), &mut pb, false), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), &mut pb, false), 0.5);
    }

    #[test]
    fn test_power_t_zero() {
        // When power_t is zero, then all optimizers behave exactly like SGD
        // So we want to test all three   
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        
        let vec_in = &lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]);
        
        // Here learning rate mechanism does not affect the results, so let's verify three different ones
        mi.optimizer = model_instance::Optimizer::AdagradFlex;

        let mut regressors: Vec<Box<Regressor>> = vec![
            //Box::new(Regressor::<optimizer::OptimizerAdagradLUT>::new(&mi)),
            Box::new(Regressor::new(&mi)),
            //Box::new(Regressor::<optimizer::OptimizerSGD>::new(&mi))
            ];
        
        let mut pb = regressors[0].new_portbuffer(&mi);
        
        for re in &mut regressors {
            assert_eq!(re.learn(vec_in, &mut pb, true), 0.5);
            assert_eq!(re.learn(vec_in, &mut pb, true), 0.48750263);
            assert_eq!(re.learn(vec_in, &mut pb, true), 0.47533244);
        }
    }

    #[test]
    fn test_double_same_feature() {
        // this is a tricky test - what happens on collision
        // depending on the order of math, results are different
        // so this is here, to make sure the math is always the same
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.optimizer = model_instance::Optimizer::AdagradLUT;
        
        let mut re = Regressor::new(&mi);
        let mut pb = re.new_portbuffer(&mi);
        let vec_in = &lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash: 1, value: 2.0}]);

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
        let mut pb = re.new_portbuffer(&mi);
        
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), &mut pb, true), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), &mut pb, true), 0.4750208);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), &mut pb, true), 0.45788094);
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
        let mut pb = re.new_portbuffer(&mi);
        let mut p: f32;
        
        p = re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), &mut pb, true);
        assert_eq!(p, 0.5);
        p = re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), &mut pb, true);
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
        let mut pb = re.new_portbuffer(&mi);
        // Here we take twice two features and then once just one
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), &mut pb, true), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), &mut pb, true), 0.45016602);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), &mut pb,  true), 0.45836908);
    }

    #[test]
    fn test_non_one_weight() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        mi.optimizer = model_instance::Optimizer::AdagradLUT;
        
        let mut re = Regressor::new(&mi);
        let mut pb = re.new_portbuffer(&mi);
        
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), &mut pb, true), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), &mut pb, true), 0.45016602);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), &mut pb, true), 0.40611085);
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
        let mut pb = re.new_portbuffer(&mi);
        
        let mut fb_instance = lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]);
        fb_instance.example_importance = 0.5;
        assert_eq!(re.learn(&fb_instance, &mut pb, true), 0.5);
        assert_eq!(re.learn(&fb_instance, &mut pb, true), 0.49375027);
        assert_eq!(re.learn(&fb_instance, &mut pb, true), 0.4875807);
    }

}

