//use std::mem::{self};
use std::any::Any;
use std::mem::{self, MaybeUninit};
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
use crate::feature_buffer::HashAndValue;
use crate::feature_buffer::HashAndValueAndSeq;
use crate::optimizer;
use crate::consts;
use optimizer::OptimizerTrait;
use crate::block_ffm::BlockFFM;
use crate::block_lr::BlockLR;
use crate::block_loss_functions::BlockSigmoid;



pub trait BlockTrait {
    fn as_any(&mut self) -> &mut dyn Any; // This enables downcasting
    fn forward_backward(&mut self, 
                         further_blocks: &mut [Box<dyn BlockTrait>], 
                         wsum: f32, 
                         fb: &feature_buffer::FeatureBuffer,
                         update:bool) -> (f32, f32);

    fn forward(&self, 
                         further_blocks: &[Box<dyn BlockTrait>], 
                         wsum: f32, 
                         fb: &feature_buffer::FeatureBuffer) -> f32;

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance);
    fn get_serialized_len(&self) -> usize;
    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>>;
    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>>;
    fn new_forward_only_without_weights(&self) -> Result<Box<dyn BlockTrait>, Box<dyn Error>>;
    fn new_without_weights(mi: &model_instance::ModelInstance) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> where Self:Sized;
    fn read_weights_from_buf_into_forward_only(&self, input_bufreader: &mut dyn io::Read, forward: &mut Box<dyn BlockTrait>) -> Result<(), Box<dyn Error>>;

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>>;
}


pub struct Regressor<'a> {
    pub regressor_name: String,
    pub blocks_boxes: Vec<Box<dyn BlockTrait>>,
    pub blocks_lista: Vec<&'a mut dyn BlockTrait>,
}

pub trait RegressorTrait {
    fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, update: bool) -> f32;
    fn predict(&self, fb: &feature_buffer::FeatureBuffer) -> f32;
    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>>;
    fn overwrite_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>>; 
    fn get_name(&self) -> String;
    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance);
    fn immutable_regressor(&mut self, mi: &model_instance::ModelInstance) -> Result<Box<dyn RegressorTrait>, Box<dyn Error>>;
    fn immutable_regressor_from_buf(&mut self, mi: &model_instance::ModelInstance, input_bufreader: &mut dyn io::Read) -> Result<Box<dyn RegressorTrait>, Box<dyn Error>>;
}


pub fn get_regressor_without_weights(mi: &model_instance::ModelInstance) -> Box<dyn RegressorTrait> {
    if mi.optimizer == model_instance::Optimizer::Adagrad {
        if mi.fastmath {
            Box::new(Regressor::new_without_weights::<optimizer::OptimizerAdagradLUT>(&mi))
        } else {
            Box::new(Regressor::new_without_weights::<optimizer::OptimizerAdagradFlex>(&mi))
        }
    } else {
        Box::new(Regressor::new_without_weights::<optimizer::OptimizerSGD>(&mi))
    }    
}

pub fn get_regressor(mi: &model_instance::ModelInstance) -> Box<dyn RegressorTrait> {
    let mut re = get_regressor_without_weights(mi);
    re.allocate_and_init_weights(mi);
    re
}

impl <'a>Regressor<'a>  {
    pub fn new_without_weights<L: optimizer::OptimizerTrait + 'static>(mi: &model_instance::ModelInstance) -> Regressor<'a>
    {

        let mut reg_lr = BlockLR::<L>::new_without_weights(mi).unwrap();
        let mut reg_ffm = BlockFFM::<L>::new_without_weights(mi).unwrap();
        let mut reg_sigmoid = BlockSigmoid::new_without_weights(mi).unwrap();

        let mut rg = Regressor{
            blocks_lista: Vec::new(),
            blocks_boxes: Vec::new(),
            regressor_name: format!("Regressor with optimizer {:?}", L::get_name()),
        };

        unsafe {
            // A bit more elaborate than necessary. Let's really make it clear what's happening
            let r1: &mut dyn BlockTrait = reg_lr.as_mut();
            let r2: &mut dyn BlockTrait = mem::transmute(&mut *r1);
            rg.blocks_boxes.push(reg_lr);
            rg.blocks_lista.push(r2);

            if mi.ffm_k > 0 {
                let r1: &mut dyn BlockTrait = reg_ffm.as_mut();
                let r2: &mut dyn BlockTrait = mem::transmute(&mut *r1);
                rg.blocks_boxes.push(reg_ffm);
                rg.blocks_lista.push(r2);
            }
                        
            let r1: &mut dyn BlockTrait = reg_sigmoid.as_mut();
            let r2: &mut dyn BlockTrait = mem::transmute(&mut *r1);
            rg.blocks_lista.push(r2);
            rg.blocks_boxes.push(reg_sigmoid);
        }

        rg
    }
    
    pub fn allocate_and_init_weights_(&mut self, mi: &model_instance::ModelInstance) {
        for rr in &mut self.blocks_boxes {
            rr.allocate_and_init_weights(mi);
        }
    }
    

    pub fn new<L: optimizer::OptimizerTrait + 'static>(mi: &model_instance::ModelInstance) -> Regressor<'a> 
    {
        let mut rg = Regressor::new_without_weights::<L>(mi);
        rg.allocate_and_init_weights(mi);
        rg
    }
}

    
impl RegressorTrait for Regressor<'_> 
{
    fn get_name(&self) -> String {
        self.regressor_name.to_owned()    
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.allocate_and_init_weights_(mi);
    }

    fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, update: bool) -> f32 {
        let update:bool = update && (fb.example_importance != 0.0);

        let blocks_list = &mut self.blocks_boxes[..];
        let (current, further_blocks) = &mut blocks_list.split_at_mut(1);
        let (prediction_probability, general_gradient) = current[0].forward_backward(further_blocks, 0.0, fb, update);
    
        return prediction_probability
    }
    
    fn predict(&self, fb: &feature_buffer::FeatureBuffer) -> f32 {
        // TODO: we should find a way of not using unsafe
        let blocks_list = &self.blocks_boxes[..];
        let (current, further_blocks) = blocks_list.split_at(1);
        let prediction_probability = current[0].forward(further_blocks, 0.0, fb);
        return prediction_probability
    }
    
    // Yeah, this is weird. I just didn't want to break the format compatibility at this point
    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        let length = self.blocks_boxes.iter().map(|block| block.get_serialized_len()).sum::<usize>() as u64;
        output_bufwriter.write_u64::<LittleEndian>(length as u64)?;

        for v in &self.blocks_boxes {
            v.write_weights_to_buf(output_bufwriter)?;
        
        }
        Ok(())
    }
    

    fn overwrite_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
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

    
    fn immutable_regressor_from_buf(&mut self, mi: &model_instance::ModelInstance, input_bufreader: &mut dyn io::Read) -> Result<Box<dyn RegressorTrait>, Box<dyn Error>> {
        // TODO Ideally we would make a copy, not based on model_instance. but this is easier at the moment
        
        let mut rg = Box::new(Regressor::new_without_weights::<optimizer::OptimizerSGD>(&mi));
    
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let expected_length = self.blocks_boxes.iter().map(|bb| bb.get_serialized_len()).sum::<usize>() as u64;
        if len != expected_length {
            return Err(format!("Lenghts of weights array in regressor file differ: got {}, expected {}", len, expected_length))?;
        }
        for (i, v) in &mut self.blocks_boxes.iter().enumerate() {
            v.read_weights_from_buf_into_forward_only(input_bufreader, &mut rg.blocks_boxes[i])?;
        }

        Ok(rg)
    }

    // Create immutable regressor from current regressor
    fn immutable_regressor(&mut self, mi: &model_instance::ModelInstance) -> Result<Box<dyn RegressorTrait>, Box<dyn Error>> {
        // Only to be used by unit tests 
        let mut rg = Box::new(Regressor::new_without_weights::<optimizer::OptimizerSGD>(&mi));

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
        let mi = model_instance::ModelInstance::new_empty().unwrap();        
        let mut re = Regressor::new::<optimizer::OptimizerAdagradLUT>(&mi);
        // Empty model: no matter how many features, prediction is 0.5
        assert_eq!(re.learn(&lr_vec(vec![]), false), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), false), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), false), 0.5);
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
        let mut regressors: Vec<Box<dyn RegressorTrait>> = vec![
            //Box::new(Regressor::<optimizer::OptimizerAdagradLUT>::new(&mi)),
            Box::new(Regressor::new::<optimizer::OptimizerAdagradFlex>(&mi)),
            //Box::new(Regressor::<optimizer::OptimizerSGD>::new(&mi))
            ];
        
        for re in &mut regressors {
            assert_eq!(re.learn(vec_in, true), 0.5);
            assert_eq!(re.learn(vec_in, true), 0.48750263);
            assert_eq!(re.learn(vec_in, true), 0.47533244);
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
        
        let mut re = Regressor::new::<optimizer::OptimizerAdagradLUT>(&mi);
        let vec_in = &lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash: 1, value: 2.0}]);

        assert_eq!(re.learn(vec_in, true), 0.5);
        assert_eq!(re.learn(vec_in, true), 0.38936076);
        assert_eq!(re.learn(vec_in, true), 0.30993468);
    }


    #[test]
    fn test_power_t_half__() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.init_acc_gradient = 0.0;
        
        let mut re = Regressor::new::<optimizer::OptimizerAdagradFlex>(&mi);
        
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true), 0.4750208);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true), 0.45788094);
    }

    #[test]
    fn test_power_t_half_fastmath() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.fastmath = true;
        mi.optimizer = model_instance::Optimizer::Adagrad;
        mi.init_acc_gradient = 0.0;
        
        let mut re = get_regressor(&mi);
        let mut p: f32;
        
        p = re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true);
        assert_eq!(p, 0.5);
        p = re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true);
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
        
        let mut re = Regressor::new::<optimizer::OptimizerAdagradFlex>(&mi);
        // Here we take twice two features and then once just one
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), true), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), true), 0.45016602);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), true), 0.45836908);
    }

    #[test]
    fn test_non_one_weight() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        
        let mut re = Regressor::new::<optimizer::OptimizerAdagradLUT>(&mi);
        
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), true), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), true), 0.45016602);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), true), 0.40611085);
    }

    #[test]
    fn test_example_importance() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        mi.optimizer = model_instance::Optimizer::Adagrad;
        mi.fastmath = true;
        
        let mut re = Regressor::new::<optimizer::OptimizerAdagradLUT>(&mi);
        
        let mut fb_instance = lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]);
        fb_instance.example_importance = 0.5;
        assert_eq!(re.learn(&fb_instance, true), 0.5);
        assert_eq!(re.learn(&fb_instance, true), 0.49375027);
        assert_eq!(re.learn(&fb_instance, true), 0.4875807);
    }

}

