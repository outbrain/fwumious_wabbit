#![allow(unused_macros)]
//use std::mem::{self};
use std::mem::{self, MaybeUninit};
use std::slice;
use std::sync::Arc;
use core::arch::x86_64::*;
use merand48::*;
use std::io;
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


#[derive(Clone, Debug)]
#[repr(C)]
pub struct Weight {
    pub weight: f32, 
}

#[derive(Clone, Debug, Copy)]
#[repr(C)]
pub struct WeightAndOptimizerData<L:OptimizerTrait> {
    pub weight: f32, 
    pub optimizer_data: L::PerWeightStore,
}

pub trait BlockTrait {
    fn forward_backwards(&mut self, 
                         further_regressors: &mut [&mut dyn BlockTrait], 
                         wsum: f32, 
                         example_num: u32, 
                         fb: &feature_buffer::FeatureBuffer,
                         update:bool) -> (f32, f32);

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance);
    fn get_weights_len(&self) -> usize;
    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>>;
    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>>;
}



pub struct Regressor<'a, L:OptimizerTrait> {
    pub reg_ffm: Box<BlockFFM<L>>,
    pub reg_lr: Box<BlockLR<L>>,
    pub reg_sig: Box<BlockSigmoid>, 
    pub vv: Vec<&'a mut dyn BlockTrait>,
}

#[derive(Clone)]
pub struct ImmutableRegressor {
    pub weights: Arc<Vec<Weight>>,
    ffm_weights_offset: u32, 
    ffm_k: u32,
}




pub trait RegressorTrait {
    fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, update: bool, example_num: u32) -> f32;
    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>>;
    fn overwrite_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>>; 
    fn get_name(&self) -> String;
    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance);
    fn immutable_regressor_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<ImmutableRegressor, Box<dyn Error>>; 
    fn immutable_regressor(&mut self) -> Result<ImmutableRegressor, Box<dyn Error>>;
}


pub fn get_regressor_without_weights(mi: &model_instance::ModelInstance) -> Box<dyn RegressorTrait> {
    if mi.optimizer == model_instance::Optimizer::Adagrad {
        if mi.fastmath {
            Box::new(Regressor::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi))
        } else {
            Box::new(Regressor::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi))
        }
    } else {
        Box::new(Regressor::<optimizer::OptimizerSGD>::new_without_weights(&mi))
    }    
}

pub fn get_regressor(mi: &model_instance::ModelInstance) -> Box<dyn RegressorTrait> {
    let mut re = get_regressor_without_weights(mi);
    re.allocate_and_init_weights(mi);
    re
}


impl <'a, L:OptimizerTrait + 'static>Regressor<'a, L> 
where <L as optimizer::OptimizerTrait>::PerWeightStore: std::clone::Clone,
L: std::clone::Clone
{
    pub fn new_without_weights(mi: &model_instance::ModelInstance) -> Regressor<'a, L> {
        let mut reg_lr = BlockLR::<L> {
            weights: Vec::new(),
            weights_len: 0, 
            optimizer_lr: L::new(),
        };
        reg_lr.optimizer_lr.init(mi.learning_rate, mi.power_t, mi.init_acc_gradient);
        reg_lr.weights_len = 1 << mi.bit_precision;


        let mut reg_ffm = BlockFFM::<L> {
            weights: Vec::new(),
            ffm_weights_len: 0, 
            local_data_ffm_indices: Vec::with_capacity(1024),
            local_data_ffm_values: Vec::with_capacity(1024),
            ffm_k: mi.ffm_k, 
            ffm_one_over_k_root: 0.0, 
            optimizer_ffm: L::new(),
        };

        if mi.ffm_k > 0 {
            reg_ffm.optimizer_ffm.init(mi.ffm_learning_rate, mi.ffm_power_t, mi.ffm_init_acc_gradient);
            // At the end we add "spillover buffer", so we can do modulo only on the base address and add offset
            reg_ffm.ffm_weights_len = (1 << mi.ffm_bit_precision) + (mi.ffm_fields.len() as u32 * reg_ffm.ffm_k);
        }
        
        let reg_sigmoid = BlockSigmoid {};

        let mut rg = Regressor::<L>{
            reg_lr: Box::new(reg_lr),
            reg_ffm: Box::new(reg_ffm),
            reg_sig: Box::new(reg_sigmoid),
            vv: Vec::new(),
        };

        unsafe {
            // A bit more elaborate than necessary. Let's really make it clear what's happening
            let r1: &mut BlockLR<L> = rg.reg_lr.as_mut();
            let r2: &mut BlockLR<L> = mem::transmute(&mut *r1);
            rg.vv.push(r2 as &mut dyn BlockTrait);

            if mi.ffm_k > 0 {
                let r1: &mut BlockFFM<L> = rg.reg_ffm.as_mut();
                let r2: &mut BlockFFM<L> = mem::transmute(&mut *r1);
                rg.vv.push(r2 as &mut dyn BlockTrait);
            }
            
            
            let r1: &mut BlockSigmoid = rg.reg_sig.as_mut();
            let r2: &mut BlockSigmoid = mem::transmute(&mut *r1);
            rg.vv.push(r2 as &mut dyn BlockTrait);
        }

        rg
    }
    
    pub fn allocate_and_init_weights_(&mut self, mi: &model_instance::ModelInstance) {
        for rr in &mut self.vv {
            rr.allocate_and_init_weights(mi);
        }
    }
    

    pub fn new(mi: &model_instance::ModelInstance) -> Regressor<'a, L> {
        let mut rg = Regressor::<L>::new_without_weights(mi);
        rg.allocate_and_init_weights(mi);
        rg
    }
}


    
impl <L:OptimizerTrait + 'static> RegressorTrait for Regressor<'_, L> 
where <L as optimizer::OptimizerTrait>::PerWeightStore: std::clone::Clone,
L: std::clone::Clone
{

    fn get_name(&self) -> String {
        format!("Regressor with optimizer {:?}", L::get_name())
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.allocate_and_init_weights_(mi);
    }

    fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, update: bool, example_num: u32) -> f32 {
        let update:bool = update && (fb.example_importance != 0.0);

        let (current, further_regressors) = &mut self.vv.split_at_mut(1);
        let (prediction_probability, general_gradient) = current[0].forward_backwards(further_regressors, 0.0, example_num, fb, update);
    
        return prediction_probability
    }
    
    // Yeah, this is weird. I just didn't want to break the format compatibility at this point
    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        // It's OK! I am a limo driver!
        output_bufwriter.write_u64::<LittleEndian>((self.reg_lr.weights.len() + self.reg_ffm.weights.len()) as u64)?;

        for v in &self.vv {
            v.write_weights_to_buf(output_bufwriter)?;
        
        }
        Ok(())
    }
    

    fn overwrite_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        // This is a bit weird format - we'll break compatibility in next release
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let expected_length = (self.reg_lr.weights.len() + self.reg_ffm.weights.len()) as u64;
        if len != expected_length {
            return Err(format!("Lenghts of weights array in regressor file differ: got {}, expected {}", len, expected_length))?;
        }
        for v in &mut self.vv {
            v.read_weights_from_buf(input_bufreader)?;
        }

        Ok(())
    }

    
    // Creates immutable regressor from current setup and weights from buffer
    fn immutable_regressor_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<ImmutableRegressor, Box<dyn Error>> {
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let expected_length = (self.reg_lr.get_weights_len() + self.reg_ffm.get_weights_len()) as u64;
        if len != expected_length {
            return Err(format!("Lenghts of weights array in regressor file differ: got {}, expected {}", len, expected_length))?;
        }
        const BUF_LEN:usize = 1024 * 1024;
        let mut in_weights = vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.reg_lr.optimizer_lr.initial_data()}; BUF_LEN as usize];
        let mut out_weights = Vec::<Weight>::new();
        
        let mut remaining_weights = expected_length as usize;
        unsafe {
            while remaining_weights > 0 {
                let chunk_size = min(remaining_weights, BUF_LEN);
                in_weights.set_len(chunk_size);
                let mut in_weights_view:&mut [u8] = slice::from_raw_parts_mut(in_weights.as_mut_ptr() as *mut u8, 
                                             chunk_size *mem::size_of::<WeightAndOptimizerData<L>>());
                input_bufreader.read_exact(&mut in_weights_view)?;
                for w in &in_weights {
                    out_weights.push(Weight{weight:w.weight});
                    
                }
                remaining_weights -= chunk_size;
            }
        }

        let fr = ImmutableRegressor {
                        weights: Arc::new(out_weights), 
                        ffm_weights_offset: self.reg_lr.get_weights_len() as u32,
                        ffm_k: self.reg_ffm.ffm_k,
        };
        Ok(fr)
    }

    // Create immutable regressor from current regressor
    fn immutable_regressor(&mut self) -> Result<ImmutableRegressor, Box<dyn Error>> {
        let mut weights = Vec::<Weight>::new();
        for w in &self.reg_lr.weights {
            weights.push(Weight{weight:w.weight});
        }
        for w in &self.reg_ffm.weights {
            weights.push(Weight{weight:w.weight});
        }
        let fr = ImmutableRegressor {
                        weights: Arc::new(weights), 
                        ffm_weights_offset: self.reg_lr.get_weights_len() as u32,
                        ffm_k: self.reg_ffm.ffm_k,
        };
        Ok(fr)
    }



}

macro_rules! specialize_k {
    ( $input_expr: expr, 
      $output_const: ident,
      $wsumbuf: ident,
      $code_block: block  ) => {
         match $input_expr {
                2 => {const $output_const:u32 = 2;   let mut $wsumbuf: [f32;$output_const as usize] = [0.0;$output_const as usize]; $code_block},
                4 => {const $output_const:u32 = 4;   let mut $wsumbuf: [f32;$output_const as usize] = [0.0;$output_const as usize]; $code_block},
                8 => {const $output_const:u32 = 8;   let mut $wsumbuf: [f32;$output_const as usize] = [0.0;$output_const as usize]; $code_block},
                val => {let $output_const:u32 = val; let mut $wsumbuf: [f32;consts::FFM_MAX_K] = [0.0;consts::FFM_MAX_K];      $code_block},
            }
    };
}


impl ImmutableRegressor {

    pub fn predict(&self, fb: &feature_buffer::FeatureBuffer, example_num: u32) -> f32 {
        let fbuf = &fb.lr_buffer;
        let mut wsum:f32 = 0.0;
        unsafe {
        for val in fbuf {
            let hash = val.hash as usize;
            let feature_value:f32 = val.value;
            wsum += self.weights.get_unchecked(hash).weight * feature_value;    
        }


        if self.ffm_k > 0 {
            let ffm_weights = &self.weights[self.ffm_weights_offset as usize..];
            specialize_k!(self.ffm_k, FFMK, wsumbuf, {                        
                for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
                    for right_hash in fb.ffm_buffer.get_unchecked(i+1 ..).iter() {
                        //if left_hash.contra_field_index == right_hash.contra_field_index {
                        //    continue	// not combining within a field
                        //}
                        let joint_value = left_hash.value * right_hash.value;
                        let lindex = (left_hash.hash + right_hash.contra_field_index) as u32;
                        let rindex = (right_hash.hash + left_hash.contra_field_index) as u32;
                        for k in 0..FFMK {
                            let left_hash_weight  = ffm_weights.get_unchecked((lindex+k) as usize).weight;
                            let right_hash_weight = ffm_weights.get_unchecked((rindex+k) as usize).weight;
                            let right_side = right_hash_weight * joint_value;
                            //wsum += left_hash_weight * right_side;
                            // We do this, so in theory Rust/LLVM could vectorize whole loop
                            // Unfortunately it does not happen in practice, but we will get there
                            // Original: wsum += left_hash_weight * right_side;
                            *wsumbuf.get_unchecked_mut(k as usize) += left_hash_weight * right_side;                        
                        }
                    }
                
                }
                for k in 0..FFMK as usize {
                    wsum += wsumbuf[k];
                }

            });

            
        }

        let prediction = -wsum;
        let mut prediction_finalized = prediction;
        if prediction_finalized.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", example_num);
            prediction_finalized = 0.0;
        } else if prediction_finalized < -50.0 {
            prediction_finalized = -50.0;
        } else if prediction_finalized > 50.0 {
            prediction_finalized = 50.0;
        }
        let prediction_probability:f32 = (1.0+(prediction_finalized).exp()).recip();
        prediction_probability
        }
    }
} 

impl RegressorTrait for ImmutableRegressor {
    fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, update: bool, example_num: u32) -> f32{
        if update == true {
            panic!("You cannot call immutable regressor with update=true");
        }
        return self.predict(fb, example_num)
    }

    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        panic!("Immutable regressor cannot be saved to a file, since optimizer weights get lost");
    }

    fn overwrite_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        panic!("Not implemented");
    }
    fn get_name(&self) -> String {
        format!("ImmutableRegressor")
    }
    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        panic!("Not implemented!");
    }
    fn immutable_regressor_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<ImmutableRegressor, Box<dyn Error>> {
        panic!("Not implemented!");
    }
    fn immutable_regressor(&mut self) -> Result<ImmutableRegressor, Box<dyn Error>> {
        panic!("Not implemented!");
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
                    lr_buffer: v,
                    ffm_buffer: Vec::new(),
                    ffm_fields_count: 0,
        }
    }


    #[test]
    fn test_learning_turned_off() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();        
        let mut re = Regressor::<optimizer::OptimizerAdagradLUT>::new(&mi);
        // Empty model: no matter how many features, prediction is 0.5
        assert_eq!(re.learn(&lr_vec(vec![]), false, 0), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), false, 0), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), false, 0), 0.5);
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
            Box::new(Regressor::<optimizer::OptimizerAdagradFlex>::new(&mi)),
            //Box::new(Regressor::<optimizer::OptimizerSGD>::new(&mi))
            ];
        
        for re in &mut regressors {
            assert_eq!(re.learn(vec_in, true, 0), 0.5);
            assert_eq!(re.learn(vec_in, true, 0), 0.48750263);
            assert_eq!(re.learn(vec_in, true, 0), 0.47533244);
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
        
        let mut re = Regressor::<optimizer::OptimizerAdagradLUT>::new(&mi);
        let vec_in = &lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash: 1, value: 2.0}]);

        assert_eq!(re.learn(vec_in, true, 0), 0.5);
        assert_eq!(re.learn(vec_in, true, 0), 0.38936076);
        assert_eq!(re.learn(vec_in, true, 0), 0.30993468);
    }


    #[test]
    fn test_power_t_half__() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.init_acc_gradient = 0.0;
        
        let mut re = Regressor::<optimizer::OptimizerAdagradFlex>::new(&mi);
        
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true, 0), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true, 0), 0.4750208);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true, 0), 0.45788094);
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
        
        p = re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true, 0);
        assert_eq!(p, 0.5);
        p = re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true, 0);
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
        
        let mut re = Regressor::<optimizer::OptimizerAdagradFlex>::new(&mi);
        // Here we take twice two features and then once just one
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), true, 0), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), true, 0), 0.45016602);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), true, 0), 0.45836908);
    }

    #[test]
    fn test_non_one_weight() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        
        let mut re = Regressor::<optimizer::OptimizerAdagradLUT>::new(&mi);
        
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), true, 0), 0.5);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), true, 0), 0.45016602);
        assert_eq!(re.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), true, 0), 0.40611085);
    }

/* FFM TESTS */
    fn ffm_vec(v:Vec<feature_buffer::HashAndValueAndSeq>, ffm_fields_count: u32) -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
                    label: 0.0,
                    example_importance: 1.0,
                    lr_buffer: Vec::new(),
                    ffm_buffer: v,
                    ffm_fields_count: ffm_fields_count,
        }
    }

    fn ffm_init<T:OptimizerTrait>(rg: &mut Regressor<T>) -> () {
        for i in 0..rg.reg_ffm.weights.len() {
            rg.reg_ffm.weights[i].weight = 1.0;
//            rg.weights[i].acc_grad = 1.0;
            rg.reg_ffm.weights[i].optimizer_data = rg.reg_ffm.optimizer_ffm.initial_data();
        }
    }


    #[test]
    fn test_ffm() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 4;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]]; // This isn't really used
        
        let mut p: f32;
        
        // Nothing can be learned from a single field
        let mut re = Regressor::<optimizer::OptimizerAdagradLUT>::new(&mi);
        let ffm_buf = ffm_vec(vec![HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0}], 1);
        p = re.learn(&ffm_buf, true, 0);
        assert_eq!(p, 0.5);
        p = re.learn(&ffm_buf, true, 0);
        assert_eq!(p, 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        let mut re = Regressor::<optimizer::OptimizerAdagradFlex>::new(&mi);
        ffm_init(&mut re);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: 1}
                                  ], 2);
        assert_eq!(re.learn(&ffm_buf, true, 0), 0.98201376); 
        assert_eq!(re.learn(&ffm_buf, true, 0), 0.96277946);

        // Two fields, use values
        let mut re = Regressor::<optimizer::OptimizerAdagradLUT>::new(&mi);
        ffm_init(&mut re);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 2.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: 1}
                                  ], 2);
        assert_eq!(re.learn(&ffm_buf, true, 0), 0.9999999);
        assert_eq!(re.learn(&ffm_buf, true, 0), 0.99685884);


    }


    #[test]
    fn test_example_importance() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        mi.optimizer = model_instance::Optimizer::Adagrad;
        mi.fastmath = true;
        
        let mut re = Regressor::<optimizer::OptimizerAdagradLUT>::new(&mi);
        
        let mut fb_instance = lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]);
        fb_instance.example_importance = 0.5;
        assert_eq!(re.learn(&fb_instance, true, 0), 0.5);
        assert_eq!(re.learn(&fb_instance, true, 0), 0.49375027);
        assert_eq!(re.learn(&fb_instance, true, 0), 0.4875807);
    }

}

