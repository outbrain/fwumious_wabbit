use std::any::Any;

use crate::optimizer;
use crate::regressor;
use crate::model_instance;
use crate::feature_buffer;
use std::io;
use core::arch::x86_64::*;
use std::error::Error;



use std::mem::{self, MaybeUninit};
use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use crate::block_helpers;
use block_helpers::{Weight, WeightAndOptimizerData};
use crate::port_buffer;


pub struct BlockLR<L:OptimizerTrait> {
    pub weights: Vec<WeightAndOptimizerData<L>>,
    pub weights_len: u32,
    pub optimizer_lr: L,
}

impl <L:OptimizerTrait + 'static> BlockTrait for BlockLR<L> 
{
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn new_without_weights(mi: &model_instance::ModelInstance) -> Result<Box<dyn BlockTrait>, Box<dyn Error>>  where Self: Sized {
        let mut reg_lr = BlockLR::<L> {
            weights: Vec::new(),
            weights_len: 0, 
            optimizer_lr: L::new(),
        };
        reg_lr.optimizer_lr.init(mi.learning_rate, mi.power_t, mi.init_acc_gradient);
        reg_lr.weights_len = 1 << mi.bit_precision;
        Ok(Box::new(reg_lr))
    }

    fn new_forward_only_without_weights(&self) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        let forwards_only = BlockLR::<optimizer::OptimizerSGD> {
            weights_len: self.weights_len,
            weights: Vec::new(),
            optimizer_lr:optimizer::OptimizerSGD::new(),
        };
        
        Ok(Box::new(forwards_only))
    }



    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights = vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.optimizer_lr.initial_data()}; self.weights_len as usize];
        
    }

    #[inline(always)]
    fn forward_backward(&mut self, 
                            further_regressors: &mut [Box<dyn BlockTrait>], 
                            wsum_input: f32, 
                            fb: &feature_buffer::FeatureBuffer, 
                            pb: &mut port_buffer::PortBuffer,                             
                            update:bool) -> (f32, f32) {
        let mut wsum:f32 = 0.0;
        unsafe {
            for hashvalue in fb.lr_buffer.iter() {
                // Prefetch couple of indexes from the future to prevent pipeline stalls due to memory latencies
//            for (i, hashvalue) in fb.lr_buffer.iter().enumerate() {
                // _mm_prefetch(mem::transmute::<&f32, &i8>(&self.weights.get_unchecked((fb.lr_buffer.get_unchecked(i+8).hash) as usize).weight), _MM_HINT_T0);  // No benefit for now
                let feature_index     = hashvalue.hash;
                let feature_value:f32 = hashvalue.value;
                let feature_weight    = self.weights.get_unchecked(feature_index as usize).weight;
                wsum += feature_weight * feature_value;
            }

            let (next_regressor, further_regressors) = further_regressors.split_at_mut(1);
            let (prediction_probability, general_gradient) = next_regressor[0].forward_backward(further_regressors, wsum_input + wsum, fb, pb, update);

            if update {
                for hashvalue in fb.lr_buffer.iter() {
                    let feature_index     = hashvalue.hash as usize;
                    let feature_value:f32 = hashvalue.value;                        
                    let gradient = general_gradient * feature_value;
                    let update = self.optimizer_lr.calculate_update(gradient, &mut self.weights.get_unchecked_mut(feature_index).optimizer_data);
                    self.weights.get_unchecked_mut(feature_index).weight += update;
                }
            }
            (prediction_probability, general_gradient)
        } // end of unsafe
    }
    
    
    fn forward(&self, 
             further_blocks: &[Box<dyn BlockTrait>], 
             wsum_input: f32, 
             fb: &feature_buffer::FeatureBuffer) -> f32 {
        let fbuf = &fb.lr_buffer;
        let mut wsum:f32 = 0.0;
        unsafe {
            for val in fbuf {
                let hash = val.hash as usize;
                let feature_value:f32 = val.value;
                wsum += self.weights.get_unchecked(hash).weight * feature_value;
            }
        }
        let (next_regressor, further_blocks) = further_blocks.split_at(1);
        let prediction_probability = next_regressor[0].forward(further_blocks, wsum + wsum_input, fb);
        prediction_probability         
    }
    
    
    fn get_serialized_len(&self) -> usize {
        return self.weights_len as usize;
    }

    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        block_helpers::read_weights_from_buf(&mut self.weights, input_bufreader)
    }

    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        block_helpers::write_weights_to_buf(&self.weights, output_bufwriter)
    }

    fn read_weights_from_buf_into_forward_only(&self, input_bufreader: &mut dyn io::Read, forward: &mut Box<dyn BlockTrait>) -> Result<(), Box<dyn Error>> {
        let mut forward = forward.as_any().downcast_mut::<BlockLR<optimizer::OptimizerSGD>>().unwrap();
        block_helpers::read_weights_only_from_buf2::<L>(self.weights_len as usize, &mut forward.weights, input_bufreader)
    }
    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>> {
        self.weights[index].weight = w[0];
        self.weights[index].optimizer_data = self.optimizer_lr.initial_data();
        Ok(())
    }


}

