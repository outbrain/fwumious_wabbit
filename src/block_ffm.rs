use std::any::Any;
use std::io;
use merand48::*;
use core::arch::x86_64::*;
use std::error::Error;
use std::mem::{self, MaybeUninit};
use serde_json::{Value, Map, Number};


use crate::optimizer;
use crate::regressor;
use crate::model_instance;
use crate::feature_buffer;
use crate::consts;
use crate::block_helpers;
use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use block_helpers::{Weight, WeightAndOptimizerData, slearn};

use crate::block_helpers::f32_to_json;


const FFM_STACK_BUF_LEN:usize= 16384;





 
use std::ops::{Deref, DerefMut};
use std::sync::Arc;


pub struct BlockFFM<L:OptimizerTrait> {
    pub optimizer_ffm: L,
    pub local_data_ffm_indices: Vec<u32>,
    pub local_data_ffm_values: Vec<f32>,
    pub ffm_k: u32,
    pub ffm_one_over_k_root: f32,
    pub ffm_weights_len: u32, 
    pub weights: Vec<WeightAndOptimizerData<L>>,
}


macro_rules! specialize_1f32 {
    ( $input_expr:expr, 
      $output_const:ident,
      $code_block:block  ) => {
          if $input_expr == 1.0 {	
              const $output_const:f32 = 1.0; 
              $code_block
          } else {
              let $output_const:f32 = $input_expr; 
              $code_block
          }
      };
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



impl <L:OptimizerTrait + 'static> BlockTrait for BlockFFM<L>

 {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn new_without_weights(mi: &model_instance::ModelInstance) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
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
        Ok(Box::new(reg_ffm))
    }

    fn new_forward_only_without_weights(&self) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        let forwards_only = BlockFFM::<optimizer::OptimizerSGD> {
            weights: Vec::new(),
            ffm_weights_len: self.ffm_weights_len, 
            local_data_ffm_indices: Vec::new(),
            local_data_ffm_values: Vec::new(),
            ffm_k: self.ffm_k, 
            ffm_one_over_k_root: self.ffm_one_over_k_root, 
            optimizer_ffm: optimizer::OptimizerSGD::new(),
        };
        
        Ok(Box::new(forwards_only))
    }



    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights =vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.optimizer_ffm.initial_data()}; self.ffm_weights_len as usize];
        if mi.ffm_k > 0 {       
            if mi.ffm_init_width == 0.0 {
                // Initialization that has showed to work ok for us, like in ffm.pdf, but centered around zero and further divided by 50
                self.ffm_one_over_k_root = 1.0 / (self.ffm_k as f32).sqrt() / 50.0;
                for i in 0..self.ffm_weights_len {
                    self.weights[i as usize].weight = (1.0 * merand48((self.ffm_weights_len as usize+ i as usize) as u64)-0.5) * self.ffm_one_over_k_root;
                    self.weights[i as usize].optimizer_data = self.optimizer_ffm.initial_data();
                }
            } else {
                let zero_half_band_width = mi.ffm_init_width * mi.ffm_init_zero_band * 0.5;
                let band_width = mi.ffm_init_width * (1.0 - mi.ffm_init_zero_band);
                for i in 0..self.ffm_weights_len {
                    let mut w = merand48(i as u64) * band_width - band_width * 0.5;
                    if w > 0.0 { 
                        w += zero_half_band_width ;
                    } else {
                        w -= zero_half_band_width;
                    }
                    w += mi.ffm_init_center;
                    self.weights[i as usize].weight = w; 
                    self.weights[i as usize].optimizer_data = self.optimizer_ffm.initial_data();
                }

            }
        }
    }


    #[inline(always)]
    fn forward_backward(&mut self, 
                        further_blocks: &mut [Box<dyn BlockTrait>], 
                        wsum_input: f32, 
                        fb: &feature_buffer::FeatureBuffer, 
                        update:bool) -> (f32, f32) {
        let mut wsum = wsum_input;
        let local_data_ffm_len = fb.ffm_buffer.len() * (self.ffm_k * fb.ffm_fields_count) as usize;

        unsafe {
            macro_rules! core_macro {
                (
                $local_data_ffm_indices:ident,
                $local_data_ffm_values:ident
                ) => {
                 
                    let mut local_data_ffm_indices = &mut $local_data_ffm_indices;
                    let mut local_data_ffm_values = &mut $local_data_ffm_values;
                        
                    let ffm_weights = &mut self.weights;
                    let fc = (fb.ffm_fields_count  * self.ffm_k) as usize;
                    let mut ifc:usize = 0;
                    for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
                        let base_weight_index = left_hash.hash;
                        for j in 0..fc as usize {
                            let addr = base_weight_index + j as u32;
                            *local_data_ffm_indices.get_unchecked_mut(ifc + j) = addr;
                            *local_data_ffm_values.get_unchecked_mut(ifc + j) = 0.0;
                            _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(addr as usize).weight), _MM_HINT_T0);  // No benefit for now
                       }
                       ifc += fc;
                    }

                    specialize_k!(self.ffm_k, FFMK, wsumbuf, {
                        let mut ifc:usize = 0;
                        for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
                            let mut right_local_index = left_hash.contra_field_index as usize + ifc;
                            for right_hash in fb.ffm_buffer.get_unchecked(i+1 ..).iter() {
                                right_local_index += fc;       
                                 
                                // Regular FFM implementation would prevent intra-field interactions
                                // But for the use case we tested this is both faster and it decreases logloss
                                /*if left_hash.contra_field_index == right_hash.contra_field_index {
                                    continue	// not combining within a field
                                }*/
                                
                                // FYI this is effectively what we calculate:
            //                    let left_local_index =  i*fc + right_hash.contra_field_index as usize;
            //                    let right_local_index = (i+1+j) * fc + left_hash.contra_field_index as usize;
                                let left_local_index = ifc + right_hash.contra_field_index as usize;
                                let joint_value = left_hash.value * right_hash.value;
                                let lindex = *local_data_ffm_indices.get_unchecked(left_local_index) as usize;
                                let rindex = *local_data_ffm_indices.get_unchecked(right_local_index) as usize;
                                specialize_1f32!(joint_value, JOINT_VALUE, {
                                    for k in 0..FFMK as usize {
                                        let llik = (left_local_index as usize + k) as usize;
                                        let rlik = (right_local_index as usize + k) as usize;
                                        let left_hash_weight  = ffm_weights.get_unchecked((lindex+k) as usize).weight;
                                        let right_hash_weight = ffm_weights.get_unchecked((rindex+k) as usize).weight;
                                        
                                        let right_side = right_hash_weight * JOINT_VALUE;
                                        *local_data_ffm_values.get_unchecked_mut(llik) += right_side; // first derivate
                                        *local_data_ffm_values.get_unchecked_mut(rlik) += left_hash_weight  * JOINT_VALUE; // first derivate
                                        // We do this, so in theory Rust/LLVM could vectorize whole loop
                                        // Original: wsum += left_hash_weight * right_side;
                                        *wsumbuf.get_unchecked_mut(k) += left_hash_weight * right_side;
                                    }
                                });
                                
                            }
                            ifc += fc;
                            
                        }
                        for k in 0..FFMK as usize {
                            wsum += wsumbuf[k];
                        }
                    });
                    
                    /*if fb.audit_mode {
                        self.audit_forward(wsum_input, wsum, fb);
                    }*/
                    
                    let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
                    let (prediction_probability, general_gradient) = next_regressor[0].forward_backward(further_blocks, wsum, fb, update);
                    
                    if update {
                       for i in 0..local_data_ffm_len {
                            let feature_value = *local_data_ffm_values.get_unchecked(i);
                            let feature_index = *local_data_ffm_indices.get_unchecked(i) as usize;
                            let gradient = general_gradient * feature_value;
                            let update = self.optimizer_ffm.calculate_update(gradient, &mut ffm_weights.get_unchecked_mut(feature_index).optimizer_data);
                            ffm_weights.get_unchecked_mut(feature_index).weight += update;
                        }
                    }
                    // The only exit point
                    return (prediction_probability, general_gradient)
                }; 
            }; // End of macro
            

            if local_data_ffm_len < FFM_STACK_BUF_LEN {
                // Fast-path - using on-stack data structures
                let mut local_data_ffm_indices: [u32; FFM_STACK_BUF_LEN as usize] = MaybeUninit::uninit().assume_init();
                let mut local_data_ffm_values: [f32; FFM_STACK_BUF_LEN as usize] = MaybeUninit::uninit().assume_init();//[0.0; FFM_STACK_BUF_LEN as usize];
                core_macro!(local_data_ffm_indices, local_data_ffm_values);

            } else {
                // Slow-path - using heap data structures
                if local_data_ffm_len > self.local_data_ffm_indices.len() {
                    self.local_data_ffm_indices.reserve(local_data_ffm_len - self.local_data_ffm_indices.len() + 1024);
                }
                if local_data_ffm_len > self.local_data_ffm_values.len() {
                    self.local_data_ffm_values.reserve(local_data_ffm_len - self.local_data_ffm_values.len() + 1024);
                }
                let mut local_data_ffm_indices = &mut self.local_data_ffm_indices;
                let mut local_data_ffm_values = &mut self.local_data_ffm_values;
            
                core_macro!(local_data_ffm_indices, local_data_ffm_values);
            }
             
        } // unsafe end
    }
    
    fn forward(&self, 
                further_blocks: &[Box<dyn BlockTrait>], 
                wsum_input: f32, 
                fb: &feature_buffer::FeatureBuffer) -> f32 {
        let mut wsum:f32 = wsum_input;
        unsafe {
            let ffm_weights = &self.weights;
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
                            //wsum += left_hash_weight * right_side;
                            // We do this, so in theory Rust/LLVM could vectorize whole loop
                            // Unfortunately it does not happen in practice, but we will get there
                            // Original: wsum += left_hash_weight * right_side;
                            *wsumbuf.get_unchecked_mut(k as usize) += left_hash_weight * right_hash_weight * joint_value;                        
                        }
                    }
                
                }
                for k in 0..FFMK as usize {
                    wsum += wsumbuf[k];
                }
            });
        }
        if fb.audit_mode {
            self.audit_forward(wsum_input, wsum, fb);
        }
        let (next_regressor, further_blocks) = further_blocks.split_at(1);
        let prediction_probability = next_regressor[0].forward(further_blocks, wsum, fb);
        prediction_probability         
                 
    }

    fn audit_forward(&self, 
        wsum_input: f32, 
        wsum_output: f32, 
        fb: &feature_buffer::FeatureBuffer) {
    }

    fn get_serialized_len(&self) -> usize {
        return self.ffm_weights_len as usize;
    }

    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        block_helpers::read_weights_from_buf(&mut self.weights, input_bufreader)
    }

    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        block_helpers::write_weights_to_buf(&self.weights, output_bufwriter)
    }

    fn read_weights_from_buf_into_forward_only(&self, input_bufreader: &mut dyn io::Read, forward: &mut Box<dyn BlockTrait>) -> Result<(), Box<dyn Error>> {
        let mut forward = forward.as_any().downcast_mut::<BlockFFM<optimizer::OptimizerSGD>>().unwrap();
        block_helpers::read_weights_only_from_buf2::<L>(self.ffm_weights_len as usize, &mut forward.weights, input_bufreader)
    }

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>> {
        self.weights[index].weight = w[0];
        self.weights[index].optimizer_data = self.optimizer_ffm.initial_data();
        Ok(())
    }
}




mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::block_loss_functions::BlockSigmoid;
    use crate::feature_buffer;
    use crate::feature_buffer::HashAndValueAndSeq;


    fn ffm_vec(v:Vec<feature_buffer::HashAndValueAndSeq>, ffm_fields_count:u32) -> feature_buffer::FeatureBuffer {
        let mut fb = feature_buffer::FeatureBuffer::new();
        fb.ffm_buffer = v;
        fb.ffm_fields_count = ffm_fields_count;
        fb
    }

    fn ffm_init<T:OptimizerTrait + 'static>(block_ffm: &mut Box<dyn BlockTrait>) -> () {
        let mut block_ffm = block_ffm.as_any().downcast_mut::<BlockFFM<T>>().unwrap();
        
        for i in 0..block_ffm.weights.len() {
            block_ffm.weights[i].weight = 1.0;
            block_ffm.weights[i].optimizer_data = block_ffm.optimizer_ffm.initial_data();
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
        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();
        
        // Nothing can be learned from a single field in FFMs
        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        let ffm_buf = ffm_vec(vec![HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0}], 1);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.5);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        let mut re = BlockFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: 1}
                                  ], 2);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.98201376); 
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.96277946);

        // Two fields, use values
        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut re);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 2.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: 1}
                                  ], 2);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.9999999);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.99685884);
    }


}



