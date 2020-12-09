

use crate::optimizer;
use crate::regressor;
use crate::model_instance;
use crate::feature_buffer;
use crate::consts;
use crate::block_helpers;
use std::io;
use merand48::*;
use core::arch::x86_64::*;
use std::error::Error;



use std::mem::{self, MaybeUninit};
use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use regressor::{Weight, WeightAndOptimizerData};


const FFM_STACK_BUF_LEN:usize= 16384;


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



impl <L:OptimizerTrait> BlockTrait for BlockFFM<L>
where <L as optimizer::OptimizerTrait>::PerWeightStore: std::clone::Clone,
L: std::clone::Clone

 {
    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights = vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.optimizer_ffm.initial_data()}; self.ffm_weights_len as usize];
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
    fn forward_backwards(&mut self, 
                        further_regressors: &mut [&mut dyn BlockTrait], 
                        wsum: f32, 
                        example_num: u32, 
                        fb: &feature_buffer::FeatureBuffer, 
                        update:bool) -> (f32, f32) {
        let mut wsum = wsum;
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
                    
                    let (next_regressor, further_regressors) = further_regressors.split_at_mut(1);
                    let (prediction_probability, general_gradient) = next_regressor[0].forward_backwards(further_regressors, wsum, example_num, fb, update);
                    
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
                    
                    let (next_regressor, further_regressors) = further_regressors.split_at_mut(1);
                    let (prediction_probability, general_gradient) = next_regressor[0].forward_backwards(further_regressors, wsum, example_num, fb, update);
                    
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
    
    fn get_weights_len(&self) -> usize {
        return self.ffm_weights_len as usize;
    }

    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        block_helpers::read_weights_from_buf(&mut self.weights, input_bufreader)
    }

    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        block_helpers::write_weights_to_buf(&self.weights, output_bufwriter)
    }

    fn read_immutable_weights_from_buf(&self, out_weights: &mut Vec<Weight>, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        block_helpers::read_immutable_weights_from_buf::<L>(self.get_weights_len(), out_weights, input_bufreader)
    }



}

