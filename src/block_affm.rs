use std::any::Any;
use std::io;
use merand48::*;
use core::arch::x86_64::*;
use std::error::Error;
use std::mem::{self, MaybeUninit};


use crate::optimizer;
use crate::regressor;
use crate::model_instance;
use crate::feature_buffer;
use crate::consts;
use crate::block_helpers;
use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use block_helpers::{Weight, WeightAndOptimizerData};


const FFM_STACK_BUF_LEN:usize= 32768;
const FFM_CONTRA_BUF_LEN:usize = 16384;


const SQRT_OF_ONE_HALF:f32 = 0.70710678118;

use std::ops::{Deref, DerefMut};
use std::sync::Arc;


pub struct BlockAFFM<L:OptimizerTrait> {
    pub optimizer_ffm: L,
    pub local_data_ffm_values: Vec<f32>,
    pub ffm_k: u32,
    pub ffm_weights_len: u32,
    pub field_embedding_len: u32,
    pub weights: Vec<WeightAndOptimizerData<L>>,
    pub field_interaction_weights: Vec<Weight>,
    pub field_interaction_weights_len: u32,

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


impl <L:OptimizerTrait + 'static> BlockTrait for BlockAFFM<L>

 {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn new_without_weights(mi: &model_instance::ModelInstance) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        let mut reg_ffm = BlockAFFM::<L> {
            weights: Vec::new(),
            ffm_weights_len: 0,
            local_data_ffm_values: Vec::with_capacity(1024),
            ffm_k: mi.ffm_k,
            field_embedding_len: mi.ffm_k * mi.ffm_fields.len() as u32,
            optimizer_ffm: L::new(),
            field_interaction_weights: Vec::new(),
            field_interaction_weights_len: 0,
        };

        if mi.ffm_k > 0 {
            reg_ffm.optimizer_ffm.init(mi.ffm_learning_rate, mi.ffm_power_t, mi.ffm_init_acc_gradient);
            // At the end we add "spillover buffer", so we can do modulo only on the base address and add offset
            reg_ffm.ffm_weights_len = (1 << mi.ffm_bit_precision) + (mi.ffm_fields.len() as u32 * reg_ffm.ffm_k);
            reg_ffm.field_interaction_weights_len = (mi.ffm_fields.len() * mi.ffm_fields.len()) as u32;
        }

        // Verify that forward pass will have enough stack for temporary buffer
        if reg_ffm.ffm_k as usize * mi.ffm_fields.len() * mi.ffm_fields.len() > FFM_CONTRA_BUF_LEN {
            return Err(format!("FFM_CONTRA_BUF_LEN is {}. It needs to be at least ffm_k * number_of_fields^2. number_of_fields: {}, ffm_k: {}, please recompile with larger constant",
                        FFM_CONTRA_BUF_LEN, mi.ffm_fields.len(), reg_ffm.ffm_k))?;
        }

        Ok(Box::new(reg_ffm))
    }

    fn new_forward_only_without_weights(&self) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        let forwards_only = BlockAFFM::<optimizer::OptimizerSGD> {
            weights: Vec::new(),
            ffm_weights_len: self.ffm_weights_len,
            local_data_ffm_values: Vec::new(),
            ffm_k: self.ffm_k,
            field_embedding_len: self.field_embedding_len,
            optimizer_ffm: optimizer::OptimizerSGD::new(),
            field_interaction_weights: Vec::new(),
            field_interaction_weights_len: self.field_interaction_weights_len,
        };

        Ok(Box::new(forwards_only))
    }



    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights =vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.optimizer_ffm.initial_data()}; self.ffm_weights_len as usize];
        self.field_interaction_weights =vec![Weight{weight:1.0}; self.field_interaction_weights_len as usize];

        if mi.ffm_k > 0 {
            if mi.ffm_init_width == 0.0 {
                // Initialization that has showed to work ok for us, like in ffm.pdf, but centered around zero and further divided by 50
                let ffm_one_over_k_root = 1.0 / (self.ffm_k as f32).sqrt() / 50.0;
                for i in 0..self.ffm_weights_len {
                    self.weights[i as usize].weight = (1.0 * merand48((self.ffm_weights_len as usize+ i as usize) as u64)-0.5) * ffm_one_over_k_root;
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

            // on command line, for each place in the matrix, we can define its own weight
            for (field_id_1, field_id_2, interaction_weight) in mi.ffm_interactions.iter() {
                // matrix has to be symetrical
                self.field_interaction_weights[*field_id_1 as usize * mi.ffm_fields.len() + *field_id_2 as usize].weight = *interaction_weight;
                self.field_interaction_weights[*field_id_2 as usize * mi.ffm_fields.len() + *field_id_1 as usize].weight = *interaction_weight;
            }

        }
    }


    #[inline(always)]
    fn forward_backward(&mut self,
                        further_blocks: &mut [Box<dyn BlockTrait>],
                        wsum_input: f32,
                        fb: &feature_buffer::FeatureBuffer,
                        update:bool) -> (f32, f32) {
        let mut wsum = 0.0;
        let local_data_ffm_len = fb.ffm_buffer.len() * (self.ffm_k * fb.ffm_fields_count) as usize;

        unsafe {
            macro_rules! core_macro {
                (
                $local_data_ffm_values:ident
                ) => {
                    let mut local_data_ffm_values = $local_data_ffm_values;
                     //   let mut local_data_ffm_values = &mut $local_data_ffm_values;

                    let ffm_weights = &mut self.weights;
                    let fc = (fb.ffm_fields_count  * self.ffm_k) as usize;
                    let mut contra_fields: [f32; FFM_CONTRA_BUF_LEN] = MaybeUninit::uninit().assume_init();
                    let field_embedding_len = self.field_embedding_len;
                    specialize_k!(self.ffm_k, FFMK, wsumbuf, {
                        /* first prepare two things:
                        - transposed contra vectors in contra_fields -
                            - for each vector we sum up all the features within a field
                            - and at the same time transpose it, so we can later directly multiply them with individual feature embeddings
                        - cache of gradients in local_data_ffm_values
                            - we will use these gradients later in backward pass
                        */
                        _mm_prefetch(mem::transmute::<&f32, &i8>(&contra_fields.get_unchecked(fb.ffm_buffer.get_unchecked(0).contra_field_index as usize)), _MM_HINT_T0);
                        let mut ffm_buffer_index = 0;
                        for field_index in 0..fb.ffm_fields_count {
                            let field_index_ffmk = field_index * FFMK;
                            let offset = (field_index_ffmk * fb.ffm_fields_count) as usize;
                            // first we handle fields with no features
                            if ffm_buffer_index >= fb.ffm_buffer.len() ||
                                fb.ffm_buffer.get_unchecked(ffm_buffer_index).contra_field_index > field_index_ffmk {
                                let mut zfc:usize = field_index_ffmk as usize;
                                for z in 0..fb.ffm_fields_count {
                                    for k in 0..FFMK as usize{
                                        *contra_fields.get_unchecked_mut(zfc + k) = 0.0;
                                    }
                                    zfc += fc;
                                }
                                continue;
                            }
                            let mut feature_num = 0;
                            while ffm_buffer_index < fb.ffm_buffer.len() && fb.ffm_buffer.get_unchecked(ffm_buffer_index).contra_field_index == field_index_ffmk {
                                _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(fb.ffm_buffer.get_unchecked(ffm_buffer_index+1).hash as usize).weight), _MM_HINT_T0);
                                let left_hash = fb.ffm_buffer.get_unchecked(ffm_buffer_index);
                                let mut addr = left_hash.hash as usize;
                                let mut zfc:usize = field_index_ffmk as usize;

                                specialize_1f32!(left_hash.value, LEFT_HASH_VALUE, {
                                    if feature_num == 0 {
                                        for z in 0..fb.ffm_fields_count {
                                            _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(addr + FFMK as usize).weight), _MM_HINT_T0);
                                            for k in 0..FFMK as usize{
                                                *contra_fields.get_unchecked_mut(zfc + k) = ffm_weights.get_unchecked(addr + k).weight * LEFT_HASH_VALUE;
                                            }
                                            zfc += fc;
                                            addr += FFMK as usize
                                        }
                                    } else {
                                        for z in 0..fb.ffm_fields_count {
                                            _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(addr + FFMK as usize).weight), _MM_HINT_T0);
                                            for k in 0..FFMK as usize{
                                                *contra_fields.get_unchecked_mut(zfc + k) += ffm_weights.get_unchecked(addr + k).weight * LEFT_HASH_VALUE;
                                            }
                                            zfc += fc;
                                            addr += FFMK as usize
                                        }
                                    }
                                });
                                ffm_buffer_index += 1;
                                feature_num += 1;
                            }
                        }

                        let mut ffm_values_offset = 0;
                        for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
                            let mut contra_offset = (left_hash.contra_field_index * fb.ffm_fields_count) as usize;
                            let contra_pure_index = contra_offset / FFMK as usize; // super not nice, but division gets optimized away
                            let mut vv = 0;
                            let left_hash_value = left_hash.value;
                            let left_hash_contra_field_index = left_hash.contra_field_index;
                            let left_hash_hash = left_hash.hash as usize;
                            //let LEFT_HASH_VALUE = left_hash_value;
                            specialize_1f32!(left_hash_value, LEFT_HASH_VALUE, {
                              for z in 0..fb.ffm_fields_count as usize {
                                  let interaction_weight = self.field_interaction_weights.get_unchecked(z + contra_pure_index).weight;
                                  if vv == left_hash_contra_field_index as usize {
                                      for k in 0..FFMK as usize {
                                          let ffm_weight = ffm_weights.get_unchecked(left_hash_hash + vv + k).weight;
                                          let contra_weight = *contra_fields.get_unchecked(contra_offset + vv + k) - ffm_weight * LEFT_HASH_VALUE;
                                          let gradient =  LEFT_HASH_VALUE * contra_weight * interaction_weight;
                                          *local_data_ffm_values.get_unchecked_mut(ffm_values_offset + k) = gradient;
                                          *wsumbuf.get_unchecked_mut(k) += ffm_weight * gradient;
                                      }
                                  } else {
                                      for k in 0..FFMK as usize {
                                          let ffm_weight = ffm_weights.get_unchecked(left_hash_hash + vv + k).weight;
                                          let contra_weight = *contra_fields.get_unchecked(contra_offset + vv + k);
                                          let gradient =  LEFT_HASH_VALUE * contra_weight * interaction_weight;
                                          *local_data_ffm_values.get_unchecked_mut(ffm_values_offset + k) = gradient;
                                          *wsumbuf.get_unchecked_mut(k) += ffm_weight * gradient;
                                      }
                                  }
                                  vv += FFMK as usize;
                                  //left_hash_hash += FFMK as usize;
                                  //contra_offset += FFMK as usize;
                                  ffm_values_offset += FFMK as usize;
                              }
                            }); // End of macro specialize_1f32! for LEFT_HASH_VALUE
                        }
                        for k in 0..FFMK as usize {
                            wsum += wsumbuf[k];
                        }
                        wsum *= 0.5;
                    });

                    let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
                    let (prediction_probability, general_gradient) = next_regressor[0].forward_backward(further_blocks, wsum + wsum_input, fb, update);

                    if update {
                        let mut local_index: usize = 0;
                        for left_hash in &fb.ffm_buffer {
                            let mut feature_index = left_hash.hash as usize;
                            for j in 0..fc as usize {
                                let feature_value = *local_data_ffm_values.get_unchecked(local_index);
                                let gradient = general_gradient * feature_value;
                                let update = self.optimizer_ffm.calculate_update(gradient, &mut ffm_weights.get_unchecked_mut(feature_index).optimizer_data);
                                ffm_weights.get_unchecked_mut(feature_index).weight += update;
                                local_index += 1;
                                feature_index += 1;
                            }
                        }
                    }
                    // The only exit point
                    return (prediction_probability, general_gradient)
                }
            }; // End of macro


            if local_data_ffm_len < FFM_STACK_BUF_LEN {
                // Fast-path - using on-stack data structures
                let mut local_data_ffm_values: [f32; FFM_STACK_BUF_LEN as usize] = MaybeUninit::uninit().assume_init();//[0.0; FFM_STACK_BUF_LEN as usize];
                core_macro!(local_data_ffm_values);

            } else {
                // Slow-path - using heap data structures
                if local_data_ffm_len > self.local_data_ffm_values.len() {
                    self.local_data_ffm_values.reserve(local_data_ffm_len - self.local_data_ffm_values.len() + 1024);
                }
                let mut local_data_ffm_values = &mut self.local_data_ffm_values;

                core_macro!(local_data_ffm_values);
            }
        } // unsafe end
    }

    fn forward(&self, further_blocks: &[Box<dyn BlockTrait>], wsum_input: f32, fb: &feature_buffer::FeatureBuffer) -> f32 {
        let mut wsum:f32 = 0.0;
        unsafe {
            let ffm_weights = &self.weights;
            if true {
                _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(fb.ffm_buffer.get_unchecked(0).hash as usize).weight), _MM_HINT_T0);
                let field_embedding_len = self.field_embedding_len as usize;
                let mut contra_fields: [f32; FFM_STACK_BUF_LEN] = MaybeUninit::uninit().assume_init();

                specialize_k!(self.ffm_k, FFMK, wsumbuf, {
                    /* We first prepare "contra_fields" or collapsed field embeddings, where we sum all individual feature embeddings
                       We need to be careful to:
                       - handle fields with zero features present
                       - handle values on diagonal - we want to be able to exclude self-interactions later (we pre-substract from wsum)
                       - optimize for just copying the embedding over when looking at first feature of the field, and add embeddings for the rest
                       - optiize for very common case of value of the feature being 1.0 - avoid multiplications
                       -
                    */

                    let mut ffm_buffer_index = 0;
                    for field_index in 0..fb.ffm_fields_count {
                        let field_index_ffmk = field_index * FFMK;
                        let offset = (field_index_ffmk * fb.ffm_fields_count) as usize;
                        // first we handle fields with no features
                        if ffm_buffer_index >= fb.ffm_buffer.len() ||
                            fb.ffm_buffer.get_unchecked(ffm_buffer_index).contra_field_index > field_index_ffmk {
                            for z in 0..field_embedding_len as usize { // first time we see this field - just overwrite
                                *contra_fields.get_unchecked_mut(offset + z) = 0.0;
                            }
                            continue;
                        }
                        let mut feature_num = 0;
                        while ffm_buffer_index < fb.ffm_buffer.len() && fb.ffm_buffer.get_unchecked(ffm_buffer_index).contra_field_index == field_index_ffmk {
                            _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(fb.ffm_buffer.get_unchecked(ffm_buffer_index+1).hash as usize).weight), _MM_HINT_T0);
                            let left_hash = fb.ffm_buffer.get_unchecked(ffm_buffer_index);
                            let left_hash_hash = left_hash.hash as usize;
                            let left_hash_value = left_hash.value;
                            specialize_1f32!(left_hash_value, LEFT_HASH_VALUE, {
                                if feature_num == 0 {
                                    for z in 0..field_embedding_len { // first feature of the field - just overwrite
                                        *contra_fields.get_unchecked_mut(offset + z) = ffm_weights.get_unchecked(left_hash_hash + z).weight * LEFT_HASH_VALUE;
                                    }
                                } else {
                                    for z in 0..field_embedding_len { // additional features of the field - addition
                                        *contra_fields.get_unchecked_mut(offset + z) += ffm_weights.get_unchecked(left_hash_hash + z).weight * LEFT_HASH_VALUE;
                                    }
                                }
                                let interaction_weight = self.field_interaction_weights.get_unchecked((field_index * fb.ffm_fields_count + field_index) as usize).weight;
                                let vv = SQRT_OF_ONE_HALF * LEFT_HASH_VALUE;     // To avoid one additional multiplication, we square root 0.5 into vv
                                for k in 0..FFMK as usize {
                                    let ss = ffm_weights.get_unchecked(left_hash_hash + field_index_ffmk as usize + k).weight * vv;
                                    *wsumbuf.get_unchecked_mut(k) -= ss * ss * interaction_weight;
                                }
                            });
                            ffm_buffer_index += 1;
                            feature_num += 1;
                        }
                    }


                    for f1 in 0..fb.ffm_fields_count as usize {
                        let f1_offset = f1 * field_embedding_len as usize;
                        let f1_ffmk = f1 * FFMK as usize;
                        let mut f2_offset_ffmk = f1_offset + f1_ffmk;
                        let mut f1_offset_ffmk = f1_offset + f1_ffmk;

                        let f1_interaction_offset = f1 * fb.ffm_fields_count as usize;
                        let interaction_weight = self.field_interaction_weights.get_unchecked(f1 + f1_interaction_offset).weight; // on diagonal

                        // This is self-interaction
                        for k in 0..FFMK as usize{
                            let v = contra_fields.get_unchecked(f1_offset_ffmk + k);
                            *wsumbuf.get_unchecked_mut(k) += v * v * 0.5 * interaction_weight;
                        }

                        for f2 in f1+1..fb.ffm_fields_count as usize {
                            let interaction_weight = self.field_interaction_weights.get_unchecked(f2 + f1_interaction_offset).weight;
                            f2_offset_ffmk += field_embedding_len as usize;
                            f1_offset_ffmk += FFMK as usize;
                            //assert_eq!(f1_offset_ffmk, f1 * field_embedding_len + f2 * FFMK as usize);
                            //assert_eq!(f2_offset_ffmk, f2 * field_embedding_len + f1 * FFMK as usize);
                            for k in 0..FFMK {
                                *wsumbuf.get_unchecked_mut(k as usize) +=
                                        contra_fields.get_unchecked(f1_offset_ffmk + k as usize) *
                                        contra_fields.get_unchecked(f2_offset_ffmk + k as usize) *
                                        interaction_weight;
                            }
                        }

                    }
                    for k in 0..FFMK as usize {
                        wsum += wsumbuf[k];
                    }
                });
            } else {
                // Old straight-forward method. As soon as we have multiple feature values per field, it is slower
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
                                *wsumbuf.get_unchecked_mut(k as usize) += left_hash_weight * right_hash_weight * joint_value;
                            }
                        }

                    }
                    for k in 0..FFMK as usize {
                        wsum += wsumbuf[k];
                    }
                });
            }
        }
        let (next_regressor, further_blocks) = further_blocks.split_at(1);
        let prediction_probability = next_regressor[0].forward(further_blocks, wsum + wsum_input, fb);
        prediction_probability

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
        let mut forward = forward.as_any().downcast_mut::<BlockAFFM<optimizer::OptimizerSGD>>().unwrap();
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
    use crate::vwmap;
    use block_helpers::{slearn, spredict};

    use crate::assert_epsilon;


    fn ffm_vec(v:Vec<feature_buffer::HashAndValueAndSeq>, ffm_fields_count: u32) -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
                    label: 0.0,
                    example_importance: 1.0,
                    example_number: 0,
                    lr_buffer: Vec::new(),
                    ffm_buffer: v,
                    ffm_fields_count: ffm_fields_count,
        }
    }

    fn ffm_init<T:OptimizerTrait + 'static>(block_ffm: &mut Box<dyn BlockTrait>) -> () {
        let mut block_ffm = block_ffm.as_any().downcast_mut::<BlockAFFM<T>>().unwrap();

        for i in 0..block_ffm.weights.len() {
            block_ffm.weights[i].weight = 1.0;
            block_ffm.weights[i].optimizer_data = block_ffm.optimizer_ffm.initial_data();
        }
    }



    #[test]
    fn test_ffm_interactions_setup() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![], vec![]]; // This isn't really used
        mi.ffm_interaction_matrix = true;
        mi.ffm_interactions.push((1, 0, 0.5));
        mi.ffm_interactions.push((0, 0, 0.1));
        mi.ffm_interactions.push((1, 1, 0.6));

        let mut re = BlockAFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();

        re.allocate_and_init_weights(&mi);
        let re2 = re.as_any().downcast_mut::<BlockAFFM<optimizer::OptimizerAdagradLUT>>().unwrap();
        assert_eq!(re2.field_interaction_weights.len(), 9);
        assert_eq!(re2.field_interaction_weights[0].weight, 0.1);
        assert_eq!(re2.field_interaction_weights[1*3 + 0].weight, 0.5);
        assert_eq!(re2.field_interaction_weights[0*3 + 1].weight, 0.5);
        assert_eq!(re2.field_interaction_weights[1*3 + 1].weight, 0.6);
        assert_eq!(re2.field_interaction_weights[8].weight, 1.0);

    }

    #[test]
    fn test_ffm_k1() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]]; // This isn't really used
        mi.ffm_interaction_matrix = true;
        mi.ffm_interactions.push((1, 0, 0.5));
        mi.ffm_interactions.push((0, 0, 0.1));
        mi.ffm_interactions.push((1, 1, 0.6));

        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();

        // Nothing can be learned from a single field in FFMs
        let mut re = BlockAFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

/*        let fb = ffm_vec(vec![HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0}],
                        1); // saying we have 1 field isn't entirely correct
        assert_epsilon!(spredict(&mut re, &mut lossf, &fb, true), 0.5);
        assert_epsilon!(slearn  (&mut re, &mut lossf, &fb, true), 0.5);
*/
        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
/*        let mut re = BlockAFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let fb = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: mi.ffm_k}
                                  ], 2);
        assert_epsilon!(spredict(&mut re, &mut lossf, &fb, true), 0.62245935);
        assert_eq!(slearn  (&mut re, &mut lossf, &fb, true), 0.62245935);

        assert_epsilon!(spredict(&mut re, &mut lossf, &fb, true), 0.6152326);
        assert_eq!(slearn  (&mut re, &mut lossf, &fb, true), 0.6152326);

*/
        // Two fields, use values
        let mut re = BlockAFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let fb = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 2.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 3.0, contra_field_index: mi.ffm_k}
                                  ], 2);

        assert_epsilon!(spredict(&mut re, &mut lossf, &fb, true), 0.95257413);
        assert_epsilon!(slearn(&mut re, &mut lossf, &fb, true), 0.95257413);
/*        println!("CCC");
        assert_epsilon!(spredict(&mut re, &mut lossf, &fb, true), 0.82205963);
        println!("DDD");
        assert_epsilon!(slearn(&mut re, &mut lossf, &fb, true), 0.82205963);*/
    }


    #[test]
    fn test_ffm_k4() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.ffm_k = 4;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]]; // This isn't really used
        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();

        // Nothing can be learned from a single field in FFMs
        let mut re = BlockAFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        let fb = ffm_vec(vec![HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0}], 4);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.5);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.5);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.5);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        let mut re = BlockAFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let fb = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: mi.ffm_k * 1}
                                  ], 2);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.98201376);
        assert_eq!(slearn  (&mut re, &mut lossf, &fb, true), 0.98201376);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.96277946);
        assert_eq!(slearn  (&mut re, &mut lossf, &fb, true), 0.96277946);

        // Two fields, use values
        let mut re = BlockAFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut re);
        let fb = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 2.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: mi.ffm_k * 1}
                                  ], 2);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.9999999);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.9999999);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.99685884);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.99685884);
    }


    #[test]
    fn test_ffm_multivalue() {
        let vw_map_string = r#"
A,featureA
B,featureB
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_power_t = 0.0;
        mi.ffm_learning_rate = 0.1;
        mi.ffm_fields = vec![vec![],vec![]];
        mi.optimizer = model_instance::Optimizer::Adagrad;
        mi.fastmath = false;
        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();

        let mut re = BlockAFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);
        let mut p: f32;

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut re);
        let fbuf = &ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:3 * 1000, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: mi.ffm_k * 1}
                                  ], 2);
        assert_epsilon!(spredict(&mut re, &mut lossf, &fbuf, true), 0.9933072);
        assert_eq!(slearn(&mut re, &mut lossf, &fbuf, true), 0.9933072);
        assert_epsilon!(spredict(&mut re, &mut lossf, &fbuf, false), 0.9395168);
        assert_eq!(slearn(&mut re, &mut lossf, &fbuf, false), 0.9395168);
        assert_epsilon!(spredict(&mut re, &mut lossf, &fbuf, false), 0.9395168);
        assert_eq!(slearn(&mut re, &mut lossf, &fbuf, false), 0.9395168);
    }

    #[test]
    fn test_ffm_multivalue_k4_nonzero_powert() {
        let vw_map_string = r#"
A,featureA
B,featureB
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.ffm_k = 4;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![],vec![]];
        mi.optimizer = model_instance::Optimizer::Adagrad;
        mi.fastmath = false;
        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();

        let mut re = BlockAFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);
        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut re);
        let fbuf = &ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:3 * 1000, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: mi.ffm_k * 1}
                                  ], 2);

        assert_eq!(spredict(&mut re, &mut lossf, &fbuf, true), 1.0);
        assert_eq!(slearn(&mut re, &mut lossf, &fbuf, true), 1.0);
        assert_eq!(spredict(&mut re, &mut lossf, &fbuf, false), 0.9949837);
        assert_eq!(slearn(&mut re, &mut lossf, &fbuf, false), 0.9949837);
        assert_eq!(slearn(&mut re, &mut lossf, &fbuf, false), 0.9949837);
    }

    #[test]
    fn test_ffm_missing_field() {
        // This test is useful to check if we don't by accient forget to initialize any of the collapsed
        // embeddings for the field, when field has no instances of a feature in it
        // We do by having three-field situation where only the middle field has features
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![], vec![]]; // This isn't really used
        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();

        // Nothing can be learned from a single field in FFMs
        let mut re = BlockAFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);


        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        let mut re = BlockAFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let fb = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:5, value: 1.0, contra_field_index: mi.ffm_k * 1},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: mi.ffm_k * 2}
                                  ], 3);
        assert_epsilon!(spredict(&mut re, &mut lossf, &fb, true), 0.95257413);
        assert_eq!(slearn  (&mut re, &mut lossf, &fb, false), 0.95257413);

        // here we intentionally have just the middle field
        let fb = ffm_vec(vec![HashAndValueAndSeq{hash:5, value: 1.0, contra_field_index: mi.ffm_k * 1}], 3);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.5);
        assert_eq!(slearn  (&mut re, &mut lossf, &fb, true), 0.5);

    }













}


