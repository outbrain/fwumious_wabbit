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
use block_helpers::{Weight, WeightAndOptimizerData, slearn};


const FFM_STACK_BUF_LEN:usize= 32768;
const FFM_CONTRA_BUF_LEN:usize = 16384;


const SQRT_OF_ONE_HALF:f32 = 0.70710678118;
 
use std::ops::{Deref, DerefMut};
use std::sync::Arc;


pub struct BlockFFM<L:OptimizerTrait> {
    pub optimizer_ffm: L,
    pub local_data_ffm_indices: Vec<u32>,
    pub local_data_ffm_values: Vec<f32>,
    pub ffm_k: u32,
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
            optimizer_ffm: L::new(),
        };

        if mi.ffm_k > 0 {
            reg_ffm.optimizer_ffm.init(mi.ffm_learning_rate, mi.ffm_power_t, mi.ffm_init_acc_gradient);
            // At the end we add "spillover buffer", so we can do modulo only on the base address and add offset
            reg_ffm.ffm_weights_len = (1 << mi.ffm_bit_precision) + (mi.ffm_fields.len() as u32 * reg_ffm.ffm_k);
        }

        // Verify that forward pass will have enough stack for temporary buffer
        if reg_ffm.ffm_k as usize * mi.ffm_fields.len() * mi.ffm_fields.len() > FFM_CONTRA_BUF_LEN {
            return Err(format!("FFM_CONTRA_BUF_LEN is {}. It needs to be at least ffm_k * number_of_fields^2. number_of_fields: {}, ffm_k: {}, please recompile with larger constant", 
                        FFM_CONTRA_BUF_LEN, mi.ffm_fields.len(), reg_ffm.ffm_k))?;
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
            optimizer_ffm: optimizer::OptimizerSGD::new(),
        };
        
        Ok(Box::new(forwards_only))
    }



    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights =vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.optimizer_ffm.initial_data()}; self.ffm_weights_len as usize];
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
                 
                    let mut local_data_ffm_values = &mut $local_data_ffm_values;
                        
                    let ffm_weights = &mut self.weights;
                    let fc = (fb.ffm_fields_count  * self.ffm_k) as usize;
                    let mut ifc:usize = 0;
                    
                    if true {
                        let mut contra_fields: [f32; FFM_CONTRA_BUF_LEN] = MaybeUninit::uninit().assume_init();
                        let mut last_contra_index = 1000000;
                        specialize_k!(self.ffm_k, FFMK, wsumbuf, {
                            /* first prepare two things:
                            - transposed contra vectors in contra_fields - 
                                - for each vector we sum up all the features within a field
                                - and at the same time transpose it, so we can later directly multiply them with individual feature embeddings
                            - cache of gradients in local_data_ffm_values 
                                - we will use these gradients later in backward pass
                            */
                            //_mm_prefetch(mem::transmute::<&f32, &i8>(&contra_fields.get_unchecked(fb.ffm_buffer.get_unchecked(0).contra_field_index as usize)), _MM_HINT_T0);
                            for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
//                                let pp = contra_fields.as_mut_ptr().add(left_hash.contra_field_index as usize);
                                let mut addr = left_hash.hash as usize;
                                 // This line is golden. Just cache the very first cache line in next iteration
                                _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(fb.ffm_buffer.get_unchecked(i+1).hash as usize).weight), _MM_HINT_T0);
                                _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(fb.ffm_buffer.get_unchecked(i+2).hash as usize).weight), _MM_HINT_T0);
                                        
                                if last_contra_index != left_hash.contra_field_index {
                                    let left_hash_value = left_hash.value;
                                    let left_hash_contra_field_index = left_hash.contra_field_index;
                                    let mut zfc:usize = left_hash.contra_field_index as usize;
                                    for z in 0..fb.ffm_fields_count {
                                        /*_mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(addr+FFMK as usize).weight), _MM_HINT_T0);
                                        */
                                        for k in 0..FFMK as usize{
                                            *contra_fields.get_unchecked_mut(zfc + k) = ffm_weights.get_unchecked(addr+k).weight * left_hash_value;
                                        }
                                        zfc += fc;
                                        addr += FFMK as usize
                                    }
                                } else {
                                    let mut zfc:usize = left_hash.contra_field_index as usize;
                                    let left_hash_value = left_hash.value;
                                    let left_hash_contra_field_index = left_hash.contra_field_index;
                                    for z in 0..fb.ffm_fields_count {
                                        /*_mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(addr+FFMK as usize).weight), _MM_HINT_T0);
                                        */
                                        for k in 0..FFMK as usize{
                                            *contra_fields.get_unchecked_mut(zfc + k) += ffm_weights.get_unchecked(addr + k).weight * left_hash_value;
                                        }
                                        //for (dest, source) in contra_fields.get_unchecked_mut(zfc..zfc+FFMK as usize).iter_mut().zip(ffm_weights.get_unchecked(addr..addr+FFMK as usize).iter()){
                                        //    *dest += source.weight * left_hash_value;
                                        //}
                                        zfc += fc;
                                        addr += FFMK as usize
                                    }
                                }
                                last_contra_index = left_hash.contra_field_index;
                            }
                            
                            let mut ffm_values_offset = 0;
                            for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
                                let mut contra_offset = (left_hash.contra_field_index * fb.ffm_fields_count) as usize;
                                let mut vv = 0;
                                let left_hash_value = left_hash.value;
                                let left_hash_contra_field_index = left_hash.contra_field_index;
                                let mut left_hash_hash = left_hash.hash as usize;
                                for z in 0..fb.ffm_fields_count as usize {
                                    {
                                        for k in 0..FFMK as usize {
                                            let ffm_weight = ffm_weights.get_unchecked(left_hash_hash + vv + k).weight;
                                            let mut contra_weight = *contra_fields.get_unchecked(contra_offset + vv + k);
                                            if vv == left_hash_contra_field_index as usize{
                                                contra_weight -= ffm_weight * left_hash_value;
                                            }
                                            let gradient =  left_hash_value * contra_weight;
                                            
                                            *local_data_ffm_values.get_unchecked_mut(ffm_values_offset + k) = gradient;
                                            *wsumbuf.get_unchecked_mut(k) += ffm_weight * gradient;
                                        }
                                    }
                                    vv += FFMK as usize;
                                    //left_hash_hash += FFMK as usize;
                                    //contra_offset += FFMK as usize;
                                    ffm_values_offset += FFMK as usize;
                                }
                            }    
                            for k in 0..FFMK as usize {
                                wsum += wsumbuf[k];
                            }
                            wsum *= 0.5;
                        });
                        
                    } else {

                        specialize_k!(self.ffm_k, FFMK, wsumbuf, {
                         
                            // This is a strange loop. We want to have just first cache line ready for each embedding
                            // Plus we need to initialize to zero.
                            let mut baddr: usize = 0;
                            for left_hash in &fb.ffm_buffer {
                                let mut addr = left_hash.hash as usize;
                                for z in 0..fb.ffm_fields_count {
                                    _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(addr as usize).weight), _MM_HINT_T0);  // No benefit for now
                                    for k in 0..FFMK {
                                        *local_data_ffm_values.get_unchecked_mut(baddr) = 0.0;
                                        addr += 1;
                                        baddr += 1;
                                    }
                                }
                            }
                           
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
                                    let joint_value = left_hash.value * right_hash.value;
                                    let left_local_index = ifc + right_hash.contra_field_index as usize;
                                    let lindex = (left_hash.hash + right_hash.contra_field_index) as usize;
                                    let rindex = (right_hash.hash + left_hash.contra_field_index) as usize;

                                    specialize_1f32!(joint_value, JOINT_VALUE, {
                                        for k in 0..FFMK as usize {
                                            let left_hash_weight  = ffm_weights.get_unchecked((lindex+k) as usize).weight;
                                            let right_hash_weight = ffm_weights.get_unchecked((rindex+k) as usize).weight;
                                            
                                            let right_side = right_hash_weight * JOINT_VALUE;
                                            *local_data_ffm_values.get_unchecked_mut(left_local_index + k) += right_side; // first derivate
                                            *local_data_ffm_values.get_unchecked_mut(right_local_index + k) += left_hash_weight  * JOINT_VALUE; // first derivate
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
                    } // End of second implementation
                    
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
                }; 
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

                let mut contra_fields: [f32; FFM_STACK_BUF_LEN] = MaybeUninit::uninit().assume_init();
                let field_embedding_len = (self.ffm_k * fb.ffm_fields_count) as usize;

                specialize_k!(self.ffm_k, FFMK, wsumbuf, {
                    let mut last_contra_index = 1000000;
                    for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
                        // This line is golden. Just cache the very first cache line in next iteration
                        _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(fb.ffm_buffer.get_unchecked(i+1).hash as usize).weight), _MM_HINT_T0);
                        let left_hash_hash = left_hash.hash as usize;
                        let contra_field_index = left_hash.contra_field_index as usize;
                        let offset = (left_hash.contra_field_index * fb.ffm_fields_count) as usize;
                        specialize_1f32!(left_hash.value, LEFT_HASH_VALUE, {
                            if last_contra_index != left_hash.contra_field_index {
                                for k in 0..field_embedding_len { // first time we see this field - just overwrite
                                    *contra_fields.get_unchecked_mut(offset + k) = ffm_weights.get_unchecked(left_hash_hash + k).weight * LEFT_HASH_VALUE;
                                }
                            } else {
                                for k in 0..field_embedding_len { // we've seen this field before - add
                                    *contra_fields.get_unchecked_mut(offset + k) += ffm_weights.get_unchecked(left_hash_hash + k).weight * LEFT_HASH_VALUE;
                                }
                            }
                            // this is part of the work to cancel-out self-interaction of a feature
                            let vv = SQRT_OF_ONE_HALF * LEFT_HASH_VALUE;     // To avoid one additional multiplication, we square root 0.5 into vv
                            for k in 0..FFMK as usize {
                                let ss = ffm_weights.get_unchecked(left_hash_hash as usize + contra_field_index + k).weight * vv;
                                let minus = ss * ss;
                                /*println!("i: {}, value: {} weight: {}, contra field index {}, ss: {}, substract: {}", 
                                        i, v, ffm_weights.get_unchecked(left_hash.hash as usize + left_hash.contra_field_index as usize).weight, 
                                        left_hash.contra_field_index,
                                        AAss, minus);
                                */
                                wsumbuf[k as usize] -= minus;
                            }
                        });
                        last_contra_index = left_hash.contra_field_index;
                    }

                    for f1 in 0..fb.ffm_fields_count as usize {
                        let f1_offset = f1 * field_embedding_len as usize;
                        let f1_ffmk = f1 * FFMK as usize;
                        let mut f2_offset_ffmk = f1_offset + f1_ffmk;
                        let mut f1_offset_ffmk = f1_offset + f1_ffmk;
                        // This is self-interaction
                        for k in 0..FFMK {
                            let v = contra_fields.get_unchecked(f1_offset_ffmk + k as usize);
                            *wsumbuf.get_unchecked_mut(k as usize) += v * v * 0.5;
                        }

                        for f2 in f1+1..fb.ffm_fields_count as usize {
                            f2_offset_ffmk += field_embedding_len;
                            f1_offset_ffmk += FFMK as usize;
                            //assert_eq!(f1_offset_ffmk, f1 * field_embedding_len + f2 * FFMK as usize);
                            //assert_eq!(f2_offset_ffmk, f2 * field_embedding_len + f1 * FFMK as usize);
                            /*let k = 0;
                            println!("F1: {}, F2: {}, f1 offset: {}, f2 offset: {}", f1, f2, f1_offset_ffmk, f2_offset_ffmk);
                            println!("c1: {} , c2: {}, wsumadd {}",   contra_fields.get_unchecked(f1_offset_ffmk + k as usize), 
                                    contra_fields.get_unchecked(f2_offset_ffmk + k as usize),
                                    contra_fields.get_unchecked(f1_offset_ffmk + k as usize) * 
                                    contra_fields.get_unchecked(f2_offset_ffmk + k as usize)
                                    );*/
                                for k in 0..FFMK {
                                    *wsumbuf.get_unchecked_mut(k as usize) += 
                                            contra_fields.get_unchecked(f1_offset_ffmk + k as usize) * 
                                            contra_fields.get_unchecked(f2_offset_ffmk + k as usize);
                                            
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
                                //wsum += left_hash_weight * right_side;
                                // We do this, so in theory Rust/LLVM could vectorize whole loop
                                // Unfortunately it does not happen in practice, but we will get there
                                // Original: wsum += left_hash_weight * right_side;
                                //println!("Left value {}, left weight {}, right value {}, right weight {}", left_hash.value, left_hash_weight,
                                //                                                            right_hash.value, right_hash_weight);
                                //println!("WsumadD: {}", left_hash_weight * right_hash_weight * joint_value);
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
        let mut block_ffm = block_ffm.as_any().downcast_mut::<BlockFFM<T>>().unwrap();
        
        for i in 0..block_ffm.weights.len() {
            block_ffm.weights[i].weight = 1.0;
            block_ffm.weights[i].optimizer_data = block_ffm.optimizer_ffm.initial_data();
        }
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
        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();
        
        // Nothing can be learned from a single field in FFMs
        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        let ffm_buf = ffm_vec(vec![HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0}], mi.ffm_k);
//        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.5);
//        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        let mut re = BlockFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: mi.ffm_k}
                                  ], 2);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.7310586); 
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.7024794);

        // Two fields, use values
        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut re);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 2.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: mi.ffm_k}
                                  ], 2);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.98201376);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.81377685);
    }


    #[test]
    fn test_ffm_k4() {
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

        let ffm_buf = ffm_vec(vec![HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0}], 4);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.5);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        let mut re = BlockFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: 4}
                                  ], 2);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.98201376); 
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.96277946);

        // Two fields, use values
        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut re);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 2.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: 4}
                                  ], 2);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.9999999);
        assert_eq!(slearn(&mut re, &mut lossf, &ffm_buf, true), 0.99685884);
    }


}



