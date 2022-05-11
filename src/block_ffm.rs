use core::arch::x86_64::*;
use merand48::*;
use serde_json::{json, Map, Number, Value};
use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::mem::{self, MaybeUninit};

use crate::block_helpers;
use crate::consts;
use crate::feature_buffer;
use crate::model_instance;
use crate::optimizer;
use crate::regressor;
use block_helpers::{Weight, WeightAndOptimizerData};
use optimizer::OptimizerTrait;
use regressor::BlockTrait;

use crate::block_helpers::f32_to_json;

const FFM_STACK_BUF_LEN: usize = 32768;
const FFM_CONTRA_BUF_LEN: usize = 16384;

const SQRT_OF_ONE_HALF: f32 = 0.70710678118;

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

pub struct BlockFFM<L: OptimizerTrait> {
    pub optimizer_ffm: L,
    pub local_data_ffm_values: Vec<f32>,
    pub ffm_k: u32,
    pub ffm_weights_len: u32,
    pub field_embedding_len: u32,
    pub weights: Vec<WeightAndOptimizerData<L>>,
}

macro_rules! specialize_1f32 {
    ( $input_expr:expr,
      $output_const:ident,
      $code_block:block  ) => {
        if $input_expr == 1.0 {
            const $output_const: f32 = 1.0;
            $code_block
        } else {
            let $output_const: f32 = $input_expr;
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
            2 => {
                const $output_const: u32 = 2;
                let mut $wsumbuf: [f32; $output_const as usize] = [0.0; $output_const as usize];
                $code_block
            }
            4 => {
                const $output_const: u32 = 4;
                let mut $wsumbuf: [f32; $output_const as usize] = [0.0; $output_const as usize];
                $code_block
            }
            8 => {
                const $output_const: u32 = 8;
                let mut $wsumbuf: [f32; $output_const as usize] = [0.0; $output_const as usize];
                $code_block
            }
            val => {
                let $output_const: u32 = val;
                let mut $wsumbuf: [f32; consts::FFM_MAX_K] = [0.0; consts::FFM_MAX_K];
                $code_block
            }
        }
    };
}

impl<L: OptimizerTrait + 'static> BlockTrait for BlockFFM<L> {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn new_without_weights(
        mi: &model_instance::ModelInstance,
    ) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        let mut reg_ffm = BlockFFM::<L> {
            weights: Vec::new(),
            ffm_weights_len: 0,
            local_data_ffm_values: Vec::with_capacity(1024),
            ffm_k: mi.ffm_k,
            field_embedding_len: mi.ffm_k * mi.ffm_fields.len() as u32,
            optimizer_ffm: L::new(),
        };

        if mi.ffm_k > 0 {
            reg_ffm.optimizer_ffm.init(
                mi.ffm_learning_rate,
                mi.ffm_power_t,
                mi.ffm_init_acc_gradient,
            );
            // At the end we add "spillover buffer", so we can do modulo only on the base address and add offset
            reg_ffm.ffm_weights_len =
                (1 << mi.ffm_bit_precision) + (mi.ffm_fields.len() as u32 * reg_ffm.ffm_k);
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
            local_data_ffm_values: Vec::new(),
            ffm_k: self.ffm_k,
            field_embedding_len: self.field_embedding_len,
            optimizer_ffm: optimizer::OptimizerSGD::new(),
        };

        Ok(Box::new(forwards_only))
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights = vec![
            WeightAndOptimizerData::<L> {
                weight: 0.0,
                optimizer_data: self.optimizer_ffm.initial_data()
            };
            self.ffm_weights_len as usize
        ];
        if mi.ffm_k > 0 {
            if mi.ffm_init_width == 0.0 {
                // Initialization that has showed to work ok for us, like in ffm.pdf, but centered around zero and further divided by 50
                let ffm_one_over_k_root = 1.0 / (self.ffm_k as f32).sqrt() / 50.0;
                for i in 0..self.ffm_weights_len {
                    self.weights[i as usize].weight =
                        (1.0 * merand48((self.ffm_weights_len as usize + i as usize) as u64) - 0.5)
                        * ffm_one_over_k_root;
                    self.weights[i as usize].optimizer_data = self.optimizer_ffm.initial_data();
                }
            } else {
                let zero_half_band_width = mi.ffm_init_width * mi.ffm_init_zero_band * 0.5;
                let band_width = mi.ffm_init_width * (1.0 - mi.ffm_init_zero_band);
                for i in 0..self.ffm_weights_len {
                    let mut w = merand48(i as u64) * band_width - band_width * 0.5;
                    if w > 0.0 {
                        w += zero_half_band_width;
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
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        wsum_input: f32,
        fb: &feature_buffer::FeatureBuffer,
        update: bool,
    ) -> (f32, f32) {
        let mut wsum = 0.0;
        let local_data_ffm_len = fb.ffm_buffer.len() * (self.ffm_k * fb.ffm_fields_count) as usize;

        unsafe {
            macro_rules! core_macro {
                (
					$local_data_ffm_values: expr
                ) => {
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
                                _mm_prefetch(mem::transmute::<&f32, &i8>(&self.weights.get_unchecked(fb.ffm_buffer.get_unchecked(ffm_buffer_index+1).hash as usize).weight), _MM_HINT_T0);
                                let left_hash = fb.ffm_buffer.get_unchecked(ffm_buffer_index);
                                let mut addr = left_hash.hash as usize;
                                let mut zfc:usize = field_index_ffmk as usize;

                                specialize_1f32!(left_hash.value, LEFT_HASH_VALUE, {
                                    if feature_num == 0 {
                                        for z in 0..fb.ffm_fields_count {
                                            _mm_prefetch(mem::transmute::<&f32, &i8>(&self.weights.get_unchecked(addr + FFMK as usize).weight), _MM_HINT_T0);
                                            for k in 0..FFMK as usize{
                                                *contra_fields.get_unchecked_mut(zfc + k) = self.weights.get_unchecked(addr + k).weight * LEFT_HASH_VALUE;
                                            }
                                            zfc += fc;
                                            addr += FFMK as usize
                                        }
                                    } else {
                                        for z in 0..fb.ffm_fields_count {
                                            _mm_prefetch(mem::transmute::<&f32, &i8>(&self.weights.get_unchecked(addr + FFMK as usize).weight), _MM_HINT_T0);
                                            for k in 0..FFMK as usize{
                                                *contra_fields.get_unchecked_mut(zfc + k) += self.weights.get_unchecked(addr + k).weight * LEFT_HASH_VALUE;
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
                            let mut vv = 0;
                            let left_hash_value = left_hash.value;
                            let left_hash_contra_field_index = left_hash.contra_field_index;
                            let left_hash_hash = left_hash.hash as usize;
                            //let LEFT_HASH_VALUE = left_hash_value;
                            specialize_1f32!(left_hash_value, LEFT_HASH_VALUE, {

								for z in 0..fb.ffm_fields_count as usize {
									if vv == left_hash_contra_field_index as usize {
										for k in 0..FFMK as usize {
											let ffm_weight = self.weights.get_unchecked(left_hash_hash + vv + k).weight;
											let contra_weight = *contra_fields.get_unchecked(contra_offset + vv + k) - ffm_weight * LEFT_HASH_VALUE;
											let gradient =  LEFT_HASH_VALUE * contra_weight;
											*$local_data_ffm_values.get_unchecked_mut(ffm_values_offset + k) = gradient;
											*wsumbuf.get_unchecked_mut(k) += ffm_weight * gradient;
										}
									} else {
										for k in 0..FFMK as usize {
											let ffm_weight = self.weights.get_unchecked(left_hash_hash + vv + k).weight;
											let contra_weight = *contra_fields.get_unchecked(contra_offset + vv + k);
											let gradient =  LEFT_HASH_VALUE * contra_weight;
											*$local_data_ffm_values.get_unchecked_mut(ffm_values_offset + k) = gradient;
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

                        let wsum_output = wsum_input + wsum;
                        if fb.audit_mode {
                            self.audit_forward(wsum_input, wsum_output, fb);
                        }

                        let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
                        let (prediction_probability, general_gradient) = next_regressor[0].forward_backward(further_blocks, wsum_output, fb, update);



                        if update {
                            let mut local_index: usize = 0;
                            for left_hash in &fb.ffm_buffer {
                                let mut feature_index = left_hash.hash as usize;
                                for j in 0..fc as usize {
                                    let feature_value = *$local_data_ffm_values.get_unchecked(local_index);
                                    let gradient = general_gradient * feature_value;
                                    let update = self.optimizer_ffm.calculate_update(gradient, &mut self.weights.get_unchecked_mut(feature_index).optimizer_data);
                                    self.weights.get_unchecked_mut(feature_index).weight += update;
                                    local_index += 1;
                                    feature_index += 1;
                                }
                            }
                        }
                        // The only exit point
                        return (prediction_probability, general_gradient)

                    });

                }
            }; // End of macro

            if local_data_ffm_len < FFM_STACK_BUF_LEN {
                // Fast-path - using on-stack data structures
                let mut local_data_ffm_values: [f32; FFM_STACK_BUF_LEN as usize] =
                    MaybeUninit::uninit().assume_init(); //[0.0; FFM_STACK_BUF_LEN as usize];
                core_macro!(local_data_ffm_values);
            } else {
                // Slow-path - using heap data structures
                if local_data_ffm_len > self.local_data_ffm_values.len() {
                    self.local_data_ffm_values
                        .reserve(local_data_ffm_len - self.local_data_ffm_values.len() + 1024);
                }
                core_macro!(self.local_data_ffm_values);
            }
        } // unsafe end
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        wsum_input: f32,
        fb: &feature_buffer::FeatureBuffer,
    ) -> f32 {
        let mut wsum: f32 = 0.0;
        unsafe {
            let ffm_weights = &self.weights;
            if true {
                _mm_prefetch(
                    mem::transmute::<&f32, &i8>(
                        &ffm_weights
                            .get_unchecked(fb.ffm_buffer.get_unchecked(0).hash as usize)
                            .weight,
                    ),
                    _MM_HINT_T0,
                );
                let field_embedding_len = self.field_embedding_len as usize;
                let mut contra_fields: [f32; FFM_STACK_BUF_LEN] =
                    MaybeUninit::uninit().assume_init();

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
                        if ffm_buffer_index >= fb.ffm_buffer.len()
                            || fb
                            .ffm_buffer
                            .get_unchecked(ffm_buffer_index)
                            .contra_field_index
                            > field_index_ffmk
                        {
                            for z in 0..field_embedding_len as usize {
                                // first time we see this field - just overwrite
                                *contra_fields.get_unchecked_mut(offset + z) = 0.0;
                            }
                            continue;
                        }
                        let mut feature_num = 0;
                        while ffm_buffer_index < fb.ffm_buffer.len()
                            && fb
                            .ffm_buffer
                            .get_unchecked(ffm_buffer_index)
                            .contra_field_index
                            == field_index_ffmk
                        {
                            _mm_prefetch(
                                mem::transmute::<&f32, &i8>(
                                    &ffm_weights
                                        .get_unchecked(
                                            fb.ffm_buffer.get_unchecked(ffm_buffer_index + 1).hash
                                                as usize,
                                        )
                                        .weight,
                                ),
                                _MM_HINT_T0,
                            );
                            let left_hash = fb.ffm_buffer.get_unchecked(ffm_buffer_index);
                            let left_hash_hash = left_hash.hash as usize;
                            let left_hash_value = left_hash.value;
                            specialize_1f32!(left_hash_value, LEFT_HASH_VALUE, {
                                if feature_num == 0 {
                                    for z in 0..field_embedding_len {
                                        // first feature of the field - just overwrite
                                        *contra_fields.get_unchecked_mut(offset + z) =
                                            ffm_weights.get_unchecked(left_hash_hash + z).weight
                                            * LEFT_HASH_VALUE;
                                    }
                                } else {
                                    for z in 0..field_embedding_len {
                                        // additional features of the field - addition
                                        *contra_fields.get_unchecked_mut(offset + z) +=
                                            ffm_weights.get_unchecked(left_hash_hash + z).weight
                                            * LEFT_HASH_VALUE;
                                    }
                                }
                                let vv = SQRT_OF_ONE_HALF * LEFT_HASH_VALUE; // To avoid one additional multiplication, we square root 0.5 into vv
                                for k in 0..FFMK as usize {
                                    let ss = ffm_weights
                                        .get_unchecked(
                                            left_hash_hash + field_index_ffmk as usize + k,
                                        )
                                        .weight
                                        * vv;
                                    *wsumbuf.get_unchecked_mut(k) -= ss * ss;
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
                        // This is self-interaction
                        for k in 0..FFMK as usize {
                            let v = contra_fields.get_unchecked(f1_offset_ffmk + k);
                            *wsumbuf.get_unchecked_mut(k) += v * v * 0.5;
                        }

                        for f2 in f1 + 1..fb.ffm_fields_count as usize {
                            f2_offset_ffmk += field_embedding_len as usize;
                            f1_offset_ffmk += FFMK as usize;
                            //assert_eq!(f1_offset_ffmk, f1 * field_embedding_len + f2 * FFMK as usize);
                            //assert_eq!(f2_offset_ffmk, f2 * field_embedding_len + f1 * FFMK as usize);
                            for k in 0..FFMK {
                                *wsumbuf.get_unchecked_mut(k as usize) += contra_fields
                                    .get_unchecked(f1_offset_ffmk + k as usize)
                                    * contra_fields.get_unchecked(f2_offset_ffmk + k as usize);
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
                        for right_hash in fb.ffm_buffer.get_unchecked(i + 1..).iter() {
                            //if left_hash.contra_field_index == right_hash.contra_field_index {
                            //    continue	// not combining within a field
                            //}
                            let joint_value = left_hash.value * right_hash.value;
                            let lindex = (left_hash.hash + right_hash.contra_field_index) as u32;
                            let rindex = (right_hash.hash + left_hash.contra_field_index) as u32;
                            for k in 0..FFMK {
                                let left_hash_weight =
                                    ffm_weights.get_unchecked((lindex + k) as usize).weight;
                                let right_hash_weight =
                                    ffm_weights.get_unchecked((rindex + k) as usize).weight;
                                *wsumbuf.get_unchecked_mut(k as usize) +=
                                    left_hash_weight * right_hash_weight * joint_value;
                            }
                        }
                    }
                    for k in 0..FFMK as usize {
                        wsum += wsumbuf[k];
                    }
                });
            }
        }

        let wsum_output = wsum_input + wsum;

        if fb.audit_mode {
            self.audit_forward(wsum_input, wsum_output, fb);
        }
        let (next_regressor, further_blocks) = further_blocks.split_at(1);
        let prediction_probability = next_regressor[0].forward(further_blocks, wsum_output, fb);
        prediction_probability
    }

    fn audit_forward(&self, wsum_input: f32, wsum_output: f32, fb: &feature_buffer::FeatureBuffer) {
        let mut map = Map::new();
        map.insert("_type".to_string(), Value::String("BlockFFM".to_string()));
        let mut features: Vec<Value> = Vec::new();
        let mut counter = 0;
        let mut mapVals = fb.fhash.borrow_mut();

        for (val, namespace_index) in fb.ffm_buffer.iter().zip(fb.ffm_buffer_audit.iter()) {
            counter += 1;
			
            let feature_hash_index = val.hash;
            let mut feature_value = val.value;			
			let feature_bin_value = val.bin_value;			
            let mut contra_fields: Vec<Value> = Vec::new();
			
            for contra_field_index in 0..fb.ffm_fields_count as usize {
                let mut weights_vec: Vec<Value> = Vec::new();
                let mut optimizer_data_vec: Vec<Value> = Vec::new();
                for k in 0..self.ffm_k as usize {
                    let weight = &self.weights[feature_hash_index as usize
											   + contra_field_index * self.ffm_k as usize
											   + k];
                    weights_vec.push(f32_to_json(weight.weight));
                    optimizer_data_vec
                        .push(self.optimizer_ffm.get_audit_data(&weight.optimizer_data));
                }
                contra_fields.push(json!({
                    "contra_field": fb.audit_aux_data.field_index_to_string[&(contra_field_index as u32)],
                    "weight": weights_vec,
                    "optimizer_data": optimizer_data_vec,
                }));
            }

			match feature_bin_value {
				Some(x) => {
					features.push(json!({
						"index": feature_hash_index,
						"value": feature_value,
						"floor_int_value": feature_bin_value.unwrap(),
						"feature": namespace_index,
						"weights": contra_fields,
					}));
					
				},
				None => {
					features.push(json!({
						"index": feature_hash_index,
						"value": feature_value,
						"feature": namespace_index,
						"weights": contra_fields,
					}));
				},
			}			
        }
        map.insert("input".to_string(), Value::Array(features));
        map.insert("output".to_string(), f32_to_json(wsum_output));
        fb.add_audit_json(map);
    }

    fn get_serialized_len(&self) -> usize {
        return self.ffm_weights_len as usize;
    }

    fn read_weights_from_buf(
        &mut self,
        input_bufreader: &mut dyn io::Read,
    ) -> Result<(), Box<dyn Error>> {
        block_helpers::read_weights_from_buf(&mut self.weights, input_bufreader)
    }

    fn write_weights_to_buf(
        &self,
        output_bufwriter: &mut dyn io::Write,
    ) -> Result<(), Box<dyn Error>> {
        block_helpers::write_weights_to_buf(&self.weights, output_bufwriter)
    }

    fn read_weights_from_buf_into_forward_only(
        &self,
        input_bufreader: &mut dyn io::Read,
        forward: &mut Box<dyn BlockTrait>,
    ) -> Result<(), Box<dyn Error>> {
        let mut forward = forward
            .as_any()
            .downcast_mut::<BlockFFM<optimizer::OptimizerSGD>>()
            .unwrap();
        block_helpers::read_weights_only_from_buf2::<L>(
            self.ffm_weights_len as usize,
            &mut forward.weights,
            input_bufreader,
        )
    }

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(
        &mut self,
        aa: i32,
        bb: i32,
        index: usize,
        w: &[f32],
    ) -> Result<(), Box<dyn Error>> {
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
    use serde_json::to_string_pretty;

    use crate::assert_epsilon;

    fn ffm_vec(
        v: Vec<feature_buffer::HashAndValueAndSeq>,
        ffm_fields_count: u32,
    ) -> feature_buffer::FeatureBuffer {
        let mut fb = feature_buffer::FeatureBuffer::new();
        fb.ffm_buffer = v;
        fb.ffm_fields_count = ffm_fields_count;
        fb
    }

    fn ffm_init<T: OptimizerTrait + 'static>(block_ffm: &mut Box<dyn BlockTrait>) -> () {
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

        let fb = ffm_vec(
            vec![HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            }],
            1,
        ); // saying we have 1 field isn't entirely correct
        assert_epsilon!(spredict(&mut re, &mut lossf, &fb, true), 0.5);
        assert_epsilon!(slearn(&mut re, &mut lossf, &fb, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        let mut re = BlockFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let fb = ffm_vec(
            vec![
                HashAndValueAndSeq {
                    hash: 1,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 100,
                    value: 1.0,
                    contra_field_index: mi.ffm_k,
                },
            ],
            2,
        );
        assert_epsilon!(spredict(&mut re, &mut lossf, &fb, true), 0.7310586);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.7310586);

        assert_epsilon!(spredict(&mut re, &mut lossf, &fb, true), 0.7024794);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.7024794);

        // Two fields, use values
        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut re);
        let fb = ffm_vec(
            vec![
                HashAndValueAndSeq {
                    hash: 1,
                    value: 2.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 100,
                    value: 2.0,
                    contra_field_index: mi.ffm_k * 1,
                },
            ],
            2,
        );
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.98201376);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.98201376);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.81377685);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.81377685);
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
        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        let fb = ffm_vec(
            vec![HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            }],
            4,
        );
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.5);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.5);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.5);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        let mut re = BlockFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let fb = ffm_vec(
            vec![
                HashAndValueAndSeq {
                    hash: 1,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 100,
                    value: 1.0,
                    contra_field_index: mi.ffm_k * 1,
                },
            ],
            2,
        );
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.98201376);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.98201376);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.96277946);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.96277946);

        // Two fields, use values
        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut re);
        let fb = ffm_vec(
            vec![
                HashAndValueAndSeq {
                    hash: 1,
                    value: 2.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 100,
                    value: 2.0,
                    contra_field_index: mi.ffm_k * 1,
                },
            ],
            2,
        );
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.9999999);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.9999999);
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.99685884);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.99685884);
    }

    #[test]
    fn test_audit() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();

        // Now prepare the "reverse resolution" data for auditing
        let vw_map_string = r#"
A,featureA
B,featureB
C,featureC
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string, (vec![], 0)).unwrap();

        mi.ffm_fields = vec![vec![0], vec![1, 2]]; // we need this in the test in order to know which fields to output

        mi.enable_audit(&vw);

        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;

        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 4;
        mi.ffm_bit_precision = 18;
        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();

        let mut re = BlockFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let mut fb = ffm_vec(
            vec![
                HashAndValueAndSeq {
                    hash: 1,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 100,
                    value: 1.0,
                    contra_field_index: mi.ffm_k * 1,
                },
            ],
            2,
        );
        fb.ffm_buffer_audit.push(0); // we have Feature A
        fb.ffm_buffer_audit.push(1); // we have Feature B
        // We don't have feature C in the input

        fb.audit_aux_data = mi.audit_aux_data.as_ref().unwrap().clone();

        fb.audit_mode = true;
        fb.reset_audit_json();
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.98201376);
        //        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.98201376);
        let audit2 = format!("{}", to_string_pretty(&fb.audit_json).unwrap());
        println!("audit: {}", audit2);
        fb.reset_audit_json();
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.98201376);
        //        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.98201376);
        let audit2 = format!("{}", to_string_pretty(&fb.audit_json).unwrap());
        println!("audit@22222222222222222: {}", audit2);
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
        mi.ffm_fields = vec![vec![], vec![]];
        mi.optimizer = model_instance::Optimizer::Adagrad;
        mi.fastmath = false;
        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();

        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);
        let mut p: f32;

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut re);
        let fbuf = &ffm_vec(
            vec![
                HashAndValueAndSeq {
                    hash: 1,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 3 * 1000,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 100,
                    value: 2.0,
                    contra_field_index: mi.ffm_k * 1,
                },
            ],
            2,
        );
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
        mi.ffm_fields = vec![vec![], vec![]];
        mi.optimizer = model_instance::Optimizer::Adagrad;
        mi.fastmath = false;
        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();

        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);
        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut re);
        let fbuf = &ffm_vec(
            vec![
                HashAndValueAndSeq {
                    hash: 1,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 3 * 1000,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 100,
                    value: 2.0,
                    contra_field_index: mi.ffm_k * 1,
                },
            ],
            2,
        );

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
        let mut re = BlockFFM::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        let mut re = BlockFFM::<optimizer::OptimizerAdagradFlex>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut re);
        let fb = ffm_vec(
            vec![
                HashAndValueAndSeq {
                    hash: 1,
                    value: 1.0,
                    contra_field_index: 0,
                },
                HashAndValueAndSeq {
                    hash: 5,
                    value: 1.0,
                    contra_field_index: mi.ffm_k * 1,
                },
                HashAndValueAndSeq {
                    hash: 100,
                    value: 1.0,
                    contra_field_index: mi.ffm_k * 2,
                },
            ],
            3,
        );
        assert_epsilon!(spredict(&mut re, &mut lossf, &fb, true), 0.95257413);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, false), 0.95257413);

        // here we intentionally have just the middle field
        let fb = ffm_vec(
            vec![HashAndValueAndSeq {
                hash: 5,
                value: 1.0,
                contra_field_index: mi.ffm_k * 1,
            }],
            3,
        );
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.5);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.5);
    }
}
