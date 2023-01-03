use std::any::Any;
use std::io;
use merand48::*;
use core::arch::x86_64::*;
use std::error::Error;
use std::mem::{self, MaybeUninit};
use std::f32::consts::PI;

use crate::optimizer;
use crate::regressor;
use crate::model_instance;
use crate::feature_buffer;
use crate::port_buffer;
use crate::consts;
use crate::block_helpers;
use crate::graph;
use crate::graph::BlockGraph;

use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use block_helpers::{WeightAndOptimizerData};


const FFM_STACK_BUF_LEN:usize= 32768;
const FFM_CONTRA_BUF_LEN:usize = 16384;


const SQRT_OF_ONE_HALF:f32 = 0.70710678118;


pub struct BlockFFM<L:OptimizerTrait> {
    pub optimizer_ffm: L,
    pub local_data_ffm_values: Vec<f32>,
    pub ffm_k: u32,
    pub ffm_weights_len: u32,
    pub ffm_num_fields: u32,
    pub field_embedding_len: u32,
    pub weights: Vec<WeightAndOptimizerData<L>>,
    pub output_offset: usize,
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
// TODO UNCOMMENT USEFUL ONES
//                2 => {const $output_const:u32 = 2;   let mut $wsumbuf: [f32;$output_const as usize] = [0.0;$output_const as usize]; $code_block},
                4 => {const $output_const:u32 = 4;   let mut $wsumbuf: [f32;$output_const as usize] = [0.0;$output_const as usize]; $code_block},
                8 => {const $output_const:u32 = 8;   let mut $wsumbuf: [f32;$output_const as usize] = [0.0;$output_const as usize]; $code_block},
                val => {let $output_const:u32 = val; let mut $wsumbuf: [f32;consts::FFM_MAX_K] = [0.0;consts::FFM_MAX_K];      $code_block},
            }
    };
}


impl<L: OptimizerTrait + 'static> BlockFFM<L> {
    fn set_weights(&mut self, lower_bound: f32, difference: f32) {
        for i in 0..self.ffm_weights_len {
            let w = difference * merand48(i as u64) as f32 + lower_bound;
            self.weights[i as usize].weight = w;
            self.weights[i as usize].optimizer_data = self.optimizer_ffm.initial_data();
        }
    }
}

pub fn new_ffm_block(
                        bg: &mut graph::BlockGraph,
                        mi: &model_instance::ModelInstance)
                        -> Result<graph::BlockPtrOutput, Box<dyn Error>> {

    let block = match mi.optimizer {
        model_instance::Optimizer::AdagradLUT => new_ffm_block_without_weights::<optimizer::OptimizerAdagradLUT>(&mi),
        model_instance::Optimizer::AdagradFlex => new_ffm_block_without_weights::<optimizer::OptimizerAdagradFlex>(&mi),
        model_instance::Optimizer::SGD => new_ffm_block_without_weights::<optimizer::OptimizerSGD>(&mi)
    }.unwrap();
    let mut block_outputs = bg.add_node(block, vec![]).unwrap();
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}



fn new_ffm_block_without_weights<L:OptimizerTrait + 'static>(mi: &model_instance::ModelInstance) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {

    let ffm_num_fields = mi.ffm_fields.len() as u32;
    let mut reg_ffm = BlockFFM::<L> {
        weights: Vec::new(),
        ffm_weights_len: 0,
        local_data_ffm_values: Vec::with_capacity(1024),
        ffm_k: mi.ffm_k,
        ffm_num_fields: ffm_num_fields,
        field_embedding_len: mi.ffm_k * ffm_num_fields,
        optimizer_ffm: L::new(),
        output_offset: usize::MAX,
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







impl <L:OptimizerTrait + 'static> BlockTrait for BlockFFM<L> {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }


    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights =vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.optimizer_ffm.initial_data()}; self.ffm_weights_len as usize];
	let initialization_type_fixed = "default";
		match initialization_type_fixed {
			"default" => {
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
			},
			"xavier_custom_mask" => {
				// MASK[U [-(1/sqrt(n)), 1/sqrt(n)]]
				let lower_bound: f32 = -1.0/(self.ffm_weights_len as f32).sqrt();
				let upper_bound: f32 = 1.0/(self.ffm_weights_len as f32).sqrt();
				let difference = upper_bound - lower_bound;

				for i in 0..self.ffm_weights_len {
					let mut w = merand48(i as u64) as f32;
					if w < (self.ffm_weights_len / 10) as f32 {
						w = 0.001;
					} else {
						w = 0.0;
					}
					self.weights[i as usize].weight = w;
					self.weights[i as usize].optimizer_data = self.optimizer_ffm.initial_data();
				}
			},
			"xavier" => {
				// U [-(1/sqrt(n)), 1/sqrt(n)]
				let lower_bound: f32 = -1.0/(self.ffm_weights_len as f32).sqrt();
				let upper_bound: f32 = 1.0/(self.ffm_weights_len as f32).sqrt();
				let difference = upper_bound - lower_bound;
				self.set_weights(lower_bound, difference);

			},
			"xavier_normalized" => {
				// U [-(sqrt(6)/sqrt(n + m)), sqrt(6)/sqrt(n + m)] + we assume symmetric input-output

				let lower_bound: f32 = -6_f32.sqrt() / (2 * self.ffm_weights_len) as f32;
				let upper_bound: f32 = 6_f32.sqrt() / (2 * self.ffm_weights_len) as f32;
				let difference = upper_bound - lower_bound;
				self.set_weights(lower_bound, difference);

			},
			"he" => {
				// G (0.0, sqrt(2/n)) + Box Muller-ish transform

				for i in 0..self.ffm_weights_len {

					// could use both, but not critical in this case
					let seed_var_first = merand48(i as u64);
					let seed_var_second = merand48(u64::pow(i as u64, 2));
					let normal_var = (-2.0 * seed_var_first.ln()).sqrt() * (2.0 * PI as f32 * seed_var_second).cos();

					self.weights[i as usize].weight = normal_var + (2.0 / self.ffm_weights_len as f32).sqrt();
					self.weights[i as usize].optimizer_data = self.optimizer_ffm.initial_data();
				}

			},
			"constant" => {
				// generic constant initialization (sanity check)

				for i in 0..self.ffm_weights_len {
					let mut w = 1.0;
					self.weights[i as usize].weight = w;
					self.weights[i as usize].optimizer_data = self.optimizer_ffm.initial_data();
				}

			},
			_ => {panic!("Please select a valid activation function.")}
		}
    }


    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert!(output.get_output_index() == 0);
        return (self.ffm_num_fields * self.ffm_num_fields) as usize;
    }

    fn get_num_output_slots(&self) -> usize { 1 }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {
        panic!("You cannnot set_input_offset() for BlockFFM");
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert!(output.get_output_index() == 0);
        self.output_offset = offset;
    }


    #[inline(always)]
    fn forward_backward(&mut self, 
                        further_blocks: &mut [Box<dyn BlockTrait>],
                        fb: &feature_buffer::FeatureBuffer,
                        pb: &mut port_buffer::PortBuffer,
                        update:bool) {
        debug_assert!(self.output_offset != usize::MAX);

        let mut wsum = 0.0;
        let local_data_ffm_len = fb.ffm_buffer.len() * (self.ffm_k * fb.ffm_fields_count) as usize;

        unsafe {

            macro_rules! core_macro {
                (
                $local_data_ffm_values:ident
                ) => {
                    // number of outputs
                    let num_outputs = (self.ffm_num_fields * self.ffm_num_fields) as usize;
                    let myslice = &mut pb.tape[self.output_offset .. (self.output_offset + num_outputs)];
                    myslice.fill(0.0); // TODO : is this really needed?

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
                            let contra_offset = (left_hash.contra_field_index * fb.ffm_fields_count) as usize;
                            let mut vv = 0;
                            let contra_offset2 = contra_offset / FFMK as usize;
                            let left_hash_value = left_hash.value;
                            let left_hash_contra_field_index = left_hash.contra_field_index;
                            let left_hash_hash = left_hash.hash as usize;
                            //let LEFT_HASH_VALUE = left_hash_value;
                            specialize_1f32!(left_hash_value, LEFT_HASH_VALUE, {

                              for z in 0..fb.ffm_fields_count as usize {
                                  if vv == left_hash_contra_field_index as usize {
                                      for k in 0..FFMK as usize {
                                          let ffm_weight = ffm_weights.get_unchecked(left_hash_hash + vv + k).weight;
                                          let contra_weight = *contra_fields.get_unchecked(contra_offset + vv + k) - ffm_weight * LEFT_HASH_VALUE;
                                          let gradient =  LEFT_HASH_VALUE * contra_weight;
                                          *local_data_ffm_values.get_unchecked_mut(ffm_values_offset + k) = gradient;
                                          *myslice.get_unchecked_mut( contra_offset2 + z ) += ffm_weight * gradient * 0.5;
                                      }
                                  } else {
                                      for k in 0..FFMK as usize {
                                          let ffm_weight = ffm_weights.get_unchecked(left_hash_hash + vv + k).weight;
                                          let contra_weight = *contra_fields.get_unchecked(contra_offset + vv + k);
                                          let gradient =  LEFT_HASH_VALUE * contra_weight;
                                          *local_data_ffm_values.get_unchecked_mut(ffm_values_offset + k) = gradient;
                                          *myslice.get_unchecked_mut(contra_offset2 + z ) += ffm_weight * gradient * 0.5;
                                      }
                                  }
                                  vv += FFMK as usize;
                                  //left_hash_hash += FFMK as usize;
                                  //contra_offset += FFMK as usize;
                                  ffm_values_offset += FFMK as usize;
                              }
                            }); // End of macro specialize_1f32! for LEFT_HASH_VALUE
                        }
                    });
                        
                    block_helpers::forward_backward(further_blocks, fb, pb, update);

                    if update {
                        let mut local_index: usize = 0;
                        let myslice = &mut pb.tape[self.output_offset..(self.output_offset + num_outputs)];

                        let wsumbuf: bool;
                        specialize_k!(self.ffm_k, FFMK, wsumbuf, {
                            for left_hash in &fb.ffm_buffer {
                                let mut feature_index = left_hash.hash as usize;
                                let mut contra_offset = (left_hash.contra_field_index * fb.ffm_fields_count) as usize;
                                let mut contra_offset2 = contra_offset / FFMK as usize;

                                for z in 0..fb.ffm_fields_count as usize {
                                    let general_gradient = myslice.get_unchecked(contra_offset2 + z);
                                    for k in 0..FFMK as usize {
                                        let feature_value = *local_data_ffm_values.get_unchecked(local_index);
                                        let gradient = general_gradient * feature_value;
                                        let update = self.optimizer_ffm.calculate_update(gradient, &mut ffm_weights.get_unchecked_mut(feature_index).optimizer_data);

                                        ffm_weights.get_unchecked_mut(feature_index).weight -= update;
                                        local_index += 1;
                                        feature_index += 1;
                                    }
                                }
                            }
                        });
                    }
                    // The only exit point
                    return
                }
            } // End of macro
            

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
    
    fn forward(&self, further_blocks: &[Box<dyn BlockTrait>],
                      fb: &feature_buffer::FeatureBuffer,
                      pb: &mut port_buffer::PortBuffer,
                      )  {
        debug_assert!(self.output_offset != usize::MAX);

        let num_outputs = (self.ffm_num_fields * self.ffm_num_fields) as usize;
        let myslice = &mut pb.tape[self.output_offset .. (self.output_offset + num_outputs)];
        myslice.fill(0.0);

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
                            let contra_offset2 = left_hash.contra_field_index / FFMK;
                            let field_embedding_len2 = field_embedding_len / FFMK as usize;
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
                                let vv = SQRT_OF_ONE_HALF * LEFT_HASH_VALUE;     // To avoid one additional multiplication, we square root 0.5 into vv
                                for k in 0..FFMK as usize {
                                    let ss = ffm_weights.get_unchecked(left_hash_hash + field_index_ffmk as usize + k).weight * vv;
                                    myslice[(contra_offset2 * (fb.ffm_fields_count + 1)) as usize] -= ss * ss;
                                }
                            });
                            ffm_buffer_index += 1;
                            feature_num += 1;
                        }
                    }
                    

                    for f1 in 0..fb.ffm_fields_count as usize {
                        let f1_offset = f1 * field_embedding_len as usize;
                        let f1_offset2 = f1 * fb.ffm_fields_count as usize;
                        let f1_ffmk = f1 * FFMK as usize;
                        let mut f2_offset_ffmk = f1_offset + f1_ffmk;
                        let mut f1_offset_ffmk = f1_offset + f1_ffmk;
                        // This is self-interaction
                        for k in 0..FFMK as usize{
                            let v = contra_fields.get_unchecked(f1_offset_ffmk + k);
                            myslice[f1_offset2 + f1] += v * v * 0.5;
                        }

                        for f2 in f1+1..fb.ffm_fields_count as usize {
                            f2_offset_ffmk += field_embedding_len as usize;
                            f1_offset_ffmk += FFMK as usize;
                            for k in 0..FFMK {
                                myslice[f1 * fb.ffm_fields_count as usize + f2] +=
                                        contra_fields.get_unchecked(f1_offset_ffmk + k as usize) *
                                        contra_fields.get_unchecked(f2_offset_ffmk + k as usize) * 0.5;
                                myslice[f2 * fb.ffm_fields_count as usize + f1] +=
                                        contra_fields.get_unchecked(f1_offset_ffmk + k as usize) * 
                                        contra_fields.get_unchecked(f2_offset_ffmk + k as usize) * 0.5;

                            }
                        }
                        
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
                });
            }
        }
        block_helpers::forward(further_blocks, fb, pb);
                 
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
    use crate::block_loss_functions;
    use crate::model_instance::Optimizer;
    use crate::feature_buffer;
    use crate::feature_buffer::HashAndValueAndSeq;
    use crate::vwmap;
    use block_helpers::{slearn2, spredict2};

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
        mi.optimizer = Optimizer::AdagradLUT;



        // Nothing can be learned from a single field in FFMs
        let mut bg = BlockGraph::new();
        let ffm_block = new_ffm_block(&mut bg, &mi).unwrap();
        let loss_block = block_loss_functions::new_logloss_block(&mut bg, ffm_block, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);
        let mut pb = bg.new_port_buffer();

        let fb = ffm_vec(vec![HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0}], 
                        1); // saying we have 1 field isn't entirely correct
        assert_epsilon!(spredict2(&mut bg, &fb, &mut pb, true), 0.5);
        assert_epsilon!(slearn2  (&mut bg, &fb, &mut pb, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        mi.optimizer = Optimizer::AdagradFlex;
        let mut bg = BlockGraph::new();

        let ffm_block = new_ffm_block(&mut bg, &mi).unwrap();
        let lossf = block_loss_functions::new_logloss_block(&mut bg, ffm_block, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);
        let mut pb = bg.new_port_buffer();

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut bg.blocks_final[0]);
        let fb = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: mi.ffm_k}
                                  ], 2);
        assert_epsilon!(spredict2(&mut bg, &fb, &mut pb, true), 0.7310586);
        assert_eq!(slearn2  (&mut bg, &fb, &mut pb, true), 0.7310586);
        
        assert_epsilon!(spredict2(&mut bg, &fb, &mut pb,true), 0.7024794);
        assert_eq!(slearn2  (&mut bg, &fb, &mut pb, true), 0.7024794);

        // Two fields, use values
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let fb = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 2.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: mi.ffm_k * 1}
                                  ], 2);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb,true), 0.98201376);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.98201376);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb,true), 0.81377685);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.81377685);
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

        // Nothing can be learned from a single field in FFMs
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);



        let mut pb = bg.new_port_buffer();

        let fb = ffm_vec(vec![HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0}], 1);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb,true), 0.5);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.5);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb,true), 0.5);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        mi.optimizer = Optimizer::AdagradFlex;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut bg.blocks_final[0]);
        let fb = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: mi.ffm_k * 1}
                                  ], 2);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb,true), 0.98201376);
        assert_eq!(slearn2  (&mut bg, &fb, &mut pb, true), 0.98201376);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb,true), 0.96277946);
        assert_eq!(slearn2  (&mut bg, &fb, &mut pb, true), 0.96277946);

        // Two fields, use values
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let fb = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 2.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: mi.ffm_k * 1}
                                  ], 2);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb,true), 0.9999999);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.9999999);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb,true), 0.99685884);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.99685884);
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

        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);


        let mut pb = bg.new_port_buffer();

        let mut p: f32;

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let fbuf = &ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:3 * 1000, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: mi.ffm_k * 1}
                                  ], 2);
        assert_epsilon!(spredict2(&mut bg, &fbuf, &mut pb,true), 0.9933072);
        assert_eq!(slearn2(&mut bg, &fbuf, &mut pb, true), 0.9933072);
        assert_epsilon!(spredict2(&mut bg, &fbuf, &mut pb,false), 0.9395168);
        assert_eq!(slearn2(&mut bg, &fbuf, &mut pb, false), 0.9395168);
        assert_epsilon!(spredict2(&mut bg, &fbuf, &mut pb,false), 0.9395168);
        assert_eq!(slearn2(&mut bg, &fbuf, &mut pb, false), 0.9395168);
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

        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let fbuf = &ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:3 * 1000, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: mi.ffm_k * 1}
                                  ], 2);

        assert_eq!(spredict2(&mut bg, &fbuf,&mut pb, true), 1.0);
        assert_eq!(slearn2(&mut bg, &fbuf, &mut pb, true), 1.0);
        assert_eq!(spredict2(&mut bg, &fbuf,&mut pb, false), 0.9949837);
        assert_eq!(slearn2(&mut bg, &fbuf, &mut pb, false), 0.9949837);
        assert_eq!(slearn2(&mut bg, &fbuf, &mut pb, false), 0.9949837);
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

        // Nothing can be learned from a single field in FFMs
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);


        let mut pb = bg.new_port_buffer();


        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        mi.optimizer = Optimizer::AdagradFlex;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut bg.blocks_final[0]);
        let fb = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:5, value: 1.0, contra_field_index: mi.ffm_k * 1},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: mi.ffm_k * 2}
                                  ], 3);
        assert_epsilon!(spredict2(&mut bg, &fb, &mut pb, true), 0.95257413);
        assert_eq!(slearn2  (&mut bg, &fb, &mut pb, false), 0.95257413);

        // here we intentionally have just the middle field
        let fb = ffm_vec(vec![HashAndValueAndSeq{hash:5, value: 1.0, contra_field_index: mi.ffm_k * 1}], 3);
        assert_eq!(spredict2(&mut bg, &fb,&mut pb,true), 0.5);
        assert_eq!(slearn2  (&mut bg, &fb, &mut pb, true), 0.5);

    }
}
