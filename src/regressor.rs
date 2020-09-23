use std::mem::{self, MaybeUninit};
//use fastapprox::fast::sigmoid; // surprisingly this doesn't work very well
use std::process;
use std::sync::Arc;
use core::arch::x86_64::*;
use merand48::*;

use crate::model_instance;
use crate::feature_buffer;
use crate::feature_buffer::HashAndValue;
use crate::feature_buffer::HashAndValueAndSeq;
use crate::learning_rate;
use learning_rate::LearningRateTrait;

#[derive(Clone, Debug)]
pub struct Weight {
    pub weight: f32, 
    pub acc_grad: f32,
}


pub struct IndexAccgradientValue {
    index: u32,
    accgradient: f32,
    value: f32,
}
pub struct IndexAccgradientValueFFM {
    index: u32,
    value: f32,
}

pub struct Regressor {
    hash_mask: u32,
    learning_rate: f32,
    minus_power_t:f32,
    pub weights: Vec<Weight>,       // all weights and gradients (has sub-spaces)
    pub ffm_weights_offset: u32, 
    ffm_k: u32,
    ffm_hashmask: u32,
    ffm_one_over_k_root: f32,
    fastmath: bool,
    ffm_iw_weights_offset: u32,
    ffm_k_threshold: f32,
    adagrad_lr: learning_rate::LearningRateAdagradLUT,
    adagrad_ffm: learning_rate::LearningRateAdagradLUT,
    local_data_lr: Vec<IndexAccgradientValue>,
    local_data_ffm: Vec<IndexAccgradientValueFFM>,
}

#[derive(Clone)]
pub struct FixedRegressor {
    hash_mask: u32,
    pub weights: Arc<Vec<Weight>>,
    ffm_weights_offset: u32, 
    ffm_k: u32,
    ffm_hashmask: u32,
}


macro_rules! specialize_boolean {
    ( $input_expr:expr, 
      $output_const:ident,
      $code_block:block  ) => {
         match $input_expr {
                true => {const $output_const:bool = true; $code_block},
                false => {const $output_const:bool = false; $code_block},
            }
    };
}
macro_rules! specialize_k {
    ( $input_expr:expr, 
      $output_const:ident,
      $code_block:block  ) => {
         match $input_expr {
                2 => {const $output_const:u32 = 2; $code_block},
                4 => {const $output_const:u32 = 4; $code_block},
                8 => {const $output_const:u32 = 8; $code_block},
                val => {let $output_const:u32 = val; $code_block},
            }
    };
}

impl Regressor {
    
    pub fn new(model_instance: &model_instance::ModelInstance) -> Regressor {
        let hash_mask = (1 << model_instance.bit_precision) -1;
        let lr_weights_len = hash_mask + 1;
        let mut rg = Regressor{
                            hash_mask: hash_mask,
                            learning_rate: model_instance.learning_rate,
                            minus_power_t : - model_instance.power_t,
                            //minimum_learning_rate: model_instance.minimum_learning_rate,
                            weights: Vec::new(), 
                            ffm_weights_offset: 0,
                            ffm_k: 0,
                            ffm_hashmask: 0,
                            ffm_one_over_k_root: 0.0,
                            fastmath: model_instance.fastmath,
                            adagrad_lr: learning_rate::LearningRateAdagradLUT {fastmath_lr_lut: [0.0; learning_rate::FASTMATH_LR_LUT_SIZE]},
                            adagrad_ffm: learning_rate::LearningRateAdagradLUT {fastmath_lr_lut: [0.0; learning_rate::FASTMATH_LR_LUT_SIZE]},
                            ffm_iw_weights_offset: 0,
                            ffm_k_threshold: model_instance.ffm_k_threshold,
                            local_data_lr: Vec::with_capacity(1024),
                            local_data_ffm: Vec::with_capacity(1024), 
                        };

        if rg.fastmath {
            rg.adagrad_lr.init(rg.learning_rate, rg.minus_power_t);
            rg.adagrad_ffm.init(model_instance.ffm_learning_rate, -model_instance.ffm_power_t);
        }

        let mut ffm_weights_len = 0;
        if model_instance.ffm_k > 0 {
            rg.ffm_weights_offset = lr_weights_len;            // Since we will align our dimensions, we need to know the number of bits for them
            rg.ffm_k = model_instance.ffm_k;
            // At the end we add "spillover buffer", so we can do modulo only on the base address and add offset
            ffm_weights_len = (1 << model_instance.ffm_bit_precision) + (model_instance.ffm_fields.len() as u32 * rg.ffm_k);
            let mut ffm_bits_for_dimensions = 0;
            while rg.ffm_k > (1 << (ffm_bits_for_dimensions)) {
                ffm_bits_for_dimensions += 1;
            }
            let dimensions_mask = (1 << ffm_bits_for_dimensions) - 1;
            // in ffm we will simply mask the lower bits, so we spare them for k
            rg.ffm_hashmask = ((1 << model_instance.ffm_bit_precision) -1) ^ dimensions_mask;
        }
        // Now allocate weights
//        let iw_weights_len =rg.ffm_k * (model_instance.ffm_fields.len() as u32 * model_instance.ffm_fields.len() as u32);
        let iw_weights_len = 0;
        rg.ffm_iw_weights_offset = lr_weights_len + ffm_weights_len;
        rg.weights = vec![Weight{weight:0.0, acc_grad:0.0}; (lr_weights_len + ffm_weights_len + iw_weights_len) as usize];

        if model_instance.ffm_k > 0 {       
            if model_instance.ffm_init_width == 0.0 {
                // Initialization, from ffm.pdf with added division by 100 and centered on zero (determined empirically)
                rg.ffm_one_over_k_root = 1.0 / (rg.ffm_k as f32).sqrt() / 10.0;
                for i in 0..ffm_weights_len {
                    rg.weights[(rg.ffm_weights_offset + i) as usize].weight = (0.2*merand48((rg.ffm_weights_offset+i) as u64)-0.1) * rg.ffm_one_over_k_root;
                    //rng.gen_range(-0.1 * rg.ffm_one_over_k_root , 0.1 * rg.ffm_one_over_k_root );
                    // we set FFM gradients to 1.0, so we avoid NaN updates due to adagrad (accumulated_squared_gradients+grad^2).powf(negative_number) * 0.0 
                    rg.weights[(rg.ffm_weights_offset + i) as usize].acc_grad = 1.0;
                }
            } else {
                for i in 0..ffm_weights_len {
                    rg.weights[(rg.ffm_weights_offset + i) as usize].weight = model_instance.ffm_init_center - model_instance.ffm_init_width * 0.5 + 
                                                                         merand48(i as u64) * model_instance.ffm_init_width;
                    //rng.gen_range(-0.1 * rg.ffm_one_over_k_root , 0.1 * rg.ffm_one_over_k_root );
                    // we set FFM gradients to 1.0, so we avoid NaN updates due to adagrad (accumulated_squared_gradients+grad^2).powf(negative_number) * 0.0 
                    rg.weights[(rg.ffm_weights_offset + i) as usize].acc_grad = 1.0;
                }

            }
        }
        rg
    }
    

    pub fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, update: bool, example_num: u32) -> f32 {
        unsafe {
        let y = fb.label; // 0.0 or 1.0

        let local_data_lr_len = fb.lr_buffer.len();
        if local_data_lr_len > self.local_data_lr.len() {
            self.local_data_lr.reserve(local_data_lr_len - self.local_data_lr.len() + 1024);
        }
        
        let local_data_ffm_len = fb.ffm_buffer.len() * (self.ffm_k * fb.ffm_fields_count) as usize;
        if local_data_ffm_len > self.local_data_ffm.len() {
            self.local_data_ffm.reserve(local_data_ffm_len - self.local_data_ffm.len() + 1024);
        }

        // local_data is writable, but weights are read-only in this section
        let local_data_lr = &mut self.local_data_lr;
        let local_data_ffm = &mut self.local_data_ffm;
//        let mut local_data_lr: [IndexAccgradientValue; BUF_LEN as usize] = MaybeUninit::uninit().assume_init() ;
//        let mut local_data_ffm: [IndexAccgradientValueFFM; BUF_LEN as usize] = MaybeUninit::uninit().assume_init() ;

        let weights = &self.weights;

        let mut wsum:f32 = 0.0;
        for (i, hashvalue) in fb.lr_buffer.iter().enumerate() {
//            _mm_prefetch(mem::transmute::<&f32, &i8>(weights.get_unchecked((fbuf.get_unchecked(i+8).hash << 1) as usize)), _MM_HINT_T0);  // No benefit for now
            let feature_index     = hashvalue.hash;
            let feature_value:f32 = hashvalue.value;
            let feature_weight       = weights.get_unchecked(feature_index as usize).weight;
            let accumulated_gradient = weights.get_unchecked(feature_index as usize).acc_grad;
            wsum += feature_weight * feature_value;
            local_data_lr.get_unchecked_mut(i).index       = feature_index;
            local_data_lr.get_unchecked_mut(i).accgradient = accumulated_gradient;
            local_data_lr.get_unchecked_mut(i).value       = feature_value;
            
        }

        
        if self.ffm_k > 0 {
            let fc = (fb.ffm_fields_count  * self.ffm_k) as usize;
            for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
                let base_weight_index = self.ffm_weights_offset + (left_hash.hash & self.ffm_hashmask);
                for j in 0..fc as usize {
                    let v = local_data_ffm.get_unchecked_mut((i * fc + j) as usize);
                    v.index = base_weight_index + j as u32;
                    v.value = 0.0;
                }
            }

            specialize_k!(self.ffm_k, FFMK, {
            for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
                for (j, right_hash) in fb.ffm_buffer[i+1 ..].iter().enumerate() {
                    if left_hash.contra_field_index == right_hash.contra_field_index {
                        continue	// not combining within a field
                    }
                    let left_local_index = i*fc as usize + right_hash.contra_field_index as usize;
                    let right_local_index = (i+1+j) * fc as usize + left_hash.contra_field_index as usize;
                    let joint_value = left_hash.value * right_hash.value;
                    let lindex = local_data_ffm.get_unchecked(left_local_index).index as usize;
                    let rindex = local_data_ffm.get_unchecked(right_local_index).index as usize;
                    for k in 0..FFMK as usize {
                        let llik = (left_local_index as usize + k) as usize;
                        let rlik = (right_local_index as usize + k) as usize;
                        let left_hash_weight  = weights.get_unchecked((lindex+k) as usize).weight;
                        let right_hash_weight = weights.get_unchecked((rindex+k) as usize).weight;
                        
                        let right_side = right_hash_weight * joint_value;
                        local_data_ffm.get_unchecked_mut(llik).value += right_side; // first derivate
                        local_data_ffm.get_unchecked_mut(rlik).value += left_hash_weight  * joint_value; // first derivate
                        
                        wsum += left_hash_weight * right_side;
                    }
                }
            }
        
            });
        }
        // Trick: instead of multiply in the updates with learning rate, multiply the result
        let prediction = -wsum;
        // vowpal compatibility
        if prediction.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", example_num);
            let prediction_finalized:f32 = 0.0;
            return (1.0+(prediction_finalized).exp()).recip();
        } else if prediction < -50.0 {
            let prediction_finalized:f32 = -50.0;
            return (1.0+(prediction_finalized).exp()).recip();
        } else if prediction > 50.0 {
            let prediction_finalized:f32 = 50.0;
            return (1.0+(prediction_finalized).exp()).recip();
        }

        let prediction_probability:f32 = (1.0+(prediction).exp()).recip();

        // Weights are now writable, but local_data is read only
        let weights = &mut self.weights;


        if update && fb.example_importance != 0.0 {
            let general_gradient = (y - prediction_probability) * fb.example_importance;
            specialize_boolean!(self.fastmath, FASTMATH, {
                for i in 0..local_data_lr_len {
                    let feature_value = local_data_lr.get_unchecked(i).value;
                    let feature_index = local_data_lr.get_unchecked(i).index as usize;
                    let gradient = general_gradient * feature_value;
                    let gradient_squared = gradient * gradient;
                    weights.get_unchecked_mut(feature_index).acc_grad += gradient_squared;
                    let accumulated_squared_gradient = local_data_lr.get_unchecked(i).accgradient;
                    if FASTMATH {
                        let update = self.adagrad_lr.calculate_update(gradient, accumulated_squared_gradient + gradient_squared);
                        weights.get_unchecked_mut(feature_index).weight += update;
                    } else {
                        let learning_rate = self.learning_rate * (accumulated_squared_gradient + gradient_squared).powf(self.minus_power_t);
                        let update = gradient * learning_rate;
                        weights.get_unchecked_mut(feature_index).weight += update;
                    }
                }
                
                for i in 0..local_data_ffm_len {
                    let feature_value = local_data_ffm.get_unchecked(i).value;
                    if feature_value == 0.0 {
                        continue; // this is basically diagonal of ffm - self combination
                    }
                    let feature_index = local_data_ffm.get_unchecked(i).index as usize;
                    let gradient = general_gradient * feature_value;
                    let gradient_squared = gradient * gradient;
                    let accumulated_gradient_squared = weights.get_unchecked_mut(feature_index).acc_grad;
                    let new_accumulated_gradient_squared = accumulated_gradient_squared + gradient_squared;
                    weights.get_unchecked_mut(feature_index).acc_grad = new_accumulated_gradient_squared;
                    if FASTMATH {
                        let update = self.adagrad_ffm.calculate_update(gradient, new_accumulated_gradient_squared);
                        weights.get_unchecked_mut(feature_index).weight += update;
                    } else {
                        let learning_rate = self.learning_rate * (new_accumulated_gradient_squared).powf(self.minus_power_t);
                        let update = gradient * learning_rate;
                        weights.get_unchecked_mut(feature_index).weight += update;
                    }
                }
                            
            });
        }
        
        prediction_probability
        }
    }
}

impl FixedRegressor {
    pub fn new(rr: Regressor) -> FixedRegressor {
        FixedRegressor {
                        hash_mask: rr.hash_mask,
                        weights: Arc::new(rr.weights),
                        ffm_weights_offset: rr.ffm_weights_offset,
                        ffm_k: rr.ffm_k,
                        ffm_hashmask: rr.ffm_hashmask,

        }
    }

    pub fn predict(&self, fb: &feature_buffer::FeatureBuffer, example_num: u32) -> f32 {
        let fbuf = &fb.lr_buffer;
        let mut wsum:f32 = 0.0;
        for val in fbuf {
            let hash = val.hash as usize;
            let feature_value:f32 = val.value;
            wsum += self.weights[hash].weight * feature_value;    
        }

        if self.ffm_k > 0 {
            let fc = fb.ffm_fields_count  as usize * self.ffm_k as usize;
            for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
                for (j, right_hash) in fb.ffm_buffer[i+1 ..].iter().enumerate() {
                    if left_hash.contra_field_index == right_hash.contra_field_index {
                        continue	// not combining within a field
                    }
                    let joint_value = left_hash.value * right_hash.value;
                    let lindex = (self.ffm_weights_offset as u32 + ((left_hash.hash & self.ffm_hashmask) + right_hash.contra_field_index)) as u32;
                    let rindex = (self.ffm_weights_offset as u32 + ((right_hash.hash & self.ffm_hashmask) + left_hash.contra_field_index)) as u32;
                    for k in 0..self.ffm_k {
                        let left_hash_weight  = self.weights[(lindex+k) as usize].weight;
                        let right_hash_weight = self.weights[(rindex+k) as usize].weight;
                        let right_side = right_hash_weight * joint_value;
                        wsum += left_hash_weight * right_side;
                    }
                }
            }

            
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
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        
        // Empty model: no matter how many features, prediction is 0.5
        p = rr.learn(&lr_vec(vec![]), false, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), false, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), false, 0);
        assert_eq!(p, 0.5);
    }

    #[test]
    fn test_power_t_zero() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), true, 0);
        assert_eq!(p, 0.48750263);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), true, 0);
        assert_eq!(p, 0.47533244);
    }

    #[test]
    fn test_double_same_feature() {
        // this is a tricky test - what happens on collision
        // depending on the order of math, results are different
        // so this is here, to make sure the math is always the same
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        let two = 2.0_f32.to_bits();
        
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash: 1, value: 2.0}]), true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash: 1, value: 2.0}]), true, 0);
        assert_eq!(p, 0.38936076);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash: 1, value: 2.0}]), true, 0);
        assert_eq!(p, 0.30993468);
    }


    #[test]
    fn test_power_t_half() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.bit_precision = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        
        p = rr.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true, 0);
        assert_eq!(p, 0.4750208);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true, 0);
        assert_eq!(p, 0.45788094);
    }

    #[test]
    fn test_power_t_half_fastmath() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.fastmath = true;
        mi.bit_precision = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        
        p = rr.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash:1, value: 1.0}]), true, 0);
        if learning_rate::FASTMATH_LR_LUT_BITS == 12 { 
            assert_eq!(p, 0.47539312); // when LUT = 12
        } else {
            assert_eq!(p, 0.475734);
        }
    }

    #[test]
    fn test_power_t_half_two_features() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.bit_precision = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        // Here we take twice two features and then once just one
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}, HashAndValue{hash:2, value: 1.0}]), true, 0);
        assert_eq!(p, 0.45016602);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]), true, 0);
        assert_eq!(p, 0.45836908);
    }

    #[test]
    fn test_non_one_weight() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        
        p = rr.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), true, 0);
        assert_eq!(p, 0.45016602);
        p = rr.learn(&lr_vec(vec![HashAndValue{hash:1, value: 2.0}]), true, 0);
        assert_eq!(p, 0.40611085);
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

    fn ffm_fixed_init(mut rg: &mut Regressor) -> () {
        for i in rg.ffm_weights_offset as usize..rg.weights.len() {
            rg.weights[i].weight = 1.0;
            rg.weights[i].acc_grad = 1.0;
        }
    }


    #[test]
    fn test_ffm() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]]; // This isn't really used
        
        let mut p: f32;
        
        // Nothing can be learned from a single field
        let mut rr = Regressor::new(&mi);
        let ffm_buf = ffm_vec(vec![HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0}], 1);
        p = rr.learn(&ffm_buf, true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&ffm_buf, true, 0);
        assert_eq!(p, 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        let mut rr = Regressor::new(&mi);
        ffm_fixed_init(&mut rr);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 1.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 1.0, contra_field_index: 1}
                                  ], 2);
        p = rr.learn(&ffm_buf, true, 0);
        assert_eq!(p, 0.7310586); 
        p = rr.learn(&ffm_buf, true, 0);
        assert_eq!(p, 0.7024794);

        // Two fields, use values
        let mut rr = Regressor::new(&mi);
        ffm_fixed_init(&mut rr);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 2.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: 1}
                                  ], 2);
        p = rr.learn(&ffm_buf, true, 0);
        assert_eq!(p, 0.98201376);
        p = rr.learn(&ffm_buf, true, 0);
        assert_eq!(p, 0.81377685);


    }


    #[test]
    fn test_example_importance() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        
        let mut fb_instance = lr_vec(vec![HashAndValue{hash: 1, value: 1.0}]);
        fb_instance.example_importance = 0.5;
        p = rr.learn(&fb_instance, true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&fb_instance, true, 0);
        assert_eq!(p, 0.49375027);
        p = rr.learn(&fb_instance, true, 0);
        assert_eq!(p, 0.4875807);
    }

}







