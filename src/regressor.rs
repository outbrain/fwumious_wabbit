#![allow(unused_macros)]
use std::mem::{self};
//use std::mem::{MaybeUninit};
use std::slice;
//use fastapprox::fast::sigmoid; // surprisingly this doesn't work very well
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
use optimizer::OptimizerTrait;

#[derive(Clone, Debug)]
#[repr(C)]
pub struct Weight {
    pub weight: f32, 
}

pub struct IndexAccgradientValue {
    index: u32,
    value: f32,
}

#[derive(Clone, Debug, Copy)]
#[repr(C)]
pub struct WeightAndOptimizerData<L:OptimizerTrait> {
    pub weight: f32, 
    pub optimizer_data: L::PerWeightStore,
}

pub struct Regressor<L:OptimizerTrait> {
    pub weights: Vec<WeightAndOptimizerData<L>>,       // all weights and gradients (has sub-spaces)
    pub weights_len: u32,
    pub ffm_weights_len: u32, 
    pub ffm_weights_offset: u32, 
    ffm_k: u32,
    ffm_hashmask: u32,
    ffm_one_over_k_root: f32,
    ffm_iw_weights_offset: u32,
    ffm_k_threshold: f32,
    optimizer_lr: L,
    pub optimizer_ffm: L,
    local_data_lr: Vec<IndexAccgradientValue>,
    local_data_ffm: Vec<IndexAccgradientValue>,
}

#[derive(Clone)]
pub struct ImmutableRegressor {
    pub weights: Arc<Vec<Weight>>,
    ffm_weights_offset: u32, 
    ffm_k: u32,
    ffm_hashmask: u32,
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

pub trait RegressorTrait {
    fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, update: bool, example_num: u32) -> f32;
    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>>;
    fn overwrite_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>>; 
    fn get_name(&self) -> &'static str;
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


impl <L:OptimizerTrait>Regressor<L> 
where <L as optimizer::OptimizerTrait>::PerWeightStore: std::clone::Clone,
L: std::clone::Clone
{
    pub fn new_without_weights(mi: &model_instance::ModelInstance) -> Regressor<L> {
        let hash_mask = (1 << mi.bit_precision) -1;
        let lr_weights_len = hash_mask + 1;
        let mut rg = Regressor::<L>{
                            //minimum_optimizer: mi.minimum_optimizer,
                            weights: Vec::new(),
                            weights_len: 0, 
                            ffm_weights_offset: 0,
                            ffm_weights_len: 0,
                            ffm_k: 0, 
                            ffm_hashmask: 0, 
                            ffm_one_over_k_root: 0.0, 
                            optimizer_lr: L::new(),
                            optimizer_ffm: L::new(),
                            ffm_iw_weights_offset: 0, ffm_k_threshold:
                            mi.ffm_k_threshold, 
                            local_data_lr: Vec::with_capacity(1024), 
                            local_data_ffm: Vec::with_capacity(1024),
                     };

        rg.optimizer_lr.init(mi.learning_rate, mi.power_t, mi.init_acc_gradient);
        rg.optimizer_ffm.init(mi.ffm_learning_rate, mi.ffm_power_t, mi.ffm_init_acc_gradient);

        if mi.ffm_k > 0 {
            
            rg.ffm_weights_offset = lr_weights_len;            // Since we will align our dimensions, we need to know the number of bits for them
            rg.ffm_k = mi.ffm_k;
            // At the end we add "spillover buffer", so we can do modulo only on the base address and add offset
            rg.ffm_weights_len = (1 << mi.ffm_bit_precision) + (mi.ffm_fields.len() as u32 * rg.ffm_k);
            let mut ffm_bits_for_dimensions = 0;
            while rg.ffm_k > (1 << (ffm_bits_for_dimensions)) {
                ffm_bits_for_dimensions += 1;
            }
            let dimensions_mask = (1 << ffm_bits_for_dimensions) - 1;
            // in ffm we will simply mask the lower bits, so we spare them for k
            rg.ffm_hashmask = ((1 << mi.ffm_bit_precision) -1) ^ dimensions_mask;
        }
        // Now allocate weights
        let iw_weights_len = 0;
        rg.ffm_iw_weights_offset = lr_weights_len + rg.ffm_weights_len;        
        rg.weights_len = lr_weights_len + rg.ffm_weights_len + iw_weights_len;
        rg
    }
    
    pub fn allocate_and_init_weights_(&mut self, mi: &model_instance::ModelInstance) {
        let rg = self;
        rg.weights = vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: rg.optimizer_lr.initial_data()}; rg.weights_len as usize];

        if mi.ffm_k > 0 {       
            if mi.ffm_init_width == 0.0 {
                // Initialization that has showed to work ok for us, like in ffm.pdf, but centered around zero and further divided by 50
                rg.ffm_one_over_k_root = 1.0 / (rg.ffm_k as f32).sqrt() / 50.0;
                for i in 0..rg.ffm_weights_len {
                    rg.weights[(rg.ffm_weights_offset + i) as usize].weight = (1.0 * merand48((rg.ffm_weights_offset+i) as u64)-0.5) * rg.ffm_one_over_k_root;
                    rg.weights[(rg.ffm_weights_offset + i) as usize].optimizer_data = rg.optimizer_ffm.initial_data();
                }
            } else {
                let zero_half_band_width = mi.ffm_init_width * mi.ffm_init_zero_band * 0.5;
                let band_width = mi.ffm_init_width * (1.0 - mi.ffm_init_zero_band);
                for i in 0..rg.ffm_weights_len {
                    let mut w = merand48(i as u64) * band_width - band_width * 0.5;
                    if w > 0.0 { 
                        w += zero_half_band_width ;
                    } else {
                        w -= zero_half_band_width;
                    }
                    w += mi.ffm_init_center;
                    rg.weights[(rg.ffm_weights_offset + i) as usize].weight = w; 
                    rg.weights[(rg.ffm_weights_offset + i) as usize].optimizer_data = rg.optimizer_ffm.initial_data();
                }

            }
        }
    }

    pub fn new(mi: &model_instance::ModelInstance) -> Regressor<L> {
        let mut rg = Regressor::<L>::new_without_weights(mi);
        rg.allocate_and_init_weights(mi);
        rg
    }
}

/* We tested standard stable logistic function, but it gives slightly 
worse logloss results than plain logistic on our data */
/*
#[inline(always)]
pub fn stable_logistic(t: f32) -> f32 {
    if t > 0.0 {
        return (1.0 +(-t).exp()).recip();
    } else {
        let texp = t.exp();
        return texp / (1.0 + texp);
    }
}
*/

#[inline(always)]
pub fn logistic(t: f32) -> f32 {
    return (1.0+(-t).exp()).recip();
}

    
impl <L:OptimizerTrait>RegressorTrait for Regressor<L> 
where <L as optimizer::OptimizerTrait>::PerWeightStore: std::clone::Clone,
L: std::clone::Clone
{

    fn get_name(&self) -> &'static str {
        L::get_name()
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.allocate_and_init_weights_(mi);
    }

    fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, update: bool, example_num: u32) -> f32 {
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
// TODO: Opportunistic use of buffers on stack instead of head - ~10% performance improvement for FFMs
//        let mut local_data_lr: [IndexAccgradientValue; BUF_LEN as usize] = MaybeUninit::uninit().assume_init() ;
//        let mut local_data_ffm: [IndexAccgradientValue; BUF_LEN as usize] = MaybeUninit::uninit().assume_init() ;

        let weights = &self.weights;

        let mut wsum:f32 = 0.0;
        for (i, hashvalue) in fb.lr_buffer.iter().enumerate() {
            // Prefetch couple of indexes from the future to prevent pipeline stalls due to memory latencies
            _mm_prefetch(mem::transmute::<&f32, &i8>(&weights.get_unchecked((fb.lr_buffer.get_unchecked(i+8).hash) as usize).weight), _MM_HINT_T0);  // No benefit for now
            let feature_index     = hashvalue.hash;
            let feature_value:f32 = hashvalue.value;
            let feature_weight    = weights.get_unchecked(feature_index as usize).weight;
            wsum += feature_weight * feature_value;
            local_data_lr.get_unchecked_mut(i).index       = feature_index;
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
                    _mm_prefetch(mem::transmute::<&f32, &i8>(&weights.get_unchecked(base_weight_index as usize + j).weight), _MM_HINT_T0);  // No benefit for now
                }
            }

            specialize_k!(self.ffm_k, FFMK, {
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
                ifc += fc;
                
            }
        
            });
        }
        // Trick: instead of multiply in the updates with learning rate, multiply the result
        let prediction = wsum;
        // vowpal compatibility
        if prediction.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", example_num);
            return logistic(0.0);
        } else if prediction < -50.0 {
            return logistic(-50.0);
        } else if prediction > 50.0 {
            return logistic(50.0);
        }

        let prediction_probability:f32 = logistic(prediction);

        // Weights are now writable, but local_data is read only
        let weights = &mut self.weights;


        if update && fb.example_importance != 0.0 {
            let general_gradient = (y - prediction_probability) * fb.example_importance;
//            println!("General gradient: {}", general_gradient);

            for i in 0..local_data_lr_len {
//                _mm_prefetch(mem::transmute::<&f32, &i8>(&weights.get_unchecked((local_data_lr.get_unchecked(i+8)).index as usize).weight), _MM_HINT_T0);  // No benefit for now
                let feature_value = local_data_lr.get_unchecked(i).value;
                let feature_index = local_data_lr.get_unchecked(i).index as usize;
                let gradient = general_gradient * feature_value;
                let update = self.optimizer_lr.calculate_update(gradient, &mut weights.get_unchecked_mut(feature_index).optimizer_data);
                weights.get_unchecked_mut(feature_index).weight += update;
            }
            
            for i in 0..local_data_ffm_len {
//                _mm_prefetch(mem::transmute::<&f32, &i8>(&weights.get_unchecked((local_data_ffm.get_unchecked(i+8)).index as usize).weight), _MM_HINT_T0);  // No benefit for now
                let feature_value = local_data_ffm.get_unchecked(i).value;
                let feature_index = local_data_ffm.get_unchecked(i).index as usize;
                let gradient = general_gradient * feature_value;
                let update = self.optimizer_ffm.calculate_update(gradient, &mut weights.get_unchecked_mut(feature_index).optimizer_data);
                weights.get_unchecked_mut(feature_index).weight += update;
            }
        }
        
        prediction_probability
        }
    }
    
    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        // It's OK! I am a limo driver!
        output_bufwriter.write_u64::<LittleEndian>(self.weights.len() as u64)?;
        unsafe {
             let buf_view:&[u8] = slice::from_raw_parts(self.weights.as_ptr() as *const u8, 
                                              self.weights.len() *mem::size_of::<WeightAndOptimizerData<L>>());
             output_bufwriter.write(buf_view)?;
        }
        
        Ok(())
    }

    fn overwrite_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        if len != self.weights.len() as u64 {
            return Err(format!("Lenghts of weights array in regressor file differ: got {}, expected {}", len, self.weights.len()))?;
        }
        unsafe {
            let mut buf_view:&mut [u8] = slice::from_raw_parts_mut(self.weights.as_mut_ptr() as *mut u8, 
                                             self.weights.len() *mem::size_of::<WeightAndOptimizerData<L>>());
            input_bufreader.read_exact(&mut buf_view)?;
        }

        Ok(())
    }

    
    // Creates immutable regressor from current setup and weights from buffer
    fn immutable_regressor_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<ImmutableRegressor, Box<dyn Error>> {
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        if len != self.weights_len as u64 {
            return Err(format!("Lenghts of weights array in regressor file differ: got {}, expected {}", len, self.weights_len))?;
        }
              
        const BUF_LEN:usize = 1024 * 1024;
        let mut in_weights = vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.optimizer_lr.initial_data()}; BUF_LEN as usize];
        let mut out_weights = Vec::<Weight>::new();
        
        let mut remaining_weights = self.weights_len as usize;
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
                        ffm_weights_offset: self.ffm_weights_offset,
                        ffm_k: self.ffm_k,
                        ffm_hashmask: self.ffm_hashmask,
        };
        Ok(fr)
    }

    fn immutable_regressor(&mut self) -> Result<ImmutableRegressor, Box<dyn Error>> {
        let mut weights = Vec::<Weight>::new();
        for w in &self.weights {
            weights.push(Weight{weight:w.weight});
        }
        let fr = ImmutableRegressor {
                        weights: Arc::new(weights), 
                        ffm_weights_offset: self.ffm_weights_offset,
                        ffm_k: self.ffm_k,
                        ffm_hashmask: self.ffm_hashmask,
        };
        Ok(fr)
    }



}

impl ImmutableRegressor {

    pub fn predict(&self, fb: &feature_buffer::FeatureBuffer, example_num: u32) -> f32 {
        let fbuf = &fb.lr_buffer;
        let mut wsum:f32 = 0.0;
        for val in fbuf {
            let hash = val.hash as usize;
            let feature_value:f32 = val.value;
            wsum += self.weights[hash].weight * feature_value;    
        }

        if self.ffm_k > 0 {
            for (i, left_hash) in fb.ffm_buffer.iter().enumerate() {
                for (_j, right_hash) in fb.ffm_buffer[i+1 ..].iter().enumerate() {
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
        for i in rg.ffm_weights_offset as usize..rg.weights.len() {
            rg.weights[i].weight = 1.0;
//            rg.weights[i].acc_grad = 1.0;
            rg.weights[i].optimizer_data = rg.optimizer_ffm.initial_data();
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
        mi.ffm_k = 1;
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
        assert_eq!(re.learn(&ffm_buf, true, 0), 0.7310586); 
        assert_eq!(re.learn(&ffm_buf, true, 0), 0.7024794);

        // Two fields, use values
        let mut re = Regressor::<optimizer::OptimizerAdagradLUT>::new(&mi);
        ffm_init(&mut re);
        let ffm_buf = ffm_vec(vec![
                                  HashAndValueAndSeq{hash:1, value: 2.0, contra_field_index: 0},
                                  HashAndValueAndSeq{hash:100, value: 2.0, contra_field_index: 1}
                                  ], 2);
        assert_eq!(re.learn(&ffm_buf, true, 0), 0.98201376);
        assert_eq!(re.learn(&ffm_buf, true, 0), 0.81377685);


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

