use std::mem::{self, MaybeUninit};
//use fastapprox::fast::sigmoid;
use std::process;
use triomphe::{UniqueArc, Arc};
//use std::fmt::Binary;
use merand48::*;

use crate::model_instance;
use crate::feature_buffer;


const ONE:u32 = 1065353216;// this is 1.0 float -> u32
const BUF_LEN:usize = 220; // We will ABORT if number of derived features for individual example is more than this.
// Why not bigger number? If we grow stack of the function too much, we end up with stack overflow protecting mechanisms


pub struct Regressor {
    hash_mask: u32,
    learning_rate: f32,
    minus_power_t:f32,
    pub weights: Vec<f32>,       // weights and gradients, interleved, for all models
    ffm_weights_offset: u32, 
    ffm_k: u32,
    ffm_hashmask: u32,
    ffm_one_over_k_root: f32,
}

#[derive(Clone)]
pub struct FixedRegressor {
    hash_mask: u32,
    learning_rate: f32,
    minus_power_t:f32,
    pub weights: Arc<Vec<f32>>,       // Both weights and gradients
    ffm_weights_offset: u32, 
    ffm_k: u32,
    ffm_hashmask: u32,
}

impl Regressor {
    pub fn new(model_instance: &model_instance::ModelInstance) -> Regressor {
        let hash_mask = (1 << model_instance.bit_precision) -1;
        let lr_weights_len = 2*(hash_mask+1);
        let mut rg = Regressor{
                            hash_mask: hash_mask,
                            learning_rate: model_instance.learning_rate,
                            minus_power_t : - model_instance.power_t,
                            weights: Vec::new(), 
                            ffm_weights_offset: 0,
                            ffm_k: 0,
                            ffm_hashmask: 0,
                            ffm_one_over_k_root: 0.0,
                        };


        let mut ffm_weights_len = 0;
        if model_instance.ffm_k > 0 {
            // To keep things simple ffm weights length will be the same as lr
            // we'll adjust everything else
            ffm_weights_len = lr_weights_len;
            rg.ffm_weights_offset = lr_weights_len;
            rg.ffm_k = model_instance.ffm_k;
            // Since we will align our dimensions, we need to know the number of bits for them
            let mut ffm_bits_for_dimensions = 0;
            while rg.ffm_k > (1 << (ffm_bits_for_dimensions)) {
                ffm_bits_for_dimensions += 1;
            }
            let dimensions_mask = (1 << ffm_bits_for_dimensions) - 1;
            // in ffm we will simply mask the lower bits, so we spare them for k
            rg.ffm_hashmask = ((1 << (model_instance.bit_precision)) -1) ^ dimensions_mask;
//            println!("ffm_k: {}, ffm_bits_for_dimesnions: {}, ffm_hashmask: {:#032b}", rg.ffm_k, ffm_bits_for_dimensions, rg.ffm_hashmask);
        }
        // Now allocate weights
        rg.weights = vec![0.0; (lr_weights_len + ffm_weights_len) as usize];

        // Initialization, from ffm.pdf, however should random distribution be cetered on zero?
        if model_instance.ffm_k > 0 {            
            rg.ffm_one_over_k_root = 1.0 / (rg.ffm_k as f32).sqrt();
            for i in 0..(hash_mask+1) {
                // rg.weights[(rg.ffm_weights_offset + i*2+1) as usize] = 1.0;
                // rg.weights[(rg.ffm_weights_offset + i*2) as usize] = rng.gen_range(-0.5* k_root , 0.5 * k_root );
            }
        }
        rg
    }
    
    pub fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, mut update: bool, example_num: u32) -> f32 {
        unsafe {
        let y = *fb.lr_buffer.get_unchecked(0) as f32; // 0.0 or 1.0
        let fbuf = &fb.lr_buffer.get_unchecked(1..fb.lr_buffer.len());
        let mut local_buf_len = fbuf.len()/2;
        if local_buf_len > BUF_LEN {
            println!("Number of features per example ({}) is higher than supported in this fw binary ({}), exiting", local_buf_len, BUF_LEN);
            process::exit(1);
        }
        let mut local_data: [f32; (BUF_LEN*4) as usize] = MaybeUninit::uninit().assume_init() ;
        let mut wsum:f32 = 0.0;
        for i in 0..local_buf_len {
            let hash = (*fbuf.get_unchecked(i*2) * 2) as usize;
            //            println!("H: {}", hash);
            let feature_value:f32 = f32::from_bits(*fbuf.get_unchecked(i*2+1));
            let w = *self.weights.get_unchecked(hash);
            wsum += w * feature_value;    
            *local_data.get_unchecked_mut(i*4) = f32::from_bits(hash as u32);
            *local_data.get_unchecked_mut(i*4+1) = *self.weights.get_unchecked(hash+1);
            *local_data.get_unchecked_mut(i*4+2) = feature_value;
        }
        
        if self.ffm_k > 0 {
            let left_feature_value = 1.0;  // we currently do not support feature values in ffm
            let right_feature_value = 1.0;
            for (i, left_fbuf) in fb.ffm_buffers.iter().enumerate() {
                for left_hash in left_fbuf {
                    let left_weight_p = (self.ffm_weights_offset + (left_hash & self.ffm_hashmask) * 2) as usize;
                    for (j, right_fbuf) in fb.ffm_buffers[i+1 ..].iter().enumerate() {
                        for right_hash in right_fbuf {
                            let right_weight_p = (self.ffm_weights_offset + (right_hash & self.ffm_hashmask) * 2) as usize;
                            for k in 0..(self.ffm_k as usize) {
                                let left_weight_a = &mut self.weights[left_weight_p + k * 2];
                                if *left_weight_a == 0.0 {
                                    *left_weight_a = merand48((left_weight_p + k *2) as u64) * self.ffm_one_over_k_root;
                                }
                                let left_weight = *left_weight_a;
                                let right_weight = self.weights[right_weight_p + k * 2];
                                wsum += left_weight * right_weight * 
                                        left_feature_value * right_feature_value;
                                // left side
                                *local_data.get_unchecked_mut(local_buf_len*4) = f32::from_bits((left_weight_p + k * 2) as u32);// store index
                                *local_data.get_unchecked_mut(local_buf_len*4+1) = self.weights[left_weight_p + k * 2 +1]; // accumulated errors
                                *local_data.get_unchecked_mut(local_buf_len*4+2) = left_feature_value * right_feature_value * right_weight; // gradient
                                local_buf_len += 1;
                                // right side
                                *local_data.get_unchecked_mut(local_buf_len*4) = f32::from_bits((right_weight_p + k * 2) as u32); // store index
                                *local_data.get_unchecked_mut(local_buf_len*4+1) = self.weights[right_weight_p + k * 2 +1]; // accumulated errors
                                *local_data.get_unchecked_mut(local_buf_len*4+2) = left_feature_value * right_feature_value * left_weight; // gradient
                                local_buf_len += 1;
                            }
                            /*
                            println!("A {} {} {} {} {} {}", i,j+i+1, left_hash, right_hash, 
                                                            self.weights[left_weight_p], 
                                                            self.weights[right_weight_p]);
                            */
                        }
                    }
                }
            }
            
        }
        // Trick: instead of multiply in the updates with learning rate, multiply the result
        let prediction = -wsum;
        // vowpal compatibility
        let mut prediction_finalized = prediction;
        if prediction_finalized.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", example_num);
            prediction_finalized = 0.0;
            update = false;
        } else if prediction_finalized < -50.0 {
            prediction_finalized = -50.0;
            update = false;
        } else if prediction_finalized > 50.0 {
            prediction_finalized = 50.0;
            update = false;
        }
        let prediction_probability:f32 = (1.0+(prediction_finalized).exp()).recip();
//        let prediction:f32 = sigmoid(wsum);      // ain't faster
        if update{
            let general_gradient = y - prediction_probability;
  //          println!("general gradient: {}, prediction {}, prediction orig: {}", general_gradient, prediction, -wsum*self.learning_rate); 
            for i in 0..local_buf_len {
                let feature_value = *local_data.get_unchecked(i*4+2);
                let gradient = general_gradient * feature_value;
                *local_data.get_unchecked_mut(i*4+3) = gradient*gradient;	// it would be easier, to just use i*4+1 at the end... but this is how vowpal does it
                *local_data.get_unchecked_mut(i*4+1) += *local_data.get_unchecked(i*4+3);
                let update_factor = gradient * (local_data.get_unchecked(i*4+1)).powf(self.minus_power_t);
                *local_data.get_unchecked_mut(i*4+2) = self.learning_rate * update_factor;    // this is how vowpal does it, first calculate, then addk
            }
            for i in 0..local_buf_len {
                let hash = local_data.get_unchecked(i*4).to_bits() as usize;
//                println!("H: {}, gradient update: {}, weight update: {}", hash, *local_data.get_unchecked(i*4+2), *local_data.get_unchecked(i*4+3));
                *self.weights.get_unchecked_mut(hash) += *local_data.get_unchecked(i*4+2);
                *self.weights.get_unchecked_mut(hash+1) += *local_data.get_unchecked(i*4+3);
            }            
        }
        prediction_probability
        }
    }
}

impl FixedRegressor {
    pub fn new(rr: Regressor) -> FixedRegressor {
        
        FixedRegressor {
                        hash_mask: rr.hash_mask,
                        learning_rate: rr.learning_rate,
                        minus_power_t: rr.minus_power_t,
                        weights: Arc::new(rr.weights),
                        ffm_weights_offset: rr.ffm_weights_offset,
                        ffm_k: rr.ffm_k,
                        ffm_hashmask: rr.ffm_hashmask,

        }
    }

    pub fn predict(&self, fb: &feature_buffer::FeatureBuffer, example_num: u32) -> f32 {
        unsafe {
        let fbuf = &fb.lr_buffer.get_unchecked(1..fb.lr_buffer.len());
        let mut wsum:f32 = 0.0;
        for i in 0..fbuf.len()/2 {     // speed of this is 4.53
            let hash = (*fbuf.get_unchecked(i*2) *2) as usize;
            let feature_value:f32 = f32::from_bits(*fbuf.get_unchecked(i*2+1));
            let w = *self.weights.get_unchecked(0);
            wsum += w * feature_value;    
        }

        if self.ffm_k > 0 {
            let left_feature_value = 1.0;  // we currently do not support feature values in ffm
            let right_feature_value = 1.0;
            for (i, left_fbuf) in fb.ffm_buffers.iter().enumerate() {
                for left_hash in left_fbuf {
                    let left_weight_p = (self.ffm_weights_offset + (left_hash & self.ffm_hashmask) * 2) as usize;
                    for (j, right_fbuf) in fb.ffm_buffers[i+1 ..].iter().enumerate() {
                        for right_hash in right_fbuf {
                            let right_weight_p = (self.ffm_weights_offset + (right_hash & self.ffm_hashmask) * 2) as usize;
                            for k in 0..(self.ffm_k as usize) {
                                let left_weight = self.weights[left_weight_p + k * 2];
                                let right_weight = self.weights[right_weight_p + k * 2];
                                wsum += left_weight * right_weight * 
                                        left_feature_value * right_feature_value;
                            }
                        }
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
} 




mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;


    #[test]
    fn test_learning_turned_off() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        // Empty model: no matter how many features, prediction is 0.5
        p = rr.learn(&vec![0], false, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE], false, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE, 2, ONE], false, 0);
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
        
        p = rr.learn(&vec![0, 1, ONE], true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE], true, 0);
        assert_eq!(p, 0.48750263);
        p = rr.learn(&vec![0, 1, ONE], true, 0);
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
        
        p = rr.learn(&vec![0, 1, ONE, 1, two,], true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE, 1, two,], true, 0);
        assert_eq!(p, 0.38936076);
        p = rr.learn(&vec![0, 1, ONE, 1, two,], true, 0);
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
        
        p = rr.learn(&vec![0, 1, ONE], true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE], true, 0);
        assert_eq!(p, 0.4750208);
        p = rr.learn(&vec![0, 1, ONE], true, 0);
        assert_eq!(p, 0.45788094);
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
        p = rr.learn(&vec![0, 1, ONE, 2, ONE], true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE, 2, ONE], true, 0);
        assert_eq!(p, 0.45016602);
        p = rr.learn(&vec![0, 1, ONE], true, 0);
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
        let two = 2.0_f32.to_bits();
        
        p = rr.learn(&vec![0, 1, two], true, 0);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, two], true, 0);
        assert_eq!(p, 0.45016602);
        p = rr.learn(&vec![0, 1, two], true, 0);
        assert_eq!(p, 0.40611085);
    }


}







