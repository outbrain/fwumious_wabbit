use std::mem::{self, MaybeUninit};
//use fastapprox::fast::sigmoid;
use std::process;
use triomphe::{UniqueArc, Arc};
use rand::Rng;

use crate::model_instance;
use crate::feature_buffer;

const ONE:u32 = 1065353216;// this is 1.0 float -> u32
const BUF_LEN:usize = 220; // We will ABORT if number of derived features for individual example is more than this.
// Why not bigger number? If we grow stack of the function too much, we end up with stack overflow protecting mechanisms


pub struct Regressor {
    hash_mask: u32,
    learning_rate: f32,
    minus_power_t:f32,
    pub weights: Vec<f32>,       // Both weights and gradients
    pub ffm_weights: Vec<f32>,
    ffm_k: u32,
    ffm_hashmask: u32,
}

#[derive(Clone)]
pub struct FixedRegressor {
    hash_mask: u32,
    learning_rate: f32,
    minus_power_t:f32,
    pub weights: Arc<Vec<f32>>,       // Both weights and gradients
    pub ffm_weights: Arc<Vec<f32>>,       // Both weights and gradients
    ffm_k: u32,
    ffm_hashmask: u32,
}
impl Regressor {
    pub fn new(model_instance: &model_instance::ModelInstance) -> Regressor {
        let hash_mask = (1 << model_instance.bit_precision) -1;
        let mut rg = Regressor{
                            hash_mask: hash_mask,
                            learning_rate: model_instance.learning_rate,
                            minus_power_t : - model_instance.power_t,
                            weights: vec![0.0; (2*(hash_mask+1)) as usize],
                            ffm_weights: Vec::new(),
                            ffm_k: 0,
                            ffm_hashmask: 0,
                        };

        if model_instance.ffm_k > 0 {
            rg.ffm_weights = vec![0.0; (2*(hash_mask+1)) as usize];
            rg.ffm_k = model_instance.ffm_k;
            // Since we will align our dimensions, we need to know the number of bits for them
            let mut ffm_bits_for_dimensions = 1;
            while rg.ffm_k < (1 << (ffm_bits_for_dimensions - 1)) {
                ffm_bits_for_dimensions += 1;
            }
            let dimensions_mask = (1 << ffm_bits_for_dimensions) - 1;
            // in ffm we will simply mask the lower bits, so we spare them for k
            rg.ffm_hashmask = ((1 << (model_instance.bit_precision)) -1) ^ dimensions_mask;
            
            // random init
            let k_root = (rg.ffm_k as f32).sqrt();
            let mut rng = rand::thread_rng();
            for i in 0..(hash_mask+1) {
                rg.ffm_weights[i as usize] = rng.gen_range(0.0, k_root);
            }
        }
        rg
    }
    
    pub fn learn(&mut self, fb: &feature_buffer::FeatureBuffer, mut update: bool, example_num: u32) -> f32 {
        unsafe {
        let y = *fb.lr_buffer.get_unchecked(0) as f32; // 0.0 or 1.0
        let fbuf = &fb.lr_buffer.get_unchecked(1..fb.lr_buffer.len());
        let fbuf_len = fbuf.len()/2;
        if fbuf_len > BUF_LEN {
            println!("Number of features per example ({}) is higher than supported in this fw binary ({}), exiting", fbuf_len, BUF_LEN);
            process::exit(1);
        }
        let mut lr_local_data: [f32; (BUF_LEN*4) as usize] = MaybeUninit::uninit().assume_init() ;
        let mut wsum:f32 = 0.0;
        for i in 0..fbuf_len {
            let hash = *fbuf.get_unchecked(i*2) as usize;
            let feature_value:f32 = f32::from_bits(*fbuf.get_unchecked(i*2+1));
            let w = *self.weights.get_unchecked(hash*2);
            wsum += w * feature_value;    
            *lr_local_data.get_unchecked_mut(i*4) = w;
            *lr_local_data.get_unchecked_mut(i*4+1) = *self.weights.get_unchecked(hash*2+1);
            *lr_local_data.get_unchecked_mut(i*4+2) = feature_value;
        }
        
        let mut ffm_local_data: [f32; (BUF_LEN*4) as usize] = MaybeUninit::uninit().assume_init() ;
        let mut fi:usize = 0;
        if self.ffm_k > 0 {
            let left_feature_value = 1.0; 
            let right_feature_value = 1.0;
            for (i, left_fbuf) in fb.ffm_buffers.iter().enumerate() {
                for left_hash in left_fbuf {
                    let left_weight_p = ((left_hash & self.ffm_hashmask) * 2) as usize;
                    for (j, right_fbuf) in fb.ffm_buffers[i+1 ..].iter().enumerate() {
                        for right_hash in right_fbuf {
                            let right_weight_p = ((right_hash & self.ffm_hashmask) * 2) as usize;
                            for k in 0..(self.ffm_k as usize) {
                                wsum += self.ffm_weights[left_weight_p + k * 2] * 
                                        self.ffm_weights[right_weight_p + k * 2] * 
                                        left_feature_value * right_feature_value;
                                *ffm_local_data.get_unchecked_mut(fi*4) = f32::from_bits((left_weight_p + k * 2) as u32);// storing index
                                *ffm_local_data.get_unchecked_mut(fi*4+1) = self.ffm_weights[left_weight_p + k * 2 +1]; // accumulated errors
                                *ffm_local_data.get_unchecked_mut(fi*4+2) = left_feature_value * right_feature_value *
                                                                            self.ffm_weights[right_weight_p + k * 2];
                                fi += 1;
                                *ffm_local_data.get_unchecked_mut(fi*4) = f32::from_bits((right_weight_p + k * 2) as u32); // we'll store index here
                                *ffm_local_data.get_unchecked_mut(fi*4+1) = self.ffm_weights[right_weight_p + k * 2 +1]; // accumulated errors
                                *ffm_local_data.get_unchecked_mut(fi*4+2) = left_feature_value * right_feature_value *
                                                                            self.ffm_weights[left_weight_p + k * 2];
                                fi += 1;
                            }
                            /*
                            println!("A {} {} {} {} {} {}", i,j+i+1, left_hash, right_hash, 
                                                            self.ffm_weights[left_weight_p], 
                                                            self.ffm_weights[right_weight_p]);
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
            for i in 0..fbuf_len {
                let feature_value = *lr_local_data.get_unchecked(i*4+2);
                let gradient = general_gradient * feature_value;
                *lr_local_data.get_unchecked_mut(i*4+3) = gradient*gradient;	// it would be easier, to just use i*4+1 at the end... but this is how vowpal does it
                *lr_local_data.get_unchecked_mut(i*4+1) += *lr_local_data.get_unchecked(i*4+3);
                let update_factor = gradient * (lr_local_data.get_unchecked(i*4+1)).powf(self.minus_power_t);
                *lr_local_data.get_unchecked_mut(i*4+2) = self.learning_rate * update_factor;    // this is how vowpal does it, first calculate, then addk
            }
            for i in 0..fbuf_len {
                let hash = *fbuf.get_unchecked(i*2) as usize;
                *self.weights.get_unchecked_mut(hash*2) += *lr_local_data.get_unchecked(i*4+2);
                *self.weights.get_unchecked_mut(hash*2+1) += *lr_local_data.get_unchecked(i*4+3);
            }
            
            if self.ffm_k > 0 {
                for i in 0..fi {
                    let feature_value = ffm_local_data.get_unchecked(i * 4 +2);
//                    println!("General: {}, feature_value : {}", general_gradient, feature_value);
                    let gradient = general_gradient * feature_value;
                    *ffm_local_data.get_unchecked_mut(i*4+3) = gradient*gradient;	// it would be easier, to just use i*4+1 at the end... but this is how vowpal does it
                    *ffm_local_data.get_unchecked_mut(i*4+1) += *lr_local_data.get_unchecked(i*4+3);
                    let update_factor = gradient * (ffm_local_data.get_unchecked(i*4+1)).powf(self.minus_power_t);
                    *ffm_local_data.get_unchecked_mut(i*4+2) = self.learning_rate * update_factor;    // this is how vowpal does it, first calculate, then addk
                }
                for i in 0..fi {
                    let hash = ffm_local_data.get_unchecked(i*4).to_bits() as usize;
                    *self.ffm_weights.get_unchecked_mut(hash) += *ffm_local_data.get_unchecked(i*4+2);
                    *self.ffm_weights.get_unchecked_mut(hash+1) += *ffm_local_data.get_unchecked(i*4+3);
                }

                    
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
                        ffm_k: rr.ffm_k,
                        ffm_hashmask: rr.ffm_hashmask,
                        ffm_weights: Arc::new(rr.ffm_weights),

        }
    }

    pub fn predict(&self, fb: &feature_buffer::FeatureBuffer, example_num: u32) -> f32 {
        unsafe {
        let fbuf = &fb.lr_buffer.get_unchecked(1..fb.lr_buffer.len());
        let fbuf_len = fbuf.len()/2;
        if fbuf_len > BUF_LEN {
            println!("Number of features per example ({}) is higher than supported in this fw binary ({}), exiting", fbuf_len, BUF_LEN);
            process::exit(1);
        }
        /* first we need a dot product, which in our case is a simple sum */
        let mut wsum:f32 = 0.0;
        for i in 0..fbuf_len {     // speed of this is 4.53
            let hash = *fbuf.get_unchecked(i*2) as usize;
            let feature_value:f32 = f32::from_bits(*fbuf.get_unchecked(i*2+1));
            let w = *self.weights.get_unchecked(hash*2);
            wsum += w * feature_value;    
        }

        if self.ffm_k > 0 {
            let left_feature_value = 1.0; 
            let right_feature_value = 1.0;
            for (i, left_fbuf) in fb.ffm_buffers.iter().enumerate() {
                for left_hash in left_fbuf {
                    let left_weight_p = ((left_hash & self.ffm_hashmask) * 2) as usize;
                    for (j, right_fbuf) in fb.ffm_buffers[i+1 ..].iter().enumerate() {
                        for right_hash in right_fbuf {
                            let right_weight_p = ((right_hash & self.ffm_hashmask) * 2) as usize;
                            for k in 0..(self.ffm_k as usize) {
                                wsum += self.ffm_weights[left_weight_p + k * 2] * 
                                        self.ffm_weights[right_weight_p + k * 2] * 
                                        left_feature_value * right_feature_value;
                            }
                            /*
                            println!("A {} {} {} {} {} {}", i,j+i+1, left_hash, right_hash, 
                                                            self.ffm_weights[left_weight_p], 
                                                            self.ffm_weights[right_weight_p]);
                            */
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







