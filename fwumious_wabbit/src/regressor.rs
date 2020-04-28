use fastapprox::fast::sigmoid;

const ONE:u32 = 1065353216;      // this is 1.0 float -> u32


use crate::model_instance;

pub struct Regressor<'a> {
    model_instance: &'a model_instance::ModelInstance,
    hash_mask: u32,
    weights: Vec<f32>,
//    gradient_sqr: Vec<f32>,
    local_data: [f32; 1024],//Vec<f32>,
     minus_power_t:f32,
}

impl<'a> Regressor<'a> {
    pub fn new(model_instance: &'a model_instance::ModelInstance) -> Regressor {
        let hash_mask = (1 << model_instance.hash_bits) -1;
                            
        let mut rg = Regressor{
                            model_instance: model_instance,
                            hash_mask: hash_mask,
                            weights: vec![0.0; (2*(hash_mask+1)) as usize],
 //                           gradient_sqr: vec![0.0; (hash_mask+1) as usize],
                            local_data: [0.0;1024], //vec![0.0; 1000],
                            minus_power_t : - model_instance.power_t,
                        };
        rg
    }
    
    
    pub fn learn(&mut self, feature_buffer: &Vec<u32>, update: bool) -> f32 {
        unsafe {
        let y = feature_buffer[0] as f32; // 0.0 or 1.0
        let fbuf = &feature_buffer.get_unchecked(1..feature_buffer.len());
        let fbuf_len = fbuf.len()/2;
        //let mut local_data = self.local_data;
        /* first we need a dot product, which in our case is a simple sum */
        let mut wsum:f32 = 0.0;
        for i in 0..fbuf_len {     // speed of this is 4.53
            let hash = (*fbuf.get_unchecked(i*2) & self.hash_mask) as usize;
            //*local_data.get_unchecked_mut(i*4) = *self.weights.get_unchecked(hash*2);
            let feature_value:f32 = f32::from_bits(*fbuf.get_unchecked(i*2+1));
            let w = *self.weights.get_unchecked(hash*2);
            wsum += w * feature_value;    
            *self.local_data.get_unchecked_mut(i*4) = w;
            *self.local_data.get_unchecked_mut(i*4+1) = *self.weights.get_unchecked(hash*2+1);
            *self.local_data.get_unchecked_mut(i*4+2) = feature_value;
            
        }
        let prediction:f32 = (1.0+(-wsum).exp()).recip();
//        let prediction:f32 = sigmoid(wsum);      // ain't faster
        if update {
            let learning_rate = self.model_instance.learning_rate;
            let general_gradient = -(prediction - y);
            for i in 0..fbuf_len {
                let feature_value = *self.local_data.get_unchecked(i*4+2);
                let gradient = general_gradient * feature_value;
//                println!("Gradient: {}", self.local_data[i*4+1]);
                *self.local_data.get_unchecked_mut(i*4+3) = gradient*gradient;	// it would be easier, to just use i*4+1 at the end... but this is how vowpal does it
                *self.local_data.get_unchecked_mut(i*4+1) += *self.local_data.get_unchecked(i*4+3);
                let update_factor = gradient * learning_rate  * (self.local_data.get_unchecked(i*4+1)).powf(self.minus_power_t);
                *self.local_data.get_unchecked_mut(i*4) = update_factor;    // this is how vowpal does it, first calculate, then addk
            }
            // Next step is: gradients = weights_vector *  
            for i in 0..fbuf_len {     // speed of this is 4.53
                let hash = (*fbuf.get_unchecked(i*2) & self.hash_mask) as usize;
                *self.weights.get_unchecked_mut(hash*2) += *self.local_data.get_unchecked(i*4);
                *self.weights.get_unchecked_mut(hash*2+1) += *self.local_data.get_unchecked(i*4+3);
            }

        }
    //    println!("S {}, {}", y, prediction);
        prediction
        }
    }

    pub fn learn_oldoptimized(&mut self, feature_buffer: &Vec<u32>, update: bool) -> f32 {
        unsafe {
        let y = feature_buffer[0] as f32; // 0.0 or 1.0
        let fbuf = &feature_buffer[1..feature_buffer.len()];
        let fbuf_len = fbuf.len()/2;
        //let mut local_data = self.local_data;
        /* first we need a dot product, which in our case is a simple sum */
        let mut wsum:f32 = 0.0;
        for i in 0..fbuf_len {     // speed of this is 4.53
            let hash = (*fbuf.get_unchecked(i*2) & self.hash_mask) as usize;
            let feature_value:f32 = f32::from_bits(*fbuf.get_unchecked(i*2+1));
            wsum += self.weights[hash*2] * feature_value;    
            //*local_data.get_unchecked_mut(i*4) = *self.weights.get_unchecked(hash*2);
            self.local_data[i*4] = *self.weights.get_unchecked(hash*2);
            
            self.local_data[i*4+1] = *self.weights.get_unchecked(hash*2+1);
            self.local_data[i*4+2] = feature_value;

        }
        let prediction:f32 = (1.0+(-wsum).exp()).recip();
//        let prediction:f32 = sigmoid(wsum);      // ain't faster
        if update {
            let learning_rate = self.model_instance.learning_rate;
            let general_gradient = -(prediction - y);
            for i in 0..fbuf_len {
                let feature_value = *self.local_data.get_unchecked(i*4+2);
                let gradient = general_gradient * feature_value;
//                println!("Gradient: {}", self.local_data[i*4+1]);
                self.local_data[i*4+3] = gradient*gradient;	// it would be easier, to just use i*4+1 at the end... but this is how vowpal does it
                self.local_data[i*4+1] += *self.local_data.get_unchecked(i*4+3);
                let update_factor = gradient * learning_rate  * (self.local_data.get_unchecked(i*4+1)).powf(self.minus_power_t);
                self.local_data[i*4] = update_factor;    // this is how vowpal does it, first calculate, then addk
            
            }
            // Next step is: gradients = weights_vector *  
            for i in 0..fbuf_len {     // speed of this is 4.53
                let hash = (*fbuf.get_unchecked(i*2) & self.hash_mask) as usize;
                self.weights[hash*2] += *self.local_data.get_unchecked(i*4);
                self.weights[hash*2+1] += *self.local_data.get_unchecked(i*4+3);
            }

        }
    //    println!("S {}, {}", y, prediction);
        prediction
        }
    }

    pub fn learn_original(&mut self, feature_buffer: &Vec<u32>, update: bool) -> f32 {
        let y = feature_buffer[0] as f32; // 0.0 or 1.0
        let fbuf = &feature_buffer[1..feature_buffer.len()];
        let fbuf_len = fbuf.len()/2;
        /* first we need a dot product, which in our case is a simple sum */
        let mut wsum:f32 = 0.0;
        for i in 0..fbuf_len {     // speed of this is 4.53
            let hash = (fbuf[i*2] & self.hash_mask) as usize;
            let feature_value:f32 = f32::from_bits(fbuf[i*2+1]);
            wsum += self.weights[hash*2] * feature_value;    
            self.local_data[i*4] = self.weights[hash*2];
            self.local_data[i*4+1] = self.weights[hash*2+1];
            self.local_data[i*4+2] = feature_value;

        }
        let prediction:f32 = (1.0+(-wsum).exp()).recip();
//        let prediction:f32 = sigmoid(wsum);      // ain't faster
        if update {
            let learning_rate = self.model_instance.learning_rate;
            let general_gradient = -(prediction - y);
            for i in 0..fbuf_len {
                let feature_value = self.local_data[i*4+2];
                let gradient = general_gradient * feature_value;
//                println!("Gradient: {}", self.local_data[i*4+1]);
                self.local_data[i*4+3] = gradient*gradient;	// it would be easier, to just use i*4+1 at the end... but this is how vowpal does it
                self.local_data[i*4+1] += self.local_data[i*4+3];
                let update_factor = gradient * learning_rate  * (self.local_data[i*4+1]).powf(self.minus_power_t);
                self.local_data[i*4] = update_factor;    // this is how vowpal does it, first calculate, then addk
            
            }
            // Next step is: gradients = weights_vector *  
            for i in 0..fbuf_len {     // speed of this is 4.53
                let hash = (fbuf[i*2] & self.hash_mask) as usize;
                self.weights[hash*2] += self.local_data[i*4];
                self.weights[hash*2+1] += self.local_data[i*4+3];
            }

        }
    //    println!("S {}, {}", y, prediction);
        prediction
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
        mi.hash_bits = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        // Empty model: no matter how many features, prediction is 0.5
        p = rr.learn(&vec![0], false);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE], false);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE, 2, ONE], false);
        assert_eq!(p, 0.5);
    }

    #[test]
    fn test_power_t_zero() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.hash_bits = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        
        p = rr.learn(&vec![0, 1, ONE], true);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE], true);
        assert_eq!(p, 0.48750263);
        p = rr.learn(&vec![0, 1, ONE], true);
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
        mi.hash_bits = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        let two = 2.0_f32.to_bits();
        
        p = rr.learn(&vec![0, 1, ONE, 1, two,], true);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE, 1, two,], true);
        assert_eq!(p, 0.38936076);
        p = rr.learn(&vec![0, 1, ONE, 1, two,], true);
        assert_eq!(p, 0.30993468);
    }


    #[test]
    fn test_power_t_half() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.hash_bits = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        
        p = rr.learn(&vec![0, 1, ONE], true);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE], true);
        assert_eq!(p, 0.4750208);
        p = rr.learn(&vec![0, 1, ONE], true);
        assert_eq!(p, 0.45788094);
    }

    #[test]
    fn test_power_t_half_two_features() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.hash_bits = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        
        // Here we take twice two features and then once just one
        p = rr.learn(&vec![0, 1, ONE, 2, ONE], true);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, ONE, 2, ONE], true);
        assert_eq!(p, 0.45016602);
        p = rr.learn(&vec![0, 1, ONE], true);
        assert_eq!(p, 0.45836908);
    }

    #[test]
    fn test_non_one_weight() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.hash_bits = 18;
        
        let mut rr = Regressor::new(&mi);
        let mut p: f32;
        let two = 2.0_f32.to_bits();
        
        p = rr.learn(&vec![0, 1, two], true);
        assert_eq!(p, 0.5);
        p = rr.learn(&vec![0, 1, two], true);
        assert_eq!(p, 0.45016602);
        p = rr.learn(&vec![0, 1, two], true);
        assert_eq!(p, 0.40611085);
    }


}







