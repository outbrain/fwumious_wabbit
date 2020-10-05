
use std::marker::PhantomData;


pub trait OptimizerTrait {
    type PerWeightStore;
    fn new() -> Self;
    fn init(&mut self, learning_rate: f32, power_t: f32);
    unsafe fn calculate_update(&self, gradient: f32, data: &mut Self::PerWeightStore) -> f32;
    fn empty_initial_data() -> Self::PerWeightStore;
    fn ffm_initial_data() -> Self::PerWeightStore;
    fn get_name() -> &'static str;
}

/******************* SGD **************************/
// This is non-adaptive fixed learning rate SGD, which is exactly the same as Vowpal when --power_t is 0.0
#[derive(Clone)]
pub struct OptimizerSGD {
    learning_rate: f32,    
}

impl OptimizerTrait for OptimizerSGD {
    type PerWeightStore = PhantomData<u32>;
    
    fn get_name() -> &'static str {
        "SGD"
    }
    
    fn new() -> Self {
        OptimizerSGD{learning_rate: 0.0}
    } 
    
    fn init(&mut self, learning_rate: f32, power_t: f32) {
        self.learning_rate = learning_rate;
    }

    #[inline(always)]
    unsafe fn calculate_update(&self, gradient: f32, data: &mut Self::PerWeightStore) -> f32 {
        return gradient * self.learning_rate;
    }

    fn empty_initial_data() -> Self::PerWeightStore {
        std::marker::PhantomData{}
    }
    fn ffm_initial_data() -> Self::PerWeightStore {
        std::marker::PhantomData{}
    }
}


/******************* Adagrad with flexible power_t  **************************/
/* Regular Adagrad always uses sqrt (power_t = 0.5)                          */
#[derive(Clone)]
pub struct OptimizerAdagradFlex {
    learning_rate: f32,   
    minus_power_t: f32,
}

impl OptimizerTrait for OptimizerAdagradFlex {
    fn get_name() -> &'static str {
        "AdagradFlex"
    }
    type PerWeightStore = f32;

    fn new() -> Self {
        OptimizerAdagradFlex{learning_rate: 0.0, minus_power_t: 0.0}
    } 

    fn init(&mut self, learning_rate: f32, power_t: f32) {
        self.learning_rate = learning_rate;
        self.minus_power_t = - power_t;
    }

    #[inline(always)]
    unsafe fn calculate_update(&self, gradient: f32, data: &mut Self::PerWeightStore) -> f32 {
        let accumulated_gradient_squared = *data;
        let gradient_squared = gradient * gradient;
        let new_accumulated_gradient_squared = accumulated_gradient_squared + gradient_squared;
        *data = new_accumulated_gradient_squared;
        let update =  gradient * self.learning_rate * (new_accumulated_gradient_squared).powf(self.minus_power_t);
        return update;
    }
    
    fn empty_initial_data() -> Self::PerWeightStore {
        0.0f32
    }
    fn ffm_initial_data() -> Self::PerWeightStore {
        1.0f32
    }
    
}



/***************** Adagrad using Look Up Table ******************/
pub const FASTMATH_LR_LUT_BITS:u8 = 11;
pub const FASTMATH_LR_LUT_SIZE:usize = 1 <<  FASTMATH_LR_LUT_BITS;

#[derive(Clone, Copy)]
pub struct OptimizerAdagradLUT {
   pub fastmath_lr_lut: [f32; FASTMATH_LR_LUT_SIZE], 
}

impl OptimizerTrait for OptimizerAdagradLUT {
    fn get_name() -> &'static str {
        "AdagradLUT"
    }
    type PerWeightStore = f32;

    fn new() -> Self {
        OptimizerAdagradLUT{fastmath_lr_lut: [0.0;FASTMATH_LR_LUT_SIZE]}
    } 
    
    fn init(&mut self, learning_rate: f32, power_t: f32) {
        println!("Calculating look-up tables for Adagrad learning rate calculation");
        let minus_power_t = -power_t;
        for x in 0..FASTMATH_LR_LUT_SIZE {
            // accumulated gradients are always positive floating points, sign is guaranteed to be zero
            // floating point: 1 bit of sign, 7 bits of signed expontent then floating point bits (mantissa)
            // we will take 7 bits of exponent + whatever most significant bits of mantissa remain
            // we take two consequtive such values, so we act as if had rounding
            let float_x = f32::from_bits((x as u32)  << (31-FASTMATH_LR_LUT_BITS));
            let float_x_plus_one = f32::from_bits(((x+1) as u32)  << (31-FASTMATH_LR_LUT_BITS));
            let mut val = learning_rate * ((float_x).powf(minus_power_t) + (float_x_plus_one).powf(minus_power_t)) * 0.5;
            // Safety measure
            /*if val > learning_rate || val.is_nan() {
                val = learning_rate;
            }*/
            
            self.fastmath_lr_lut[x] = val;
        }
    }
    
    #[inline(always)]
    unsafe fn calculate_update(&self, gradient: f32, data: &mut Self::PerWeightStore) -> f32 {
        let accumulated_gradient_squared = *data;
        debug_assert!(accumulated_gradient_squared >= 0.0);
        let gradient_squared = gradient * gradient;
        let new_accumulated_gradient_squared = accumulated_gradient_squared + gradient_squared;
        *data = new_accumulated_gradient_squared;
        let key = new_accumulated_gradient_squared.to_bits() >> (31-FASTMATH_LR_LUT_BITS);
        let update = gradient * *self.fastmath_lr_lut.get_unchecked(key as usize);
        return update;
    }

    fn empty_initial_data() -> Self::PerWeightStore {
        0.0f32
    }
    fn ffm_initial_data() -> Self::PerWeightStore {
        1.0f32
    }

}





mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_sgd() {
        let mut l = OptimizerSGD::new();
        l.init(0.15, 0.4);
        unsafe {
            let mut acc: PhantomData<u32> = std::marker::PhantomData{};
            let p = l.calculate_update(0.1, &mut acc);
            assert_eq!(p, 0.1* 0.15);
        }
    }

    #[test]
    fn test_adagradflex() {
        let mut l = OptimizerAdagradFlex::new();
        l.init(0.15, 0.4);
        unsafe {
            let mut acc: f32;
            acc = 0.9;
            let p = l.calculate_update(0.1, &mut acc);
            assert_eq!(p, 0.015576674);
            assert_eq!(acc, 0.9 + 0.1*0.1);

            acc = 0.0;
            let p = l.calculate_update(0.1, &mut acc);
            assert_eq!(p, 0.09464361);
            assert_eq!(acc, 0.1*0.1);
            
        }
    }

    #[test]
    fn test_adagradlut() {
        let mut l = OptimizerAdagradLUT::new();
        l.init(0.15, 0.4);
        unsafe {
            let mut acc: f32;
            acc = 0.9;
            let p = l.calculate_update(0.1, &mut acc);
            assert_eq!(p, 0.015607622);
            assert_eq!(acc, 0.9 + 0.1*0.1);

            acc = 0.0;
            let p = l.calculate_update(0.1, &mut acc);
            assert_eq!(p, 0.09375872);
            assert_eq!(acc, 0.1*0.1);
            
        }
    }


    
    #[test]
    fn test_adagradlut_comparison() {
        // Here we test that our implementation of LUT has small enough relative error
        let mut l_lut = OptimizerAdagradFlex::new();
        let mut l_flex = OptimizerAdagradLUT::new();
        l_lut.init(0.15, 0.4);
        l_flex.init(0.15, 0.4);
        let test_gradients = [-1.0, -0.9, -0.1, -0.00001, 0.0, 0.00001, 0.1, 0.5, 0.9, 1.0];
        let test_accumulations = [0.0000000001, 0.00001, 0.1, 0.5, 1.1, 2.0, 20.0, 200.0, 2000.0, 200000.0, 2000000.0];

        unsafe {
            let mut error_sum = 0.0;
            for gradient in test_gradients.iter() {
                for accumulation in test_accumulations.iter() {
                    let mut acc_flex: f32 = *accumulation;
                    let p_flex = l_flex.calculate_update(*gradient, &mut acc_flex);
                    let mut acc_lut: f32 = *accumulation;
                    let p_lut = l_lut.calculate_update(*gradient, &mut acc_lut);
                    let error = (p_flex - p_lut).abs();
                    let mut relative_error:f32;
                    if p_flex != 0.0 {
                        relative_error = error / p_flex.abs();
                    } else {
                        relative_error = error; // happens when the update is 0.0
                    }
//                    println!("Relative error {}", relative_error);
//                    println!("Err: {} - p_flex: {}, p_lut: {}, gradient: {}, accumulation {}", error, p_flex, p_lut, *gradient, *accumulation);
                    assert!(relative_error < 0.05); 
                }
            }
        }
    }


}




