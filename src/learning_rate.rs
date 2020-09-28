

pub trait LearningRateTrait {
    fn new() -> Self;
    fn init(&mut self, learning_rate: f32, power_t: f32);
    unsafe fn calculate_update(&self, update: f32, accumulated_squared_gradient: f32) -> f32;
}

/******************* SGD **************************/
// This is non-adaptive fixed learning rate SGD, which is exactly the same as Vowpal when --power_t is 0.0
pub struct LearningRateSGD {
    learning_rate: f32,    
}

impl LearningRateTrait for LearningRateSGD {
    fn new() -> Self {
        LearningRateSGD{learning_rate: 0.0}
    } 
    fn init(&mut self, learning_rate: f32, power_t: f32) {
        self.learning_rate = learning_rate;
    }

    #[inline(always)]
    unsafe fn calculate_update(&self, update: f32, accumulated_squared_gradient: f32) -> f32{
        return update * self.learning_rate;
    }
}


/******************* Adagrad with flexible power_t  **************************/
pub struct LearningRateAdagradFlex {
    learning_rate: f32,   
    minus_power_t: f32,
}

impl LearningRateTrait for LearningRateAdagradFlex {
    fn new() -> Self {
        LearningRateAdagradFlex{learning_rate: 0.0, minus_power_t: 0.0}
    } 

    fn init(&mut self, learning_rate: f32, power_t: f32) {
        self.learning_rate = learning_rate;
        self.minus_power_t = - power_t;
    }

    #[inline(always)]
    unsafe fn calculate_update(&self, update: f32, accumulated_squared_gradient: f32) -> f32{
        let learning_rate = self.learning_rate * (accumulated_squared_gradient).powf(self.minus_power_t);
        return update * learning_rate;
    }
}



/***************** Adagrad using Look Up Table ******************/
pub const FASTMATH_LR_LUT_BITS:u8 = 11;
pub const FASTMATH_LR_LUT_SIZE:usize = 1 <<  FASTMATH_LR_LUT_BITS;

pub struct LearningRateAdagradLUT {
   pub fastmath_lr_lut: [f32; FASTMATH_LR_LUT_SIZE], 
}

impl LearningRateTrait for LearningRateAdagradLUT {
    fn new() -> Self {
        LearningRateAdagradLUT{fastmath_lr_lut: [0.0;FASTMATH_LR_LUT_SIZE]}
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
    unsafe fn calculate_update(&self, update: f32, accumulated_squared_gradient: f32) -> f32 {
        debug_assert!(accumulated_squared_gradient >= 0.0);
        let key = accumulated_squared_gradient.to_bits() >> (31-FASTMATH_LR_LUT_BITS);
        return update * *self.fastmath_lr_lut.get_unchecked(key as usize);
    }
}




mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_sgd() {
        let mut l = LearningRateSGD::new();
        l.init(0.15, 0.4);
        unsafe {
            let p = l.calculate_update(0.1, 0.9);
            assert_eq!(p, 0.1* 0.15);
        }
    }

    #[test]
    fn test_adagradflex() {
        let mut l = LearningRateAdagradFlex::new();
        l.init(0.15, 0.4);
        unsafe {
            let p = l.calculate_update(0.1, 0.9);
            assert_eq!(p, 0.015645673);
            let p = l.calculate_update(0.1, 0.0);
            assert_eq!(p, f32::INFINITY);
        }
    }
    
    #[test]
    fn test_adagradlut() {
        let mut l = LearningRateAdagradLUT::new();
        l.init(0.15, 0.4);
        unsafe {
            let p = l.calculate_update(0.1, 0.9);
            assert_eq!(p, 0.015607622);         
            let p = l.calculate_update(0.1, 0.0);
            assert_eq!(p, f32::INFINITY);
        }
    }

    #[test]
    fn test_adagradlut_comparison() {
        // Here we test that our implementation of LUT has small enough relative error
        let mut l_lut = LearningRateAdagradFlex::new();
        let mut l_flex = LearningRateAdagradLUT::new();
        l_lut.init(0.15, 0.4);
        l_flex.init(0.15, 0.4);
        let test_gradients = [-1.0, -0.9, -0.1, -0.00001, 0.00001, 0.1, 0.5, 0.9, 1.0];
        let test_accumulations = [0.0000000001, 0.00001, 0.1, 0.5, 1.1, 2.0, 20.0, 200.0, 2000.0, 200000.0, 2000000.0];

        unsafe {
            let mut error_sum = 0.0;
            for gradient in test_gradients.iter() {
                for accumulation in test_accumulations.iter() {
                    let p_flex = l_flex.calculate_update(*gradient, *accumulation);
                    let p_lut = l_lut.calculate_update(*gradient, *accumulation);
                    let error = (p_flex - p_lut).abs();
                    let relative_error = error / p_flex.abs();
                    println!("Relative error {}", relative_error);
                    assert!(relative_error < 0.05); 
//                    println!("Err: {} - p_flex: {}, p_lut: {}, gradient: {}, accumulation {}", error, p_flex, p_lut, *gradient, *accumulation);
                }
            }
        }
    }


}




