

pub trait LearningRateTrait {
    fn new() -> Self;
    fn init(&mut self, learning_rate: f32, minus_power_t: f32);
    unsafe fn calculate_update(&self, update: f32, accumulated_squared_gradient: f32) -> f32;
}

/******************* SGD **************************/
pub struct LearningRateSGD {
    learning_rate: f32,    
}

impl LearningRateTrait for LearningRateSGD {
    fn new() -> Self {
        LearningRateSGD{learning_rate: 0.0}
    } 
    fn init(&mut self, learning_rate: f32, minus_power_t: f32) {
        self.learning_rate = learning_rate;
    }

    #[inline(always)]
    unsafe fn calculate_update(&self, update: f32, accumulated_squared_gradient: f32) -> f32{
   //     println!("LR: {}", self.learning_rate);
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

    fn init(&mut self, learning_rate: f32, minus_power_t: f32) {
        self.learning_rate = learning_rate;
        self.minus_power_t = minus_power_t;
    }

    #[inline(always)]
    unsafe fn calculate_update(&self, update: f32, accumulated_squared_gradient: f32) -> f32{
        let learning_rate = self.learning_rate * (accumulated_squared_gradient).powf(self.minus_power_t);
      //  println("Flex: {}", learning_rate);
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
    
    fn init(&mut self, learning_rate: f32, minus_power_t: f32) {
        println!("Calculating look-up tables for Adagrad learning rate calculation");
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



