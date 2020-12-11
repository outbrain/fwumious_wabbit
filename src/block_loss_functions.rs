use std::error::Error;
use std::io;

use crate::regressor;
use crate::feature_buffer;
use crate::model_instance;
use regressor::BlockTrait;
use regressor::{Weight};


//use fastapprox::fast::sigmoid; // surprisingly this doesn't work very well

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



pub struct BlockSigmoid {
}

impl BlockTrait for BlockSigmoid {

    #[inline(always)]
    fn forward_backward(&mut self, 
                    further_regressors: &mut [&mut dyn BlockTrait], 
                    wsum: f32, 
                    example_num: u32, 
                    fb: &feature_buffer::FeatureBuffer, 
                    update:bool) -> (f32, f32) {
        if further_regressors.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }
        
        // vowpal compatibility
        if wsum.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", example_num);
            return (logistic(0.0), 0.0);
        } else if wsum < -50.0 {
            return (logistic(-50.0), 0.0);
        } else if wsum > 50.0 {
            return (logistic(50.0), 0.0);
        }        

        let prediction_probability = logistic(wsum);
        let general_gradient = (fb.label - prediction_probability) * fb.example_importance;
        //println!("General gradient: {}", general_gradient);
        (prediction_probability, general_gradient)
    }
    
    fn forward(&self, 
                     further_blocks: &mut [&dyn BlockTrait], 
                     wsum: f32, 
                     example_num: u32, 
                     fb: &feature_buffer::FeatureBuffer) -> f32 {

        if further_blocks.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }
        
        // vowpal compatibility
        if wsum.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", example_num);
            return logistic(0.0);
        } else if wsum < -50.0 {
            return logistic(-50.0);
        } else if wsum > 50.0 {
            return logistic(50.0);
        }        

        let prediction_probability = logistic(wsum);
        prediction_probability
    }

    
    
    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        // empty
    }
    fn get_weights_len(&self) -> usize {
        return 0
    }

    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn read_immutable_weights_from_buf(&self, out_weights: &mut Vec<Weight>, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn get_forwards_only_version(&self) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        Ok(Box::new(BlockSigmoid{}))
    }

}

