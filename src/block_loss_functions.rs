use std::any::Any;
use std::error::Error;
use std::io;

use crate::feature_buffer;
use crate::model_instance;
use crate::regressor;
use regressor::BlockTrait;

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
    return (1.0 + (-t).exp()).recip();
}

pub struct BlockSigmoid {}

impl BlockTrait for BlockSigmoid {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn new_without_weights(
        mi: &model_instance::ModelInstance,
    ) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        Ok(Box::new(BlockSigmoid {}))
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_regressors: &mut [Box<dyn BlockTrait>],
        wsum: f32,
        fb: &feature_buffer::FeatureBuffer,
        update: bool,
    ) -> (f32, f32) {
        if further_regressors.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }

        // vowpal compatibility
        if wsum.is_nan() {
            eprintln!(
                "NAN prediction in example {}, forcing 0.0",
                fb.example_number
            );
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

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        wsum: f32,
        fb: &feature_buffer::FeatureBuffer,
    ) -> f32 {
        if further_blocks.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }

        // vowpal compatibility
        if wsum.is_nan() {
            eprintln!(
                "NAN prediction in example {}, forcing 0.0",
                fb.example_number
            );
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
    fn get_serialized_len(&self) -> usize {
        return 0;
    }

    fn read_weights_from_buf(
        &mut self,
        input_bufreader: &mut dyn io::Read,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn write_weights_to_buf(
        &self,
        output_bufwriter: &mut dyn io::Write,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn read_weights_from_buf_into_forward_only(
        &self,
        input_bufreader: &mut dyn io::Read,
        forward: &mut Box<dyn BlockTrait>,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn new_forward_only_without_weights(&self) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        Ok(Box::new(BlockSigmoid {}))
    }
    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(
        &mut self,
        aa: i32,
        bb: i32,
        index: usize,
        w: &[f32],
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}
