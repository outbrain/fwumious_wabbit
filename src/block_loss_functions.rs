use std::any::Any;
use std::error::Error;
use std::io;

use crate::regressor;
use crate::feature_buffer;
use crate::port_buffer;
use crate::model_instance;

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
    return (1.0+(-t).exp()).recip();
}



pub struct BlockSigmoid {
    num_inputs: u32,
    input_tape_index: i32,
    output_tape_index: i32,
}

impl BlockTrait for BlockSigmoid {

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn new_without_weights(mi: &model_instance::ModelInstance) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        Ok(Box::new(BlockSigmoid {num_inputs: 0,
                                    input_tape_index: -1,
                                    output_tape_index: -1}))
    }


    fn get_num_outputs(&self) -> u32 {
        return 0
    }
    
    fn set_num_inputs(&mut self, num_inputs: u32) {
        if num_inputs <= 0 {
          panic!("set_num_inputs(): num_inputs for BlockSigmoid has to be greater than 0");
        }
        self.num_inputs = num_inputs;
    }

    fn set_input_tape_index(&mut self, input_tape_index: i32) {
        self.input_tape_index = input_tape_index;
    }

    fn set_output_tape_index(&mut self, output_tape_index: i32) {
        self.output_tape_index = output_tape_index;
    }

    #[inline(always)]
    fn forward_backward(&mut self, 
                    further_regressors: &mut [Box<dyn BlockTrait>], 
                    fb: &feature_buffer::FeatureBuffer, 
                    pb: &mut port_buffer::PortBuffer, 
                    update:bool) -> (f32, f32) {
        if further_regressors.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }
        

        let len = pb.tapes[self.input_tape_index as usize].len();
        // Technically it needs to be longer. but for debugging we want to consume all of them
        if (self.num_inputs as usize) != len {
            panic!("BlockSigmoid::forward_backward() Number of inputs is different than number of values on the input tape");
        }
        
//        println!("AAA: {}", len);
        let wsum:f32 = pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize..].iter().sum();

        // vowpal compatibility
        if wsum.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", fb.example_number);
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
                     further_blocks: &[Box<dyn BlockTrait>], 
                     wsum: f32, 
                     fb: &feature_buffer::FeatureBuffer) -> f32 {

        if further_blocks.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }
        
        // vowpal compatibility
        if wsum.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", fb.example_number);
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
        return 0
    }

    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn read_weights_from_buf_into_forward_only(&self, input_bufreader: &mut dyn io::Read, forward: &mut Box<dyn BlockTrait>) -> Result<(), Box<dyn Error>> {
        Ok(())        
    }

    fn new_forward_only_without_weights(&self) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        Ok(Box::new(BlockSigmoid{num_inputs: 0,
                                input_tape_index: -1,
                                output_tape_index: -1}))
    }
    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

}

