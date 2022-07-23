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
    copy_to_result: bool
}

pub fn new_without_weights(mi: &model_instance::ModelInstance, 
                            num_inputs: u32,
                            copy_to_result: bool) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    Ok(Box::new(BlockSigmoid {num_inputs: num_inputs,
                                input_tape_index: -1,
                                output_tape_index: -1,
                                copy_to_result: copy_to_result}))
}


impl BlockTrait for BlockSigmoid {

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_num_outputs(&self) -> u32 {
        return 0
    }
    
    fn set_input_tape_index(&mut self, input_tape_index: i32) {
        self.input_tape_index = input_tape_index;
    }

    fn set_output_tape_index(&mut self, output_tape_index: i32) {
        self.output_tape_index = output_tape_index;
    }


    #[inline(always)]
    fn forward_backward(&mut self, 
                    further_blocks: &mut [Box<dyn BlockTrait>], 
                    fb: &feature_buffer::FeatureBuffer, 
                    pb: &mut port_buffer::PortBuffer, 
                    update:bool) {

        if further_blocks.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }
        debug_assert!(self.output_tape_index >= 0);
        debug_assert!(self.input_tape_index >= 0);
        debug_assert!(self.input_tape_index != self.output_tape_index);


        let len = pb.tapes[self.input_tape_index as usize].len();
        // Technically it needs to be longer. but for debugging we want to consume all of them
        if (self.num_inputs as usize) != len {
            panic!("BlockSigmoid::forward_backward() Number of inputs is different than number of values on the input tape: self.num_inputs: {} input tape: {}", self.num_inputs, len);
        }
        
//        println!("AAA: {}", len);
        let wsum:f32 = {
            let myslice = &pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize..];
            myslice.iter().sum()
        };
        // vowpal compatibility
        
        let mut prediction_probability: f32;
        let mut general_gradient: f32;
        
        if wsum.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", fb.example_number);
            prediction_probability = logistic(0.0);
            general_gradient = 0.0;
        } else if wsum < -50.0 {
            prediction_probability = logistic(-50.0);
            general_gradient = 0.0;
        } else if wsum > 50.0 {
            prediction_probability = logistic(50.0);
            general_gradient = 0.0;
        } else {
            prediction_probability = logistic(wsum);
            general_gradient = - (fb.label - prediction_probability) * fb.example_importance;
        }
        //println!("General gradient: {}", general_gradient);
        pb.tapes[self.output_tape_index as usize].push(prediction_probability);
        if self.copy_to_result {
            pb.results.push(prediction_probability);
        }
        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);
        }


        {
            // replace inputs with their gradients
            let mut myslice = &mut pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize..];
            for s in myslice.iter_mut() {
                *s = general_gradient;
            }
        }
    }

    fn forward(&self, 
                     further_blocks: &[Box<dyn BlockTrait>], 
                     fb: &feature_buffer::FeatureBuffer,
                     pb: &mut port_buffer::PortBuffer, ) {

        if further_blocks.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }
        debug_assert!(self.output_tape_index >= 0);
        debug_assert!(self.input_tape_index >= 0);
        debug_assert!(self.input_tape_index != self.output_tape_index);


        let len = pb.tapes[self.input_tape_index as usize].len();
        // Technically it needs to be longer. but for debugging we want to consume all of them
        if (self.num_inputs as usize) != len {
            panic!("BlockSigmoid::forward_backward() Number of inputs is different than number of values on the input tape: self.num_inputs: {} input tape: {}", self.num_inputs, len);
        }
        
        let wsum:f32 = {
            let myslice = &pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize..];
            myslice.iter().sum()
        };
        
        let prediction_probability:f32;
        if wsum.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", fb.example_number);
            prediction_probability = logistic(0.0);
        } else if wsum < -50.0 {
            prediction_probability = logistic(-50.0);
        } else if wsum > 50.0 {
            prediction_probability = logistic(50.0);
        } else {
            prediction_probability = logistic(wsum);
        }
        
        pb.tapes[self.output_tape_index as usize].push(prediction_probability);
        if self.copy_to_result {
            pb.results.push(prediction_probability);
        }
        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at(1);
            next_regressor[0].forward(further_blocks, fb, pb);
        }
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

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}





pub struct BlockResult {
    num_inputs: u32,
    input_tape_index: i32,
    replace_input_with: f32,
}

pub fn new_result_block(num_inputs: u32, replace_input_with: f32) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    Ok(Box::new(BlockResult {num_inputs: num_inputs,
                             input_tape_index: -1,
                             replace_input_with: replace_input_with}))
}


impl BlockTrait for BlockResult {
    // Warning: It does not confirm to regular clean-up after itself

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_num_outputs(&self) -> u32 {
        // this means outputs on regular tapes
        return 0
    }
    
    fn set_input_tape_index(&mut self, input_tape_index: i32) {
        self.input_tape_index = input_tape_index;
    }

    fn set_output_tape_index(&mut self, output_tape_index: i32) {
        panic!("Output tape of the BlockResult is automatically in result");
    }


    #[inline(always)]
    fn forward_backward(&mut self, 
                    further_blocks: &mut [Box<dyn BlockTrait>], 
                    fb: &feature_buffer::FeatureBuffer, 
                    pb: &mut port_buffer::PortBuffer, 
                    update:bool) {
        debug_assert!(self.input_tape_index >= 0);

        let len = pb.tapes[self.input_tape_index as usize].len();
        // Technically it needs to be longer. but for debugging we want to consume all of them
        if (self.num_inputs as usize) != len {
            panic!("BlockResult::forward_backward() Number of inputs is different than number of values on the input tape");
        }
        
        // copy inputs to result and replace inputs with whatever we have
        // copy inputs to result
        for x in 0..(self.num_inputs as usize) {
            let s = pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize + x];
            pb.results.push(s);
        }

        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);
        }
        
        // replace inputs with whatever we wanted
        for x in 0..(self.num_inputs as usize) {
            pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize + x] = self.replace_input_with;
        }

    }

    fn forward(&self, 
                     further_blocks: &[Box<dyn BlockTrait>], 
                     fb: &feature_buffer::FeatureBuffer,
                     pb: &mut port_buffer::PortBuffer, ) {
        debug_assert!(self.input_tape_index >= 0);

        let len = pb.tapes[self.input_tape_index as usize].len();
        // Technically it needs to be longer. but for debugging we want to consume all of them
        if (self.num_inputs as usize) != len {
            panic!("BlockSigmoid::forward_backward() Number of inputs is different than number of values on the input tape");
        }
        
        // copy inputs to result
        for x in 0..(self.num_inputs as usize) {
            let s = pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize + x];
            pb.results.push(s);
        }
        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at(1);
            next_regressor[0].forward(further_blocks, fb, pb);
        }

        // replace inputs with whatever we wanted
        for x in 0..(self.num_inputs as usize) {
            pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize + x] = self.replace_input_with;
        }


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

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}





pub struct BlockZeros {
    num_outputs: u32,
    output_tape_index: i32,
}

pub fn new_zero_block(num_outputs: u32) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    Ok(Box::new(BlockZeros {num_outputs: num_outputs,
                                output_tape_index: -1}))
}


impl BlockTrait for BlockZeros {
    // Warning: It does not confirm to regular clean-up after itself

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_num_outputs(&self) -> u32 {
        self.num_outputs
    }
    
    fn set_input_tape_index(&mut self, input_tape_index: i32) {
        panic!("You cannnot set input_tape_index for BlockZeros");

    }

    fn set_output_tape_index(&mut self, output_tape_index: i32) {
        self.output_tape_index = output_tape_index;
    }


    #[inline(always)]
    fn forward_backward(&mut self, 
                    further_blocks: &mut [Box<dyn BlockTrait>], 
                    fb: &feature_buffer::FeatureBuffer, 
                    pb: &mut port_buffer::PortBuffer, 
                    update:bool) {
        debug_assert!(self.output_tape_index >= 0);

        let original_size = pb.tapes[self.output_tape_index as usize].len();
        // Technically it needs to be longer. but for debugging we want to consume all of them

        for x in 0..self.num_outputs as usize {
            pb.tapes[self.output_tape_index as usize].push(0.0);
        }
        let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
        next_regressor[0].forward_backward(further_blocks, fb, pb, update);
        pb.tapes[self.output_tape_index as usize].truncate(original_size );
    }

    fn forward(&self, 
                     further_blocks: &[Box<dyn BlockTrait>], 
                     fb: &feature_buffer::FeatureBuffer,
                     pb: &mut port_buffer::PortBuffer, ) {
        debug_assert!(self.output_tape_index >= 0);

        let original_size = pb.tapes[self.output_tape_index as usize].len();
        // Technically it needs to be longer. but for debugging we want to consume all of them

        for x in 0..self.num_outputs as usize {
            pb.tapes[self.output_tape_index as usize].push(0.0);
        }
        let (next_regressor, further_blocks) = further_blocks.split_at(1);
        next_regressor[0].forward(further_blocks, fb, pb);
        pb.tapes[self.output_tape_index as usize].truncate(original_size );
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

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}
























