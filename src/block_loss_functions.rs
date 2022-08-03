use std::any::Any;
use std::error::Error;
use std::io;

use crate::regressor;
use crate::feature_buffer;
use crate::port_buffer;
use crate::model_instance;
use crate::graph;
use crate::block_helpers;
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
    num_inputs: usize,
    input_offset: usize,
    output_offset: usize,
    copy_to_result: bool
}


pub fn new_logloss_block(  bg: &mut graph::BlockGraph, 
                           input: graph::BlockPtrOutput,
                           copy_to_result: bool) 
                        -> Result<graph::BlockPtrOutput, Box<dyn Error>> {    
    let num_inputs = bg.get_num_output_values(vec![&input]);
    let block = Box::new(BlockSigmoid {num_inputs: num_inputs as usize,
                                input_offset: usize::MAX,
                                output_offset: usize::MAX,
                                copy_to_result: copy_to_result});
    let mut block_outputs = bg.add_node(block, vec![input]).unwrap();
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())

}


pub fn new_without_weights(mi: &model_instance::ModelInstance, 
                            num_inputs: u32,
                            copy_to_result: bool) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    Ok(Box::new(BlockSigmoid {num_inputs: num_inputs as usize,
                                input_offset: usize::MAX,
                                output_offset: usize::MAX,
                                copy_to_result: copy_to_result}))
}


impl BlockTrait for BlockSigmoid {

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_num_output_slots(&self) -> usize {1}   


    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert!(output.get_output_index() == 0);
        1
    }
    
    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize)  {
        assert!(input.get_input_index() == 0);
        assert!(self.input_offset == usize::MAX); // We only allow a single call
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize)  {
        assert!(self.output_offset == usize::MAX); // We only allow a single call
        assert!(output.get_output_index() == 0);
        self.output_offset = offset;
    }


    #[inline(always)]
    fn forward_backward(&mut self, 
                    further_blocks: &mut [Box<dyn BlockTrait>], 
                    fb: &feature_buffer::FeatureBuffer, 
                    pb: &mut port_buffer::PortBuffer, 
                    update:bool) {

        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.output_offset != usize::MAX);

        unsafe {

            let wsum:f32 = {
                let myslice = &pb.tape.get_unchecked(self.input_offset .. (self.input_offset + self.num_inputs));
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
            *pb.tape.get_unchecked_mut(self.output_offset) = prediction_probability;
            if self.copy_to_result {
                pb.observations.push(prediction_probability);
            }
            block_helpers::forward_backward(further_blocks, fb, pb, update);
//            general_gradient *= *pb.tape.get_unchecked(self.output_offset);
        // replace inputs with their gradients
            pb.tape.get_unchecked_mut(self.input_offset .. (self.input_offset + self.num_inputs)).fill(general_gradient);
        }
    }

    fn forward(&self, 
                     further_blocks: &[Box<dyn BlockTrait>], 
                     fb: &feature_buffer::FeatureBuffer,
                     pb: &mut port_buffer::PortBuffer, ) {

/*        if further_blocks.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }*/
        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.output_offset != usize::MAX);

        let wsum:f32 = {
            let myslice = &pb.tape[self.input_offset .. (self.input_offset + self.num_inputs)];
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
        
        pb.tape[self.output_offset] = prediction_probability;
        if self.copy_to_result {
            pb.observations.push(prediction_probability);
        }
        block_helpers::forward(further_blocks, fb, pb);
    }

}






