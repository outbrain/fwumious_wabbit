use std::any::Any;
use std::io;
use merand48::*;
use core::arch::x86_64::*;
use std::error::Error;
use std::mem::{self, MaybeUninit};


use crate::optimizer;
use crate::regressor;
use crate::model_instance;
use crate::feature_buffer;
use crate::port_buffer;
use crate::consts;
use crate::block_helpers;
use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use block_helpers::{Weight, WeightAndOptimizerData};


#[derive(PartialEq)]
pub enum NeuronType {
    WeightedSum,
    LimitedWeightedSum,
}

#[derive(PartialEq)]
pub enum InitType {
    Random,
    RandomFirstNeuron1,
    RandomFirstNeuron10
}



pub struct BlockRELU {    
    pub num_inputs: u32,
    pub input_tape_index: i32,
    pub output_tape_index: i32,
}


pub fn new_without_weights(mi: &model_instance::ModelInstance, 
                                                    num_inputs: u32, 
                                                    ) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    assert!(num_inputs != 0);
    let mut rg = BlockRELU {
        output_tape_index: -1,
        input_tape_index: -1,
        num_inputs: num_inputs,
    };
    Ok(Box::new(rg))
}






impl BlockTrait for BlockRELU

 {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
    }

    fn get_num_outputs(&self) -> u32 {
        return self.num_inputs
    }
    
    fn set_input_tape_index(&mut self, input_tape_index: i32) {
        self.input_tape_index = input_tape_index;
    }

    fn set_output_tape_index(&mut self, output_tape_index: i32) {
        self.output_tape_index = output_tape_index;
    }

    fn get_output_tape_index(&self) -> i32 {
        self.output_tape_index
    }



    #[inline(always)]
    fn forward_backward(&mut self, 
                        further_blocks: &mut [Box<dyn BlockTrait>], 
                        fb: &feature_buffer::FeatureBuffer, 
                        pb: &mut port_buffer::PortBuffer, 
                        update:bool) {
        debug_assert!(self.output_tape_index >= 0);
        debug_assert!(self.input_tape_index >= 0);
        debug_assert!(self.input_tape_index != self.output_tape_index);
        debug_assert!(self.num_inputs > 0);
        
        unsafe {
            let len = pb.tapes[self.input_tape_index as usize].len();
            let output_tape_start = pb.tapes[self.output_tape_index as usize].len();
            let input_tape_start = pb.tapes[self.input_tape_index as usize].len() - self.num_inputs as usize; 

            for i in 0..self.num_inputs as usize {                                 
                let w = *(pb.tapes.get_unchecked_mut(self.input_tape_index as usize)).get_unchecked_mut(input_tape_start + i);
                if w < 0.0 {
                    pb.tapes.get_unchecked_mut(self.output_tape_index as usize).push(0.0);
                    // immediately change input tape to zero which will be the final gradient
                    *(pb.tapes.get_unchecked_mut(self.input_tape_index as usize)).get_unchecked_mut(input_tape_start + i) = 0.0; 
                } else {
                    pb.tapes.get_unchecked_mut(self.output_tape_index as usize).push(w);
                    *(pb.tapes.get_unchecked_mut(self.input_tape_index as usize)).get_unchecked_mut(input_tape_start + i) = 1.0;
                }
            }
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);

            if update {
                for i in 0..self.num_inputs as usize {
                    let w = *(pb.tapes.get_unchecked_mut(self.input_tape_index as usize)).get_unchecked_mut(input_tape_start + i);
                    let gradient = pb.tapes.get_unchecked(self.output_tape_index as usize).get_unchecked(output_tape_start + i);
                    *(pb.tapes.get_unchecked_mut(self.input_tape_index as usize)).get_unchecked_mut(input_tape_start + i) = gradient * w;
                }

            }
            pb.tapes[self.output_tape_index as usize].truncate(output_tape_start);
            
            // The only exit point
            return
            
        } // unsafe end
    }
    
    fn forward(&self, further_blocks: &[Box<dyn BlockTrait>], 
                        fb: &feature_buffer::FeatureBuffer, 
                        pb: &mut port_buffer::PortBuffer, 
                        ) {
        assert!(false, "Unimplemented");    
    }
    
    fn get_serialized_len(&self) -> usize {
        return 0;
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










mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::block_loss_functions;
    use crate::feature_buffer;
    use crate::feature_buffer::HashAndValueAndSeq;
    use crate::vwmap;
    use block_helpers::{slearn, spredict};

    use crate::assert_epsilon;

    fn fb_vec() -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
                    label: 0.0,
                    example_importance: 1.0,
                    example_number: 0,
                    lr_buffer: Vec::new(),
                    ffm_buffer: Vec::new(),
                    ffm_fields_count: 0,
        }
    }


    #[test]
    fn test_simple() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.optimizer = model_instance::Optimizer::SGD;
        
        
        let mut re = new_without_weights(&mi, 
                                            1, 
                                        ).unwrap();
        re.set_input_tape_index(0);
        re.set_output_tape_index(1);
        re.allocate_and_init_weights(&mi);
        
        let mut ib = block_loss_functions::new_identity_block(&mi, 1).unwrap();
        ib.set_input_tape_index(1);
        ib.set_output_tape_index(2);

        
        let mut pb = port_buffer::PortBuffer::new(&mi);
        let fb = fb_vec();
        pb.tapes[0].push(2.0);
        assert_epsilon!(slearn  (&mut re, &mut ib, &fb, &mut pb, true), 2.0);
        // what do we expect:
        // on tape 0 input of 2.0 will be replaced with the gradient of 1.0
        // on tape 1 input has been consumed by returning function
        // on tape 2 the output was consumed by slearn
        assert_eq!(pb.tapes[0][0], 1.0);
        assert_eq!(pb.tapes[1].len(), 0);
        assert_eq!(pb.tapes[2].len(), 0);
        pb.reset();
        pb.tapes[0].push(2.0);
        assert_epsilon!(slearn  (&mut re, &mut ib, &fb, &mut pb, true), 2.0); // relu desnt learn
        

    }


}



