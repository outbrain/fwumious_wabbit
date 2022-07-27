use std::any::Any;
use std::error::Error;
use std::io;

use crate::regressor;
use crate::feature_buffer;
use crate::port_buffer;
use crate::model_instance;
use crate::graph;

use regressor::BlockTrait;

pub struct BlockResult {
    num_inputs: usize,
    input_offset: usize,
    replace_input_with: f32,
}


pub fn new_result_block2(bg: &mut graph::BlockGraph,
                        input: graph::BlockPtrOutput,
                        replace_input_with: f32) 
                        -> Result<(), Box<dyn Error>> {

    let num_inputs = bg.get_num_outputs(vec![&input]);
    println!("Inputs: {} vec: {:?}", num_inputs, input);
    let block = Box::new(BlockResult {
                         num_inputs: num_inputs as usize,
                         input_offset: usize::MAX,
                         replace_input_with: replace_input_with});
    let block_outputs = bg.add_node(block, vec![input]);
    assert_eq!(block_outputs.len(), 0);
    Ok(())
}


pub fn new_result_block(num_inputs: usize, replace_input_with: f32) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    Ok(Box::new(BlockResult {num_inputs: num_inputs,
                             input_offset: usize::MAX,
                             replace_input_with: replace_input_with}))
}


impl BlockTrait for BlockResult {
    // Warning: It does not confirm to regular clean-up after itself

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_num_output_tapes(&self) -> usize {0}   

    fn get_num_outputs(&self, output: graph::BlockOutput) -> usize {
        assert!(output.get_output_id() == 0);
        // this means outputs on regular tapes
        return self.num_inputs
    }

    fn set_input_offset(&mut self, input: graph::BlockInput, offset: usize)  {
        assert!(input.get_input_id() == 0);
        self.input_offset = offset;
    }

    
    #[inline(always)]
    fn forward_backward(&mut self, 
                    further_blocks: &mut [Box<dyn BlockTrait>], 
                    fb: &feature_buffer::FeatureBuffer, 
                    pb: &mut port_buffer::PortBuffer, 
                    update:bool) {
        debug_assert!(self.input_offset != usize::MAX);

        // copy inputs to result
        println!("Result block with Num inputs: {}", self.num_inputs);
        for i in 0..self.num_inputs {
            let s = pb.tape[self.input_offset + i];
            pb.results.push(s);
        }

        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);
        }
        
        // replace inputs with whatever we wanted
        for i in 0..self.num_inputs {
            pb.tape[self.input_offset + i] = self.replace_input_with;
        }

    }

    fn forward(&self, 
                     further_blocks: &[Box<dyn BlockTrait>], 
                     fb: &feature_buffer::FeatureBuffer,
                     pb: &mut port_buffer::PortBuffer
                     ) {
        debug_assert!(self.input_offset != usize::MAX);
        
        // copy inputs to result
        for i in 0..self.num_inputs {
            let s = pb.tape[self.input_offset + i];
            pb.results.push(s);
        }
        
        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at(1);
            next_regressor[0].forward(further_blocks, fb, pb);
        }

        // replace inputs with whatever we wanted
        for i in 0..self.num_inputs {
            pb.tape[self.input_offset + i] = self.replace_input_with;
        }


    }

}





pub struct BlockConsts {
    pub output_offset: usize,
    consts: Vec<f32>,
    
}
/*
pub fn new_const_block(consts: Vec<f32>) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    Ok(Box::new(BlockConsts {   output_tape_index: -1,
                                consts: consts}))
}*/

pub fn new_const_block( bg: &mut graph::BlockGraph, 
                        consts: Vec<f32>) 
                        -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let block = Box::new(BlockConsts {   output_offset: usize::MAX,
                                         consts: consts});
    let mut block_outputs = bg.add_node(block, vec![]);
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())


}



impl BlockTrait for BlockConsts {
    // Warning: It does not confirm to regular clean-up after itself

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_num_output_tapes(&self) -> usize {1}   

    fn get_num_outputs(&self, output: graph::BlockOutput) -> usize {
        assert!(output.get_output_id() == 0);
        self.consts.len() as usize
    }
    
    fn set_input_offset(&mut self, input: graph::BlockInput, offset: usize)  {
        panic!("You cannnot set input_tape_index for BlockConsts");
    }

    fn set_output_offset(&mut self, output: graph::BlockOutput, offset: usize) {
        assert!(output.get_output_id() == 0, "Only supports a single output for BlockConsts");
        self.output_offset = offset;
    }

    #[inline(always)]
    fn forward_backward(&mut self, 
                    further_blocks: &mut [Box<dyn BlockTrait>], 
                    fb: &feature_buffer::FeatureBuffer, 
                    pb: &mut port_buffer::PortBuffer, 
                    update:bool) {
        debug_assert!(self.output_offset != usize::MAX);

        pb.tape[self.output_offset..(self.output_offset + self.consts.len())].copy_from_slice(&self.consts);

        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);
        }
    }

    fn forward(&self, 
                     further_blocks: &[Box<dyn BlockTrait>], 
                     fb: &feature_buffer::FeatureBuffer,
                     pb: &mut port_buffer::PortBuffer, ) {

        debug_assert!(self.output_offset != usize::MAX);
        pb.tape[self.output_offset..(self.output_offset + self.consts.len())].copy_from_slice(&self.consts);

        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at(1);
            next_regressor[0].forward(further_blocks, fb, pb);
        }

    }

}


pub struct BlockCopy {    
    pub num_inputs: usize,
    pub input_offset: usize,
    pub output_offset: usize,
}


pub fn new_copy_block2(bg: &mut graph::BlockGraph,
                       input: graph::BlockPtrOutput
                       ) -> Result<Vec<graph::BlockPtrOutput>, Box<dyn Error>> {
    let num_inputs = bg.get_num_outputs(vec![&input]);
    assert!(num_inputs != 0);

    let mut block = Box::new(BlockCopy {
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs: num_inputs as usize,
    });
    let block_outputs = bg.add_node(block, vec![input]);
    assert_eq!(block_outputs.len(), 2);
    Ok(block_outputs)
}

pub fn new_copy_block(mi: &model_instance::ModelInstance, 
                                                    num_inputs: u32, 
                                                    ) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    assert!(num_inputs != 0);
    let mut rg = BlockCopy {
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs: num_inputs as usize,
    };
    Ok(Box::new(rg))
}






impl BlockTrait for BlockCopy

 {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
    }

    fn get_num_output_tapes(&self) -> usize {2}   

    fn get_num_outputs(&self, output_id: graph::BlockOutput) -> usize {
        assert!(output_id.get_output_id() <= 1);
        self.num_inputs
    }

    fn set_input_offset(&mut self, input: graph::BlockInput, offset: usize)  {
        assert!(input.get_input_id() == 0);
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::BlockOutput, offset: usize) {
/*        if output.get_output_id() == 0 {
            assert!(self.input_offset == offset)
        } else */
        if output.get_output_id() == 1 {
            self.output_offset = offset;
        } else {
            panic!("only two outputs supported for BlockCopy");
        }
    }






    #[inline(always)]
    fn forward_backward(&mut self, 
                        further_blocks: &mut [Box<dyn BlockTrait>], 
                        fb: &feature_buffer::FeatureBuffer, 
                        pb: &mut port_buffer::PortBuffer, 
                        update:bool) {
        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.num_inputs > 0);
        
        unsafe {
            // plain copy from input to output
            pb.tape.copy_within(self.input_offset .. (self.input_offset + self.num_inputs), self.output_offset);
                        
            //pb.tapes[self.output_tape_index as usize].extend_from_slice(pb.tapes.get_unchecked(self.input_tape_index as usize).get_unchecked(input_tape_start .. input_tape_start + self.num_inputs as usize));
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);

            if update {
                // Sum up the gradients from output to input
                for i in 0..self.num_inputs as usize {
                    let w = *pb.tape.get_unchecked(self.output_offset + i);
//                    println!("AAAAAA: {}, initial: {}", w, *(pb.tapes.get_unchecked_mut(self.input_tape_index as usize)).get_unchecked_mut(input_tape_start + i));
                    *pb.tape.get_unchecked_mut(self.input_offset + i) += w;
                }

            }
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
    
}




pub struct BlockJoin {    
    pub num_inputs: usize,
    pub input_offset: usize,
    pub output_offset: usize,
}


pub fn new_join_block2(bg: &mut graph::BlockGraph,
                       inputs: Vec<graph::BlockPtrOutput>,
                       ) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let num_inputs = bg.get_num_outputs(inputs.iter().collect());
    assert!(num_inputs != 0);

    let mut block = Box::new(BlockJoin {
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs: num_inputs,
    });
    let mut block_outputs = bg.add_node(block, inputs);
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

impl BlockTrait for BlockJoin

 {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
    }

    fn get_num_output_tapes(&self) -> usize {1}
    
    fn get_num_outputs(&self, output_id: graph::BlockOutput) -> usize {
        assert!(output_id.get_output_id() == 0);
        self.num_inputs
    }

    fn set_input_offset(&mut self, input: graph::BlockInput, offset: usize)  {
        assert!(input.get_input_id() <= 1);
        if input.get_input_id() == 0 {
            self.input_offset = offset;
        } else if input.get_input_id() == 1 {
            assert!(self.input_offset == offset);
        }
        
    } 

    fn set_output_offset(&mut self, output: graph::BlockOutput, offset: usize) {
        if output.get_output_id() == 0 {
            self.output_offset = offset;
        } else {
            panic!("only two outputs supported for BlockCopy");
        }
    }


    #[inline(always)]
    fn forward_backward(&mut self, 
                        further_blocks: &mut [Box<dyn BlockTrait>], 
                        fb: &feature_buffer::FeatureBuffer, 
                        pb: &mut port_buffer::PortBuffer, 
                        update:bool) {
        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.num_inputs > 0);
        
        let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
        next_regressor[0].forward_backward(further_blocks, fb, pb, update);
    }
    
    fn forward(&self, further_blocks: &[Box<dyn BlockTrait>], 
                        fb: &feature_buffer::FeatureBuffer, 
                        pb: &mut port_buffer::PortBuffer, 
                        ) {
        assert!(false, "Unimplemented");    
    }
    
}













