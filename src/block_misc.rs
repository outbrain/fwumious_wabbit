use std::any::Any;
use std::error::Error;
use std::io;

use crate::regressor;
use crate::feature_buffer;
use crate::port_buffer;
use crate::model_instance;
use crate::graph;

use regressor::BlockTrait;

#[derive(PartialEq)]
pub enum Observe {
    Forward,
    Backward
}

pub struct BlockObserve {
    num_inputs: usize,
    input_offset: usize,
    observe: Observe,
    replace_backward_with: Option<f32>,
}


pub fn new_observe_block(bg: &mut graph::BlockGraph,
                        input: graph::BlockPtrOutput,
                        observe: Observe,
                        replace_backward_with: Option<f32>) 
                        -> Result<(), Box<dyn Error>> {

    let num_inputs = bg.get_num_output_values(vec![&input]);
//    println!("Inputs: {} vec: {:?}", num_inputs, input);
    let block = Box::new(BlockObserve {
                         num_inputs: num_inputs as usize,
                         input_offset: usize::MAX,
                         observe: observe,
                         replace_backward_with: replace_backward_with});
    let block_outputs = bg.add_node(block, vec![input])?;
    assert_eq!(block_outputs.len(), 1);
    Ok(())
}



impl BlockTrait for BlockObserve {
    // Warning: It does not confirm to regular clean-up after itself

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_block_type(&self) -> graph::BlockType {graph::BlockType::Observe}  

    fn get_num_output_slots(&self) -> usize {1} // It is a pass-through   

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert!(output.get_output_index() == 0);
        // this means outputs on regular tapes
        return self.num_inputs
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize)  {
        assert!(input.get_input_index() == 0);
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize)  {
        assert!(output.get_output_index() == 0);
        assert_eq!(self.input_offset, offset); // this block type has special treatment
    }

    fn get_input_offset(&mut self, input: graph::InputSlot) -> Result<usize, Box<dyn Error>> {
        assert!(input.get_input_index() == 0);
        Ok(self.input_offset)
    }



    
    #[inline(always)]
    fn forward_backward(&mut self, 
                    further_blocks: &mut [Box<dyn BlockTrait>], 
                    fb: &feature_buffer::FeatureBuffer, 
                    pb: &mut port_buffer::PortBuffer, 
                    update:bool) {
        debug_assert!(self.input_offset != usize::MAX);

        // copy inputs to result
//        println!("Result block with Num inputs: {}", self.num_inputs);
        if self.observe == Observe::Forward {
            pb.observations.extend_from_slice(&pb.tape[self.input_offset .. (self.input_offset + self.num_inputs)]);
        }

        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);
        }
        
        if self.observe == Observe::Backward {
            pb.observations.extend_from_slice(&pb.tape[self.input_offset .. (self.input_offset + self.num_inputs)]);
        }

        // replace inputs with whatever we wanted
        match self.replace_backward_with {
            Some(value) => pb.tape[self.input_offset..(self.input_offset + self.num_inputs)].fill(value),
            None => {},
        }

    }

    fn forward(&self, 
                     further_blocks: &[Box<dyn BlockTrait>], 
                     fb: &feature_buffer::FeatureBuffer,
                     pb: &mut port_buffer::PortBuffer
                     ) {
        debug_assert!(self.input_offset != usize::MAX);
        
        if self.observe == Observe::Forward {
            pb.observations.extend_from_slice(&pb.tape[self.input_offset .. (self.input_offset + self.num_inputs)]);
        }

        
        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at(1);
            next_regressor[0].forward(further_blocks, fb, pb);
        }


        if self.observe == Observe::Backward {
            pb.observations.extend_from_slice(&pb.tape[self.input_offset .. (self.input_offset + self.num_inputs)]);
        }

        // replace inputs with whatever we wanted
        match self.replace_backward_with {
            Some(value) => pb.tape[self.input_offset..(self.input_offset + self.num_inputs)].fill(value),
            None => {},
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
    let mut block_outputs = bg.add_node(block, vec![])?;
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())


}



impl BlockTrait for BlockConsts {
    // Warning: It does not confirm to regular clean-up after itself

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_num_output_slots(&self) -> usize {1}   

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert!(output.get_output_index() == 0);
        self.consts.len() as usize
    }
    
    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize)  {
        panic!("You cannnot set input_tape_index for BlockConsts");
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert!(output.get_output_index() == 0, "Only supports a single output for BlockConsts");
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


pub fn new_copy_block(bg: &mut graph::BlockGraph,
                       input: graph::BlockPtrOutput
                       ) -> Result<Vec<graph::BlockPtrOutput>, Box<dyn Error>> {
    let num_inputs = bg.get_num_output_values(vec![&input]);
    assert!(num_inputs != 0);

    let mut block = Box::new(BlockCopy {
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs: num_inputs as usize,
    });
    let block_outputs = bg.add_node(block, vec![input])?;
    assert_eq!(block_outputs.len(), 2);
    Ok(block_outputs)
}





impl BlockTrait for BlockCopy

 {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_block_type(&self) -> graph::BlockType {graph::BlockType::Copy}  

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
    }

    fn get_num_output_slots(&self) -> usize {2}   

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert!(output.get_output_index() <= 1);
        self.num_inputs
    }

    fn get_input_offset(&mut self, input: graph::InputSlot) -> Result<usize, Box<dyn Error>> {
        assert!(input.get_input_index() == 0);
        Ok(self.input_offset)
    }


    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize)  {
        assert!(input.get_input_index() == 0);
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        if output.get_output_index() == 0 {
            assert!(self.input_offset == offset)
        } else 
        if output.get_output_index() == 1 {
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


pub fn new_join_block(bg: &mut graph::BlockGraph,
                       inputs: Vec<graph::BlockPtrOutput>,
                       ) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let num_inputs = bg.get_num_output_values(inputs.iter().collect());
    assert!(num_inputs != 0);

    let mut block = Box::new(BlockJoin {
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs: num_inputs,
    });
    let mut block_outputs = bg.add_node(block, inputs)?;
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

impl BlockTrait for BlockJoin {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
    
    fn get_block_type(&self) -> graph::BlockType {graph::BlockType::Join}  

    fn get_num_output_slots(&self) -> usize {1}
    
    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert!(output.get_output_index() == 0);
        self.num_inputs
    }

    fn get_input_offset(&mut self, input: graph::InputSlot) -> Result<usize, Box<dyn Error>> {
        assert!(input.get_input_index() <= 1);
        Ok(self.input_offset)
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize)  {
        assert!(input.get_input_index() <= 1);
        if input.get_input_index() == 0 {
            self.input_offset = offset;
        } else if input.get_input_index() == 1 {
            assert!(self.input_offset <= offset, "Output 1, error 1: Input offset: {}, num_inputs: {}, offset: {}", self.input_offset, self.num_inputs, offset);
            assert!(self.input_offset + self.num_inputs >= offset, "Output 1, error 2: Input offset: {}, num_inputs: {}, offset: {}", self.input_offset, self.num_inputs, offset);
        }
        
    } 

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert!(output.get_output_index() == 0);
        self.output_offset = offset;
    }

    // WARNING: These two functions are automatically removed from the graph when executing, since they are a no-op
    #[inline(always)]
    fn forward_backward(&mut self, 
                        further_blocks: &mut [Box<dyn BlockTrait>], 
                        fb: &feature_buffer::FeatureBuffer, 
                        pb: &mut port_buffer::PortBuffer, 
                        update:bool) {
        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.num_inputs > 0);
        
        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);
        }
    }
    
    fn forward(&self, further_blocks: &[Box<dyn BlockTrait>], 
                        fb: &feature_buffer::FeatureBuffer, 
                        pb: &mut port_buffer::PortBuffer, 
                        ) {
        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at(1);
            next_regressor[0].forward(further_blocks, fb, pb);
        }

    }
    
}


pub struct BlockSum {    
    pub num_inputs: usize,
    pub input_offset: usize,
    pub output_offset: usize,

}


fn new_sum_without_weights(
                          num_inputs: usize, 
                          ) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    assert!(num_inputs > 0);
    let mut rg = BlockSum {
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs: num_inputs,
    };
    Ok(Box::new(rg))
}




pub fn new_sum_block(   bg: &mut graph::BlockGraph, 
                        input: graph::BlockPtrOutput,
                        ) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {    
    let num_inputs = bg.get_num_output_values(vec![&input]);
    let block = new_sum_without_weights(num_inputs).unwrap(); 
    let mut block_outputs = bg.add_node(block, vec![input]).unwrap();
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}







impl BlockTrait for BlockSum {
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
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize)  {
        assert!(output.get_output_index() == 0);
        self.output_offset = offset;
    }

    #[inline(always)]
    fn forward_backward(&mut self, 
                        further_blocks: &mut [Box<dyn BlockTrait>], 
                        fb: &feature_buffer::FeatureBuffer, 
                        pb: &mut port_buffer::PortBuffer, 
                        update:bool) {
        debug_assert!(self.num_inputs > 0);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);
        
        let wsum:f32 = pb.tape[self.input_offset .. (self.input_offset + self.num_inputs as usize)].iter().sum();
        pb.tape[self.output_offset as usize] = wsum;
        
        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);
        }

        let general_gradient = pb.tape[self.output_offset];
        if update {
            pb.tape[self.input_offset .. (self.input_offset + self.num_inputs as usize)].fill(general_gradient);
        } // unsafe end
    }
    
    fn forward(		    &self, 		
                            further_blocks: &[Box<dyn BlockTrait>], 
                            fb: &feature_buffer::FeatureBuffer,
                            pb: &mut port_buffer::PortBuffer, 
                           ) {
        debug_assert!(self.num_inputs > 0);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);
        
        let wsum:f32 = pb.tape[self.input_offset .. (self.input_offset + self.num_inputs as usize)].iter().sum();
        pb.tape[self.output_offset as usize] = wsum;

        if further_blocks.len() > 0 {
            let (next_regressor, further_blocks) = further_blocks.split_at(1);
            next_regressor[0].forward(further_blocks, fb, pb);
        }
    }
}










mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::block_loss_functions;
    use crate::block_misc;
    use crate::model_instance::Optimizer;
    use crate::feature_buffer;
    use crate::feature_buffer::HashAndValueAndSeq;
    use crate::vwmap;
    use crate::block_helpers::{slearn2, spredict2};
    use crate::graph;
    use crate::graph::{BlockGraph};
    use crate::block_misc::Observe;
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
    fn test_sum_block() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.optimizer = Optimizer::SGD;
        
       
        let mut bg = BlockGraph::new();
        let input_block = block_misc::new_const_block(&mut bg, vec![2.0]).unwrap();
        let sum_block = new_sum_block(&mut bg, input_block).unwrap();
        let observe_block = block_misc::new_observe_block(&mut bg, sum_block, Observe::Forward, Some(1.0)).unwrap();
        bg.schedule();
        bg.allocate_and_init_weights(&mi);
        
        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();
        assert_epsilon!(slearn2  (&mut bg, &fb, &mut pb, true), 2.0);
        assert_eq!(pb.observations.len(), 1);
        assert_epsilon!(slearn2  (&mut bg, &fb, &mut pb, true), 2.0);
        

    }


}








