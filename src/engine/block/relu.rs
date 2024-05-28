use std::any::Any;
use std::error::Error;

use crate::engine::block::iterators;
use crate::engine::graph::{BlockGraph, BlockPtrOutput, InputSlot, OutputSlot};
use crate::engine::port_buffer::PortBuffer;
use crate::engine::regressor::BlockCache;
use crate::engine::regressor::BlockTrait;
use crate::model_instance;
use crate::namespace::feature_buffer::FeatureBuffer;

pub struct BlockRELU {
    pub num_inputs: usize,
    pub input_offset: usize,
    pub output_offset: usize,
}

pub fn new_relu_block(
    bg: &mut BlockGraph,
    _mi: &model_instance::ModelInstance,
    input: BlockPtrOutput,
) -> Result<BlockPtrOutput, Box<dyn Error>> {
    let num_inputs = bg.get_num_output_values(vec![&input]);
    assert_ne!(num_inputs, 0);
    let block = Box::new(BlockRELU {
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs,
    });
    let mut block_outputs = bg.add_node(block, vec![input])?;
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

impl BlockRELU {
    #[inline(always)]
    fn internal_forward(&self, pb: &mut PortBuffer) {
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.num_inputs > 0);

        unsafe {
            for i in 0..self.num_inputs {
                let w = *pb.tape.get_unchecked_mut(self.input_offset + i);
                if w < 0.0 {
                    *pb.tape.get_unchecked_mut(self.output_offset + i) = 0.0;
                } else {
                    *pb.tape.get_unchecked_mut(self.output_offset + i) = w;
                }
            }
        }
    }
}

impl BlockTrait for BlockRELU {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_num_output_values(&self, output: OutputSlot) -> usize {
        assert_eq!(output.get_output_index(), 0);
        self.num_inputs
    }

    fn set_input_offset(&mut self, input: InputSlot, offset: usize) {
        assert_eq!(input.get_input_index(), 0);
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: OutputSlot, offset: usize) {
        assert_eq!(output.get_output_index(), 0);
        self.output_offset = offset;
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        update: bool,
    ) {
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.num_inputs > 0);

        unsafe {
            for i in 0..self.num_inputs {
                let w = *pb.tape.get_unchecked_mut(self.input_offset + i);
                if w < 0.0 {
                    *pb.tape.get_unchecked_mut(self.output_offset + i) = 0.0;
                    *pb.tape.get_unchecked_mut(self.input_offset + i) = 0.0;
                } else {
                    *pb.tape.get_unchecked_mut(self.output_offset + i) = w;
                    *pb.tape.get_unchecked_mut(self.input_offset + i) = 1.0;
                }
            }

            iterators::forward_backward(further_blocks, fb, pb, update);

            if update {
                for i in 0..self.num_inputs {
                    let gradient = *pb.tape.get_unchecked(self.output_offset + i);
                    *pb.tape.get_unchecked_mut(self.input_offset + i) *= gradient;
                }
            }
        }
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
    ) {
        self.internal_forward(pb);
        iterators::forward(further_blocks, fb, pb);
    }

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        self.internal_forward(pb);
        iterators::forward_with_cache(further_blocks, fb, pb, caches);
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::assert_epsilon;
    use crate::engine::block::misc;
    use crate::engine::block::test;
    use misc::Observe;
    use test::slearn2;

    fn fb_vec() -> FeatureBuffer {
        FeatureBuffer {
            label: 0.0,
            example_importance: 1.0,
            example_number: 0,
            lr_buffer: Vec::new(),
            ffm_buffer: Vec::new(),
        }
    }

    #[test]
    fn test_simple_positive() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block = misc::new_const_block(&mut bg, vec![2.0]).unwrap();
        let relu_block = new_relu_block(&mut bg, &mi, input_block).unwrap();
        let _observe_block =
            misc::new_observe_block(&mut bg, relu_block, Observe::Forward, Some(1.0)).unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        let fb = fb_vec();
        assert_epsilon!(slearn2(&mut bg, &fb, &mut pb, true), 2.0);
        assert_epsilon!(slearn2(&mut bg, &fb, &mut pb, true), 2.0); // relu desnt learn
    }
}
