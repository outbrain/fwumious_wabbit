#![allow(dead_code,unused_imports)]

use std::any::Any;
use std::error::Error;

use crate::block_helpers;
use crate::feature_buffer;
use crate::graph;
use crate::port_buffer;
use crate::regressor;

use crate::feature_buffer::FeatureBuffer;
use crate::port_buffer::PortBuffer;
use crate::regressor::BlockCache;
use regressor::BlockTrait;

#[derive(PartialEq)]
pub enum Observe {
    Forward,
    Backward,
}

pub struct BlockObserve {
    num_inputs: usize,
    input_offset: usize,
    observe: Observe,
    replace_backward_with: Option<f32>,
}

pub fn new_observe_block(
    bg: &mut graph::BlockGraph,
    input: graph::BlockPtrOutput,
    observe: Observe,
    replace_backward_with: Option<f32>,
) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let num_inputs = bg.get_num_output_values(vec![&input]);
    let block = Box::new(BlockObserve {
        num_inputs,
        input_offset: usize::MAX,
        observe,
        replace_backward_with,
    });
    let mut block_outputs = bg.add_node(block, vec![input])?;
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

impl BlockTrait for BlockObserve {
    // Warning: It does not confirm to regular clean-up after itself

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_block_type(&self) -> graph::BlockType {
        graph::BlockType::Observe
    }

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert_eq!(output.get_output_index(), 0);
        self.num_inputs
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {
        assert_eq!(input.get_input_index(), 0);
        assert_eq!(self.input_offset, usize::MAX); // We only allow a single call
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert_eq!(output.get_output_index(), 0);
        assert_eq!(self.input_offset, offset); // this block type has special treatment
    }

    fn get_input_offset(&mut self, input: graph::InputSlot) -> Result<usize, Box<dyn Error>> {
        assert_eq!(input.get_input_index(), 0);
        Ok(self.input_offset)
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) {
        debug_assert!(self.input_offset != usize::MAX);

        if self.observe == Observe::Forward {
            pb.observations.extend_from_slice(
                &pb.tape[self.input_offset..(self.input_offset + self.num_inputs)],
            );
        }

        block_helpers::forward_backward(further_blocks, fb, pb, update);

        if self.observe == Observe::Backward {
            pb.observations.extend_from_slice(
                &pb.tape[self.input_offset..(self.input_offset + self.num_inputs)],
            );
        }

        // replace inputs with whatever we wanted
        if let Some(value) = self.replace_backward_with {
            pb.tape[self.input_offset..(self.input_offset + self.num_inputs)].fill(value)
        }
    }

    #[inline(always)]
    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        debug_assert!(self.input_offset != usize::MAX);

        if self.observe == Observe::Forward {
            pb.observations.extend_from_slice(
                &pb.tape[self.input_offset..(self.input_offset + self.num_inputs)],
            );
        }

        block_helpers::forward(further_blocks, fb, pb);

        if self.observe == Observe::Backward {
            pb.observations.extend_from_slice(
                &pb.tape[self.input_offset..(self.input_offset + self.num_inputs)],
            );
        }

        // replace inputs with whatever we wanted
        if let Some(value) = self.replace_backward_with {
            pb.tape[self.input_offset..(self.input_offset + self.num_inputs)].fill(value)
        }
    }

    #[inline(always)]
    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);

        if self.observe == Observe::Forward {
            pb.observations.extend_from_slice(
                &pb.tape[self.input_offset..(self.input_offset + self.num_inputs)],
            );
        }

        block_helpers::forward_with_cache(further_blocks, fb, pb, caches);

        if self.observe == Observe::Backward {
            pb.observations.extend_from_slice(
                &pb.tape[self.input_offset..(self.input_offset + self.num_inputs)],
            );
        }

        // replace inputs with whatever we wanted
        if let Some(value) = self.replace_backward_with {
            pb.tape[self.input_offset..(self.input_offset + self.num_inputs)].fill(value)
        }
    }
}

pub enum SinkType {
    Zero,
    Untouched,
}

pub struct BlockSink {
    num_inputs: usize,
    input_offset: usize,
    sink_type: SinkType,
}

pub fn new_sink_block(
    bg: &mut graph::BlockGraph,
    input: graph::BlockPtrOutput,
    sink_type: SinkType,
) -> Result<(), Box<dyn Error>> {
    let num_inputs = bg.get_num_output_values(vec![&input]);
    let block = Box::new(BlockSink {
        input_offset: usize::MAX,
        num_inputs,
        sink_type,
    });
    let block_outputs = bg.add_node(block, vec![input])?;
    assert_eq!(block_outputs.len(), 0);
    Ok(())
}

impl BlockTrait for BlockSink {
    // Warning: It does not confirm to regular clean-up after itself

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_block_type(&self) -> graph::BlockType {
        graph::BlockType::Regular
    } // It is regular, as there is no special functionality.

    fn get_num_output_slots(&self) -> usize {
        0
    } // It is a pass-through

    fn get_num_output_values(&self, _output: graph::OutputSlot) -> usize {
        assert!(false, "No output values in BlockSink");
        0
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {
        assert_eq!(input.get_input_index(), 0);
        assert_eq!(self.input_offset, usize::MAX); // We only allow a single call
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, _output: graph::OutputSlot, _offset: usize) {
        assert!(false, "No outputs in BlockSink");
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) {
        debug_assert!(self.input_offset != usize::MAX);

        block_helpers::forward_backward(further_blocks, fb, pb, update);
        match self.sink_type {
            SinkType::Zero => {
                pb.tape[self.input_offset..(self.input_offset + self.num_inputs)].fill(0.0);
            }
            SinkType::Untouched => {}
        }
    }

    #[inline(always)]
    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        debug_assert!(self.input_offset != usize::MAX);
        block_helpers::forward(further_blocks, fb, pb);
    }

    #[inline(always)]
    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        debug_assert!(self.input_offset != usize::MAX);
        block_helpers::forward_with_cache(further_blocks, fb, pb, caches);
    }
}

pub struct BlockConsts {
    pub output_offset: usize,
    consts: Vec<f32>,
}

pub fn new_const_block(
    bg: &mut graph::BlockGraph,
    consts: Vec<f32>,
) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let block = Box::new(BlockConsts {
        output_offset: usize::MAX,
        consts,
    });
    let mut block_outputs = bg.add_node(block, vec![])?;
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

impl BlockConsts {
    fn internal_forward(&self, pb: &mut port_buffer::PortBuffer) {
        debug_assert!(self.output_offset != usize::MAX);
        pb.tape[self.output_offset..(self.output_offset + self.consts.len())]
            .copy_from_slice(&self.consts);
    }
}

impl BlockTrait for BlockConsts {
    // Warning: It does not confirm to regular clean-up after itself

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert_eq!(output.get_output_index(), 0);
        self.consts.len()
    }

    fn set_input_offset(&mut self, _input: graph::InputSlot, _offset: usize) {
        panic!("You cannot set input_tape_index for BlockConsts");
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert_eq!(
            output.get_output_index(),
            0,
            "Only supports a single output for BlockConsts"
        );
        self.output_offset = offset;
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) {
        self.internal_forward(pb);
        block_helpers::forward_backward(further_blocks, fb, pb, update);
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        self.internal_forward(pb);
        block_helpers::forward(further_blocks, fb, pb);
    }

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        self.internal_forward(pb);
        block_helpers::forward_with_cache(further_blocks, fb, pb, caches);
    }
}

pub struct BlockCopy {
    pub num_inputs: usize,
    pub input_offset: usize,
    pub output_offsets: Vec<usize>,
}

pub fn new_copy_block(
    bg: &mut graph::BlockGraph,
    input: graph::BlockPtrOutput,
    num_output_slots: usize,
) -> Result<Vec<graph::BlockPtrOutput>, Box<dyn Error>> {
    let num_inputs = bg.get_num_output_values(vec![&input]);
    assert_ne!(num_inputs, 0);

    let block = Box::new(BlockCopy {
        output_offsets: vec![usize::MAX; num_output_slots],
        input_offset: usize::MAX,
        num_inputs,
    });
    let block_outputs = bg.add_node(block, vec![input])?;
    assert_eq!(block_outputs.len(), num_output_slots);
    Ok(block_outputs)
}

pub fn new_copy_block_2(
    bg: &mut graph::BlockGraph,
    input: graph::BlockPtrOutput,
) -> Result<(graph::BlockPtrOutput, graph::BlockPtrOutput), Box<dyn Error>> {
    let mut outputs = new_copy_block(bg, input, 2)?;
    assert_eq!(outputs.len(), 2);
    let output_2 = outputs.pop().unwrap();
    let output_1 = outputs.pop().unwrap();
    Ok((output_1, output_2))
}

impl BlockTrait for BlockCopy {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_block_type(&self) -> graph::BlockType {
        graph::BlockType::Copy
    }

    fn get_num_output_slots(&self) -> usize {
        self.output_offsets.len()
    }

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert!(
            output.get_output_index() < self.output_offsets.len(),
            "output.get_output_index(): {}, self.output_offsets.len(): {}",
            output.get_output_index(),
            self.output_offsets.len()
        );
        self.num_inputs // all output slots have the same number of output values
    }

    fn get_input_offset(&mut self, input: graph::InputSlot) -> Result<usize, Box<dyn Error>> {
        assert_eq!(input.get_input_index(), 0);
        Ok(self.input_offset)
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {
        assert_eq!(input.get_input_index(), 0);
        assert_eq!(self.input_offset, usize::MAX); // We only allow a single call
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert!(output.get_output_index() < self.output_offsets.len());
        if output.get_output_index() == 0 {
            // output index 0 is special - as it is zero copy from input
            self.output_offsets[0] = offset;
        } else {
            assert_eq!(self.output_offsets[output.get_output_index()], usize::MAX); // We only allow a single call
            self.output_offsets[output.get_output_index()] = offset;
        }
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) {
        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.num_inputs > 0);

        unsafe {
            // CopyBlock supports two modes:
            // If input is the same as the first output offset, there is just one copy to be done.
            //
            self.internal_forward(pb);
            block_helpers::forward_backward(further_blocks, fb, pb, update);

            if update {
                let output_offset_0 = *self.output_offsets.get_unchecked(0);
                if self.input_offset != output_offset_0 {
                    pb.tape.copy_within(
                        output_offset_0..(output_offset_0 + self.num_inputs),
                        self.input_offset,
                    );
                }

                // Sum up the gradients from output to input
                for &output_offset in self.output_offsets.get_unchecked(1..) {
                    let (input_tape, output_tape) = block_helpers::get_input_output_borrows(
                        &mut pb.tape,
                        self.input_offset,
                        self.num_inputs,
                        output_offset,
                        self.num_inputs,
                    );
                    for i in 0..self.num_inputs {
                        *input_tape.get_unchecked_mut(i) += *output_tape.get_unchecked(i);
                    }
                }
            }
        } // unsafe end
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        self.internal_forward(pb);
        block_helpers::forward(further_blocks, fb, pb);
    }

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        self.internal_forward(pb);
        block_helpers::forward_with_cache(further_blocks, fb, pb, caches);
    }
}

impl BlockCopy {
    #[inline(always)]
    fn internal_forward(&self, pb: &mut port_buffer::PortBuffer) {
        unsafe {
            let output_offset_0 = *self.output_offsets.get_unchecked(0);
            if self.input_offset != output_offset_0 {
                pb.tape.copy_within(
                    self.input_offset..(self.input_offset + self.num_inputs),
                    output_offset_0,
                );
            }

            for &output_offset in self.output_offsets.get_unchecked(1..) {
                pb.tape.copy_within(
                    self.input_offset..(self.input_offset + self.num_inputs),
                    output_offset,
                );
            }
        }
    }
}

pub struct BlockJoin {
    pub num_inputs: usize,
    pub input_offset: usize,
    pub output_offset: usize,
}

pub fn new_join_block(
    bg: &mut graph::BlockGraph,
    inputs: Vec<graph::BlockPtrOutput>,
) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let num_inputs = bg.get_num_output_values(inputs.iter().collect());
    assert_ne!(num_inputs, 0);

    let block = Box::new(BlockJoin {
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs,
    });
    let mut block_outputs = bg.add_node(block, inputs)?;
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

impl BlockTrait for BlockJoin {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_block_type(&self) -> graph::BlockType {
        graph::BlockType::Join
    }

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert_eq!(output.get_output_index(), 0);
        self.num_inputs
    }

    fn get_input_offset(&mut self, input: graph::InputSlot) -> Result<usize, Box<dyn Error>> {
        assert!(input.get_input_index() <= 1);
        Ok(self.input_offset)
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {
        if input.get_input_index() == 0 {
            assert_eq!(self.input_offset, usize::MAX); // We only allow a single call
            self.input_offset = offset;
        } else if input.get_input_index() >= 1 {
            assert!(
                self.input_offset <= offset,
                "Output 1, error 1: Input offset: {}, num_inputs: {}, offset: {}",
                self.input_offset,
                self.num_inputs,
                offset
            );
            assert!(
                self.input_offset + self.num_inputs >= offset,
                "Output 1, error 2: Input offset: {}, num_inputs: {}, offset: {}",
                self.input_offset,
                self.num_inputs,
                offset
            );
        }
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert_eq!(output.get_output_index(), 0);
        assert_eq!(self.output_offset, usize::MAX); // We only allow a single call
        self.output_offset = offset;
    }

    // WARNING: These two functions are automatically removed from the graph when executing, since they are a no-op
    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) {
        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.num_inputs > 0);

        block_helpers::forward_backward(further_blocks, fb, pb, update);
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        block_helpers::forward(further_blocks, fb, pb);
    }

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        block_helpers::forward_with_cache(further_blocks, fb, pb, caches);
    }
}

pub struct BlockSum {
    pub num_inputs: usize,
    pub input_offset: usize,
    pub output_offset: usize,
}

fn new_sum_without_weights(num_inputs: usize) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    assert!(num_inputs > 0);
    let rg = BlockSum {
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs,
    };
    Ok(Box::new(rg))
}

pub fn new_sum_block(
    bg: &mut graph::BlockGraph,
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

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert_eq!(output.get_output_index(), 0);
        1
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {
        assert_eq!(input.get_input_index(), 0);
        assert_eq!(self.input_offset, usize::MAX); // We only allow a single call
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert_eq!(output.get_output_index(), 0);
        assert_eq!(self.output_offset, usize::MAX); // We only allow a single call
        self.output_offset = offset;
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) {
        self.internal_forward(pb);

        block_helpers::forward_backward(further_blocks, fb, pb, update);

        unsafe {
            let general_gradient = pb.tape[self.output_offset];
            if update {
                pb.tape
                    .get_unchecked_mut(self.input_offset..(self.input_offset + self.num_inputs))
                    .fill(general_gradient);
            }
        }
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        self.internal_forward(pb);

        block_helpers::forward(further_blocks, fb, pb);
    }

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        self.internal_forward(pb);
        block_helpers::forward_with_cache(further_blocks, fb, pb, caches);
    }
}

impl BlockSum {
    #[inline(always)]
    fn internal_forward(&self, pb: &mut port_buffer::PortBuffer) {
        debug_assert!(self.num_inputs > 0);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);

        let wsum: f32 = pb.tape[self.input_offset..(self.input_offset + self.num_inputs)]
            .iter()
            .sum();
        pb.tape[self.output_offset] = wsum;
    }
}

// From a square only keep weights that are on the lower left triangle + diagonal
// Why is this useful?
// Because in FFM you get a square matrix of outputs, but it is symetrical across the diagonal
// This means that when you bring FFM to the first neural layer, you have double parameters needlessly -slowing things down
// It would be possible to modify BlockFFM output, but that is excessively complex to do.

pub struct BlockTriangle {
    pub square_width: usize,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub input_offset: usize,
    pub output_offset: usize,
}

pub fn new_triangle_block(
    bg: &mut graph::BlockGraph,
    input: graph::BlockPtrOutput,
) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let num_inputs = bg.get_num_output_values(vec![&input]);
    assert_ne!(num_inputs, 0);

    let num_inputs_sqrt = (num_inputs as f32).sqrt() as usize;
    if num_inputs_sqrt * num_inputs_sqrt != num_inputs {
        panic!("Triangle has to have number of inputs as square number, instead we have: {} whose square is {}", num_inputs, num_inputs_sqrt);
    }
    let square_width = num_inputs_sqrt;
    let num_outputs = square_width * (square_width + 1) / 2;
    let block = Box::new(BlockTriangle {
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs: square_width * square_width,
        num_outputs,
        square_width,
    });
    let mut block_outputs = bg.add_node(block, vec![input])?;
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

impl BlockTrait for BlockTriangle {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert_eq!(output.get_output_index(), 0);
        self.num_outputs
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {
        assert_eq!(input.get_input_index(), 0);
        assert_eq!(self.input_offset, usize::MAX); // We only allow a single call
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert_eq!(output.get_output_index(), 0);
        assert_eq!(self.output_offset, usize::MAX); // We only allow a single call
        self.output_offset = offset;
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) {
        debug_assert!(self.input_offset != usize::MAX);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.num_inputs > 0);

        self.internal_forward(pb);

        unsafe {
            block_helpers::forward_backward(further_blocks, fb, pb, update);

            if update {
                let (input_tape, output_tape) = block_helpers::get_input_output_borrows(
                    &mut pb.tape,
                    self.input_offset,
                    self.num_inputs,
                    self.output_offset,
                    self.num_outputs,
                );

                let mut output_index: usize = 0;
                for i in 0..self.square_width {
                    for j in 0..i + 1 {
                        *input_tape.get_unchecked_mut(i * self.square_width + j) =
                            *output_tape.get_unchecked(output_index);
                        *input_tape.get_unchecked_mut(j * self.square_width + i) =
                            *output_tape.get_unchecked(output_index);
                        output_index += 1;
                    }
                }
            }
        }
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        self.internal_forward(pb);
        block_helpers::forward(further_blocks, fb, pb);
    }

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        self.internal_forward(pb);
        block_helpers::forward_with_cache(further_blocks, fb, pb, caches);
    }
}

impl BlockTriangle {
    #[inline(always)]
    fn internal_forward(&self, pb: &mut port_buffer::PortBuffer) {
        unsafe {
            let (input_tape, output_tape) = block_helpers::get_input_output_borrows(
                &mut pb.tape,
                self.input_offset,
                self.num_inputs,
                self.output_offset,
                self.num_outputs,
            );

            let mut output_index: usize = 0;
            for i in 0..self.square_width {
                for j in 0..i {
                    *output_tape.get_unchecked_mut(output_index) =
                        *input_tape.get_unchecked(i * self.square_width + j) * 2.0;
                    output_index += 1;
                }
                *output_tape.get_unchecked_mut(output_index) =
                    *input_tape.get_unchecked(i * self.square_width + i);
                output_index += 1;
            }
        }
    }
}

mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::block_helpers::{slearn2, spredict2};
    use crate::block_misc;
    use crate::block_misc::Observe;
    use crate::feature_buffer;
    use crate::graph::BlockGraph;
    use crate::model_instance;

    fn fb_vec() -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
            label: 0.0,
            example_importance: 1.0,
            example_number: 0,
            lr_buffer: Vec::new(),
            ffm_buffer: Vec::new(),
        }
    }

    #[test]
    fn test_sum_block() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block = block_misc::new_const_block(&mut bg, vec![2.0, 3.0]).unwrap();
        let observe_block_backward =
            block_misc::new_observe_block(&mut bg, input_block, Observe::Backward, None).unwrap();
        let sum_block = new_sum_block(&mut bg, observe_block_backward).unwrap();
        let _observe_block_forward =
            block_misc::new_observe_block(&mut bg, sum_block, Observe::Forward, Some(1.0)).unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();
        slearn2(&mut bg, &fb, &mut pb, true);
        assert_eq!(
            pb.observations,
            vec![
                5.0, // forward part, just a sum of 2 and 3
                1.0, 1.0
            ]
        ); // backward part -- 1 is distributed to both inputs

        spredict2(&mut bg, &fb, &mut pb);
        assert_eq!(
            pb.observations,
            vec![
                5.0, // forward part, just a sum of 2 and 3
                2.0, 3.0
            ]
        ); // backward part -- nothing gets updated
    }

    #[test]
    fn test_triangle_block() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block = block_misc::new_const_block(&mut bg, vec![2.0, 4.0, 4.0, 5.0]).unwrap();
        let observe_block_backward =
            block_misc::new_observe_block(&mut bg, input_block, Observe::Backward, None).unwrap();
        let triangle_block = new_triangle_block(&mut bg, observe_block_backward).unwrap();
        let observe_block_forward =
            block_misc::new_observe_block(&mut bg, triangle_block, Observe::Forward, None).unwrap();
        block_misc::new_sink_block(
            &mut bg,
            observe_block_forward,
            block_misc::SinkType::Untouched,
        )
        .unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();
        slearn2(&mut bg, &fb, &mut pb, true);
        assert_eq!(
            pb.observations,
            vec![
                2.0, 8.0, 5.0, // forward part
                2.0, 8.0, 8.0, 5.0
            ]
        ); // backward part -- 3.0 gets turned into 4.0 since that is its transpose
    }

    #[test]
    fn test_copy_block() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block = block_misc::new_const_block(&mut bg, vec![2.0, 3.0]).unwrap();
        let observe_block_backward =
            block_misc::new_observe_block(&mut bg, input_block, Observe::Backward, None).unwrap();
        let (copy_block_1, copy_block_2) =
            new_copy_block_2(&mut bg, observe_block_backward).unwrap();
        let _observe_block_1_forward =
            block_misc::new_observe_block(&mut bg, copy_block_1, Observe::Forward, Some(5.0))
                .unwrap();
        let _observe_block_2_forward =
            block_misc::new_observe_block(&mut bg, copy_block_2, Observe::Forward, Some(6.0))
                .unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();
        slearn2(&mut bg, &fb, &mut pb, true);
        assert_eq!(
            pb.observations,
            vec![
                2.0, 3.0, // 1st copy of forward parts
                2.0, 3.0, // 2nd copy of forward
                11.0, 11.0,
            ]
        );

        spredict2(&mut bg, &fb, &mut pb);
        assert_eq!(
            pb.observations,
            vec![
                2.0, 3.0, // 1st copy of forward parts
                2.0, 3.0, // 2nd copy of forward
                2.0, 3.0,
            ]
        ); // backward part isn't touched, it will contain whatever observe block_1 put there
    }

    #[test]
    fn test_copy_block_cascade() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block = block_misc::new_const_block(&mut bg, vec![2.0, 3.0]).unwrap();
        let observe_block_backward =
            block_misc::new_observe_block(&mut bg, input_block, Observe::Backward, None).unwrap();
        let (copy_block_1, copy_block_2) =
            new_copy_block_2(&mut bg, observe_block_backward).unwrap();
        let (copy_block_3, copy_block_4) = new_copy_block_2(&mut bg, copy_block_1).unwrap();

        let _observe_block_1_forward =
            block_misc::new_observe_block(&mut bg, copy_block_2, Observe::Forward, Some(5.0))
                .unwrap();
        let _observe_block_2_forward =
            block_misc::new_observe_block(&mut bg, copy_block_3, Observe::Forward, Some(6.0))
                .unwrap();
        let _observe_block_3_forward =
            block_misc::new_observe_block(&mut bg, copy_block_4, Observe::Forward, Some(7.0))
                .unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();
        slearn2(&mut bg, &fb, &mut pb, true);
        assert_eq!(
            pb.observations,
            vec![
                2.0, 3.0, // 1st copy of forward parts
                2.0, 3.0, // 2nd copy of forward
                2.0, 3.0, 18.0, 18.0,
            ]
        );

        spredict2(&mut bg, &fb, &mut pb);
        assert_eq!(
            pb.observations,
            vec![
                2.0, 3.0, // 1st copy of forward parts
                2.0, 3.0, // 2nd copy of forward
                2.0, 3.0, 2.0, 3.0
            ]
        ); // backward part isn't touched, it will contain whatever observe block_1 put there
           // it is from copy_block_3 since that is the last one where observe_block_2_forward does its work
    }

    #[test]
    fn test_join_1_block() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block_1 = block_misc::new_const_block(&mut bg, vec![2.0, 3.0]).unwrap();
        let join_block = new_join_block(&mut bg, vec![input_block_1]).unwrap();
        let _observe_block =
            block_misc::new_observe_block(&mut bg, join_block, Observe::Forward, Some(6.0))
                .unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();
        slearn2(&mut bg, &fb, &mut pb, true);
        assert_eq!(pb.observations, vec![2.0, 3.0,]); // join actually doesn't do anything

        spredict2(&mut bg, &fb, &mut pb);
        assert_eq!(pb.observations, vec![2.0, 3.0,]); // join actually doesn't do anything
    }

    #[test]
    fn test_join_2_blocks() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block_1 = block_misc::new_const_block(&mut bg, vec![2.0, 3.0]).unwrap();
        let input_block_2 = block_misc::new_const_block(&mut bg, vec![4.0, 5.0, 6.0]).unwrap();
        let join_block = new_join_block(&mut bg, vec![input_block_1, input_block_2]).unwrap();
        let _observe_block =
            block_misc::new_observe_block(&mut bg, join_block, Observe::Forward, Some(6.0))
                .unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();
        slearn2(&mut bg, &fb, &mut pb, true);
        assert_eq!(pb.observations, vec![2.0, 3.0, 4.0, 5.0, 6.0]); // join actually doesn't do anything

        spredict2(&mut bg, &fb, &mut pb);
        assert_eq!(pb.observations, vec![2.0, 3.0, 4.0, 5.0, 6.0]); // join actually doesn't do anything
    }

    #[test]
    fn test_join_3_blocks() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block_3 = block_misc::new_const_block(&mut bg, vec![1.0, 2.0]).unwrap();
        let input_block_2 = block_misc::new_const_block(&mut bg, vec![3.0, 4.0, 5.0]).unwrap();
        let input_block_1 = block_misc::new_const_block(&mut bg, vec![6.0, 7.0]).unwrap();
        let join_block =
            new_join_block(&mut bg, vec![input_block_1, input_block_2, input_block_3]).unwrap();
        let _observe_block =
            block_misc::new_observe_block(&mut bg, join_block, Observe::Forward, Some(6.0))
                .unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();
        slearn2(&mut bg, &fb, &mut pb, true);
        // Order depends on the input parameters order, not on order of adding to graph
        assert_eq!(pb.observations, vec![6.0, 7.0, 3.0, 4.0, 5.0, 1.0, 2.0]); // join actually doesn't do anything

        spredict2(&mut bg, &fb, &mut pb);
        assert_eq!(pb.observations, vec![6.0, 7.0, 3.0, 4.0, 5.0, 1.0, 2.0]); // join actually doesn't do anything
    }

    #[test]
    fn test_join_cascading() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block_1 = block_misc::new_const_block(&mut bg, vec![1.0, 2.0]).unwrap();
        let input_block_2 = block_misc::new_const_block(&mut bg, vec![3.0, 4.0, 5.0]).unwrap();
        let join_block_1 = new_join_block(&mut bg, vec![input_block_1, input_block_2]).unwrap();
        let input_block_3 = block_misc::new_const_block(&mut bg, vec![6.0, 7.0]).unwrap();
        let join_block_2 = new_join_block(&mut bg, vec![input_block_3, join_block_1]).unwrap();
        let _observe_block =
            block_misc::new_observe_block(&mut bg, join_block_2, Observe::Forward, Some(6.0))
                .unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();

        // Order depends on the input parameters order, not on order of adding to graph ( join actually doesn't do anything)
        slearn2(&mut bg, &fb, &mut pb, true);
        assert_eq!(pb.observations, vec![6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        spredict2(&mut bg, &fb, &mut pb);
        assert_eq!(pb.observations, vec![6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_copy_to_join() {
        let mi = model_instance::ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block_1 = block_misc::new_const_block(&mut bg, vec![2.0, 3.0]).unwrap();
        let observe_block_backward =
            block_misc::new_observe_block(&mut bg, input_block_1, Observe::Backward, None).unwrap();
        let (copy_1, copy_2) =
            block_misc::new_copy_block_2(&mut bg, observe_block_backward).unwrap();
        let join_block = new_join_block(&mut bg, vec![copy_1, copy_2]).unwrap();
        let _observe_block =
            block_misc::new_observe_block(&mut bg, join_block, Observe::Forward, Some(6.0))
                .unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();
        slearn2(&mut bg, &fb, &mut pb, true);
        assert_eq!(pb.observations, vec![2.0, 3.0, 2.0, 3.0, 12.0, 12.0]); // correct backwards pass

        spredict2(&mut bg, &fb, &mut pb);
        assert_eq!(pb.observations, vec![2.0, 3.0, 2.0, 3.0, 2.0, 3.0]); // on backward pass this are leftovers
    }
}
