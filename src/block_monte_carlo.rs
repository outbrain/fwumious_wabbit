use std::any::Any;
use std::error::Error;

use rand::distributions::{Distribution, Uniform};

use regressor::BlockTrait;

use crate::block_helpers;
use crate::feature_buffer::FeatureBuffer;
use crate::graph;
use crate::port_buffer::{MonteCarloStats, PortBuffer};
use crate::regressor;
use crate::regressor::BlockCache;

pub fn new_monte_carlo_block(
    bg: &mut graph::BlockGraph,
    input: graph::BlockPtrOutput,
    num_iterations: usize,
    dropout_rate: f32,
) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let num_inputs = bg.get_num_output_values(vec![&input]);
    assert_ne!(num_inputs, 0);

    let skip_index_generator = Uniform::from(0..num_inputs);
    let number_of_inputs_to_skip = (dropout_rate * num_inputs as f32) as usize;

    let mut block = Box::new(BlockMonteCarlo {
        num_iterations,
        number_of_inputs_to_skip,
        skip_index_generator,
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs,
    });
    let mut block_outputs = bg.add_node(block, vec![input])?;
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

pub struct BlockMonteCarlo {
    num_iterations: usize,
    number_of_inputs_to_skip: usize,
    skip_index_generator: Uniform<usize>,

    pub num_inputs: usize,
    pub input_offset: usize,
    pub output_offset: usize,
}

impl BlockTrait for BlockMonteCarlo {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        update: bool,
    ) {
        self.copy_input_tape_to_output_tape(pb);

        block_helpers::forward_backward(further_blocks, fb, pb, update);

        if update {
            let (input_tape, output_tape) = block_helpers::get_input_output_borrows(
                &mut pb.tape,
                self.input_offset,
                self.num_inputs,
                self.output_offset,
                self.num_inputs,
            );

            input_tape.copy_from_slice(output_tape);
        }
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
    ) {
        unsafe {
            if self.number_of_inputs_to_skip == 0 {
                self.copy_input_tape_to_output_tape(pb);
                block_helpers::forward(further_blocks, fb, pb);
                return;
            }

            let mut rng = rand::thread_rng();

            let input_tape = pb
                .tape
                .get_unchecked_mut(self.input_offset..self.input_offset + self.num_inputs)
                .to_vec();
            for _ in 0..self.num_iterations {
                let (_, output_tape) = block_helpers::get_input_output_borrows(
                    &mut pb.tape,
                    self.input_offset,
                    self.num_inputs,
                    self.output_offset,
                    self.num_inputs,
                );

                output_tape.copy_from_slice(&input_tape);
                for _ in 0..self.number_of_inputs_to_skip {
                    let skip_index = self.skip_index_generator.sample(&mut rng);
                    *output_tape.get_unchecked_mut(skip_index) = 0.0;
                }
                block_helpers::forward(further_blocks, fb, pb);
            }

            self.fill_stats(pb);
        }
    }

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        unsafe {
            if self.number_of_inputs_to_skip == 0 {
                self.copy_input_tape_to_output_tape(pb);
                block_helpers::forward_with_cache(further_blocks, fb, pb, caches);
                return;
            }

            let mut rng = rand::thread_rng();

            let input_tape = pb
                .tape
                .get_unchecked_mut(self.input_offset..self.input_offset + self.num_inputs)
                .to_vec();
            for _ in 0..self.num_iterations {
                let (_, output_tape) = block_helpers::get_input_output_borrows(
                    &mut pb.tape,
                    self.input_offset,
                    self.num_inputs,
                    self.output_offset,
                    self.num_inputs,
                );

                output_tape.copy_from_slice(&input_tape);
                for _ in 0..self.number_of_inputs_to_skip {
                    let skip_index = self.skip_index_generator.sample(&mut rng);
                    *output_tape.get_unchecked_mut(skip_index) = 0.0;
                }
                block_helpers::forward_with_cache(further_blocks, fb, pb, caches);
            }

            self.fill_stats(pb);
        }
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
        assert_eq!(self.output_offset, usize::MAX); // We only allow a single call
        self.output_offset = offset;
    }
}

impl BlockMonteCarlo {
    fn copy_input_tape_to_output_tape(&self, pb: &mut PortBuffer) {
        let (input_tape, output_tape) = block_helpers::get_input_output_borrows(
            &mut pb.tape,
            self.input_offset,
            self.num_inputs,
            self.output_offset,
            self.num_inputs,
        );

        output_tape.copy_from_slice(input_tape);
    }

    fn fill_stats(&self, pb: &mut PortBuffer) {
        let mean: f32 = pb.observations.iter().sum::<f32>() / self.num_iterations as f32;
        let variance = pb
            .observations
            .iter()
            .map(|prediction| {
                let diff = mean - prediction;
                diff * diff
            })
            .sum::<f32>()
            / self.num_iterations as f32;
        let standard_deviation = variance.sqrt();

        pb.monte_carlo_stats = Some(MonteCarloStats {
            mean,
            variance,
            standard_deviation,
        });
    }
}
