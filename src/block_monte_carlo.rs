use std::any::Any;
use std::error::Error;

use rand::distributions::{Distribution, Uniform};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

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

fn create_seed_from_input_tape(input_tape: &[f32]) -> u64 {
    (input_tape.iter().sum::<f32>() * 1000.0) as u64
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

            let input_tape = pb
                .tape
                .get_unchecked_mut(self.input_offset..self.input_offset + self.num_inputs)
                .to_vec();

            let seed = create_seed_from_input_tape(&input_tape);
            let mut rng = Xoshiro256Plus::seed_from_u64(seed);

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

            let input_tape = pb
                .tape
                .get_unchecked_mut(self.input_offset..self.input_offset + self.num_inputs)
                .to_vec();

            let seed = create_seed_from_input_tape(&input_tape);
            let mut rng = Xoshiro256Plus::seed_from_u64(seed);

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

mod tests {
    use crate::assert_epsilon;
    use crate::block_helpers::{slearn2, spredict2};
    use crate::block_misc::Observe;
    use crate::block_monte_carlo::new_monte_carlo_block;
    use crate::graph::BlockGraph;
    use crate::model_instance::ModelInstance;
    use crate::{block_misc, feature_buffer::FeatureBuffer};

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
    fn test_monte_carlo_block() {
        let mut mi = ModelInstance::new_empty().unwrap();
        let mut bg = BlockGraph::new();
        let input_block = block_misc::new_const_block(&mut bg, vec![2.0, 4.0, 4.0, 5.0]).unwrap();
        let observe_block_backward =
            block_misc::new_observe_block(&mut bg, input_block, Observe::Backward, None).unwrap();
        let triangle_block =
            new_monte_carlo_block(&mut bg, observe_block_backward, 3, 0.3).unwrap();
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
        assert_eq!(pb.observations, [2.0, 4.0, 4.0, 5.0, 2.0, 4.0, 4.0, 5.0]);

        spredict2(&mut bg, &fb, &mut pb, false);
        let expected_observations = [2.0, 4.0, 0.0, 5.0, 0.0, 4.0, 4.0, 5.0, 2.0, 4.0, 4.0, 0.0, 2.0, 4.0, 4.0, 5.0];
        assert_eq!(pb.observations.len(), expected_observations.len());
        assert_eq!(pb.observations, expected_observations);

        assert_ne!(pb.monte_carlo_stats, None);

        let monte_carlo_stats = pb.monte_carlo_stats.unwrap();
        assert_epsilon!(monte_carlo_stats.mean, 11.333333);
        assert_epsilon!(monte_carlo_stats.variance, 302.88885);
        assert_epsilon!(monte_carlo_stats.standard_deviation, 17.403702);
    }
}
