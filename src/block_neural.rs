#![allow(dead_code,unused_imports)]

use rand_distr::{Distribution, Normal, Uniform};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::any::Any;
use std::error::Error;
use std::io;
use std::io::Error as IOError;
use std::io::ErrorKind;
use std::mem::MaybeUninit;

use crate::block_helpers;
use crate::block_misc;
use crate::feature_buffer;
use crate::graph;
use crate::model_instance;
use crate::optimizer;
use crate::port_buffer;
use crate::regressor;
use block_helpers::OptimizerData;
use optimizer::OptimizerTrait;
use regressor::BlockTrait;

use crate::feature_buffer::FeatureBuffer;
use crate::port_buffer::PortBuffer;
use crate::regressor::BlockCache;
use blas::*;

const MAX_NUM_INPUTS: usize = 16000;

#[derive(PartialEq, Debug)]
pub enum NeuronType {
    WeightedSum,
    Sum,
}

#[derive(PartialEq, Debug)]
pub enum InitType {
    Xavier,
    Hu,
    //    RandomFirst1,
    //    RandomFirst10,
    One,
    Zero,
}

pub struct BlockNeuronLayer<L: OptimizerTrait> {
    pub num_inputs: usize,
    pub input_offset: usize,
    pub output_offset: usize,
    pub weights_len: u32,
    // While FFM part keeps weight and accumulation together (since memory locality is the issue)
    // for NN part it is actually preferrable to have it separately
    pub weights: Vec<f32>,
    pub weights_optimizer: Vec<OptimizerData<L>>,
    pub optimizer: L,
    pub neuron_type: NeuronType,
    pub num_neurons: usize,
    pub init_type: InitType,
    pub dropout: f32,
    pub dropout_inv: f32,
    pub max_norm: f32,
    pub layer_norm: bool,
    rng: Xoshiro256PlusPlus,
    rng_scratchpad: Vec<u32>,
    dropout_threshold: u32,
    bias_offset: usize,
}

fn new_neuronlayer_without_weights<L: OptimizerTrait + 'static>(
    mi: &model_instance::ModelInstance,
    num_inputs: usize,
    ntype: NeuronType,
    num_neurons: usize,
    init_type: InitType,
    dropout: f32,
    max_norm: f32,
    layer_norm: bool,
) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    assert!(num_neurons > 0);
    assert!(num_inputs < MAX_NUM_INPUTS);
    assert_ne!(num_inputs, 0);

    if dropout != 0.0 {
        panic!("Dropout in this binary is not supported due to bizzare side effects on inner loop unrolling");
    }

    let weights_len = ((num_inputs + 1) * num_neurons) as u32; // +1 is for bias term

    let bias_offset = num_inputs * num_neurons;

    let mut rg = BlockNeuronLayer::<L> {
        weights: Vec::new(),
        weights_optimizer: Vec::new(),
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs,
        optimizer: L::new(),
        weights_len,
        neuron_type: ntype,
        num_neurons,
        init_type,
        dropout,
        dropout_inv: 1.0 / (1.0 - dropout),
        max_norm,
        layer_norm,
        rng: Xoshiro256PlusPlus::seed_from_u64(0_u64),
        rng_scratchpad: Vec::new(),
        dropout_threshold: ((u32::MAX as f64) * (dropout as f64)) as u32,
        bias_offset,
    };

    rg.optimizer
        .init(mi.nn_learning_rate, mi.nn_power_t, mi.nn_init_acc_gradient);
    Ok(Box::new(rg))
}

pub fn new_neuronlayer_block(
    bg: &mut graph::BlockGraph,
    mi: &model_instance::ModelInstance,
    input: graph::BlockPtrOutput,
    ntype: NeuronType,
    num_neurons: usize,
    init_type: InitType,
    dropout: f32,
    max_norm: f32,
    layer_norm: bool,
) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let num_inputs = bg.get_num_output_values(vec![&input]);
    if ntype == NeuronType::Sum {
        return Err(Box::new(IOError::new(ErrorKind::Other, "You should not use new_neuronlayer_block with the type NeuronType::Sum, it makes no sense - use block_misc::new_sum_block()")));
    }
    let block = match mi.optimizer {
        model_instance::Optimizer::AdagradLUT => {
            new_neuronlayer_without_weights::<optimizer::OptimizerAdagradLUT>(
                mi,
                num_inputs,
                ntype,
                num_neurons,
                init_type,
                dropout,
                max_norm,
                layer_norm,
            )
        }
        model_instance::Optimizer::AdagradFlex => {
            new_neuronlayer_without_weights::<optimizer::OptimizerAdagradFlex>(
                mi,
                num_inputs,
                ntype,
                num_neurons,
                init_type,
                dropout,
                max_norm,
                layer_norm,
            )
        }
        model_instance::Optimizer::SGD => {
            new_neuronlayer_without_weights::<optimizer::OptimizerSGD>(
                mi,
                num_inputs,
                ntype,
                num_neurons,
                init_type,
                dropout,
                max_norm,
                layer_norm,
            )
        }
    }
    .unwrap();

    let mut block_outputs = bg.add_node(block, vec![input]).unwrap();
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

pub fn new_neuron_block(
    bg: &mut graph::BlockGraph,
    mi: &model_instance::ModelInstance,
    input: graph::BlockPtrOutput,
    ntype: NeuronType,
    init_type: InitType,
) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    match ntype {
        NeuronType::Sum => block_misc::new_sum_block(bg, input),
        _ => new_neuronlayer_block(
            bg, mi, input, ntype, 1, // a single neuron
            init_type, 0.0,   // dropout
            0.0,   // maxnorm
            false, // layer norm
        ),
    }
}

impl<L: OptimizerTrait + 'static> BlockNeuronLayer<L> {
    #[inline(always)]
    fn internal_forward(&self, pb: &mut port_buffer::PortBuffer, alpha: f32) {
        unsafe {
            let (input_tape, output_tape) = block_helpers::get_input_output_borrows(
                &mut pb.tape,
                self.input_offset,
                self.num_inputs,
                self.output_offset,
                self.num_neurons,
            );

            // This is actually speed things up considerably.
            output_tape.copy_from_slice(self.weights.get_unchecked(self.bias_offset..));
            sgemv(
                b'T',                               //   trans: u8,
                self.num_inputs as i32,             //   m: i32,
                self.num_neurons as i32,            //   n: i32,
                alpha,                              //   alpha: f32,
                self.weights.get_unchecked(0..),    //  a: &[f32],
                self.num_inputs as i32,             //lda: i32,
                input_tape.get_unchecked(0..),      //   x: &[f32],
                1,                                  //incx: i32,
                1.0,                                // beta: f32,
                output_tape.get_unchecked_mut(0..), //y: &mut [f32],
                1,                                  //incy: i32
            );
        }
    }
}

impl<L: OptimizerTrait + 'static> BlockTrait for BlockNeuronLayer<L> {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) {
        debug_assert!(self.num_inputs > 0);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);

        // If we are in pure prediction mode (
        let dropout_inv = match update {
            true => self.dropout_inv,
            false => 1.0,
        };

        self.internal_forward(pb, dropout_inv);

        block_helpers::forward_backward(further_blocks, fb, pb, update);

        unsafe {
            if update && self.neuron_type == NeuronType::WeightedSum {
                // first we need to initialize inputs to zero

                let mut output_errors: [f32; MAX_NUM_INPUTS] = [0.0; MAX_NUM_INPUTS];

                let (input_tape, output_tape) = block_helpers::get_input_output_borrows(
                    &mut pb.tape,
                    self.input_offset,
                    self.num_inputs,
                    self.output_offset,
                    self.num_neurons,
                );

                for j in 0..self.num_neurons {
                    if self.dropout != 0.0
                        && *self.rng_scratchpad.get_unchecked(j) < self.dropout_threshold
                    {
                        continue;
                    }

                    let general_gradient = output_tape.get_unchecked(j) * self.dropout_inv;
                    // if this is zero, subsequent multiplications make no sense
                    if general_gradient == 0.0 {
                        continue;
                    }

                    let j_offset = j * self.num_inputs;
                    for i in 0..self.num_inputs {
                        let feature_value = input_tape.get_unchecked(i);
                        let gradient = general_gradient * feature_value;
                        let update = self.optimizer.calculate_update(
                            gradient,
                            &mut self
                                .weights_optimizer
                                .get_unchecked_mut(i + j_offset)
                                .optimizer_data,
                        );
                        *output_errors.get_unchecked_mut(i) +=
                            self.weights.get_unchecked(i + j_offset) * general_gradient;
                        *self.weights.get_unchecked_mut(i + j_offset) -= update;
                    }
                    {
                        // Updating bias term:
                        let gradient = general_gradient * 1.0;
                        let update = self.optimizer.calculate_update(
                            gradient,
                            &mut self
                                .weights_optimizer
                                .get_unchecked_mut(self.bias_offset + j)
                                .optimizer_data,
                        );
                        *self.weights.get_unchecked_mut(self.bias_offset + j) -= update;
                    }

                    if self.max_norm != 0.0 && fb.example_number % 10 == 0 {
                        let mut wsquaredsum: f32 = 0.000001; // Epsilon
                        for i in 0..self.num_inputs {
                            let w = *self.weights.get_unchecked(i + j_offset);
                            wsquaredsum += w * w;
                        }
                        let norm = wsquaredsum.sqrt();
                        if norm > self.max_norm {
                            let scaling = self.max_norm / norm;
                            for i in 0..self.num_inputs {
                                *self.weights.get_unchecked_mut(i + j_offset) *= scaling;
                            }
                        }
                    }
                }
                if self.layer_norm && fb.example_number % 10 == 0 {
                    let mut sum: f32 = 0.0;
                    let mut sumsqr: f32 = 0.0;
                    let k = 100.0;
                    for i in 0..self.bias_offset {
                        let w = self.weights.get_unchecked(i) - k;
                        sum += w;
                        sumsqr += w * w;
                    }
                    let var1 =
                        (sumsqr - sum * sum / self.bias_offset as f32) / self.bias_offset as f32;
                    let var2 = var1.sqrt();
                    for i in 0..self.bias_offset {
                        *self.weights.get_unchecked_mut(i) /= var2;
                    }
                }

                input_tape.copy_from_slice(output_errors.get_unchecked(0..self.num_inputs));
            }
        }
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        self.internal_forward(pb, 1.0);

        block_helpers::forward(further_blocks, fb, pb);
    }

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        self.internal_forward(pb, 1.0);

        block_helpers::forward_with_cache(further_blocks, fb, pb, caches);
    }

    fn allocate_and_init_weights(&mut self, _mi: &model_instance::ModelInstance) {
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);

        assert_ne!(
            self.weights_len, 0,
            "allocate_and_init_weights(): Have you forgotten to call set_num_inputs()?"
        );
        self.weights = vec![1.0; self.weights_len as usize];
        self.weights_optimizer = vec![
            OptimizerData::<L> {
                optimizer_data: self.optimizer.initial_data()
            };
            self.weights_len as usize
        ];
        self.rng_scratchpad = vec![0; self.num_neurons];
        // We need to seed each layer with a separate seed... how?
        // by the time we call this function input_offset and output_offset are set and are unique. L
        self.rng = Xoshiro256PlusPlus::seed_from_u64(
            (self.input_offset * self.output_offset + self.num_inputs + self.weights_len as usize)
                as u64,
        );

        self.bias_offset = self.num_inputs * self.num_neurons;

        match self.init_type {
            InitType::Xavier => {
                let bound = 6.0_f64.sqrt() / ((self.bias_offset) as f64).sqrt();
                let normal = Uniform::new(-bound, bound);

                for i in 0..self.bias_offset {
                    self.weights[i] = normal.sample(&mut self.rng) as f32;
                }
            }
            InitType::Hu => {
                let normal = Normal::new(0.0, (2.0 / self.num_inputs as f64).sqrt()).unwrap();

                for i in 0..self.bias_offset {
                    self.weights[i] = normal.sample(&mut self.rng) as f32;
                }
            }
            InitType::One => {
                for i in 0..self.weights_len {
                    self.weights[i as usize] = 1.0
                }
            }
            InitType::Zero => {
                for i in 0..self.weights_len {
                    self.weights[i as usize] = 0.0
                }
            }
        }

        // Bias terms are always initialized to zero
        for i in 0..self.num_neurons {
            self.weights[self.bias_offset + i] = 0.0
        }
    }

    fn get_serialized_len(&self) -> usize {
        return self.weights_len as usize;
    }

    fn write_weights_to_buf(
        &self,
        output_bufwriter: &mut dyn io::Write,
    ) -> Result<(), Box<dyn Error>> {
        block_helpers::write_weights_to_buf(&self.weights, output_bufwriter)?;
        block_helpers::write_weights_to_buf(&self.weights_optimizer, output_bufwriter)?;
        Ok(())
    }

    fn read_weights_from_buf(
        &mut self,
        input_bufreader: &mut dyn io::Read,
    ) -> Result<(), Box<dyn Error>> {
        block_helpers::read_weights_from_buf(&mut self.weights, input_bufreader)?;
        block_helpers::read_weights_from_buf(&mut self.weights_optimizer, input_bufreader)?;
        Ok(())
    }

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert_eq!(output.get_output_index(), 0);
        self.num_neurons
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {
        assert_eq!(input.get_input_index(), 0);
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert_eq!(output.get_output_index(), 0);
        self.output_offset = offset;
    }

    fn read_weights_from_buf_into_forward_only(
        &self,
        input_bufreader: &mut dyn io::Read,
        forward: &mut Box<dyn BlockTrait>,
    ) -> Result<(), Box<dyn Error>> {
        let forward = forward
            .as_any()
            .downcast_mut::<BlockNeuronLayer<optimizer::OptimizerSGD>>()
            .unwrap();
        block_helpers::read_weights_from_buf(&mut forward.weights, input_bufreader)?;
        block_helpers::skip_weights_from_buf::<OptimizerData<L>>(
            self.weights_len as usize,
            input_bufreader,
        )?;
        Ok(())
    }
}

mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::assert_epsilon;
    use crate::block_misc;
    use crate::block_misc::Observe;
    use crate::feature_buffer;
    use crate::graph::BlockGraph;
    use crate::model_instance::Optimizer;
    use block_helpers::slearn2;

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
    fn test_simple() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.nn_learning_rate = 0.1;
        mi.nn_power_t = 0.0;
        mi.optimizer = Optimizer::SGD;

        let mut bg = BlockGraph::new();
        let input_block = block_misc::new_const_block(&mut bg, vec![2.0]).unwrap();
        let neuron_block = new_neuronlayer_block(
            &mut bg,
            &mi,
            input_block,
            NeuronType::WeightedSum,
            1,
            InitType::One,
            0.0, // dropout
            0.0, // max norm
            false,
        )
        .unwrap();
        let _observe_block =
            block_misc::new_observe_block(&mut bg, neuron_block, Observe::Forward, Some(1.0))
                .unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        let fb = fb_vec();
        assert_epsilon!(slearn2(&mut bg, &fb, &mut pb, true), 2.0);
        assert_epsilon!(slearn2(&mut bg, &fb, &mut pb, true), 1.5);
    }

    #[test]
    fn test_two_neurons() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.nn_learning_rate = 0.1;
        mi.nn_power_t = 0.0;
        mi.optimizer = Optimizer::SGD;

        let num_neurons = 2;
        let mut bg = BlockGraph::new();
        let input_block = block_misc::new_const_block(&mut bg, vec![2.0]).unwrap();
        let neuron_block = new_neuronlayer_block(
            &mut bg,
            &mi,
            input_block,
            NeuronType::WeightedSum,
            num_neurons,
            InitType::One,
            0.0,   // dropout
            0.0,   // max norm
            false, // layer norm
        )
        .unwrap();
        let _observe_block =
            block_misc::new_observe_block(&mut bg, neuron_block, Observe::Forward, Some(1.0))
                .unwrap();
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        let fb = fb_vec();
        assert_epsilon!(slearn2(&mut bg, &fb, &mut pb, true), 2.0);
        // what do we expect:
        // on tape 0 input of 2.0 will be replaced with the gradient of 2.0
        // on tape 1 input has been consumed by returning function
        // on tape 2 the output was consumed by slearn
        assert_eq!(pb.observations.len(), num_neurons);
        assert_eq!(pb.observations[0], 2.0); // since we are using identity loss function, only one was consumed by slearn
        assert_eq!(pb.observations[1], 2.0); // since we are using identity loss function, only one was consumed by slearn

        assert_epsilon!(slearn2(&mut bg, &fb, &mut pb, false), 1.5);
    }
}
