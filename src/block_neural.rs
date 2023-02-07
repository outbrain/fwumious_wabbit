use rand_distr::{Distribution, Normal, Uniform};
use rand_xoshiro::rand_core::RngCore;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::any::Any;
use std::error::Error;
use std::io;
use std::io::Error as IOError;
use std::io::ErrorKind;
use std::mem::{self, MaybeUninit};
use std::slice;

use crate::block_helpers;
use crate::block_misc;
use crate::feature_buffer;
use crate::graph;
use crate::model_instance;
use crate::optimizer;
use crate::port_buffer;
use crate::regressor;
use block_helpers::{OptimizerData, Weight};
use optimizer::OptimizerTrait;
use regressor::BlockTrait;

use blas::*;

const MAX_NUM_INPUTS: usize = 16000;
const USE_BLAS: bool = true;
const WEIGHT_NN_DEFAULT_FALLBACK: f32 = 0.000001;
const WEIGHT_NN_UPPER_BOUND: f32 = 1.5;

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
    pub weights: Vec<Weight>,
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
    assert!((num_inputs as usize) < MAX_NUM_INPUTS);
    assert!(num_inputs != 0);

    if dropout != 0.0 {
        panic!("Dropout in this binary is not supported due to bizzare side effects on inner loop unrolling");
    }

    let weights_len = ((num_inputs + 1) * num_neurons as usize) as u32; // +1 is for bias term

    let mut rg = BlockNeuronLayer::<L> {
        weights: Vec::new(),
        weights_optimizer: Vec::new(),
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs: num_inputs,
        optimizer: L::new(),
        weights_len: weights_len,
        neuron_type: ntype,
        num_neurons: num_neurons,
        init_type: init_type,
        dropout: dropout,
        dropout_inv: 1.0 / (1.0 - dropout),
        max_norm: max_norm,
        layer_norm: layer_norm,
        rng: Xoshiro256PlusPlus::seed_from_u64(0 as u64),
        rng_scratchpad: Vec::new(),
        dropout_threshold: ((u32::MAX as f64) * (dropout as f64)) as u32,
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
                &mi,
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
                &mi,
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
                &mi,
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

impl<L: OptimizerTrait + 'static> BlockTrait for BlockNeuronLayer<L> {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);

        assert!(
            self.weights_len != 0,
            "allocate_and_init_weights(): Have you forgotten to call set_num_inputs()?"
        );
        self.weights = vec![Weight { weight: 1.0 }; self.weights_len as usize];
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

        match self.init_type {
            InitType::Xavier => {
                let bound = 6.0_f64.sqrt() / ((self.num_inputs + self.num_neurons) as f64).sqrt();
                let normal = Uniform::new(-bound, bound);
                for i in 0..self.num_neurons * self.num_inputs {
                    self.weights[i as usize].weight = normal.sample(&mut self.rng) as f32;
                }
            }
            InitType::Hu => {
                let normal =
                    Normal::new(0.0, (2.0 / self.num_inputs as f64).sqrt() as f64).unwrap();
                for i in 0..self.num_neurons * self.num_inputs {
                    self.weights[i as usize].weight = normal.sample(&mut self.rng) as f32;
                }
            }
            //            InitType::RandomFirst1 => { for i in 0..self.num_inputs { self.weights[i as usize].weight = 1.0}},
            //            InitType::RandomFirst10 => { for i in 0..self.num_inputs { self.weights[i as usize].weight = 0.0}; self.weights[0].weight = 1.0;},
            InitType::One => {
                for i in 0..self.weights_len {
                    self.weights[i as usize].weight = 1.0
                }
            }
            InitType::Zero => {
                for i in 0..self.weights_len {
                    self.weights[i as usize].weight = 0.0
                }
            }
        }

        // Bias terms are always initialized to zero
        for i in 0..self.num_neurons {
            self.weights[(self.num_neurons * self.num_inputs + i) as usize].weight = 0.0
        }
    }

    fn get_num_output_slots(&self) -> usize {
        1
    }

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert!(output.get_output_index() == 0);
        self.num_neurons
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {
        assert!(input.get_input_index() == 0);
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert!(output.get_output_index() == 0);
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
        debug_assert!(self.num_inputs > 0);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);

        unsafe {
            let bias_offset = self.num_inputs * self.num_neurons;
            {
                let (input_tape, output_tape) = block_helpers::get_input_output_borrows(
                    &mut pb.tape,
                    self.input_offset,
                    self.num_inputs,
                    self.output_offset,
                    self.num_neurons,
                );

                // If we are in pure prediction mode (
                let dropout_inv = match update {
                    true => self.dropout_inv,
                    false => 1.0,
                };
                if !USE_BLAS {
                    let mut j_offset: usize = 0;
                    for j in 0..self.num_neurons {
                        let mut wsum: f32 = self.weights.get_unchecked(bias_offset + j).weight; // bias term
                        for i in 0..self.num_inputs {
                            wsum += input_tape.get_unchecked(i)
                                * self.weights.get_unchecked(i + j_offset as usize).weight;
                        }
                        j_offset += self.num_inputs;
                        *output_tape.get_unchecked_mut(j) = wsum * self.dropout_inv;
                    }
                } else {
                    // This is actually speed things up considerably.
                    output_tape.copy_from_slice(std::mem::transmute::<&[Weight], &[f32]>(
                        self.weights.get_unchecked(bias_offset..),
                    ));
                    sgemv(
                        b'T',                                                                      //   trans: u8,
                        self.num_inputs as i32,  //   m: i32,
                        self.num_neurons as i32, //   n: i32,
                        dropout_inv,             //   alpha: f32,
                        std::mem::transmute::<&[Weight], &[f32]>(self.weights.get_unchecked(0..)), //  a: &[f32],
                        self.num_inputs as i32,             // lda: i32,
                        &input_tape.get_unchecked(0..),     // x: &[f32],
                        1,                                  // incx: i32,
                        1.0,                                // beta: f32,
                        output_tape.get_unchecked_mut(0..), //y: &mut [f32],
                        1,                                  // incy: i32
                    )
                }

                if update {
                    // In case we are doing doing learning, and we have a dropout

                    if false && self.dropout != 0.0 {
                        let mut fill_view: &mut [u8] = slice::from_raw_parts_mut(
                            self.rng_scratchpad.as_mut_ptr() as *mut u8,
                            self.num_neurons * mem::size_of::<u32>(),
                        );

                        //let mut fill_view: [u8; 100] = [0; 100];
                        // This one is from the "You won't believe it till you measure it 10 times and even then you will suspect yourself not the compiler"
                        // For some reason call to this function in a block that NEVER executes, slows down the code by 50%
                        // Specifically it prevents loop unrolling of the weight optimizer 50 lines below ... discovered through disassembly
                        // I thought It might be related to from_raw_parts above, but it isn't

                        // Attempted fix of going through a #[inline(never)] function did not work
                        self.rng.fill_bytes(&mut fill_view);
                        //fill_bytes(&mut self.rng, &mut fill_view);
                        for j in 0..self.num_neurons {
                            if *self.rng_scratchpad.get_unchecked(j) < self.dropout_threshold {
                                *output_tape.get_unchecked_mut(j) = 0.0;
                            }
                        }
                    }
                }
            }

            block_helpers::forward_backward(further_blocks, fb, pb, update);

            if update {
                if self.neuron_type == NeuronType::WeightedSum {
                    // first we need to initialize inputs to zero
                    // TODO - what to think about this buffer
                    let mut output_errors: [f32; MAX_NUM_INPUTS] =
                        MaybeUninit::uninit().assume_init();
                    output_errors
                        .get_unchecked_mut(0..self.num_inputs)
                        .fill(0.0);

                    let (input_tape, output_tape) = block_helpers::get_input_output_borrows(
                        &mut pb.tape,
                        self.input_offset,
                        self.num_inputs,
                        self.output_offset,
                        self.num_neurons,
                    );

                    for j in 0..self.num_neurons as usize {
                        if self.dropout != 0.0
                            && *self.rng_scratchpad.get_unchecked(j) < self.dropout_threshold
                        {
                            //          println!("B:j {}", j);
                            continue;
                        }

                        let general_gradient = output_tape.get_unchecked(j) * self.dropout_inv;

                        let j_offset = j * self.num_inputs as usize;
                        for i in 0..self.num_inputs as usize {
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
                                self.weights.get_unchecked(i + j_offset).weight * general_gradient;


			    if self.weights.get_unchecked_mut(i + j_offset).weight.abs() > WEIGHT_NN_UPPER_BOUND {
				self.weights.get_unchecked_mut(i + j_offset).weight = WEIGHT_NN_DEFAULT_FALLBACK;
			    }
			    
                            self.weights.get_unchecked_mut(i + j_offset).weight -= update;
                        }
                        {
                            // Updating bias term:
                            let gradient = general_gradient * 1.0;
                            let update = self.optimizer.calculate_update(
                                gradient,
                                &mut self
                                    .weights_optimizer
                                    .get_unchecked_mut(bias_offset + j)
                                    .optimizer_data,
                            );

			    if self.weights.get_unchecked_mut(bias_offset + j).weight.abs() > WEIGHT_NN_UPPER_BOUND {
				self.weights.get_unchecked_mut(bias_offset + j).weight = WEIGHT_NN_DEFAULT_FALLBACK;
			    }
			    
                            self.weights.get_unchecked_mut(bias_offset + j).weight -= update;
                        }

                        if self.max_norm != 0.0 && fb.example_number % 10 == 0 {
                            let mut wsquaredsum = 0.000001; // Epsilon
                            for i in 0..self.num_inputs as usize {
                                let w = self.weights.get_unchecked_mut(i + j_offset).weight;
                                wsquaredsum += w * w;
                            }
                            let norm = wsquaredsum.sqrt();
                            if norm > self.max_norm {
                                let scaling = self.max_norm / norm;
                                for i in 0..self.num_inputs as usize {
                                    self.weights.get_unchecked_mut(i + j_offset).weight *= scaling;
                                }
                            }
                        }
                    }
                    if self.layer_norm && fb.example_number % 10 == 0 {
                        let mut sum: f32 = 0.0;
                        let mut sumsqr: f32 = 0.0;
                        let K = 100.0;
                        for i in 0..bias_offset {
                            let w = self.weights.get_unchecked(i).weight - K;
                            sum += w;
                            sumsqr += w * w;
                        }
                        let var1 = (sumsqr - sum * sum / bias_offset as f32) / bias_offset as f32;
                        let var2 = var1.sqrt();
                        for i in 0..bias_offset {
                            self.weights.get_unchecked_mut(i).weight /= var2;
                        }
                    }

                    input_tape.copy_from_slice(output_errors.get_unchecked(0..self.num_inputs));
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
        unsafe {
            let frandseed = fb.example_number * fb.example_number;
            let bias_offset = self.num_inputs * self.num_neurons;
            let (input_tape, output_tape) = block_helpers::get_input_output_borrows(
                &mut pb.tape,
                self.input_offset,
                self.num_inputs,
                self.output_offset,
                self.num_neurons,
            );

            if !USE_BLAS {
                let mut j_offset: usize = 0;
                for j in 0..self.num_neurons {
                    let mut wsum: f32 = self.weights.get_unchecked(bias_offset + j).weight; // bias term
                    for i in 0..self.num_inputs {
                        wsum += input_tape.get_unchecked(i)
                            * self.weights.get_unchecked(i + j_offset as usize).weight;
                    }
                    j_offset += self.num_inputs;
                    *output_tape.get_unchecked_mut(j) = wsum;
                }
            } else {
                // This is actually speed things up considerably.
                output_tape.copy_from_slice(std::mem::transmute::<&[Weight], &[f32]>(
                    self.weights.get_unchecked(bias_offset..),
                ));
                sgemv(
                    b'T',                                                                      //   trans: u8,
                    self.num_inputs as i32,  //   m: i32,
                    self.num_neurons as i32, //   n: i32,
                    1.0,                     //   alpha: f32,
                    std::mem::transmute::<&[Weight], &[f32]>(self.weights.get_unchecked(0..)), //  a: &[f32],
                    self.num_inputs as i32,             //lda: i32,
                    &input_tape.get_unchecked(0..),     //   x: &[f32],
                    1,                                  //incx: i32,
                    1.0,                                // beta: f32,
                    output_tape.get_unchecked_mut(0..), //y: &mut [f32],
                    1,                                  //incy: i32
                )
            }
            block_helpers::forward(further_blocks, fb, pb);
        } // unsafe end
    }

    fn get_serialized_len(&self) -> usize {
        return self.weights_len as usize;
    }

    fn read_weights_from_buf(
        &mut self,
        input_bufreader: &mut dyn io::Read,
    ) -> Result<(), Box<dyn Error>> {
        block_helpers::read_weights_from_buf(&mut self.weights, input_bufreader)?;
        block_helpers::read_weights_from_buf(&mut self.weights_optimizer, input_bufreader)?;
        Ok(())
    }

    fn write_weights_to_buf(
        &self,
        output_bufwriter: &mut dyn io::Write,
    ) -> Result<(), Box<dyn Error>> {
        block_helpers::write_weights_to_buf(&self.weights, output_bufwriter)?;
        block_helpers::write_weights_to_buf(&self.weights_optimizer, output_bufwriter)?;
        Ok(())
    }

    fn read_weights_from_buf_into_forward_only(
        &self,
        input_bufreader: &mut dyn io::Read,
        forward: &mut Box<dyn BlockTrait>,
    ) -> Result<(), Box<dyn Error>> {
        let mut forward = forward
            .as_any()
            .downcast_mut::<BlockNeuronLayer<optimizer::OptimizerSGD>>()
            .unwrap();
        block_helpers::read_weights_from_buf(&mut forward.weights, input_bufreader)?;
        block_helpers::skip_weights_from_buf(
            self.weights_len as usize,
            &self.weights_optimizer,
            input_bufreader,
        )?;
        Ok(())
    }

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(
        &mut self,
        aa: i32,
        bb: i32,
        index: usize,
        w: &[f32],
    ) -> Result<(), Box<dyn Error>> {
        self.weights[index].weight = w[0];
        self.weights_optimizer[index].optimizer_data = self.optimizer.initial_data();
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
            ffm_fields_count: 0,
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
        let observe_block =
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

        let NUM_NEURONS = 2;
        let mut bg = BlockGraph::new();
        let input_block = block_misc::new_const_block(&mut bg, vec![2.0]).unwrap();
        let neuron_block = new_neuronlayer_block(
            &mut bg,
            &mi,
            input_block,
            NeuronType::WeightedSum,
            NUM_NEURONS,
            InitType::One,
            0.0,   // dropout
            0.0,   // max norm
            false, // layer norm
        )
        .unwrap();
        let observe_block =
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
        assert_eq!(pb.observations.len(), NUM_NEURONS as usize);
        assert_eq!(pb.observations[0], 2.0); // since we are using identity loss function, only one was consumed by slearn
        assert_eq!(pb.observations[1], 2.0); // since we are using identity loss function, only one was consumed by slearn

        assert_epsilon!(slearn2(&mut bg, &fb, &mut pb, false), 1.5);
    }

    // #[test]
    // fn test_neuron() {
    //     let mut mi = model_instance::ModelInstance::new_empty().unwrap();
    //     mi.nn_learning_rate = 0.1;
    //     mi.nn_power_t = 0.0;
    //     mi.nn_init_acc_gradient = mi.init_acc_gradient;
    //     mi.optimizer = Optimizer::SGD;

    //     let mut bg = BlockGraph::new();
    //     let input_block = block_misc::new_const_block(&mut bg, vec![2.0]).unwrap();
    //     let neuron_block = new_neuron_block(&mut bg, &mi, input_block, NeuronType::WeightedSum, InitType::One).unwrap();
    //     let observe_block = block_misc::new_observe_block(&mut bg, neuron_block, Observe::Forward, Some(1.0)).unwrap();
    //     bg.finalize();
    //     bg.allocate_and_init_weights(&mi);

    //     let mut pb = bg.new_port_buffer();
    //     let fb = fb_vec();
    //     assert_epsilon!(slearn2  (&mut bg, &fb, &mut pb, true), 2.0);
    //     assert_eq!(pb.observations.len(), 1);
    //     assert_epsilon!(slearn2  (&mut bg, &fb, &mut pb, true), 1.5);
    // }
    /*
        #[test]
        fn test_dropout() {
            let mut mi = model_instance::ModelInstance::new_empty().unwrap();

            let NUM_NEURONS = 6;
            let mut bg = BlockGraph::new();
            let input_block = block_misc::new_const_block(&mut bg, vec![3.0]).unwrap();
            let observe_block_backward = block_misc::new_observe_block(&mut bg, input_block, Observe::Backward, None).unwrap();
            let neuron_block = new_neuronlayer_block(&mut bg,
                                                &mi,
                                                observe_block_backward,
                                                NeuronType::WeightedSum,
                                                NUM_NEURONS,
                                                InitType::One,
                                                0.5, // dropout
                                                0.0, // max norm
                                                false,
                                                ).unwrap();
            let observe_block = block_misc::new_observe_block(&mut bg, neuron_block, Observe::Forward, Some(3.0)).unwrap();
            bg.finalize();
            bg.allocate_and_init_weights(&mi);

            let mut pb = bg.new_port_buffer();

            let fb = fb_vec();

            spredict2  (&mut bg, &fb, &mut pb, false);
            assert_eq!(pb.observations, vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, // mostly deterministic due to specific PRNG use,
                                            3.0 			       // Untouched output when forward-only
                                                        ]);

            slearn2  (&mut bg, &fb, &mut pb, true);
            assert_eq!(pb.observations, vec![6.0, 0.0, 0.0, 0.0, 6.0, 6.0, // mostly deterministic due to specific PRNG use
                                            18.0 			       // scaled backprop
                                                        ]);
    //        println!("O: {:?}", pb.observations);



        }

    */

    /*    #[test]
        fn test_segm() {
            unsafe {
            let input_matrix:Vec<f32> = vec![1.0, 2.0,
                                             3.0, 4.0,
                                             5.0, 6.0];
            let input_vec: Vec<f32> = vec![2.0, 1.0];
            let mut output_vec : Vec<f32> = vec![0.0, 0.0, 0.0];

                        sgemv(
                         b'T',//   trans: u8,
                         2, //   m: i32,          // num_neurons
                         3, //   n: i32, 	      // num inputs
                         1.0, //   alpha: f32,
                         &input_matrix, //  a: &[f32],
                         2,   //lda: i32,
                         &input_vec,//   x: &[f32],
                            1, //incx: i32,
                            1.0, // beta: f32,
                            &mut output_vec, //y: &mut [f32],
                            1,//incy: i32
                        );

        println!("Output : {:?}", output_vec);

        }
        }
    */
}
