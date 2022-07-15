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


const MAX_NUM_INPUTS:usize= 16000;


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



pub struct BlockNeuronLayer<L:OptimizerTrait> {    
    pub num_inputs: u32,
    pub input_tape_index: i32,
    pub output_tape_index: i32,
    pub weights_len: u32, 
    pub weights: Vec<WeightAndOptimizerData<L>>,
    pub optimizer: L,
    pub neuron_type: NeuronType,
    pub num_neurons: u32,
    pub init_type: InitType,
}


pub fn new_without_weights(mi: &model_instance::ModelInstance, 
                            num_inputs: u32, 
                            ntype: NeuronType, 
                            num_neurons: u32,
                            init_type: InitType) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    match mi.optimizer {
        model_instance::Optimizer::AdagradLUT => new_without_weights_2::<optimizer::OptimizerAdagradLUT>(&mi, num_inputs, ntype, num_neurons, init_type),
        model_instance::Optimizer::AdagradFlex => new_without_weights_2::<optimizer::OptimizerAdagradFlex>(&mi, num_inputs, ntype, num_neurons, init_type),
        model_instance::Optimizer::SGD => new_without_weights_2::<optimizer::OptimizerSGD>(&mi, num_inputs, ntype, num_neurons, init_type)
    }
}


fn new_without_weights_2<L:OptimizerTrait + 'static>(mi: &model_instance::ModelInstance, 
                                                    num_inputs: u32, 
                                                    ntype: NeuronType, 
                                                    num_neurons: u32,
                                                    init_type: InitType) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    assert!(num_neurons > 0);
    assert!((num_inputs as usize )< MAX_NUM_INPUTS);
    assert!(num_inputs != 0);


    let weights_len = (num_inputs + 1) * num_neurons; // +1 is for bias term

    let mut rg = BlockNeuronLayer::<L> {
        weights: Vec::new(),
        output_tape_index: -1,
        input_tape_index: -1,
        num_inputs: num_inputs,
        optimizer: L::new(),
        weights_len: weights_len,
        neuron_type: ntype,
        num_neurons: num_neurons,
        init_type: init_type,
    };
    rg.optimizer.init(mi.learning_rate, mi.power_t, mi.init_acc_gradient);

    Ok(Box::new(rg))
}






impl <L:OptimizerTrait + 'static> BlockTrait for BlockNeuronLayer<L>

 {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        assert!(self.weights_len != 0, "allocate_and_init_weights(): Have you forgotten to call set_num_inputs()?");
        self.weights =vec![WeightAndOptimizerData::<L>{weight:1.0, optimizer_data: self.optimizer.initial_data()}; self.weights_len as usize];
        // now set bias terms to zero
        
        // first neuron is always set to 1.0  
        for i in 0..self.num_neurons * self.num_inputs {
            self.weights[i as usize].weight = (2.0 * merand48(((i*i+i) as usize) as u64)-1.0) * 0.001;
        }
        
        match self.init_type {
            InitType::Random => {},
            InitType::RandomFirstNeuron1 => { for i in 0..self.num_inputs { self.weights[i as usize].weight = 1.0}},
            InitType::RandomFirstNeuron10 => { for i in 0..self.num_inputs { self.weights[i as usize].weight = 0.0}; self.weights[0].weight = 1.0;},
        }
        
        
//      
        
        for i in 0..self.num_neurons {
            self.weights[(self.num_neurons * self.num_inputs + i) as usize].weight = 0.0
        }
        
    }

    fn get_num_outputs(&self) -> u32 {
        return self.num_neurons
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

//          println!("len: {}, num inputs: {}, input_tape_indeX: {}", len, self.num_inputs, self.input_tape_index);

            {

                for j in 0..self.num_neurons {
                    let mut wsum:f32 = self.weights.get_unchecked((self.num_inputs * self.num_neurons + j) as usize).weight; // bias term
                    let input_tape = pb.tapes.get_unchecked(self.input_tape_index as usize).get_unchecked(input_tape_start..);

                    for i in 0..input_tape.len() {                                 
                            wsum += input_tape.get_unchecked(i) * self.weights.get_unchecked((i as u32 + j * self.num_inputs) as usize).weight;
                    }
                    pb.tapes.get_unchecked_mut(self.output_tape_index as usize).push(wsum);
//                    println!("wsum: {}", wsum);
                }
            }
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);

            if update {
            {
//                let general_gradient = pb.tapes[self.output_tape_index as usize].pop().unwrap();
            
                if self.neuron_type == NeuronType::WeightedSum {
                    //let mut myslice = &mut pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize..];
                    // first we need to initialize inputs to zero
                    // TODO - what to think about this buffer
                    let mut output_errors: [f32; MAX_NUM_INPUTS] = MaybeUninit::uninit().assume_init();
                    for i in 0..self.num_inputs as usize {
                        output_errors[i] = 0.0; 
                    }

                    let output_tape = pb.tapes.get_unchecked(self.output_tape_index as usize).get_unchecked(output_tape_start..);
                    let input_tape = pb.tapes.get_unchecked(self.input_tape_index as usize).get_unchecked(input_tape_start..);
                    
                    for j in 0..self.num_neurons as usize {
                        let general_gradient = output_tape.get_unchecked(j);
                        let j_offset = j * self.num_inputs as usize;
//                        println!("General gradient: {}", general_gradient);
                        for i in 0..self.num_inputs as usize {
                            let feature_value = input_tape.get_unchecked(i);
  //                          println!("input tape index: {}, input tape start: {}, i: {}", self.input_tape_index, input_tape_start, i);
  //                          println!("Wieght: {}, feature value: {}", w, feature_value);
                            let gradient = general_gradient * feature_value;
    //                        println!("Final gradient: {}", gradient);
                            let update = self.optimizer.calculate_update(gradient, 
                                                                    &mut self.weights.get_unchecked_mut(i + j_offset).optimizer_data);
    //                        println!
                            *output_errors.get_unchecked_mut(i)  += self.weights.get_unchecked(i + j_offset).weight * general_gradient;
                            self.weights.get_unchecked_mut(i + j_offset).weight -= update;
                            
                        }
                        {
                            // Updating bias term:
                            let gradient = general_gradient * 1.0;
                            let update = self.optimizer.calculate_update(gradient, 
                                                                        &mut self.weights.get_unchecked_mut(((self.num_inputs* self.num_neurons) as usize + j) as usize).optimizer_data);
                            self.weights.get_unchecked_mut(((self.num_inputs * self.num_neurons) as usize + j) as usize).weight -= update;
                        }
                     }
                     
                     // TODO: Implement bias term update
                    for i in 0..self.num_inputs as usize {
                        *(pb.tapes.get_unchecked_mut(self.input_tape_index as usize)).get_unchecked_mut(input_tape_start + i) = *output_errors.get_unchecked(i);
                    }


                
                } else if self.neuron_type == NeuronType::LimitedWeightedSum {
                }
/*                    // Here it is like WeightedSum, but weights are limited to the maximum
                    let mut myslice = &mut pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize..];
                    for i in 0..myslice.len() {
                        let w = self.weights.get_unchecked(i).weight;
                        let feature_value = myslice.get_unchecked(i);
                        let gradient = general_gradient * feature_value;
                        let update = self.optimizer.calculate_update(gradient, &mut self.weights.get_unchecked_mut(i).optimizer_data);
                        self.weights.get_unchecked_mut(i).weight -= update;
                        if self.weights.get_unchecked_mut(i).weight > 1.0 {
                            self.weights.get_unchecked_mut(i).weight = 1.0;
                        } else if self.weights.get_unchecked_mut(i).weight < -1.0 {
                            self.weights.get_unchecked_mut(i).weight = -1.0;
                        }
                        
                        *myslice.get_unchecked_mut(i) = w * general_gradient;    // put the gradient on the tape in place of the value
                     }
                    
                }*/
                pb.tapes[self.output_tape_index as usize].truncate(output_tape_start);

            }
            
            // The only exit point
            return
        }
            
        } // unsafe end
    }
    
    fn forward(&self, further_blocks: &[Box<dyn BlockTrait>], 
                        fb: &feature_buffer::FeatureBuffer, 
                        pb: &mut port_buffer::PortBuffer, 
                        ) {
        assert!(false, "Unimplemented");    
    }
    
    fn get_serialized_len(&self) -> usize {
        return self.weights_len as usize;
    }

    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        block_helpers::read_weights_from_buf(&mut self.weights, input_bufreader)
    }

    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        block_helpers::write_weights_to_buf(&self.weights, output_bufwriter)
    }

    fn read_weights_from_buf_into_forward_only(&self, input_bufreader: &mut dyn io::Read, forward: &mut Box<dyn BlockTrait>) -> Result<(), Box<dyn Error>> {
        let mut forward = forward.as_any().downcast_mut::<BlockNeuronLayer<optimizer::OptimizerSGD>>().unwrap();
        block_helpers::read_weights_only_from_buf2::<L>(self.weights_len as usize, &mut forward.weights, input_bufreader)
    }

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>> {
        self.weights[index].weight = w[0];
        self.weights[index].optimizer_data = self.optimizer.initial_data();
        Ok(())
    }
}










mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::block_loss_functions;
    use crate::model_instance::Optimizer;
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
        mi.optimizer = Optimizer::SGD;
        
        
        let mut re = new_without_weights(&mi, 1, NeuronType::WeightedSum, 1).unwrap();
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

        pb.tapes[0].push(2.0);
        assert_epsilon!(slearn  (&mut re, &mut ib, &fb, &mut pb, true), 1.6);
        

    }

    #[test]
    fn test_two_neurons() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.optimizer = Optimizer::SGD;
        
        
        let NUM_NEURONS = 2;
        let mut re = new_without_weights(&mi, 1, NeuronType::WeightedSum, NUM_NEURONS).unwrap();
        re.set_input_tape_index(0);
        re.set_output_tape_index(1);
        re.allocate_and_init_weights(&mi);
        
        let mut ib = block_loss_functions::new_identity_block(&mi, NUM_NEURONS).unwrap();
        ib.set_input_tape_index(1);
        ib.set_output_tape_index(2);

        
        let mut pb = port_buffer::PortBuffer::new(&mi);
        let fb = fb_vec();
        pb.tapes[0].push(2.0);
        assert_epsilon!(slearn  (&mut re, &mut ib, &fb, &mut pb, true), 2.0);
        // what do we expect:
        // on tape 0 input of 2.0 will be replaced with the gradient of 2.0
        // on tape 1 input has been consumed by returning function
        // on tape 2 the output was consumed by slearn
        assert_eq!(pb.tapes[0][0], 2.0);
        assert_eq!(pb.tapes[1].len(), 0);
        assert_eq!(pb.tapes[2].len(), 1); // since we are using identity loss function, only one was consumed by slearn

        pb.reset();
        pb.tapes[0].push(2.0);
        assert_epsilon!(slearn  (&mut re, &mut ib, &fb, &mut pb, false), 1.6);
        

    }


}



