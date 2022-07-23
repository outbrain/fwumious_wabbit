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
    Sum,
    WeightedSum,
    LimitedWeightedSum,
}



pub struct BlockNeuron<L:OptimizerTrait> {    
    pub num_inputs: u32,
    pub input_tape_index: i32,
    pub output_tape_index: i32,
    pub weights_len: u32, 
    pub weights: Vec<WeightAndOptimizerData<L>>,
    pub optimizer: L,
    pub bias_term: bool,
    pub neuron_type: NeuronType,

}


pub fn new_without_weights(mi: &model_instance::ModelInstance, 
                            num_inputs: u32,
                            ntype: NeuronType) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    match mi.optimizer {
        model_instance::Optimizer::AdagradLUT => new_without_weights_2::<optimizer::OptimizerAdagradLUT>(&mi, num_inputs, ntype),
        model_instance::Optimizer::AdagradFlex => new_without_weights_2::<optimizer::OptimizerAdagradFlex>(&mi, num_inputs, ntype),
        model_instance::Optimizer::SGD => new_without_weights_2::<optimizer::OptimizerSGD>(&mi, num_inputs, ntype)
    }
}


fn new_without_weights_2<L:OptimizerTrait + 'static>(mi: &model_instance::ModelInstance, 
                                                    num_inputs: u32, 
                                                    ntype: NeuronType) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    assert!(num_inputs > 0);
    let weights_len = num_inputs;
    let mut rg = BlockNeuron::<L> {
        weights: Vec::new(),
        output_tape_index: -1,
        input_tape_index: -1,
        num_inputs: num_inputs,
        optimizer: L::new(),
        weights_len: weights_len,
        bias_term: false,
        neuron_type: ntype,
    };
    rg.optimizer.init(mi.learning_rate, mi.power_t, mi.init_acc_gradient);

    Ok(Box::new(rg))
}






impl <L:OptimizerTrait + 'static> BlockTrait for BlockNeuron<L>

 {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        assert!(self.weights_len != 0, "allocate_and_init_weights(): Have you forgotten to call set_num_inputs()?");
        self.weights =vec![WeightAndOptimizerData::<L>{weight:1.0, optimizer_data: self.optimizer.initial_data()}; self.weights_len as usize];
        
    }

    fn get_num_output_tapes(&self) -> usize {1}   


    fn get_num_outputs(&self) -> u32 {
        return 1
    }
    

    fn set_input_tape_index(&mut self, input_tape_index: i32) {
        self.input_tape_index = input_tape_index;
    }

    fn set_output_tape_index(&mut self, output_tape_index: i32) {
        self.output_tape_index = output_tape_index;
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
            let mut wsum:f32 = 0.0;
            let len = pb.tapes[self.input_tape_index as usize].len();
//            println!("len: {}, num inputs: {}, input_tape_indeX: {}", len, self.num_inputs, self.input_tape_index);

            {
                let myslice = &pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize..];
                for i in 0..myslice.len() {
                    wsum += myslice.get_unchecked(i) * self.weights.get_unchecked(i).weight;
                }
//                println!("wsum: {}", wsum);
            }
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            pb.tapes[self.output_tape_index as usize].push(wsum);
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);
            let general_gradient = pb.tapes[self.output_tape_index as usize].pop().unwrap();
  //          println!("GG: {}", general_gradient);
            if update {
            {
                if self.neuron_type == NeuronType::WeightedSum {
                    let mut myslice = &mut pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize..];
                    for i in 0..myslice.len() {
                        let w = self.weights.get_unchecked(i).weight;
                        let feature_value = myslice.get_unchecked(i);
                        let gradient = general_gradient * feature_value;
                        let update = self.optimizer.calculate_update(gradient, &mut self.weights.get_unchecked_mut(i).optimizer_data);
                        self.weights.get_unchecked_mut(i).weight -= update;
                        *myslice.get_unchecked_mut(i) = w * general_gradient;    // put the gradient on the tape in place of the value
                     }
                } else if self.neuron_type == NeuronType::Sum {
                    // Do nothing, as we don't update the weights away from 1.0
                    let mut myslice = &mut pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize..];
                    for i in 0..myslice.len() {
                        *myslice.get_unchecked_mut(i) = general_gradient;    // put the gradient on the tape in place of the value
                     }


                } else if self.neuron_type == NeuronType::LimitedWeightedSum {
                    // Here it is like WeightedSum, but weights are limited to the maximum
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
                    
                }
            }
            // The only exit point
            return
        }
            
        } // unsafe end
    }
    
    fn forward(		    &self, 		
                            further_blocks: &[Box<dyn BlockTrait>], 
                            fb: &feature_buffer::FeatureBuffer,
                            pb: &mut port_buffer::PortBuffer, 
                           ) {
//            assert!(false, "Not implemented yet");
        debug_assert!(self.output_tape_index >= 0);
        debug_assert!(self.input_tape_index >= 0);
        debug_assert!(self.input_tape_index != self.output_tape_index);
        debug_assert!(self.num_inputs > 0);
        
        unsafe {
            let mut wsum:f32 = 0.0;
            let len = pb.tapes[self.input_tape_index as usize].len();
            {
                let myslice = &pb.tapes[self.input_tape_index as usize][len - self.num_inputs as usize..];
                for i in 0..myslice.len() {
                    wsum += myslice.get_unchecked(i) * self.weights.get_unchecked(i).weight;
                }
            }
            let (next_regressor, further_blocks) = further_blocks.split_at(1);
            pb.tapes[self.output_tape_index as usize].push(wsum);
            next_regressor[0].forward(further_blocks, fb, pb);
            pb.tapes[self.output_tape_index as usize].pop().unwrap();
            return
            
        } // unsafe end


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
        let mut forward = forward.as_any().downcast_mut::<BlockNeuron<optimizer::OptimizerSGD>>().unwrap();
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
        
        
        let mut re = new_without_weights(&mi, 1, NeuronType::WeightedSum).unwrap();
        re.set_input_tape_index(0);
        re.set_output_tape_index(1);
        re.allocate_and_init_weights(&mi);
        
        let mut ib = block_loss_functions::new_result_block(1, 1.0).unwrap();
        ib.set_input_tape_index(1);

        
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
        assert_eq!(pb.results.len(), 1);

        pb.reset();
        pb.tapes[0].push(2.0);
        assert_epsilon!(slearn  (&mut re, &mut ib, &fb, &mut pb, true), 1.6);
        

    }


}



