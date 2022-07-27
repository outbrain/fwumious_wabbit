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
use crate::graph;
use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use block_helpers::{Weight, WeightAndOptimizerData};
use crate::block_misc;

#[derive(PartialEq)]
pub enum NeuronType {
    Sum,
    WeightedSum,
    LimitedWeightedSum,
}



pub struct BlockNeuron<L:OptimizerTrait> {    
    pub num_inputs: usize,
    pub input_offset: usize,
    pub output_offset: usize,
    pub weights_len: u32, 
    pub weights: Vec<WeightAndOptimizerData<L>>,
    pub optimizer: L,
    pub bias_term: bool,
    pub neuron_type: NeuronType,

}


pub fn new_without_weights(mi: &model_instance::ModelInstance, 
                            num_inputs: usize,
                            ntype: NeuronType) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    match mi.optimizer {
        model_instance::Optimizer::AdagradLUT => new_without_weights_2::<optimizer::OptimizerAdagradLUT>(&mi, num_inputs, ntype),
        model_instance::Optimizer::AdagradFlex => new_without_weights_2::<optimizer::OptimizerAdagradFlex>(&mi, num_inputs, ntype),
        model_instance::Optimizer::SGD => new_without_weights_2::<optimizer::OptimizerSGD>(&mi, num_inputs, ntype)
    }
}


fn new_without_weights_2<L:OptimizerTrait + 'static>(mi: &model_instance::ModelInstance, 
                                                    num_inputs: usize, 
                                                    ntype: NeuronType) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    assert!(num_inputs > 0);
    let mut rg = BlockNeuron::<L> {
        weights: Vec::new(),
        output_offset: usize::MAX,
        input_offset: usize::MAX,
        num_inputs: num_inputs,
        optimizer: L::new(),
        weights_len: num_inputs as u32,
        bias_term: false,
        neuron_type: ntype,
    };
    rg.optimizer.init(mi.learning_rate, mi.power_t, mi.init_acc_gradient);

    Ok(Box::new(rg))
}



pub fn new_neuron_block2<L:OptimizerTrait + 'static>(
                        bg: &mut graph::BlockGraph, 
                        mi: &model_instance::ModelInstance,
                        input: graph::BlockPtrOutput,
                        ntype: NeuronType
                        ) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {    
    let num_inputs = bg.get_num_outputs(vec![&input]);
    let block = new_without_weights_2::<L>(&mi, num_inputs, ntype).unwrap();
    let mut block_outputs = bg.add_node(block, vec![input]);
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

pub fn new_neuron_block(bg: &mut graph::BlockGraph, 
                        mi: &model_instance::ModelInstance,
                        input: graph::BlockPtrOutput,
                        ntype: NeuronType)
                        -> Result<graph::BlockPtrOutput, Box<dyn Error>> {

    match mi.optimizer {
        model_instance::Optimizer::AdagradLUT => new_neuron_block2::<optimizer::OptimizerAdagradLUT>(bg, &mi, input, ntype),
        model_instance::Optimizer::AdagradFlex => new_neuron_block2::<optimizer::OptimizerAdagradFlex>(bg, &mi, input, ntype),
        model_instance::Optimizer::SGD => new_neuron_block2::<optimizer::OptimizerSGD>(bg, &mi, input, ntype)
    }
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

    fn get_num_output_slots(&self) -> usize {1}   

    fn get_num_outputs(&self, output_id: graph::BlockOutput) -> usize {
        assert!(output_id.get_output_id() == 0);
        1
    }


    fn set_input_offset(&mut self, input: graph::BlockInput, offset: usize)  {
        assert!(input.get_input_id() == 0);
        self.input_offset = offset;
    }

    fn set_output_offset(&mut self, output: graph::BlockOutput, offset: usize)  {
        assert!(output.get_output_id() == 0);
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
        
        unsafe {
            let mut wsum:f32 = 0.0;
//            println!("len: {}, num inputs: {}, input_tape_indeX: {}", len, self.num_inputs, self.input_tape_index);

            {
                let myslice = &pb.tape[self.input_offset .. (self.input_offset + self.num_inputs as usize)];
                for i in 0..myslice.len() {
                    wsum += myslice.get_unchecked(i) * self.weights.get_unchecked(i).weight;
                }
//                println!("wsum: {}", wsum);
            }
            let (next_regressor, further_blocks) = further_blocks.split_at_mut(1);
            pb.tape[self.output_offset as usize] = wsum;
            next_regressor[0].forward_backward(further_blocks, fb, pb, update);
            let general_gradient = pb.tape[self.output_offset];
  //          println!("GG: {}", general_gradient);
            if update {
            {
                let mut myslice = &mut pb.tape[self.input_offset..self.input_offset + self.num_inputs as usize];
                if self.neuron_type == NeuronType::WeightedSum {
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
                    for i in 0..myslice.len() {
                        *myslice.get_unchecked_mut(i) = general_gradient;    // put the gradient on the tape in place of the value
                     }


                } else if self.neuron_type == NeuronType::LimitedWeightedSum {
                    // Here it is like WeightedSum, but weights are limited to the maximum
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
        debug_assert!(self.num_inputs > 0);
        debug_assert!(self.output_offset != usize::MAX);
        debug_assert!(self.input_offset != usize::MAX);
        
        unsafe {
            let mut wsum:f32 = 0.0;
            {
                let mut myslice = &mut pb.tape[self.input_offset..self.input_offset + self.num_inputs as usize];
                for i in 0..myslice.len() {
                    wsum += myslice.get_unchecked(i) * self.weights.get_unchecked(i).weight;
                }
            }
            let (next_regressor, further_blocks) = further_blocks.split_at(1);
            pb.tape[self.output_offset] = wsum;
            next_regressor[0].forward(further_blocks, fb, pb);
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
    use crate::block_misc;
    use crate::model_instance::Optimizer;
    use crate::feature_buffer;
    use crate::feature_buffer::HashAndValueAndSeq;
    use crate::vwmap;
    use block_helpers::{slearn2, spredict2};
    use crate::graph;
    use crate::graph::{BlockGraph};

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
        
       
        let mut bg = BlockGraph::new();
        let input_block = block_misc::new_const_block(&mut bg, vec![2.0]).unwrap();
//        println!("Graph1: {:?}", bg.nodes); 
        let neuron_block = new_neuron_block(&mut bg, &mi, input_block, NeuronType::WeightedSum).unwrap();
//        println!("Graph2: {:?}", bg.nodes); 
        let result_block = block_misc::new_result_block2(&mut bg, neuron_block, 1.0).unwrap();
//        println!("Graph3: {:?}", bg.nodes); 
        bg.schedule();
        bg.allocate_and_init_weights(&mi);
        
        let mut pb = bg.new_port_buffer();
        let fb = fb_vec();
        assert_epsilon!(slearn2  (&mut bg, &fb, &mut pb, true), 2.0);
        assert_eq!(pb.results.len(), 1);
        
        assert_epsilon!(slearn2  (&mut bg, &fb, &mut pb, true), 1.6);
        

    }


}



