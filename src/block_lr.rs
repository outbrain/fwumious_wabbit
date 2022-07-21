use std::any::Any;

use crate::optimizer;
use crate::regressor;
use crate::model_instance;
use crate::feature_buffer;
use std::io;
use core::arch::x86_64::*;
use std::error::Error;



use std::mem::{self, MaybeUninit};
use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use crate::block_helpers;
use block_helpers::{Weight, WeightAndOptimizerData};
use crate::port_buffer;


pub struct BlockLR<L:OptimizerTrait> {
    pub weights: Vec<WeightAndOptimizerData<L>>,
    pub weights_len: u32,
    pub optimizer_lr: L,
    pub output_tape_index: i32,
    pub num_combos: u32,
}

pub fn new_without_weights(mi: &model_instance::ModelInstance) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    match mi.optimizer {
        model_instance::Optimizer::AdagradLUT => new_without_weights_2::<optimizer::OptimizerAdagradLUT>(&mi),
        model_instance::Optimizer::AdagradFlex => new_without_weights_2::<optimizer::OptimizerAdagradFlex>(&mi),
        model_instance::Optimizer::SGD => new_without_weights_2::<optimizer::OptimizerSGD>(&mi)
    }
}

fn new_without_weights_2<L:OptimizerTrait + 'static>(mi: &model_instance::ModelInstance) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    let mut num_combos = mi.feature_combo_descs.len() as u32;
    if mi.add_constant_feature {
        num_combos += 1;
    }
    let mut reg_lr = BlockLR::<L> {
        weights: Vec::new(),
        weights_len: 0, 
        optimizer_lr: L::new(),
        output_tape_index: -1, 
        num_combos: num_combos,

    };
    reg_lr.optimizer_lr.init(mi.learning_rate, mi.power_t, mi.init_acc_gradient);
    reg_lr.weights_len = 1 << mi.bit_precision;
    Ok(Box::new(reg_lr))
}

impl <L:OptimizerTrait + 'static> BlockTrait for BlockLR<L> 
{
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights = vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.optimizer_lr.initial_data()}; self.weights_len as usize];
        
    }

    fn get_num_outputs(&self) -> u32 {
        return self.num_combos;
    }
    
    fn set_input_tape_index(&mut self, output_tape_index: i32) {
        panic!("You cannnot set input_tape_index for BlockLR");
    }

    fn set_output_tape_index(&mut self, output_tape_index: i32) {
        self.output_tape_index = output_tape_index;
    }

    fn get_output_tape_index(&self) -> i32 {
        self.output_tape_index
    }

    #[inline(always)]
    fn forward_backward(&mut self, 
                            further_regressors: &mut [Box<dyn BlockTrait>], 
                            fb: &feature_buffer::FeatureBuffer, 
                            pb: &mut port_buffer::PortBuffer,                             
                            update:bool) {
        debug_assert!(self.output_tape_index >= 0);

        let mut wsum:f32 = 0.0;
        unsafe {
            let original_size = pb.tapes[self.output_tape_index as usize].len();
            {
                pb.tapes[self.output_tape_index as usize].resize_with(original_size + self.num_combos as usize, || {0.0});
                let myslice = &mut pb.tapes.get_unchecked_mut(self.output_tape_index as usize).get_unchecked_mut(original_size..);

                for hashvalue in fb.lr_buffer.iter() {
                    // Prefetch couple of indexes from the future to prevent pipeline stalls due to memory latencies
    //            for (i, hashvalue) in fb.lr_buffer.iter().enumerate() {
                    // _mm_prefetch(mem::transmute::<&f32, &i8>(&self.weights.get_unchecked((fb.lr_buffer.get_unchecked(i+8).hash) as usize).weight), _MM_HINT_T0);  // No benefit for now
                    let feature_index     = hashvalue.hash;
                    let feature_value:f32 = hashvalue.value;
                    let combo_index = hashvalue.combo_index;
                    let feature_weight    = self.weights.get_unchecked(feature_index as usize).weight;
                    *myslice.get_unchecked_mut(combo_index as usize) += feature_weight * feature_value;
                }
            }
            let (next_regressor, further_regressors) = further_regressors.split_at_mut(1);
//            pb.tapes[self.output_tape_index as usize].push(wsum);
//            println!("Pushing feature value: {}", wsum);
            next_regressor[0].forward_backward(further_regressors, fb, pb, update);
//            let general_gradient = pb.tapes[self.output_tape_index as usize].pop().unwrap();

            if update {
                let myslice = &mut pb.tapes.get_unchecked(self.output_tape_index as usize).get_unchecked(original_size..);

                for hashvalue in fb.lr_buffer.iter() {
                    let feature_index     = hashvalue.hash as usize;
                    let feature_value:f32 = hashvalue.value;                        
                    let general_gradient = myslice.get_unchecked(hashvalue.combo_index as usize);
                    let gradient = general_gradient * feature_value;
                    let update = self.optimizer_lr.calculate_update(gradient, &mut self.weights.get_unchecked_mut(feature_index).optimizer_data);
                    self.weights.get_unchecked_mut(feature_index).weight -= update;
                }
            }
            pb.tapes[self.output_tape_index as usize].truncate(original_size );

            return
        } // end of unsafe
    }
    
    
    fn forward(&self, 
             further_blocks: &[Box<dyn BlockTrait>], 
             fb: &feature_buffer::FeatureBuffer,
             pb: &mut port_buffer::PortBuffer) {
        let fbuf = &fb.lr_buffer;
        let mut wsum:f32 = 0.0;
        
        let original_size = pb.tapes[self.output_tape_index as usize].len();
        pb.tapes[self.output_tape_index as usize].resize_with(original_size + self.num_combos as usize, || {0.0});
        {

            unsafe {
                let myslice = &mut pb.tapes.get_unchecked_mut(self.output_tape_index as usize).get_unchecked_mut(original_size..);
                for val in fbuf {
                    let hash = val.hash as usize;
                    let feature_value:f32 = val.value;
                    *myslice.get_unchecked_mut(val.combo_index as usize) += self.weights.get_unchecked(hash).weight * feature_value;
                }
            }
        }
        let (next_regressor, further_blocks) = further_blocks.split_at(1);
//        pb.tapes[self.output_tape_index as usize].push(wsum);
        next_regressor[0].forward(further_blocks, fb, pb);
//        pb.tapes[self.output_tape_index as usize].pop().unwrap();
        pb.tapes[self.output_tape_index as usize].truncate(original_size );
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
        let mut forward = forward.as_any().downcast_mut::<BlockLR<optimizer::OptimizerSGD>>().unwrap();
        block_helpers::read_weights_only_from_buf2::<L>(self.weights_len as usize, &mut forward.weights, input_bufreader)
    }
    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>> {
        self.weights[index].weight = w[0];
        self.weights[index].optimizer_data = self.optimizer_lr.initial_data();
        Ok(())
    }


}

