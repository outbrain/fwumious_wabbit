
use crate::optimizer;
use crate::regressor;
use crate::model_instance;
use crate::feature_buffer;
use crate::consts;
use std::io;
use std::slice;
use core::arch::x86_64::*;
use std::error::Error;



use std::mem::{self, MaybeUninit};
use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use regressor::WeightAndOptimizerData;


pub struct BlockLR<L:OptimizerTrait> {
    pub weights: Vec<WeightAndOptimizerData<L>>,
    pub weights_len: u32,
    pub optimizer_lr: L,
}

impl <L:OptimizerTrait> BlockTrait for BlockLR<L> 
where <L as optimizer::OptimizerTrait>::PerWeightStore: std::clone::Clone,
L: std::clone::Clone
{
    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights = vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.optimizer_lr.initial_data()}; self.weights_len as usize];
        
    }


    #[inline(always)]
    fn forward_backwards(&mut self, 
                            further_regressors: &mut [&mut dyn BlockTrait], 
                            wsum_in: f32, 
                            example_num: u32, 
                            fb: &feature_buffer::FeatureBuffer, 
                            update:bool) -> (f32, f32) {
        let mut wsum:f32 = 0.0;
        unsafe {
            for (i, hashvalue) in fb.lr_buffer.iter().enumerate() {
                // Prefetch couple of indexes from the future to prevent pipeline stalls due to memory latencies
                // _mm_prefetch(mem::transmute::<&f32, &i8>(&self.weights.get_unchecked((fb.lr_buffer.get_unchecked(i+8).hash) as usize).weight), _MM_HINT_T0);  // No benefit for now
                let feature_index     = hashvalue.hash;
                let feature_value:f32 = hashvalue.value;
                let feature_weight    = self.weights.get_unchecked(feature_index as usize).weight;
                wsum += feature_weight * feature_value;
            }

            let (next_regressor, further_regressors) = further_regressors.split_at_mut(1);
            let (prediction_probability, general_gradient) = next_regressor[0].forward_backwards(further_regressors, wsum_in + wsum, example_num, fb, update);

            if update {
                for hashvalue in fb.lr_buffer.iter() {
                    let feature_index     = hashvalue.hash as usize;
                    let feature_value:f32 = hashvalue.value;                        
                    let gradient = general_gradient * feature_value;
                    let update = self.optimizer_lr.calculate_update(gradient, &mut self.weights.get_unchecked_mut(feature_index).optimizer_data);
                    self.weights.get_unchecked_mut(feature_index).weight += update;
                }
            }
            (prediction_probability, general_gradient)
        } // end of unsafe
    }
    
    fn get_weights_len(&self) -> usize {
        return self.weights_len as usize;
    }

    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        if self.weights.len() == 0 {
            return Err(format!("Loading weights to unallocated weighs buffer"))?;
        }
        unsafe {
            let mut buf_view:&mut [u8] = slice::from_raw_parts_mut(self.weights.as_mut_ptr() as *mut u8, 
                                             self.weights.len() *mem::size_of::<WeightAndOptimizerData<L>>());
            input_bufreader.read_exact(&mut buf_view)?;
        }
        Ok(())
    }

    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        if self.weights.len() == 0 {
            return Err(format!("Writing weights of unallocated weights buffer"))?;
        }
        unsafe {
             let buf_view:&[u8] = slice::from_raw_parts(self.weights.as_ptr() as *const u8, 
                                              self.weights.len() *mem::size_of::<WeightAndOptimizerData<L>>());
             output_bufwriter.write(buf_view)?;
        }
        Ok(())
    }
}

