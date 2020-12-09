use std::error::Error;
use crate::optimizer::OptimizerTrait;
use crate::regressor::{Weight, WeightAndOptimizerData};
use std::io;
use std::slice;
use std::mem::{self, MaybeUninit};
use std::cmp::min;

pub fn read_weights_from_buf<L:OptimizerTrait>(weights: &mut Vec<WeightAndOptimizerData<L>>, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
    if weights.len() == 0 {
        return Err(format!("Loading weights to unallocated weighs buffer"))?;
    }
    unsafe {
        let mut buf_view:&mut [u8] = slice::from_raw_parts_mut(weights.as_mut_ptr() as *mut u8, 
                                         weights.len() *mem::size_of::<WeightAndOptimizerData<L>>());
        input_bufreader.read_exact(&mut buf_view)?;
    }
    Ok(())
}


pub fn write_weights_to_buf<L:OptimizerTrait>(weights: &Vec<WeightAndOptimizerData<L>>, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
    if weights.len() == 0 {
        return Err(format!("Writing weights of unallocated weights buffer"))?;
    }
    unsafe {
         let buf_view:&[u8] = slice::from_raw_parts(weights.as_ptr() as *const u8, 
                                          weights.len() *mem::size_of::<WeightAndOptimizerData<L>>());
         output_bufwriter.write(buf_view)?;
    }
    Ok(())
}

pub fn read_immutable_weights_from_buf<L:OptimizerTrait>(weights_len: usize, out_weights: &mut Vec<Weight>, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
    const BUF_LEN:usize = 1024 * 1024;
    let mut in_weights: Vec<WeightAndOptimizerData::<L>> = Vec::with_capacity(BUF_LEN as usize);
    let mut remaining_weights = weights_len;
    unsafe {
        while remaining_weights > 0 {
            let chunk_size = min(remaining_weights, BUF_LEN);
            in_weights.set_len(chunk_size);
            let mut in_weights_view:&mut [u8] = slice::from_raw_parts_mut(in_weights.as_mut_ptr() as *mut u8, 
                                         chunk_size * mem::size_of::<WeightAndOptimizerData<L>>());
            input_bufreader.read_exact(&mut in_weights_view)?;
            for w in &in_weights {
                out_weights.push(Weight{weight:w.weight});           
            }
            remaining_weights -= chunk_size;
        }
    }    
    Ok(())
}

