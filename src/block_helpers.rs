use std::error::Error;
use crate::optimizer::OptimizerTrait;
use std::io;
use std::slice;
use std::mem::{self, MaybeUninit};
use std::cmp::min;
use std::ops::Deref;
use crate::optimizer::OptimizerSGD;
use std::marker::PhantomData;
use serde_json::{Value, Number};
use crate::feature_buffer;
use crate::regressor::BlockTrait;

#[derive(Clone, Debug)]
#[repr(C)]
pub struct Weight {
    pub weight: f32, 
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct WeightAndOptimizerData<L:OptimizerTrait> {
    pub weight: f32, 
    pub optimizer_data: L::PerWeightStore,
}

#[macro_export]
macro_rules! assert_epsilon {
    ($x:expr, $y:expr) => {
        if !($x - $y < 0.0000001 || $y - $x < 0.0000001) { panic!(); }
    }
}




// It's OK! I am a limo driver!
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
         output_bufwriter.write_all(buf_view)?;
    }
    Ok(())
}


pub fn read_weights_only_from_buf2<L:OptimizerTrait>(weights_len: usize, out_weights: &mut Vec<WeightAndOptimizerData<OptimizerSGD>>, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
    const BUF_LEN:usize = 1024 * 1024;
    let mut in_weights: Vec<WeightAndOptimizerData::<L>> = Vec::with_capacity(BUF_LEN as usize);
    let mut remaining_weights = weights_len;
    let mut out_idx: usize = 0;
    if weights_len != out_weights.len() {
        return Err(format!("read_weights_only_from_buf2 - number of weights to read ({}) and number of weights allocated ({}) isn't the same", weights_len, out_weights.len()))?;
    }

    unsafe {
        while remaining_weights > 0 {
            let chunk_size = min(remaining_weights, BUF_LEN);
            in_weights.set_len(chunk_size);
            let mut in_weights_view:&mut [u8] = slice::from_raw_parts_mut(in_weights.as_mut_ptr() as *mut u8, 
                                         chunk_size * mem::size_of::<WeightAndOptimizerData<L>>());
            input_bufreader.read_exact(&mut in_weights_view)?;
            for w in &in_weights {
                //out_weights.push(WeightAndOptimizerData{weight:w.weight, optimizer_data: std::marker::PhantomData{}});
                out_weights.get_unchecked_mut(out_idx).weight = w.weight;
                out_idx += 1;
            }
            remaining_weights -= chunk_size;
        }
    }    
    Ok(())
}


/// This function is used only in tests to run a single block with given loss function
pub fn slearn<'a>(block_run: &mut Box<dyn BlockTrait>, 
                    block_loss_function: &mut Box<dyn BlockTrait>,
                    fb: &feature_buffer::FeatureBuffer, 
                    update: bool) -> f32 {

    unsafe {
        let block_loss_function: Box<dyn BlockTrait> = mem::transmute(& *block_loss_function.deref().deref());
        let mut further_blocks_v: Vec<Box<dyn BlockTrait>> = vec![block_loss_function];
        let further_blocks = &mut further_blocks_v[..];
        let (prediction_probability, general_gradient) = block_run.forward_backward(further_blocks, 0.0, fb, update);
        // black magic here: forget about further blocks that we got through transmute:
        further_blocks_v.set_len(0);
        return prediction_probability
    }
}

/// This function is used only in tests to run a single block with given loss function
pub fn spredict<'a>(block_run: &mut Box<dyn BlockTrait>, 
                    block_loss_function: &mut Box<dyn BlockTrait>,
                    fb: &feature_buffer::FeatureBuffer, 
                    update: bool) -> f32 {

    unsafe {
        let block_loss_function: Box<dyn BlockTrait> = mem::transmute(& *block_loss_function.deref().deref());
        let mut further_blocks_v: Vec<Box<dyn BlockTrait>> = vec![block_loss_function];
        let further_blocks = & further_blocks_v[..];
        let prediction_probability = block_run.forward(further_blocks, 0.0, fb);
        // black magic here: forget about further blocks that we got through transmute:
        further_blocks_v.set_len(0);
        return prediction_probability
    }
}

pub fn f32_to_json(f: f32) -> Value {
    let n = Number::from_f64(f as f64);
    match n {
        Some(v) => return Value::Number(v),
        None => return Value::String(format!("{}", f).to_string()) 
    };
}


