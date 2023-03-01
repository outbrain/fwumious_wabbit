use crate::optimizer::OptimizerTrait;
use std::error::Error;
use std::io;
use std::io::Read;

use crate::feature_buffer;
use crate::graph;
use crate::optimizer::OptimizerSGD;
use crate::port_buffer;
use crate::regressor::BlockTrait;
use std::cmp::min;
use std::mem::{self};
use std::slice;

#[derive(Clone, Debug)]
#[repr(C)]
pub struct Weight {
    pub weight: f32,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct OptimizerData<L: OptimizerTrait> {
    pub optimizer_data: L::PerWeightStore,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct WeightAndOptimizerData<L: OptimizerTrait> {
    pub weight: f32,
    pub optimizer_data: L::PerWeightStore,
}

#[macro_export]
macro_rules! assert_epsilon {
    ($x:expr, $y:expr) => {
        let x = $x; // Make sure we evaluate only once
        let y = $y;
        if !(x - y < 0.000005 && y - x < 0.000005) {
            println!("Expectation: {}, Got: {}", y, x);
            panic!();
        }
    };
}

// It's OK! I am a limo driver!
pub fn read_weights_from_buf<L>(
    weights: &mut Vec<L>,
    input_bufreader: &mut dyn io::Read,
) -> Result<(), Box<dyn Error>> {
    if weights.len() == 0 {
        return Err(format!("Loading weights to unallocated weighs buffer"))?;
    }
    unsafe {
        let mut buf_view: &mut [u8] = slice::from_raw_parts_mut(
            weights.as_mut_ptr() as *mut u8,
            weights.len() * mem::size_of::<L>(),
        );
        input_bufreader.read_exact(&mut buf_view)?;
    }
    Ok(())
}

// We get a vec here just so we easily know the type...
// Skip amount of bytes that a weights vector would be
pub fn skip_weights_from_buf<L>(
    weights_len: usize,
    weights: &Vec<L>,
    input_bufreader: &mut dyn Read,
) -> Result<(), Box<dyn Error>> {
    let bytes_skip = weights_len * mem::size_of::<L>();
    io::copy(
        &mut input_bufreader.take(bytes_skip as u64),
        &mut io::sink(),
    )?;
    Ok(())
}

pub fn write_weights_to_buf<L>(
    weights: &Vec<L>,
    output_bufwriter: &mut dyn io::Write,
) -> Result<(), Box<dyn Error>> {
    if weights.len() == 0 {
        assert!(false);
        return Err(format!("Writing weights of unallocated weights buffer"))?;
    }
    unsafe {
        let buf_view: &[u8] = slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * mem::size_of::<L>(),
        );
        output_bufwriter.write_all(buf_view)?;
    }
    Ok(())
}

pub fn read_weights_only_from_buf2<L: OptimizerTrait>(
    weights_len: usize,
    out_weights: &mut Vec<WeightAndOptimizerData<OptimizerSGD>>,
    input_bufreader: &mut dyn io::Read,
) -> Result<(), Box<dyn Error>> {
    const BUF_LEN: usize = 1024 * 1024;
    let mut in_weights: Vec<WeightAndOptimizerData<L>> = Vec::with_capacity(BUF_LEN as usize);
    let mut remaining_weights = weights_len;
    let mut out_idx: usize = 0;
    if weights_len != out_weights.len() {
        return Err(format!("read_weights_only_from_buf2 - number of weights to read ({}) and number of weights allocated ({}) isn't the same", weights_len, out_weights.len()))?;
    }

    unsafe {
        while remaining_weights > 0 {
            let chunk_size = min(remaining_weights, BUF_LEN);
            in_weights.set_len(chunk_size);
            let mut in_weights_view: &mut [u8] = slice::from_raw_parts_mut(
                in_weights.as_mut_ptr() as *mut u8,
                chunk_size * mem::size_of::<WeightAndOptimizerData<L>>(),
            );
            input_bufreader.read_exact(&mut in_weights_view)?;
            for w in &in_weights {
		println!("WEIGHT\t{:?}", w.weight);
                out_weights.get_unchecked_mut(out_idx).weight = w.weight;
                out_idx += 1;
            }
            remaining_weights -= chunk_size;
        }
    }
    Ok(())
}

#[inline(always)]
pub fn get_input_output_borrows(
    i: &mut Vec<f32>,
    start1: usize,
    len1: usize,
    start2: usize,
    len2: usize,
) -> (&mut [f32], &mut [f32]) {
    debug_assert!(
        (start1 >= start2 + len2) || (start2 >= start1 + len1),
        "start1: {}, len1: {}, start2: {}, len2 {}",
        start1,
        len1,
        start2,
        len2
    );
    unsafe {
        if start2 > start1 {
            let (rest, second) = i.split_at_mut(start2);
            let (rest, first) = rest.split_at_mut(start1);
            return (
                first.get_unchecked_mut(0..len1),
                second.get_unchecked_mut(0..len2),
            );
        } else {
            let (rest, first) = i.split_at_mut(start1);
            let (rest, second) = rest.split_at_mut(start2);
            return (
                first.get_unchecked_mut(0..len1),
                second.get_unchecked_mut(0..len2),
            );
        }
    }
}

pub fn slearn2<'a>(
    bg: &mut graph::BlockGraph,
    fb: &feature_buffer::FeatureBuffer,
    pb: &mut port_buffer::PortBuffer,
    update: bool,
) -> f32 {
    pb.reset();
    let (block_run, further_blocks) = bg.blocks_final.split_at_mut(1);
    block_run[0].forward_backward(further_blocks, fb, pb, update);
    let prediction_probability = pb.observations[0];
    return prediction_probability;
}

pub fn spredict2<'a>(
    bg: &mut graph::BlockGraph,
    fb: &feature_buffer::FeatureBuffer,
    pb: &mut port_buffer::PortBuffer,
    update: bool,
) -> f32 {
    pb.reset();
    let (block_run, further_blocks) = bg.blocks_final.split_at(1);
    block_run[0].forward(further_blocks, fb, pb);
    let prediction_probability = pb.observations[0];
    return prediction_probability;
}

#[inline(always)]
pub fn forward_backward(
    further_blocks: &mut [Box<dyn BlockTrait>],
    fb: &feature_buffer::FeatureBuffer,
    pb: &mut port_buffer::PortBuffer,
    update: bool,
) {
    match further_blocks.split_first_mut() {
        Some((next_regressor, further_blocks)) => {
            next_regressor.forward_backward(further_blocks, fb, pb, update)
        }
        None => {}
    }
}

#[inline(always)]
pub fn forward(
    further_blocks: &[Box<dyn BlockTrait>],
    fb: &feature_buffer::FeatureBuffer,
    pb: &mut port_buffer::PortBuffer,
) {
    match further_blocks.split_first() {
        Some((next_regressor, further_blocks)) => next_regressor.forward(further_blocks, fb, pb),
        None => {}
    }
}
