use std::error::Error;
use std::{io, mem, slice};
use std::cmp::min;
use std::io::{Read, Write};
use crate::engine::block::iterators::WeightAndOptimizerData;
use crate::engine::optimizer::{OptimizerSGD, OptimizerTrait};


pub fn read_weights_from_buf<L>(
    weights: &mut Vec<L>,
    input_bufreader: &mut dyn io::Read,
    _use_quantization: bool
) -> Result<(), Box<dyn Error>> {
    if weights.is_empty() {
        return Err("Loading weights to unallocated weighs buffer".to_string())?;
    }
    unsafe {
        let buf_view: &mut [u8] = slice::from_raw_parts_mut(
            weights.as_mut_ptr() as *mut u8,
            weights.len() * mem::size_of::<L>(),
        );
        input_bufreader.read_exact(buf_view)?;
    }
    Ok(())
}

pub fn skip_weights_from_buf<L>(
    weights_len: usize,
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
    _use_quantization: bool
) -> Result<(), Box<dyn Error>> {
    if weights.is_empty() {
        assert!(false);
        return Err("Writing weights of unallocated weights buffer".to_string())?;
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
    let mut in_weights: Vec<WeightAndOptimizerData<L>> = Vec::with_capacity(BUF_LEN);
    let mut remaining_weights = weights_len;
    let mut out_idx: usize = 0;
    if weights_len != out_weights.len() {
        return Err(format!("read_weights_only_from_buf2 - number of weights to read ({}) and number of weights allocated ({}) isn't the same", weights_len, out_weights.len()))?;
    }

    unsafe {
        while remaining_weights > 0 {
            let chunk_size = min(remaining_weights, BUF_LEN);
            in_weights.set_len(chunk_size);
            let in_weights_view: &mut [u8] = slice::from_raw_parts_mut(
                in_weights.as_mut_ptr() as *mut u8,
                chunk_size * mem::size_of::<WeightAndOptimizerData<L>>(),
            );
            input_bufreader.read_exact(in_weights_view)?;
            for w in &in_weights {
                out_weights.get_unchecked_mut(out_idx).weight = w.weight;
                out_idx += 1;
            }
            remaining_weights -= chunk_size;
        }
    }
    Ok(())
}