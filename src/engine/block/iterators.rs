use crate::engine::optimizer::OptimizerTrait;

use crate::feature_buffer;
use crate::port_buffer;
use crate::engine::regressor::{BlockCache, BlockTrait};


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
        return if start2 > start1 {
            let (rest, second) = i.split_at_mut(start2);
            let (_, first) = rest.split_at_mut(start1);
            (
                first.get_unchecked_mut(0..len1),
                second.get_unchecked_mut(0..len2),
            )
        } else {
            let (rest, first) = i.split_at_mut(start1);
            let (_, second) = rest.split_at_mut(start2);
            (
                first.get_unchecked_mut(0..len1),
                second.get_unchecked_mut(0..len2),
            )
        };
    }
}

#[inline(always)]
pub fn forward_backward(
    further_blocks: &mut [Box<dyn BlockTrait>],
    fb: &feature_buffer::FeatureBuffer,
    pb: &mut port_buffer::PortBuffer,
    update: bool,
) {
    if let Some((next_regressor, further_blocks)) = further_blocks.split_first_mut() {
        next_regressor.forward_backward(further_blocks, fb, pb, update)
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

#[inline(always)]
pub fn forward_with_cache(
    further_blocks: &[Box<dyn BlockTrait>],
    fb: &feature_buffer::FeatureBuffer,
    pb: &mut port_buffer::PortBuffer,
    caches: &[BlockCache],
) {
    if let Some((next_regressor, further_blocks)) = further_blocks.split_first() {
        next_regressor.forward_with_cache(further_blocks, fb, pb, caches)
    }
}

#[inline(always)]
pub fn prepare_forward_cache(
    further_blocks: &mut [Box<dyn BlockTrait>],
    fb: &feature_buffer::FeatureBuffer,
    caches: &mut [BlockCache],
) {
    if let Some((next_regressor, further_blocks)) = further_blocks.split_first_mut() {
        next_regressor.prepare_forward_cache(further_blocks, fb, caches)
    }
}

#[inline(always)]
pub fn create_forward_cache(
    further_blocks: &mut [Box<dyn BlockTrait>],
    caches: &mut Vec<BlockCache>,
) {
    if let Some((next_regressor, further_blocks)) = further_blocks.split_first_mut() {
        next_regressor.create_forward_cache(further_blocks, caches)
    }
}
