#[cfg(test)]
use crate::{namespace::feature_buffer, engine:: {graph, port_buffer, regressor::BlockCache}};

#[cfg(test)]
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

#[cfg(test)]
pub fn slearn2(
    bg: &mut graph::BlockGraph,
    fb: &feature_buffer::FeatureBuffer,
    pb: &mut port_buffer::PortBuffer,
    update: bool,
) -> f32 {
    pb.reset();
    let (block_run, further_blocks) = bg.blocks_final.split_at_mut(1);
    block_run[0].forward_backward(further_blocks, fb, pb, update);

    pb.observations[0]
}

#[cfg(test)]
pub fn ssetup_cache2(
    bg: &mut graph::BlockGraph,
    cache_fb: &feature_buffer::FeatureBuffer,
    caches: &mut Vec<BlockCache>,
) {
    let (create_block_run, create_further_blocks) = bg.blocks_final.split_at_mut(1);
    create_block_run[0].create_forward_cache(create_further_blocks, caches);

    let (prepare_block_run, prepare_further_blocks) = bg.blocks_final.split_at_mut(1);
    prepare_block_run[0].prepare_forward_cache(
        prepare_further_blocks,
        cache_fb,
        caches.as_mut_slice(),
    );
}


#[cfg(test)]
pub fn spredict2_with_cache(
    bg: &mut graph::BlockGraph,
    fb: &feature_buffer::FeatureBuffer,
    pb: &mut port_buffer::PortBuffer,
    caches: &[BlockCache],
) -> f32 {
    pb.reset();
    let (block_run, further_blocks) = bg.blocks_final.split_at(1);
    block_run[0].forward_with_cache(further_blocks, fb, pb, caches);

    pb.observations[0]
}

#[cfg(test)]
pub fn spredict2(
    bg: &mut graph::BlockGraph,
    fb: &feature_buffer::FeatureBuffer,
    pb: &mut port_buffer::PortBuffer,
) -> f32 {
    pb.reset();
    let (block_run, further_blocks) = bg.blocks_final.split_at(1);
    block_run[0].forward(further_blocks, fb, pb);
    pb.observations[0]
}