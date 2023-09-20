#![allow(dead_code,unused_imports, unused_mut, invalid_value)]

use core::arch::x86_64::*;
use rustc_hash::FxHashSet;
use std::any::Any;
use std::error::Error;
use std::mem::{self, MaybeUninit};
use std::ops::Bound::Included;
use std::sync::Mutex;
use std::time::Instant;
use std::{io, ptr};

use merand48::*;

use optimizer::OptimizerTrait;
use regressor::BlockTrait;

use crate::block_helpers;
use crate::block_helpers::OptimizerData;
use crate::feature_buffer;
use crate::feature_buffer::{FeatureBuffer, HashAndValueAndSeq};
use crate::graph;
use crate::model_instance;
use crate::optimizer;
use crate::port_buffer;
use crate::port_buffer::PortBuffer;
use crate::regressor;
use crate::regressor::{BlockCache, FFM_CONTRA_BUF_LEN};

const FFM_STACK_BUF_LEN: usize = 131072;
const FFM_CONTRA_CACHE_BUF_LEN: usize = 1024;
const STEP: usize = 4;
const ZEROES: [f32; STEP] = [0.0; STEP];

pub struct BlockFFM<L: OptimizerTrait> {
    pub optimizer_ffm: L,
    pub local_data_ffm_values: Vec<f32>,
    pub ffm_k: u32,
    pub ffm_weights_len: u32,
    pub ffm_num_fields: u32,
    pub field_embedding_len: u32,
    pub weights: Vec<f32>,
    pub optimizer: Vec<OptimizerData<L>>,
    pub output_offset: usize,
    mutex: Mutex<()>,
}

pub fn new_ffm_block(
    bg: &mut graph::BlockGraph,
    mi: &model_instance::ModelInstance,
) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let block = match mi.optimizer {
        model_instance::Optimizer::AdagradLUT => {
            new_ffm_block_without_weights::<optimizer::OptimizerAdagradLUT>(mi)
        }
        model_instance::Optimizer::AdagradFlex => {
            new_ffm_block_without_weights::<optimizer::OptimizerAdagradFlex>(mi)
        }
        model_instance::Optimizer::SGD => {
            new_ffm_block_without_weights::<optimizer::OptimizerSGD>(mi)
        }
    }
    .unwrap();
    let mut block_outputs = bg.add_node(block, vec![]).unwrap();
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

fn new_ffm_block_without_weights<L: OptimizerTrait + 'static>(
    mi: &model_instance::ModelInstance,
) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    let ffm_num_fields = mi.ffm_fields.len() as u32;
    let field_embedding_len = mi.ffm_k * ffm_num_fields as u32;

    let mut reg_ffm = BlockFFM::<L> {
        weights: Vec::new(),
        optimizer: Vec::new(),
        ffm_weights_len: 0,
        local_data_ffm_values: Vec::with_capacity(1024),
        ffm_k: mi.ffm_k,
        ffm_num_fields,
        field_embedding_len,
        optimizer_ffm: L::new(),
        output_offset: usize::MAX,
        mutex: Mutex::new(()),
    };

    if mi.ffm_k > 0 {
        reg_ffm.optimizer_ffm.init(
            mi.ffm_learning_rate,
            mi.ffm_power_t,
            mi.ffm_init_acc_gradient,
        );
        // At the end we add "spillover buffer", so we can do modulo only on the base address and add offset
        reg_ffm.ffm_weights_len =
            (1 << mi.ffm_bit_precision) + (mi.ffm_fields.len() as u32 * reg_ffm.ffm_k);
    }

    // Verify that forward pass will have enough stack for temporary buffer
    if reg_ffm.ffm_k as usize * mi.ffm_fields.len() * mi.ffm_fields.len() > FFM_CONTRA_BUF_LEN {
        return Err(format!("FFM_CONTRA_BUF_LEN is {}. It needs to be at least ffm_k * number_of_fields^2. number_of_fields: {}, ffm_k: {}, please recompile with larger constant",
                           FFM_CONTRA_BUF_LEN, mi.ffm_fields.len(), reg_ffm.ffm_k))?;
    }

    Ok(Box::new(reg_ffm))
}

#[inline(always)]
unsafe fn hadd_ps(r4: __m128) -> f32 {
    let r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    // Add 2 lower values into the final result
    let r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
    // Return the lowest lane of the result vector.
    // The intrinsic below compiles into noop, modern compilers return floats in the lowest lane of xmm0 register.
    _mm_cvtss_f32(r1)
}

impl<L: OptimizerTrait + 'static> BlockTrait for BlockFFM<L> {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) {
        debug_assert!(self.output_offset != usize::MAX);

        unsafe {
            macro_rules! core_macro {
                (
                $local_data_ffm_values:ident
                ) => {
                    // number of outputs
                    let num_outputs = (self.ffm_num_fields * self.ffm_num_fields) as usize;
                    let myslice = &mut pb.tape[self.output_offset .. (self.output_offset + num_outputs)];
                    myslice.fill(0.0);

                    let mut local_data_ffm_values = $local_data_ffm_values;

                    let ffm_weights = &mut self.weights;

                    let ffmk: u32 = self.ffm_k;
                    let ffmk_as_usize: usize = ffmk as usize;

                    let ffm_fields_count: u32 = self.ffm_num_fields;
                    let ffm_fields_count_as_usize: usize = ffm_fields_count as usize;

                    let fc: usize = ffm_fields_count_as_usize * ffmk_as_usize;

                    let mut contra_fields: [f32; FFM_CONTRA_BUF_LEN] = MaybeUninit::uninit().assume_init();

                    /* first prepare two things:
                       - transposed contra vectors in contra_fields -
                           - for each vector we sum up all the features within a field
                           - and at the same time transpose it, so we can later directly multiply them with individual feature embeddings
                       - cache of gradients in local_data_ffm_values
                           - we will use these gradients later in backward pass
                    */

                    _mm_prefetch(mem::transmute::<&f32, &i8>(&contra_fields.get_unchecked(fb.ffm_buffer.get_unchecked(0).contra_field_index as usize)), _MM_HINT_T0);
                    let mut ffm_buffer_index = 0;
                    for field_index in 0..ffm_fields_count {
                        let field_index_ffmk = field_index * ffmk;
                        // first we handle fields with no features
                        if ffm_buffer_index >= fb.ffm_buffer.len() ||
                            fb.ffm_buffer.get_unchecked(ffm_buffer_index).contra_field_index > field_index_ffmk
                        {
                            let mut offset: usize = field_index_ffmk as usize;
                            for _z in 0..ffm_fields_count_as_usize {
                                for k in offset..offset + ffmk_as_usize {
                                    *contra_fields.get_unchecked_mut(k) = 0.0;
                                }

                                offset += fc;
                            }
                            continue;
                        }

                        let mut is_first_feature = true;
                        while ffm_buffer_index < fb.ffm_buffer.len() && fb.ffm_buffer.get_unchecked(ffm_buffer_index).contra_field_index == field_index_ffmk {
                            _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(fb.ffm_buffer.get_unchecked(ffm_buffer_index + 1).hash as usize)), _MM_HINT_T0);

                            let feature = fb.ffm_buffer.get_unchecked(ffm_buffer_index);
                            let feature_value = feature.value as f32;

                            let mut feature_index = feature.hash as usize;
                            let mut offset: usize = field_index_ffmk as usize;

                            if is_first_feature {
                                for _z in 0..ffm_fields_count_as_usize {
                                    _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(feature_index + ffmk_as_usize)), _MM_HINT_T0);
                                    for k in 0..ffmk_as_usize {
                                        *contra_fields.get_unchecked_mut(offset + k) = ffm_weights.get_unchecked(feature_index + k) * feature_value;
                                    }

                                    offset += fc;
                                    feature_index += ffmk_as_usize;
                                }
                                is_first_feature = false;
                            } else {
                                for _z in 0..ffm_fields_count_as_usize {
                                    _mm_prefetch(mem::transmute::<&f32, &i8>(&ffm_weights.get_unchecked(feature_index + ffmk_as_usize)), _MM_HINT_T0);
                                    for k in 0..ffmk_as_usize {
                                        *contra_fields.get_unchecked_mut(offset + k) += ffm_weights.get_unchecked(feature_index + k) * feature_value;
                                    }

                                    offset += fc;
                                    feature_index += ffmk_as_usize;
                                }
                            }

                            ffm_buffer_index += 1;
                        }
                    }

                    let mut ffm_values_offset = 0;
                    for feature in &fb.ffm_buffer {
                        let feature_value = feature.value;
                        let feature_index = feature.hash as usize;
                        let feature_contra_field_index = feature.contra_field_index as usize;

                        let contra_offset = feature_contra_field_index * ffm_fields_count_as_usize;

                        let contra_offset2 = contra_offset / ffmk_as_usize;

                        let mut vv = 0;
                        for z in 0..ffm_fields_count_as_usize {
                            let mut correction = 0.0;

                            let vv_feature_index = feature_index + vv;
                            let vv_contra_offset = contra_offset + vv;

                            if vv == feature_contra_field_index {
                                for k in 0..ffmk_as_usize {
                                    let ffm_weight = ffm_weights.get_unchecked(vv_feature_index + k);
                                    let contra_weight = *contra_fields.get_unchecked(vv_contra_offset + k) - ffm_weight * feature_value;
                                    let gradient = feature_value * contra_weight;
                                    *local_data_ffm_values.get_unchecked_mut(ffm_values_offset + k) = gradient;

                                    correction += ffm_weight * gradient;
                                }
                            } else {
                                for k in 0..ffmk_as_usize {
                                    let contra_weight = *contra_fields.get_unchecked(vv_contra_offset + k);
                                    let gradient = feature_value * contra_weight;

                                    *local_data_ffm_values.get_unchecked_mut(ffm_values_offset + k) = gradient;

                                    let ffm_weight = ffm_weights.get_unchecked(vv_feature_index + k);
                                    correction += ffm_weight * gradient;
                                }
                            }

                            *myslice.get_unchecked_mut(contra_offset2 + z) += correction * 0.5;
                            vv += ffmk_as_usize;
                            ffm_values_offset += ffmk_as_usize;
                        }
                    }

                    block_helpers::forward_backward(further_blocks, fb, pb, update);

                    if update {
                        let mut local_index: usize = 0;
                        let myslice = &mut pb.tape[self.output_offset..(self.output_offset + num_outputs)];

                        for feature in &fb.ffm_buffer {
                            let mut feature_index = feature.hash as usize;
                            let contra_offset = (feature.contra_field_index * ffm_fields_count) as usize / ffmk_as_usize;

                            for z in 0..ffm_fields_count_as_usize {
                                let general_gradient = myslice.get_unchecked(contra_offset + z);

                                for _ in 0.. ffmk_as_usize {
                                    let feature_value = *local_data_ffm_values.get_unchecked(local_index);
                                    let gradient = general_gradient * feature_value;
                                    let update = self.optimizer_ffm.calculate_update(gradient,
                                        &mut self.optimizer.get_unchecked_mut(feature_index).optimizer_data);

                                    *ffm_weights.get_unchecked_mut(feature_index) -= update;
                                    local_index += 1;
                                    feature_index += 1;
                                }
                            }
                        }
                    }
                    // The only exit point
                    return
                }
            } // End of macro

            let local_data_ffm_len =
                fb.ffm_buffer.len() * (self.ffm_k * self.ffm_num_fields) as usize;
            if local_data_ffm_len < FFM_STACK_BUF_LEN {
                // Fast-path - using on-stack data structures
                let local_data_ffm_values: [f32; FFM_STACK_BUF_LEN as usize] =
                    MaybeUninit::uninit().assume_init();
                core_macro!(local_data_ffm_values);
            } else {
                // Slow-path - using heap data structures
                log::warn!("FFM data too large, allocating on the heap (slow path)!");
                let _guard = self.mutex.lock().unwrap(); // following operations are not thread safe
                if local_data_ffm_len > self.local_data_ffm_values.len() {
                    self.local_data_ffm_values
                        .reserve(local_data_ffm_len - self.local_data_ffm_values.len() + 1024);
                }
                let local_data_ffm_values = &mut self.local_data_ffm_values;

                core_macro!(local_data_ffm_values);
            }
        }
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        debug_assert!(self.output_offset != usize::MAX);

        let num_outputs = (self.ffm_num_fields * self.ffm_num_fields) as usize;
        let myslice = &mut pb.tape[self.output_offset..(self.output_offset + num_outputs)];
        myslice.fill(0.0);

        unsafe {
            let ffm_weights = &self.weights;
            _mm_prefetch(
                mem::transmute::<&f32, &i8>(
                    &ffm_weights.get_unchecked(fb.ffm_buffer.get_unchecked(0).hash as usize),
                ),
                _MM_HINT_T0,
            );

            /* We first prepare "contra_fields" or collapsed field embeddings, where we sum all individual feature embeddings
              We need to be careful to:
              - handle fields with zero features present
              - handle values on diagonal - we want to be able to exclude self-interactions later (we pre-substract from wsum)
              - optimize for just copying the embedding over when looking at first feature of the field, and add embeddings for the rest
              - optimize for very common case of value of the feature being 1.0 - avoid multiplications
            */

            let ffmk: u32 = self.ffm_k;
            let ffmk_as_usize: usize = ffmk as usize;

            let ffm_fields_count: u32 = self.ffm_num_fields;
            let ffm_fields_count_as_usize: usize = ffm_fields_count as usize;
            let ffm_fields_count_plus_one = ffm_fields_count + 1;

            let field_embedding_len_as_usize = self.field_embedding_len as usize;
            let field_embedding_len_end =
                field_embedding_len_as_usize - field_embedding_len_as_usize % STEP;

            let mut contra_fields: [f32; FFM_CONTRA_BUF_LEN] = MaybeUninit::uninit().assume_init();

            let mut ffm_buffer_index = 0;

            for field_index in 0..ffm_fields_count {
                let field_index_ffmk = field_index * ffmk;
                let field_index_ffmk_as_usize = field_index_ffmk as usize;
                let offset = (field_index_ffmk * ffm_fields_count) as usize;
                // first we handle fields with no features
                if ffm_buffer_index >= fb.ffm_buffer.len()
                    || fb
                        .ffm_buffer
                        .get_unchecked(ffm_buffer_index)
                        .contra_field_index
                        > field_index_ffmk
                {
                    // first feature of the field - just overwrite
                    for z in (offset..offset + field_embedding_len_end).step_by(STEP) {
                        contra_fields
                            .get_unchecked_mut(z..z + STEP)
                            .copy_from_slice(&ZEROES);
                    }

                    for z in offset + field_embedding_len_end..offset + field_embedding_len_as_usize
                    {
                        *contra_fields.get_unchecked_mut(z) = 0.0;
                    }

                    continue;
                }

                let ffm_index = (field_index * ffm_fields_count_plus_one) as usize;

                let mut is_first_feature = true;
                while ffm_buffer_index < fb.ffm_buffer.len()
                    && fb
                        .ffm_buffer
                        .get_unchecked(ffm_buffer_index)
                        .contra_field_index
                        == field_index_ffmk
                {
                    _mm_prefetch(
                        mem::transmute::<&f32, &i8>(ffm_weights.get_unchecked(
                            fb.ffm_buffer.get_unchecked(ffm_buffer_index + 1).hash as usize,
                        )),
                        _MM_HINT_T0,
                    );
                    let feature = fb.ffm_buffer.get_unchecked(ffm_buffer_index);
                    let feature_index = feature.hash as usize;
                    let feature_value = feature.value;

                    self.prepare_contra_fields(
                        feature,
                        contra_fields.as_mut_slice(),
                        ffm_weights,
                        offset,
                        field_embedding_len_as_usize,
                        &mut is_first_feature,
                    );

                    let feature_field_index = feature_index + field_index_ffmk_as_usize;

                    let mut correction = 0.0;
                    for k in feature_field_index..feature_field_index + ffmk_as_usize {
                        correction += ffm_weights.get_unchecked(k) * ffm_weights.get_unchecked(k);
                    }

                    *myslice.get_unchecked_mut(ffm_index) -=
                        correction * 0.5 * feature_value * feature_value;

                    ffm_buffer_index += 1;
                }
            }

            self.calculate_interactions(
                myslice,
                contra_fields.as_slice(),
                ffmk_as_usize,
                ffm_fields_count_as_usize,
                field_embedding_len_as_usize,
            );
        }

        block_helpers::forward(further_blocks, fb, pb);
    }

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &FeatureBuffer,
        pb: &mut PortBuffer,
        caches: &[BlockCache],
    ) {
        debug_assert!(self.output_offset != usize::MAX);

        let Some((next_cache, further_caches)) = caches.split_first() else {
            log::warn!("Expected caches, but non available, executing forward pass without cache");
            self.forward(further_blocks, fb, pb);
            return;
        };

        let BlockCache::FFM {
            contra_fields,
            features_present,
            ffm,
        } = next_cache else {
            log::warn!("Unable to downcast cache to BlockFFMCache, executing forward pass without cache");
            self.forward(further_blocks, fb, pb);
            return;
        };

        unsafe {
            let num_outputs = (self.ffm_num_fields * self.ffm_num_fields) as usize;
            let ffm_slice = &mut pb.tape[self.output_offset..(self.output_offset + num_outputs)];
            ptr::copy_nonoverlapping(ffm.as_ptr(), ffm_slice.as_mut_ptr(), num_outputs);

            let cached_contra_fields = contra_fields;

            let ffm_weights = &self.weights;
            _mm_prefetch(
                mem::transmute::<&f32, &i8>(
                    ffm_weights.get_unchecked(fb.ffm_buffer.get_unchecked(0).hash as usize),
                ),
                _MM_HINT_T0,
            );

            /* We first prepare "contra_fields" or collapsed field embeddings, where we sum all individual feature embeddings
              We need to be careful to:
              - handle fields with zero features present
              - handle values on diagonal - we want to be able to exclude self-interactions later (we pre-substract from wsum)
              - optimize for just copying the embedding over when looking at first feature of the field, and add embeddings for the rest
              - optimize for very common case of value of the feature being 1.0 - avoid multiplications
            */

            let ffmk: u32 = self.ffm_k;
            let ffmk_as_usize: usize = ffmk as usize;

            let ffm_fields_count: u32 = self.ffm_num_fields;
            let ffm_fields_count_as_usize: usize = ffm_fields_count as usize;
            let ffm_fields_count_plus_one = ffm_fields_count + 1;

            let field_embedding_len_as_usize = self.field_embedding_len as usize;
            let field_embedding_len_end =
                field_embedding_len_as_usize - field_embedding_len_as_usize % STEP;

            let mut contra_fields: [f32; FFM_CONTRA_BUF_LEN] = MaybeUninit::uninit().assume_init();

            let mut ffm_buffer_index = 0;

            for field_index in 0..ffm_fields_count {
                let field_index_ffmk = field_index * ffmk;
                let field_index_ffmk_as_usize = field_index_ffmk as usize;
                let offset = field_index_ffmk_as_usize * ffm_fields_count_as_usize;
                // first we handle fields with no features
                if ffm_buffer_index >= fb.ffm_buffer.len()
                    || fb
                        .ffm_buffer
                        .get_unchecked(ffm_buffer_index)
                        .contra_field_index
                        > field_index_ffmk
                {
                    // first feature of the field - just overwrite
                    for z in (offset..offset + field_embedding_len_end).step_by(STEP) {
                        contra_fields
                            .get_unchecked_mut(z..z + STEP)
                            .copy_from_slice(&ZEROES);
                    }

                    for z in offset + field_embedding_len_end..offset + field_embedding_len_as_usize
                    {
                        *contra_fields.get_unchecked_mut(z) = 0.0;
                    }

                    continue;
                }

                let ffm_index = (field_index * ffm_fields_count_plus_one) as usize;

                let mut contra_fields_copied = false;
                let mut is_first_feature = true;
                while ffm_buffer_index < fb.ffm_buffer.len()
                    && fb
                        .ffm_buffer
                        .get_unchecked(ffm_buffer_index)
                        .contra_field_index
                        == field_index_ffmk
                {
                    _mm_prefetch(
                        mem::transmute::<&f32, &i8>(ffm_weights.get_unchecked(
                            fb.ffm_buffer.get_unchecked(ffm_buffer_index + 1).hash as usize,
                        )),
                        _MM_HINT_T0,
                    );
                    let feature = fb.ffm_buffer.get_unchecked(ffm_buffer_index);

                    let ffm_feature = feature.into();
                    if features_present.contains(&ffm_feature) {
                        if is_first_feature {
                            is_first_feature = false;
                            contra_fields_copied = true;
                            // Copy only once, skip other copying as the data for all features of that contra_index is already calculated
                            ptr::copy_nonoverlapping(
                                cached_contra_fields.as_ptr().add(offset),
                                contra_fields.as_mut_ptr().add(offset),
                                field_embedding_len_as_usize,
                            );
                        } else if !contra_fields_copied {
                            contra_fields_copied = true;

                            const LANES: usize = STEP * 4;
                            let field_embedding_len_end = field_embedding_len_as_usize
                                - (field_embedding_len_as_usize % LANES);

                            let mut contra_fields_ptr = contra_fields.as_mut_ptr().add(offset);
                            let mut cached_contra_fields_ptr =
                                cached_contra_fields.as_ptr().add(offset);

                            for _ in (0..field_embedding_len_end).step_by(LANES) {
                                add_cached_contra_field(
                                    contra_fields_ptr,
                                    cached_contra_fields_ptr,
                                );
                                contra_fields_ptr = contra_fields_ptr.add(STEP);
                                cached_contra_fields_ptr = cached_contra_fields_ptr.add(STEP);

                                add_cached_contra_field(
                                    contra_fields_ptr,
                                    cached_contra_fields_ptr,
                                );
                                contra_fields_ptr = contra_fields_ptr.add(STEP);
                                cached_contra_fields_ptr = cached_contra_fields_ptr.add(STEP);

                                add_cached_contra_field(
                                    contra_fields_ptr,
                                    cached_contra_fields_ptr,
                                );
                                contra_fields_ptr = contra_fields_ptr.add(STEP);
                                cached_contra_fields_ptr = cached_contra_fields_ptr.add(STEP);

                                add_cached_contra_field(
                                    contra_fields_ptr,
                                    cached_contra_fields_ptr,
                                );
                                contra_fields_ptr = contra_fields_ptr.add(STEP);
                                cached_contra_fields_ptr = cached_contra_fields_ptr.add(STEP);
                            }

                            for z in field_embedding_len_end..field_embedding_len_as_usize {
                                *contra_fields.get_unchecked_mut(offset + z) +=
                                    cached_contra_fields.get_unchecked(offset + z);
                            }
                        }
                    } else {
                        let feature_index = feature.hash as usize;
                        let feature_value = feature.value;

                        self.prepare_contra_fields(
                            feature,
                            contra_fields.as_mut_slice(),
                            ffm_weights,
                            offset,
                            field_embedding_len_as_usize,
                            &mut is_first_feature,
                        );

                        let feature_field_index = feature_index + field_index_ffmk_as_usize;

                        let mut correction = 0.0;
                        for k in feature_field_index..feature_field_index + ffmk_as_usize {
                            correction +=
                                ffm_weights.get_unchecked(k) * ffm_weights.get_unchecked(k);
                        }

                        *ffm_slice.get_unchecked_mut(ffm_index) -=
                            correction * 0.5 * feature_value * feature_value;
                    }
                    ffm_buffer_index += 1;
                }
            }

            self.calculate_interactions(
                ffm_slice,
                contra_fields.as_slice(),
                ffmk_as_usize,
                ffm_fields_count_as_usize,
                field_embedding_len_as_usize,
            );
        }
        block_helpers::forward_with_cache(further_blocks, fb, pb, further_caches);
    }

    fn create_forward_cache(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        caches: &mut Vec<BlockCache>,
    ) {
        unsafe {
            caches.push(BlockCache::FFM {
                contra_fields: MaybeUninit::uninit().assume_init(),
                features_present: FxHashSet::default(),
                ffm: vec![0.0; (self.ffm_num_fields * self.ffm_num_fields) as usize],
            });
        }

        block_helpers::create_forward_cache(further_blocks, caches);
    }

    fn prepare_forward_cache(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        caches: &mut [BlockCache],
    ) {
        let Some((next_cache, further_caches)) = caches.split_first_mut() else {
            log::warn!("Expected BlockFFMCache caches, but non available, skipping cache preparation");
            return;
        };

        let BlockCache::FFM {
            contra_fields,
            features_present,
            ffm,
        } = next_cache else {
            log::warn!("Unable to downcast cache to BlockFFMCache, skipping cache preparation");
            return;
        };

        unsafe {
            let ffm_slice = ffm.as_mut_slice();
            ffm_slice.fill(0.0);

            features_present.clear();

            let ffm_weights = &self.weights;
            _mm_prefetch(
                mem::transmute::<&f32, &i8>(
                    ffm_weights.get_unchecked(fb.ffm_buffer.get_unchecked(0).hash as usize),
                ),
                _MM_HINT_T0,
            );

            /* We first prepare "contra_fields" or collapsed field embeddings, where we sum all individual feature embeddings
              We need to be careful to:
              - handle fields with zero features present
              - handle values on diagonal - we want to be able to exclude self-interactions later (we pre-substract from wsum)
              - optimize for just copying the embedding over when looking at first feature of the field, and add embeddings for the rest
              - optimize for very common case of value of the feature being 1.0 - avoid multiplications
            */

            let ffmk: u32 = self.ffm_k;
            let ffmk_as_usize: usize = ffmk as usize;

            let ffm_fields_count: u32 = self.ffm_num_fields;
            let ffm_fields_count_plus_one = ffm_fields_count + 1;

            let field_embedding_len_as_usize = self.field_embedding_len as usize;

            let mut ffm_buffer_index = 0;

            for field_index in 0..ffm_fields_count {
                let field_index_ffmk = field_index * ffmk;
                let field_index_ffmk_as_usize = field_index_ffmk as usize;
                let offset = (field_index_ffmk * ffm_fields_count) as usize;
                // first we handle fields with no features
                if ffm_buffer_index >= fb.ffm_buffer.len()
                    || fb
                        .ffm_buffer
                        .get_unchecked(ffm_buffer_index)
                        .contra_field_index
                        > field_index_ffmk
                {
                    continue;
                }

                let ffm_index = (field_index * ffm_fields_count_plus_one) as usize;

                let mut is_first_feature = true;
                while ffm_buffer_index < fb.ffm_buffer.len()
                    && fb
                        .ffm_buffer
                        .get_unchecked(ffm_buffer_index)
                        .contra_field_index
                        == field_index_ffmk
                {
                    _mm_prefetch(
                        mem::transmute::<&f32, &i8>(ffm_weights.get_unchecked(
                            fb.ffm_buffer.get_unchecked(ffm_buffer_index + 1).hash as usize,
                        )),
                        _MM_HINT_T0,
                    );
                    let feature = fb.ffm_buffer.get_unchecked(ffm_buffer_index);
                    features_present.insert(feature.into());
                    let feature_index = feature.hash as usize;
                    let feature_value = feature.value;

                    self.prepare_contra_fields(
                        feature,
                        contra_fields.as_mut_slice(),
                        ffm_weights,
                        offset,
                        field_embedding_len_as_usize,
                        &mut is_first_feature,
                    );

                    let feature_field_index = feature_index + field_index_ffmk_as_usize;

                    let mut correction = 0.0;
                    for k in feature_field_index..feature_field_index + ffmk_as_usize {
                        correction += ffm_weights.get_unchecked(k) * ffm_weights.get_unchecked(k);
                    }

                    *ffm_slice.get_unchecked_mut(ffm_index) -=
                        correction * 0.5 * feature_value * feature_value;

                    ffm_buffer_index += 1;
                }
            }
        }

        block_helpers::prepare_forward_cache(further_blocks, fb, further_caches);
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights = vec![0.0; self.ffm_weights_len as usize];
        self.optimizer = vec![
            OptimizerData::<L> {
                optimizer_data: self.optimizer_ffm.initial_data(),
            };
            self.ffm_weights_len as usize
        ];

        match mi.ffm_initialization_type.as_str() {
            "default" => {
                if mi.ffm_k > 0 {
                    if mi.ffm_init_width == 0.0 {
                        // Initialization that has showed to work ok for us, like in ffm.pdf, but centered around zero and further divided by 50
                        let ffm_one_over_k_root = 1.0 / (self.ffm_k as f32).sqrt() / 50.0;
                        for i in 0..self.ffm_weights_len {
                            self.weights[i as usize] = (1.0
                                * merand48((self.ffm_weights_len as usize + i as usize) as u64)
                                - 0.5)
                                * ffm_one_over_k_root;
                            self.optimizer[i as usize].optimizer_data =
                                self.optimizer_ffm.initial_data();
                        }
                    } else {
                        let zero_half_band_width = mi.ffm_init_width * mi.ffm_init_zero_band * 0.5;
                        let band_width = mi.ffm_init_width * (1.0 - mi.ffm_init_zero_band);
                        for i in 0..self.ffm_weights_len {
                            let mut w = merand48(i as u64) * band_width - band_width * 0.5;
                            if w > 0.0 {
                                w += zero_half_band_width;
                            } else {
                                w -= zero_half_band_width;
                            }
                            w += mi.ffm_init_center;
                            self.weights[i as usize] = w;
                            self.optimizer[i as usize].optimizer_data =
                                self.optimizer_ffm.initial_data();
                        }
                    }
                }
            }
            _ => {
                panic!("Please select a valid activation function.")
            }
        }
    }

    fn get_serialized_len(&self) -> usize {
        self.ffm_weights_len as usize
    }

    fn write_weights_to_buf(
        &self,
        output_bufwriter: &mut dyn io::Write,
    ) -> Result<(), Box<dyn Error>> {
        block_helpers::write_weights_to_buf(&self.weights, output_bufwriter)?;
        block_helpers::write_weights_to_buf(&self.optimizer, output_bufwriter)?;
        Ok(())
    }

    fn read_weights_from_buf(
        &mut self,
        input_bufreader: &mut dyn io::Read,
    ) -> Result<(), Box<dyn Error>> {
        block_helpers::read_weights_from_buf(&mut self.weights, input_bufreader)?;
        block_helpers::read_weights_from_buf(&mut self.optimizer, input_bufreader)?;
        Ok(())
    }

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert_eq!(output.get_output_index(), 0);
        (self.ffm_num_fields * self.ffm_num_fields) as usize
    }

    fn set_input_offset(&mut self, _input: graph::InputSlot, _offset: usize) {
        panic!("You cannot set_input_offset() for BlockFFM");
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert_eq!(output.get_output_index(), 0);
        self.output_offset = offset;
    }

    fn read_weights_from_buf_into_forward_only(
        &self,
        input_bufreader: &mut dyn io::Read,
        forward: &mut Box<dyn BlockTrait>,
    ) -> Result<(), Box<dyn Error>> {
        let forward = forward
            .as_any()
            .downcast_mut::<BlockFFM<optimizer::OptimizerSGD>>()
            .unwrap();
        block_helpers::read_weights_from_buf(&mut forward.weights, input_bufreader)?;
        block_helpers::skip_weights_from_buf::<OptimizerData<L>>(
            self.ffm_weights_len as usize,
            input_bufreader,
        )?;
        Ok(())
    }
}

#[inline(always)]
unsafe fn add_cached_contra_field(
    contra_fields_ptr: *mut f32,
    cached_contra_fields_ptr: *const f32,
) {
    let contra_fields = _mm_loadu_ps(contra_fields_ptr);
    let cached_contra_fields = _mm_loadu_ps(cached_contra_fields_ptr);
    _mm_storeu_ps(
        contra_fields_ptr,
        _mm_add_ps(cached_contra_fields, contra_fields),
    );
}

#[inline(always)]
unsafe fn prepare_first_contra_field(
    contra_fields_ptr: *mut f32,
    ffm_weights_ptr: *const f32,
    feature_value: __m128,
) {
    let acc = _mm_mul_ps(_mm_loadu_ps(ffm_weights_ptr), feature_value);
    _mm_storeu_ps(contra_fields_ptr, acc);
}

#[inline(always)]
unsafe fn prepare_contra_field_without_feature_value(
    contra_fields_ptr: *mut f32,
    ffm_weights_ptr: *const f32,
) {
    let contra_fields = _mm_loadu_ps(contra_fields_ptr);
    let ffm_weights = _mm_loadu_ps(ffm_weights_ptr);
    _mm_storeu_ps(contra_fields_ptr, _mm_add_ps(ffm_weights, contra_fields));
}

#[inline(always)]
unsafe fn prepare_contra_field_with_feature_value(
    contra_fields_ptr: *mut f32,
    ffm_weights_ptr: *const f32,
    feature_value: __m128,
) {
    let contra_fields = _mm_loadu_ps(contra_fields_ptr);
    let ffm_weights = _mm_loadu_ps(ffm_weights_ptr);
    let acc = _mm_fmadd_ps(ffm_weights, feature_value, contra_fields);
    _mm_storeu_ps(contra_fields_ptr, acc);
}

impl<L: OptimizerTrait + 'static> BlockFFM<L> {
    #[inline(always)]
    unsafe fn prepare_contra_fields(
        &self,
        feature: &HashAndValueAndSeq,
        contra_fields: &mut [f32],
        ffm_weights: &[f32],
        offset: usize,
        field_embedding_len: usize,
        is_first_feature: &mut bool,
    ) {
        let feature_index = feature.hash as usize;
        let feature_value = feature.value;
        const LANES: usize = STEP * 4;
        if *is_first_feature {
            *is_first_feature = false;
            if feature_value == 1.0 {
                ptr::copy_nonoverlapping(
                    ffm_weights.as_ptr().add(feature_index),
                    contra_fields.as_mut_ptr().add(offset),
                    field_embedding_len,
                );
            } else {
                let feature_value_mm_128 = _mm_set1_ps(feature_value);

                let field_embedding_len_end = field_embedding_len - (field_embedding_len % LANES);

                let mut contra_fields_ptr = contra_fields.as_mut_ptr().add(offset);
                let mut ffm_weights_ptr = ffm_weights.as_ptr().add(feature_index);
                for _ in (0..field_embedding_len_end).step_by(LANES) {
                    prepare_first_contra_field(
                        contra_fields_ptr,
                        ffm_weights_ptr,
                        feature_value_mm_128,
                    );
                    contra_fields_ptr = contra_fields_ptr.add(STEP);
                    ffm_weights_ptr = ffm_weights_ptr.add(STEP);

                    prepare_first_contra_field(
                        contra_fields_ptr,
                        ffm_weights_ptr,
                        feature_value_mm_128,
                    );
                    contra_fields_ptr = contra_fields_ptr.add(STEP);
                    ffm_weights_ptr = ffm_weights_ptr.add(STEP);

                    prepare_first_contra_field(
                        contra_fields_ptr,
                        ffm_weights_ptr,
                        feature_value_mm_128,
                    );
                    contra_fields_ptr = contra_fields_ptr.add(STEP);
                    ffm_weights_ptr = ffm_weights_ptr.add(STEP);

                    prepare_first_contra_field(
                        contra_fields_ptr,
                        ffm_weights_ptr,
                        feature_value_mm_128,
                    );
                    contra_fields_ptr = contra_fields_ptr.add(STEP);
                    ffm_weights_ptr = ffm_weights_ptr.add(STEP);
                }

                for z in field_embedding_len_end..field_embedding_len {
                    *contra_fields.get_unchecked_mut(offset + z) =
                        ffm_weights.get_unchecked(feature_index + z) * feature_value;
                }
            }
        } else if feature_value == 1.0 {
            let field_embedding_len_end = field_embedding_len - (field_embedding_len % LANES);

            let mut contra_fields_ptr = contra_fields.as_mut_ptr().add(offset);
            let mut ffm_weights_ptr = ffm_weights.as_ptr().add(feature_index);

            for _ in (0..field_embedding_len_end).step_by(LANES) {
                prepare_contra_field_without_feature_value(contra_fields_ptr, ffm_weights_ptr);
                contra_fields_ptr = contra_fields_ptr.add(STEP);
                ffm_weights_ptr = ffm_weights_ptr.add(STEP);

                prepare_contra_field_without_feature_value(contra_fields_ptr, ffm_weights_ptr);
                contra_fields_ptr = contra_fields_ptr.add(STEP);
                ffm_weights_ptr = ffm_weights_ptr.add(STEP);

                prepare_contra_field_without_feature_value(contra_fields_ptr, ffm_weights_ptr);
                contra_fields_ptr = contra_fields_ptr.add(STEP);
                ffm_weights_ptr = ffm_weights_ptr.add(STEP);

                prepare_contra_field_without_feature_value(contra_fields_ptr, ffm_weights_ptr);
                contra_fields_ptr = contra_fields_ptr.add(STEP);
                ffm_weights_ptr = ffm_weights_ptr.add(STEP);
            }

            for z in field_embedding_len_end..field_embedding_len {
                *contra_fields.get_unchecked_mut(offset + z) +=
                    *ffm_weights.get_unchecked(feature_index + z);
            }
        } else {
            let feature_value_mm_128 = _mm_set1_ps(feature_value);

            let field_embedding_len_end = field_embedding_len - (field_embedding_len % LANES);

            let mut contra_fields_ptr = contra_fields.as_mut_ptr().add(offset);
            let mut ffm_weights_ptr = ffm_weights.as_ptr().add(feature_index);
            for _ in (0..field_embedding_len_end).step_by(LANES) {
                prepare_contra_field_with_feature_value(
                    contra_fields_ptr,
                    ffm_weights_ptr,
                    feature_value_mm_128,
                );
                contra_fields_ptr = contra_fields_ptr.add(STEP);
                ffm_weights_ptr = ffm_weights_ptr.add(STEP);

                prepare_contra_field_with_feature_value(
                    contra_fields_ptr,
                    ffm_weights_ptr,
                    feature_value_mm_128,
                );
                contra_fields_ptr = contra_fields_ptr.add(STEP);
                ffm_weights_ptr = ffm_weights_ptr.add(STEP);

                prepare_contra_field_with_feature_value(
                    contra_fields_ptr,
                    ffm_weights_ptr,
                    feature_value_mm_128,
                );
                contra_fields_ptr = contra_fields_ptr.add(STEP);
                ffm_weights_ptr = ffm_weights_ptr.add(STEP);

                prepare_contra_field_with_feature_value(
                    contra_fields_ptr,
                    ffm_weights_ptr,
                    feature_value_mm_128,
                );
                contra_fields_ptr = contra_fields_ptr.add(STEP);
                ffm_weights_ptr = ffm_weights_ptr.add(STEP);
            }

            for z in field_embedding_len_end..field_embedding_len {
                *contra_fields.get_unchecked_mut(offset + z) +=
                    ffm_weights.get_unchecked(feature_index + z) * feature_value;
            }
        }
    }

    #[inline(always)]
    unsafe fn calculate_interactions(
        &self,
        ffm_slice: &mut [f32],
        contra_fields: &[f32],
        ffmk_as_usize: usize,
        ffm_fields_count_as_usize: usize,
        field_embedding_len_as_usize: usize,
    ) {
        const LANES: usize = STEP * 2;

        let ffmk_end_as_usize = ffmk_as_usize - ffmk_as_usize % LANES;

        for f1 in 0..ffm_fields_count_as_usize {
            let f1_offset = f1 * field_embedding_len_as_usize;
            let f1_ffmk = f1 * ffmk_as_usize;

            let mut f1_offset_ffmk = f1_offset + f1_ffmk;
            // This is self-interaction
            let mut contra_field = 0.0;
            let mut contra_fields_ptr = contra_fields.as_ptr().add(f1_offset_ffmk);
            if ffmk_as_usize == LANES {
                let contra_field_0 = _mm_loadu_ps(contra_fields_ptr);
                let contra_field_1 = _mm_loadu_ps(contra_fields_ptr.add(STEP));

                let acc_0 = _mm_mul_ps(contra_field_0, contra_field_0);
                let acc_1 = _mm_mul_ps(contra_field_1, contra_field_1);

                contra_field = hadd_ps(_mm_add_ps(acc_0, acc_1));
            } else {
                for _ in (0..ffmk_end_as_usize).step_by(LANES) {
                    let contra_field_0 = _mm_loadu_ps(contra_fields_ptr);
                    contra_fields_ptr = contra_fields_ptr.add(STEP);
                    let contra_field_1 = _mm_loadu_ps(contra_fields_ptr);
                    contra_fields_ptr = contra_fields_ptr.add(STEP);

                    let acc_0 = _mm_mul_ps(contra_field_0, contra_field_0);
                    let acc_1 = _mm_mul_ps(contra_field_1, contra_field_1);

                    contra_field += hadd_ps(_mm_add_ps(acc_0, acc_1));
                }

                for k in ffmk_end_as_usize..ffmk_as_usize {
                    contra_field += contra_fields.get_unchecked(f1_offset_ffmk + k)
                        * contra_fields.get_unchecked(f1_offset_ffmk + k);
                }
            }
            *ffm_slice.get_unchecked_mut(f1 * ffm_fields_count_as_usize + f1) += contra_field * 0.5;

            let mut f2_offset_ffmk = f1_offset + f1_ffmk;
            for f2 in f1 + 1..ffm_fields_count_as_usize {
                f2_offset_ffmk += field_embedding_len_as_usize;
                f1_offset_ffmk += ffmk_as_usize;

                let mut contra_field = 0.0;
                let mut contra_fields_ptr_1 = contra_fields.as_ptr().add(f1_offset_ffmk);
                let mut contra_fields_ptr_2 = contra_fields.as_ptr().add(f2_offset_ffmk);
                if ffmk_as_usize == LANES {
                    let contra_field_0 = _mm_loadu_ps(contra_fields_ptr_1);
                    let contra_field_1 = _mm_loadu_ps(contra_fields_ptr_2);
                    let acc_0 = _mm_mul_ps(contra_field_0, contra_field_1);

                    let contra_field_2 = _mm_loadu_ps(contra_fields_ptr_1.add(STEP));
                    let contra_field_3 = _mm_loadu_ps(contra_fields_ptr_2.add(STEP));
                    let acc_1 = _mm_mul_ps(contra_field_2, contra_field_3);

                    contra_field = hadd_ps(_mm_add_ps(acc_0, acc_1));
                } else {
                    for _ in (0..ffmk_end_as_usize).step_by(LANES) {
                        let contra_field_0 = _mm_loadu_ps(contra_fields_ptr_1);
                        let contra_field_1 = _mm_loadu_ps(contra_fields_ptr_2);
                        let acc_0 = _mm_mul_ps(contra_field_0, contra_field_1);
                        contra_fields_ptr_1 = contra_fields_ptr_1.add(STEP);
                        contra_fields_ptr_2 = contra_fields_ptr_2.add(STEP);

                        let contra_field_2 = _mm_loadu_ps(contra_fields_ptr_1);
                        let contra_field_3 = _mm_loadu_ps(contra_fields_ptr_2);
                        let acc_1 = _mm_mul_ps(contra_field_2, contra_field_3);
                        contra_fields_ptr_1 = contra_fields_ptr_1.add(STEP);
                        contra_fields_ptr_2 = contra_fields_ptr_2.add(STEP);

                        contra_field += hadd_ps(_mm_add_ps(acc_0, acc_1));
                    }

                    for k in ffmk_end_as_usize..ffmk_as_usize {
                        contra_field += contra_fields.get_unchecked(f1_offset_ffmk + k)
                            * contra_fields.get_unchecked(f2_offset_ffmk + k);
                    }
                }
                contra_field *= 0.5;

                *ffm_slice.get_unchecked_mut(f1 * ffm_fields_count_as_usize + f2) += contra_field;
                *ffm_slice.get_unchecked_mut(f2 * ffm_fields_count_as_usize + f1) += contra_field;
            }
        }
    }
}

mod tests {
    use block_helpers::{slearn2, spredict2, spredict2_with_cache};

    use crate::assert_epsilon;
    use crate::block_helpers::ssetup_cache2;
    use crate::block_loss_functions;
    use crate::feature_buffer;
    use crate::feature_buffer::HashAndValueAndSeq;
    use crate::graph::BlockGraph;
    use crate::model_instance::Optimizer;

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    fn ffm_vec(v: Vec<feature_buffer::HashAndValueAndSeq>) -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer {
            label: 0.0,
            example_importance: 1.0,
            example_number: 0,
            lr_buffer: Vec::new(),
            ffm_buffer: v,
        }
    }

    fn ffm_init<T: OptimizerTrait + 'static>(block_ffm: &mut Box<dyn BlockTrait>) {
        let block_ffm = block_ffm.as_any().downcast_mut::<BlockFFM<T>>().unwrap();

        for i in 0..block_ffm.weights.len() {
            block_ffm.weights[i] = 1.0;
            block_ffm.optimizer[i].optimizer_data = block_ffm.optimizer_ffm.initial_data();
        }
    }

    #[test]
    fn test_ffm_k1() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]]; // This isn't really used
        mi.optimizer = Optimizer::AdagradLUT;

        // Nothing can be learned from a single field in FFMs
        let mut bg = BlockGraph::new();
        let ffm_block = new_ffm_block(&mut bg, &mi).unwrap();
        let _loss_block = block_loss_functions::new_logloss_block(&mut bg, ffm_block, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);
        let mut pb = bg.new_port_buffer();

        let fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 1,
            value: 1.0,
            contra_field_index: 0,
        }]);
        // saying we have 1 field isn't entirely correct
        assert_epsilon!(spredict2(&mut bg, &fb, &mut pb), 0.5);
        assert_epsilon!(slearn2(&mut bg, &fb, &mut pb, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        mi.optimizer = Optimizer::AdagradFlex;
        let mut bg = BlockGraph::new();

        let ffm_block = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, ffm_block, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);
        let mut pb = bg.new_port_buffer();

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut bg.blocks_final[0]);
        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 1.0,
                contra_field_index: mi.ffm_k,
            },
        ]);
        assert_epsilon!(spredict2(&mut bg, &fb, &mut pb), 0.7310586);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.7310586);

        assert_epsilon!(spredict2(&mut bg, &fb, &mut pb), 0.7024794);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.7024794);

        // Two fields, use values
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 2.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 2.0,
                contra_field_index: mi.ffm_k,
            },
        ]);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb), 0.98201376);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.98201376);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb), 0.81377685);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.81377685);
    }

    #[test]
    fn test_ffm_k1_with_cache() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]]; // This isn't really used
        mi.optimizer = Optimizer::AdagradLUT;

        // Nothing can be learned from a single field in FFMs
        let mut bg = BlockGraph::new();
        let ffm_block = new_ffm_block(&mut bg, &mi).unwrap();
        let _loss_block = block_loss_functions::new_logloss_block(&mut bg, ffm_block, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);
        let mut pb = bg.new_port_buffer();

        let mut caches: Vec<BlockCache> = Vec::default();
        let cache_fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 1,
            value: 1.0,
            contra_field_index: 0,
        }]); // saying we have 1 field isn't entirely correct

        let fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 1,
            value: 1.0,
            contra_field_index: 0,
        }]); // saying we have 1 field isn't entirely correct
        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(spredict2_with_cache(&mut bg, &fb, &mut pb, &caches), 0.5);
        assert_epsilon!(slearn2(&mut bg, &fb, &mut pb, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        mi.optimizer = Optimizer::AdagradFlex;
        let mut bg = BlockGraph::new();

        let ffm_block = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, ffm_block, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);
        let mut pb = bg.new_port_buffer();

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut bg.blocks_final[0]);
        let mut caches: Vec<BlockCache> = Vec::default();
        let cache_fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 1,
            value: 1.0,
            contra_field_index: 0,
        }]);

        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 1.0,
                contra_field_index: mi.ffm_k,
            },
        ]);
        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.7310586
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.7310586);

        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.7024794
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.7024794);

        // Two fields, use values
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let mut caches: Vec<BlockCache> = Vec::default();
        let cache_fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 100,
            value: 2.0,
            contra_field_index: mi.ffm_k,
        }]);
        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 2.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 2.0,
                contra_field_index: mi.ffm_k,
            },
        ]);
        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.98201376
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.98201376);

        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.81377685
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.81377685);
    }

    #[test]
    fn test_ffm_k4() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.ffm_k = 4;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]]; // This isn't really used

        // Nothing can be learned from a single field in FFMs
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        let fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 1,
            value: 1.0,
            contra_field_index: 0,
        }]);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb), 0.5);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.5);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb), 0.5);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        mi.optimizer = Optimizer::AdagradFlex;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut bg.blocks_final[0]);
        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 1.0,
                contra_field_index: mi.ffm_k,
            },
        ]);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb), 0.98201376);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.98201376);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb), 0.96277946);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.96277946);

        // Two fields, use values
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 2.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 2.0,
                contra_field_index: mi.ffm_k,
            },
        ]);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb), 0.9999999);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.9999999);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb), 0.99685884);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.99685884);
    }

    #[test]
    fn test_ffm_k4_with_cache() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.ffm_k = 4;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]]; // This isn't really used

        // Nothing can be learned from a single field in FFMs
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();
        let mut caches: Vec<BlockCache> = Vec::default();
        let cache_fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 1,
            value: 1.0,
            contra_field_index: 0,
        }]);
        let fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 1,
            value: 1.0,
            contra_field_index: 0,
        }]);
        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(spredict2_with_cache(&mut bg, &fb, &mut pb, &caches), 0.5);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.5);

        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(spredict2_with_cache(&mut bg, &fb, &mut pb, &caches), 0.5);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.5);

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        mi.optimizer = Optimizer::AdagradFlex;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut bg.blocks_final[0]);
        let mut caches: Vec<BlockCache> = Vec::default();
        let cache_fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 100,
            value: 1.0,
            contra_field_index: mi.ffm_k,
        }]);
        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 1.0,
                contra_field_index: mi.ffm_k,
            },
        ]);
        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.98201376
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.98201376);

        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.96277946
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.96277946);

        // Two fields, use values
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let mut caches: Vec<BlockCache> = Vec::default();
        let cache_fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 100,
            value: 2.0,
            contra_field_index: mi.ffm_k,
        }]);
        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 2.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 2.0,
                contra_field_index: mi.ffm_k,
            },
        ]);
        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.9999999
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.9999999);

        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.99685884
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.99685884);
    }

    #[test]
    fn test_ffm_multivalue() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_power_t = 0.0;
        mi.ffm_learning_rate = 0.1;
        mi.ffm_fields = vec![vec![], vec![]];

        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let fbuf = &ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 3 * 1000,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 2.0,
                contra_field_index: mi.ffm_k,
            },
        ]);
        assert_epsilon!(spredict2(&mut bg, fbuf, &mut pb), 0.9933072);
        assert_eq!(slearn2(&mut bg, fbuf, &mut pb, true), 0.9933072);
        assert_epsilon!(spredict2(&mut bg, fbuf, &mut pb), 0.9395168);
        assert_eq!(slearn2(&mut bg, fbuf, &mut pb, false), 0.9395168);
        assert_epsilon!(spredict2(&mut bg, fbuf, &mut pb), 0.9395168);
        assert_eq!(slearn2(&mut bg, fbuf, &mut pb, false), 0.9395168);
    }

    #[test]
    fn test_ffm_multivalue_with_cache() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_power_t = 0.0;
        mi.ffm_learning_rate = 0.1;
        mi.ffm_fields = vec![vec![], vec![]];

        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let mut caches: Vec<BlockCache> = Vec::default();
        let cache_fb = &ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 3 * 1000,
                value: 1.0,
                contra_field_index: 0,
            },
        ]);

        let fb = &ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 3 * 1000,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 2.0,
                contra_field_index: mi.ffm_k * 1,
            },
        ]);
        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.9933072
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.9933072);

        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.9395168
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, false), 0.9395168);

        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.9395168
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, false), 0.9395168);
    }

    #[test]
    fn test_ffm_multivalue_k4_nonzero_powert() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.ffm_k = 4;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]];

        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let fbuf = &ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 3 * 1000,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 2.0,
                contra_field_index: mi.ffm_k,
            },
        ]);

        assert_eq!(spredict2(&mut bg, fbuf, &mut pb), 1.0);
        assert_eq!(slearn2(&mut bg, fbuf, &mut pb, true), 1.0);
        assert_eq!(spredict2(&mut bg, fbuf, &mut pb), 0.9949837);
        assert_eq!(slearn2(&mut bg, fbuf, &mut pb, false), 0.9949837);
        assert_eq!(slearn2(&mut bg, fbuf, &mut pb, false), 0.9949837);
    }

    #[test]
    fn test_ffm_multivalue_k4_nonzero_powert_with_cache() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.ffm_k = 4;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]];

        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        ffm_init::<optimizer::OptimizerAdagradLUT>(&mut bg.blocks_final[0]);
        let mut caches: Vec<BlockCache> = Vec::default();
        let cache_fb = &ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 3 * 1000,
                value: 1.0,
                contra_field_index: 0,
            },
        ]);

        let fb = &ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 3 * 1000,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 2.0,
                contra_field_index: mi.ffm_k,
            },
        ]);

        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(spredict2_with_cache(&mut bg, &fb, &mut pb, &caches), 1.0);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 1.0);

        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.9949837
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, false), 0.9949837);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, false), 0.9949837);
    }

    #[test]
    fn test_ffm_missing_field() {
        // This test is useful to check if we don't by accient forget to initialize any of the collapsed
        // embeddings for the field, when field has no instances of a feature in it
        // We do by having three-field situation where only the middle field has features
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![], vec![]]; // This isn't really used

        // Nothing can be learned from a single field in FFMs
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        mi.optimizer = Optimizer::AdagradFlex;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut bg.blocks_final[0]);
        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 5,
                value: 1.0,
                contra_field_index: mi.ffm_k,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 1.0,
                contra_field_index: mi.ffm_k * 2,
            },
        ]);
        assert_epsilon!(spredict2(&mut bg, &fb, &mut pb), 0.95257413);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, false), 0.95257413);

        // here we intentionally have just the middle field
        let fb = ffm_vec(vec![HashAndValueAndSeq {
            hash: 5,
            value: 1.0,
            contra_field_index: mi.ffm_k,
        }]);
        assert_eq!(spredict2(&mut bg, &fb, &mut pb), 0.5);
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.5);
    }

    #[test]
    fn test_ffm_missing_field_with_cache() {
        // This test is useful to check if we don't by accient forget to initialize any of the collapsed
        // embeddings for the field, when field has no instances of a feature in it
        // We do by having three-field situation where only the middle field has features
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![], vec![]]; // This isn't really used

        // Nothing can be learned from a single field in FFMs
        mi.optimizer = Optimizer::AdagradLUT;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        let mut pb = bg.new_port_buffer();

        // With two fields, things start to happen
        // Since fields depend on initial randomization, these tests are ... peculiar.
        mi.optimizer = Optimizer::AdagradFlex;
        let mut bg = BlockGraph::new();
        let re_ffm = new_ffm_block(&mut bg, &mi).unwrap();
        let _lossf = block_loss_functions::new_logloss_block(&mut bg, re_ffm, true);
        bg.finalize();
        bg.allocate_and_init_weights(&mi);

        ffm_init::<optimizer::OptimizerAdagradFlex>(&mut bg.blocks_final[0]);
        let mut caches: Vec<BlockCache> = Vec::default();
        let cache_fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 1.0,
                contra_field_index: mi.ffm_k * 2,
            },
        ]);
        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 5,
                value: 1.0,
                contra_field_index: mi.ffm_k,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 1.0,
                contra_field_index: mi.ffm_k * 2,
            },
        ]);
        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_epsilon!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.95257413
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, false), 0.95257413);

        // here we intentionally have missing fields
        let fb = ffm_vec(vec![
            HashAndValueAndSeq {
                hash: 1,
                value: 1.0,
                contra_field_index: 0,
            },
            HashAndValueAndSeq {
                hash: 100,
                value: 1.0,
                contra_field_index: mi.ffm_k * 2,
            },
        ]);
        ssetup_cache2(&mut bg, &cache_fb, &mut caches);
        assert_eq!(
            spredict2_with_cache(&mut bg, &fb, &mut pb, &caches),
            0.7310586
        );
        assert_eq!(slearn2(&mut bg, &fb, &mut pb, true), 0.7310586);
    }
}
