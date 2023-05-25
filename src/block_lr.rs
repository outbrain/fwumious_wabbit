use std::any::Any;
use std::collections::HashSet;

use crate::feature_buffer;
use crate::graph;
use crate::model_instance;
use crate::optimizer;
use crate::regressor;

use std::error::Error;
use std::io;

use crate::block_helpers;
use crate::port_buffer;
use block_helpers::WeightAndOptimizerData;
use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use crate::block_helpers::szudziki_pair;

#[derive(Default)]
struct BlockLRCache {
    lr: Vec<f32>,
    buffer: HashSet<u32>
}

pub struct BlockLR<L: OptimizerTrait> {
    pub weights: Vec<WeightAndOptimizerData<L>>,
    pub weights_len: u32,
    pub optimizer_lr: L,
    pub output_offset: usize,
    pub num_combos: u32,
    cache: BlockLRCache,
}

impl<L: OptimizerTrait + 'static> BlockLR<L> {

    fn internal_forward(
        &self,
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        debug_assert!(self.output_offset != usize::MAX);

        {
            unsafe {
                let myslice = &mut pb.tape
                    [self.output_offset..(self.output_offset + self.num_combos as usize)];
                myslice.fill(0.0);
                for feature in fb.lr_buffer.iter() {
                    let feature_index = feature.hash as usize;
                    let feature_value = feature.value;
                    let combo_index = feature.combo_index as usize;
                    *myslice.get_unchecked_mut(combo_index) +=
                        self.weights.get_unchecked(feature_index).weight * feature_value;
                }
            }
        }
    }
}

fn new_lr_block_without_weights<L: OptimizerTrait + 'static>(
    mi: &model_instance::ModelInstance,
) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
    let mut num_combos = mi.feature_combo_descs.len() as u32;
    if mi.add_constant_feature {
        num_combos += 1;
    }
    let mut reg_lr = BlockLR::<L> {
        weights: Vec::new(),
        weights_len: 0,
        optimizer_lr: L::new(),
        output_offset: usize::MAX,
        num_combos,
        cache: BlockLRCache::default(),
    };
    reg_lr
        .optimizer_lr
        .init(mi.learning_rate, mi.power_t, mi.init_acc_gradient);
    reg_lr.weights_len = 1 << mi.bit_precision;
    Ok(Box::new(reg_lr))
}

pub fn new_lr_block(
    bg: &mut graph::BlockGraph,
    mi: &model_instance::ModelInstance,
) -> Result<graph::BlockPtrOutput, Box<dyn Error>> {
    let block = match mi.optimizer {
        model_instance::Optimizer::AdagradLUT => {
            new_lr_block_without_weights::<optimizer::OptimizerAdagradLUT>(&mi)
        }
        model_instance::Optimizer::AdagradFlex => {
            new_lr_block_without_weights::<optimizer::OptimizerAdagradFlex>(&mi)
        }
        model_instance::Optimizer::SGD => {
            new_lr_block_without_weights::<optimizer::OptimizerSGD>(&mi)
        }
    }
    .unwrap();
    let mut block_outputs = bg.add_node(block, vec![])?;
    assert_eq!(block_outputs.len(), 1);
    Ok(block_outputs.pop().unwrap())
}

impl<L: OptimizerTrait + 'static> BlockTrait for BlockLR<L> {
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights = vec![
            WeightAndOptimizerData::<L> {
                weight: 0.0,
                optimizer_data: self.optimizer_lr.initial_data()
            };
            self.weights_len as usize
        ];
    }

    fn get_num_output_slots(&self) -> usize {
        1
    }

    fn get_num_output_values(&self, output: graph::OutputSlot) -> usize {
        assert_eq!(output.get_output_index(), 0);
        self.num_combos as usize
    }

    fn set_input_offset(&mut self, input: graph::InputSlot, offset: usize) {
        panic!("You cannnot set_input_offset() for BlockLR");
    }

    fn set_output_offset(&mut self, output: graph::OutputSlot, offset: usize) {
        assert_eq!(output.get_output_index(), 0);
        debug_assert!(self.output_offset == usize::MAX); // We only allow a single call
        self.output_offset = offset;
    }

    #[inline(always)]
    fn forward_backward(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
        update: bool,
    ) {
        unsafe {
            self.internal_forward(fb, pb);

            block_helpers::forward_backward(further_blocks, fb, pb, update);

            if update {
                let myslice = &mut pb.tape.get_unchecked(
                    self.output_offset..(self.output_offset + self.num_combos as usize),
                );

                for feature in fb.lr_buffer.iter() {
                    let feature_index = feature.hash as usize;
                    let feature_value = feature.value;
                    let gradient = myslice.get_unchecked(feature.combo_index as usize) * feature_value;
                    let update = self.optimizer_lr.calculate_update(
                        gradient,
                        &mut self.weights.get_unchecked_mut(feature_index).optimizer_data,
                    );
                    self.weights.get_unchecked_mut(feature_index).weight -= update;
                }
            }
        } // end of unsafe
    }

    fn forward(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        self.internal_forward(fb, pb);
        block_helpers::forward(further_blocks, fb, pb);
    }

    fn forward_with_cache(
        &self,
        further_blocks: &[Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
        pb: &mut port_buffer::PortBuffer,
    ) {
        unsafe {
            let lr_slice = &mut pb.tape
                [self.output_offset..(self.output_offset + self.num_combos as usize)];
            lr_slice.copy_from_slice(self.cache.lr.as_slice());

            for feature in fb.lr_buffer.iter() {
                let feature_index = feature.hash as usize;
                let combo_index = feature.combo_index as usize;
                if self.cache.buffer.contains(&szudziki_pair(feature.combo_index, feature.hash)) {
                    continue
                }
                let feature_value = feature.value;
                *lr_slice.get_unchecked_mut(combo_index) +=
                    self.weights.get_unchecked(feature_index).weight * feature_value;
            }
        }

        block_helpers::forward_with_cache(further_blocks, fb, pb);
    }

    fn prepare_forward_cache(
        &mut self,
        further_blocks: &mut [Box<dyn BlockTrait>],
        fb: &feature_buffer::FeatureBuffer,
    ) {
        unsafe {
            self.cache.buffer = fb.lr_buffer.iter()
                .map(|feature| szudziki_pair(feature.combo_index, feature.hash))
                .collect();
            self.cache.lr = vec![0.0; self.num_combos as usize];

            let mut lr_slice = self.cache.lr.as_mut_slice();

            for feature in fb.lr_buffer.iter() {
                let feature_index = feature.hash as usize;
                let feature_value = feature.value;
                let combo_index = feature.combo_index as usize;
                *lr_slice.get_unchecked_mut(combo_index) +=
                    self.weights.get_unchecked(feature_index).weight * feature_value;
            }
        }

        block_helpers::prepare_forward_cache(further_blocks, fb);
    }

    fn get_serialized_len(&self) -> usize {
        return self.weights_len as usize;
    }

    fn read_weights_from_buf(
        &mut self,
        input_bufreader: &mut dyn io::Read,
    ) -> Result<(), Box<dyn Error>> {
        block_helpers::read_weights_from_buf(&mut self.weights, input_bufreader)
    }

    fn write_weights_to_buf(
        &self,
        output_bufwriter: &mut dyn io::Write,
    ) -> Result<(), Box<dyn Error>> {
        block_helpers::write_weights_to_buf(&self.weights, output_bufwriter)
    }

    fn read_weights_from_buf_into_forward_only(
        &self,
        input_bufreader: &mut dyn io::Read,
        forward: &mut Box<dyn BlockTrait>,
    ) -> Result<(), Box<dyn Error>> {
        let mut forward = forward
            .as_any()
            .downcast_mut::<BlockLR<optimizer::OptimizerSGD>>()
            .unwrap();
        block_helpers::read_weights_only_from_buf2::<L>(
            self.weights_len as usize,
            &mut forward.weights,
            input_bufreader,
        )
    }

    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(
        &mut self,
        aa: i32,
        bb: i32,
        index: usize,
        w: &[f32],
    ) -> Result<(), Box<dyn Error>> {
        self.weights[index].weight = w[0];
        self.weights[index].optimizer_data = self.optimizer_lr.initial_data();
        Ok(())
    }
}
