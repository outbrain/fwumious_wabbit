use std::any::Any;

use crate::optimizer;
use crate::regressor;
use crate::model_instance;
use crate::feature_buffer;
use std::io;
use core::arch::x86_64::*;
use std::error::Error;
use serde_json::{Value, Map, json};
use std::sync::Arc;

use std::mem::{self, MaybeUninit};
use optimizer::OptimizerTrait;
use regressor::BlockTrait;
use crate::block_helpers;
use block_helpers::{Weight, WeightAndOptimizerData, f32_to_json};


pub struct BlockLR<L:OptimizerTrait> {
    pub weights: Vec<WeightAndOptimizerData<L>>,
    pub weights_len: u32,
    pub optimizer_lr: L,
}

impl <L:OptimizerTrait + 'static> BlockTrait for BlockLR<L> 
{
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn new_without_weights(mi: &model_instance::ModelInstance) -> Result<Box<dyn BlockTrait>, Box<dyn Error>>  where Self: Sized {
        let mut reg_lr = BlockLR::<L> {
            weights: Vec::new(),
            weights_len: 0, 
            optimizer_lr: L::new(),
        };
        reg_lr.optimizer_lr.init(mi.learning_rate, mi.power_t, mi.init_acc_gradient);
        reg_lr.weights_len = 1 << mi.bit_precision;
        Ok(Box::new(reg_lr))
    }

    fn new_forward_only_without_weights(&self) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        let forwards_only = BlockLR::<optimizer::OptimizerSGD> {
            weights_len: self.weights_len,
            weights: Vec::new(),
            optimizer_lr:optimizer::OptimizerSGD::new(),
        };
        
        Ok(Box::new(forwards_only))
    }



    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        self.weights = vec![WeightAndOptimizerData::<L>{weight:0.0, optimizer_data: self.optimizer_lr.initial_data()}; self.weights_len as usize];
        
    }

    #[inline(always)]
    fn forward_backward(&mut self, 
                            further_regressors: &mut [Box<dyn BlockTrait>], 
                            wsum_input: f32, 
                            fb: &feature_buffer::FeatureBuffer, 
                            update:bool) -> (f32, f32) {
        let mut wsum:f32 = 0.0;
        unsafe {
            for hashvalue in fb.lr_buffer.iter() {
                // Prefetch couple of indexes from the future to prevent pipeline stalls due to memory latencies
//            for (i, hashvalue) in fb.lr_buffer.iter().enumerate() {
                // _mm_prefetch(mem::transmute::<&f32, &i8>(&self.weights.get_unchecked((fb.lr_buffer.get_unchecked(i+8).hash) as usize).weight), _MM_HINT_T0);  // No benefit for now
                let feature_index     = hashvalue.hash;
                let feature_value:f32 = hashvalue.value;
                let feature_weight    = self.weights.get_unchecked(feature_index as usize).weight;
                wsum += feature_weight * feature_value;
            }
            let wsum_output = wsum + wsum_input;

            if fb.audit_mode {
                self.audit_forward(wsum_input, wsum_output, fb)
            }

            let (next_regressor, further_regressors) = further_regressors.split_at_mut(1);
            let (prediction_probability, general_gradient) = next_regressor[0].forward_backward(further_regressors, wsum_output, fb, update);

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
    
    
    fn forward(&self, 
             further_blocks: &[Box<dyn BlockTrait>], 
             wsum_input: f32, 
             fb: &feature_buffer::FeatureBuffer) -> f32 {
        let mut wsum:f32 = 0.0;
        unsafe {
            for val in &fb.lr_buffer {
                let feature_hash_index = val.hash as usize;
                let feature_value:f32 = val.value;
                wsum += self.weights.get_unchecked(feature_hash_index).weight * feature_value;
            }
        }
        let wsum_output = wsum_input + wsum;
        
        if fb.audit_mode {
            self.audit_forward(wsum_input, wsum_output, fb)
        }

        let (next_regressor, further_blocks) = further_blocks.split_at(1);
        let prediction_probability = next_regressor[0].forward(further_blocks, wsum_output, fb);
        prediction_probability
    }
    
    fn audit_forward(&self, 
        wsum_input: f32, 
        output: f32, 
        fb: &feature_buffer::FeatureBuffer) {
    
        let mut map = Map::new();
            map.insert("_type".to_string(), Value::String("BlockLR".to_string()));
            let mut features: Vec<Value> = Vec::new();
//            println!("A: {}. B: {}", fb.lr_buffer.len(), fb.lr_buffer_audit.len());
            for (val, combo_number) in fb.lr_buffer.iter().zip(fb.lr_buffer_audit.iter()) {
                let feature_hash_index = val.hash;
                let feature_value = val.value;
                features.push(json!({
                    "index": feature_hash_index,
                    "value": feature_value,
                    "weight": self.weights[feature_hash_index as usize].weight,
                    "feature": fb.audit_aux_data.combo_index_to_string[combo_number],
                    "optimizer_data": self.optimizer_lr.get_audit_data(&self.weights[feature_hash_index as usize].optimizer_data),
                    }));
            }
            map.insert("input".to_string(), Value::Array(features));
            map.insert("output".to_string(), f32_to_json(output));
            fb.add_audit_json(map);
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


mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use serde_json::to_string_pretty;
    use crate::feature_buffer::HashAndValue;
    use crate::block_loss_functions::BlockSigmoid;
    use crate::block_helpers::{spredict, slearn};
    use crate::vwmap;


    /* LR TESTS */
    fn lr_vec(v:Vec<feature_buffer::HashAndValue>) -> feature_buffer::FeatureBuffer {
        let mut fb = feature_buffer::FeatureBuffer::new();
        fb.lr_buffer = v;
        fb
    }

    
    #[test]
    fn test_basic_audit() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.power_t = 0.5;
        mi.init_acc_gradient = 0.0;
        // Now prepare the "reverse resolution" data for auditing
        let vw_map_string = r#"
A,featureA
B,featureB
C,featureC
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string, (vec![], 0)).unwrap();

        mi.feature_combo_descs.push(model_instance::FeatureComboDesc {
                                                        feature_indices: vec![0, 2], 
                                                        weight: 1.0});
        mi.enable_audit(&vw); 
        
                
        let fb = &mut lr_vec(vec![HashAndValue{hash:15, value: 1.0}]);
        fb.lr_buffer_audit.push(0); // we have one feature combo 
        fb.audit_aux_data = mi.audit_aux_data.as_ref().unwrap().clone();

        let mut lossf = BlockSigmoid::new_without_weights(&mi).unwrap();
        lossf.allocate_and_init_weights(&mi);
        
        let mut re = BlockLR::<optimizer::OptimizerAdagradLUT>::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        assert_eq!(slearn(&mut re, &mut lossf, &fb, true), 0.5);
        assert_eq!(slearn(&mut re, &mut lossf, &fb, false), 0.475734);
        fb.audit_mode = true;
        fb.reset_audit_json();
        assert_eq!(spredict(&mut re, &mut lossf, &fb, true), 0.475734);
        let audit1 = format!("{}", to_string_pretty(&fb.audit_json).unwrap());
        println!("{}", audit1);
        fb.reset_audit_json();
        assert_eq!(slearn(&mut re, &mut lossf, &fb, false), 0.475734);
        let audit2 = format!("{}", to_string_pretty(&fb.audit_json).unwrap());
        assert_eq!(audit1, audit2);     // both have to be equal, no matter if spredict or slearn was used


    }


}





















