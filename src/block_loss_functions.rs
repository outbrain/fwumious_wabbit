use std::any::Any;
use std::error::Error;
use std::io;
use serde_json::{Value, Map, Number};
use std::mem;
use std::mem::size_of;

use crate::regressor;
use crate::feature_buffer;
use crate::model_instance;
use regressor::BlockTrait;
use crate::block_helpers::f32_to_json;

//use fastapprox::fast::sigmoid; // surprisingly this doesn't work very well

/* We tested standard stable logistic function, but it gives slightly 
worse logloss results than plain logistic on our data */
/*
#[inline(always)]
pub fn stable_logistic(t: f32) -> f32 {
    if t > 0.0 {
        return (1.0 +(-t).exp()).recip();
    } else {
        let texp = t.exp();
        return texp / (1.0 + texp);
    }
}
*/

#[inline(always)]
pub fn logistic(t: f32) -> f32 {
    return (1.0+(-t).exp()).recip();
}



pub struct BlockSigmoid {
}

impl BlockTrait for BlockSigmoid {

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn new_without_weights(mi: &model_instance::ModelInstance) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        Ok(Box::new(BlockSigmoid {}))
    }

    #[inline(always)]
    fn forward_backward(&mut self, 
                    further_regressors: &mut [Box<dyn BlockTrait>], 
                    wsum_input: f32, 
                    fb: &feature_buffer::FeatureBuffer, 
                    update:bool) -> (f32, f32) {
        if further_regressors.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }
        
        // vowpal compatibility
        let prediction_probability: f32;
        if wsum_input.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", fb.example_number);
            prediction_probability = logistic(0.0);
        } else if wsum_input < -50.0 {
            prediction_probability = logistic(-50.0);
        } else if wsum_input > 50.0 {
            prediction_probability = logistic(50.0);
        } else {        
          prediction_probability = logistic(wsum_input);
        }
        
        if fb.audit_mode {
            self.audit_forward(wsum_input, prediction_probability, fb);
        }

        let general_gradient = (fb.label - prediction_probability) * fb.example_importance;
        //println!("General gradient: {}", general_gradient);
        (prediction_probability, general_gradient)
    }

    fn forward(&self, 
                     further_blocks: &[Box<dyn BlockTrait>], 
                     wsum_input: f32, 
                     fb: &feature_buffer::FeatureBuffer) -> f32 {

        if further_blocks.len() != 0 {
            panic!("RegSigmoid can only be at the end of the chain!");
        }
        
        // vowpal compatibility
        let mut prediction_probability: f32;
        if wsum_input.is_nan() {
            eprintln!("NAN prediction in example {}, forcing 0.0", fb.example_number);
            prediction_probability = logistic(0.0);
        } else if wsum_input < -50.0 {
            prediction_probability = logistic(-50.0);
        } else if wsum_input > 50.0 {
            prediction_probability = logistic(50.0);
        } else {        
          prediction_probability = logistic(wsum_input);
        }
        
        if fb.audit_mode {
            self.audit_forward(wsum_input, prediction_probability, fb);
        }
        prediction_probability
    }
    fn audit_forward(&self, 
        wsum_input: f32, 
        output: f32, 
        fb: &feature_buffer::FeatureBuffer) {

        let mut map = Map::new();
        map.insert("_type".to_string(), Value::String("BlockSigmoid".to_string()));
        map.insert("input".to_string(), f32_to_json(wsum_input));
        map.insert("output".to_string(), f32_to_json(output));
        fb.add_audit_json(map);

    }

    fn allocate_and_init_weights(&mut self, mi: &model_instance::ModelInstance) {
        // empty
    }
    fn get_serialized_len(&self) -> usize {
        return 0
    }

    fn read_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn read_weights_from_buf_into_forward_only(&self, input_bufreader: &mut dyn io::Read, forward: &mut Box<dyn BlockTrait>) -> Result<(), Box<dyn Error>> {
        Ok(())        
    }

    fn new_forward_only_without_weights(&self) -> Result<Box<dyn BlockTrait>, Box<dyn Error>> {
        Ok(Box::new(BlockSigmoid{}))
    }
    /// Sets internal state of weights based on some completely object-dependent parameters
    fn testing_set_weights(&mut self, aa: i32, bb: i32, index: usize, w: &[f32]) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

}


mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use serde_json::to_string_pretty;
    use crate::block_helpers::slearn;

    fn in_vec() -> feature_buffer::FeatureBuffer {
        feature_buffer::FeatureBuffer::new()
    }

    #[test]
    fn test_basic() {
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();        
        mi.learning_rate = 0.1;
        mi.ffm_learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.ffm_power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 4;
        mi.ffm_bit_precision = 18;
        mi.ffm_fields = vec![vec![], vec![]]; // This isn't really used
        let mut re = BlockSigmoid::new_without_weights(&mi).unwrap();
        re.allocate_and_init_weights(&mi);

        let mut in_buf = in_vec();
        in_buf.audit_mode = true;
        let result = re.forward(&[], 0.1, &in_buf);
        assert_eq!(result, 0.5249792);
        assert_eq!(to_string_pretty(&in_buf.audit_json).unwrap(),
r#"{
  "_type": "BlockSigmoid",
  "input": 0.10000000149011612,
  "output": 0.5249791741371155,
  "predcessor": null
}"#);
    }
}



