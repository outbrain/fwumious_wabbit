
mod block_ffm;
mod block_helpers;
mod block_loss_functions;
mod block_lr;
mod cache;
mod cmdline;
mod consts;
mod feature_buffer;
mod feature_transform_executor;
mod feature_transform_implementations;
mod feature_transform_parser;
mod model_instance;
mod multithread_helpers;
mod optimizer;
mod parser;
mod persistence;
mod regressor;
mod serving;
mod version;
mod vwmap;

use regressor::Regressor;

#[no_mangle]
pub extern "C" fn fw_predict(ptr: *mut Regressor, fb: &feature_buffer::FeatureBuffer) -> f32 {
    let fw_regressor = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    return fw_regressor.predict(fb);
}
