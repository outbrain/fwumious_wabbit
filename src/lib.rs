
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

use std::ffi::CStr;
use std::io::Cursor;
use std::os::raw::c_char;
use shellwords;
use crate::feature_buffer::FeatureBufferTranslator;
use crate::multithread_helpers::BoxedRegressorTrait;
use crate::parser::VowpalParser;
use crate::regressor::Regressor;

#[repr(C)]
pub struct FfiPredictor {
    _marker: core::marker::PhantomData<Predictor>,
}

pub struct Predictor {
    feature_buffer_translator: FeatureBufferTranslator,
    vw_parser: VowpalParser,
    regressor: BoxedRegressorTrait,
}

impl Predictor {

    unsafe fn predict(&mut self, input_buffer: &str) -> f32 {
        let mut buffered_input = Cursor::new(input_buffer);
        let reading_result = self.vw_parser.next_vowpal(&mut buffered_input);
        let buffer = match reading_result {
            Ok([]) => return -1.0, // EOF
            Ok(buffer2) => buffer2,
            Err(_e) => return -1.0
        };
        self.feature_buffer_translator.translate(buffer, 0);
        self.regressor.predict(&self.feature_buffer_translator.feature_buffer)
    }
}


#[no_mangle]
pub extern "C" fn new_fw_predictor_prototype(command: *const c_char) -> *mut FfiPredictor {
    let str_command = c_char_to_str(command);
    let words = shellwords::split(str_command).unwrap();
    let cmd_matches = cmdline::create_expected_args().get_matches_from(words);
    let weights_filename = match cmd_matches.value_of("initial_regressor") {
            Some(filename) => filename,
            None => panic!("Cannot resolve input weights file name")
    };
    let (model_instance, vw_namespace_map, regressor) = persistence::new_regressor_from_filename(weights_filename, true, Some(&cmd_matches)).unwrap();
    let feature_buffer_translator = FeatureBufferTranslator::new(&model_instance);
    let vw_parser = VowpalParser::new(&vw_namespace_map);
    let sharable_regressor = BoxedRegressorTrait::new(Box::new(regressor));
    let predictor = Predictor {
        feature_buffer_translator,
        vw_parser,
        regressor: sharable_regressor
    };
    Box::into_raw(Box::new(predictor)).cast()
}

#[no_mangle]
pub unsafe extern "C" fn clone_lite(prototype: *mut FfiPredictor) -> *mut FfiPredictor {
    let prototype: &mut Predictor = from_ptr(prototype);
    let lite_predictor = Predictor {
        feature_buffer_translator: prototype.feature_buffer_translator.clone(),
        vw_parser: prototype.vw_parser.clone(),
        regressor: prototype.regressor.clone()
    };
    Box::into_raw(Box::new(lite_predictor)).cast()
}

#[no_mangle]
pub unsafe extern "C" fn fw_predict(ptr: *mut FfiPredictor, input_buffer: *const c_char) -> f32 {
    let str_buffer = c_char_to_str(input_buffer);
    let predictor: &mut Predictor = from_ptr(ptr);
    predictor.predict(str_buffer)
}

#[no_mangle]
pub unsafe extern "C" fn free_predictor(ptr: *mut FfiPredictor) {
    drop::<Box<Predictor>>(Box::from_raw(from_ptr(ptr)));
}

unsafe fn from_ptr<'a>(ptr: *mut FfiPredictor) -> &'a mut Predictor
{
    if ptr.is_null() {
        eprintln!("Fatal error, got NULL `Context` pointer");
        std::process::abort();
    }
    &mut *(ptr.cast())
}

fn c_char_to_str<'a>(input_buffer: *const c_char) -> &'a str {
    let c_str = unsafe {
        assert!(!input_buffer.is_null());
        CStr::from_ptr(input_buffer)
    };
    let str_buffer = c_str.to_str().unwrap();
    str_buffer
}
