#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(non_snake_case)]
#![allow(redundant_semicolons)]
#![allow(dead_code,unused_imports)]

use crate::hogwild::HogwildTrainer;
use crate::multithread_helpers::BoxedRegressorTrait;
use flate2::read::MultiGzDecoder;
use std::collections::VecDeque;
use std::error::Error;
use std::f32;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

extern crate blas;
extern crate intel_mkl_src;

#[macro_use]
extern crate nom;
extern crate core;

mod block_ffm;
mod block_helpers;
mod block_loss_functions;
mod block_lr;
mod block_misc;
mod block_neural;
mod block_normalize;
mod block_relu;
mod cache;
mod cmdline;
mod feature_buffer;
mod feature_transform_executor;
mod feature_transform_implementations;
mod feature_transform_parser;
mod graph;
mod hogwild;
mod logging_layer;
mod model_instance;
mod multithread_helpers;
mod optimizer;
mod parser;
mod persistence;
mod port_buffer;
mod radix_tree;
mod regressor;
mod serving;
mod version;
mod vwmap;

fn main() {
    logging_layer::initialize_logging_layer();

    if let Err(e) = main2() {
        log::error!("Global error: {:?}", e);
        std::process::exit(1)
    }
}

fn build_cache_without_training(cl: clap::ArgMatches) -> Result<(), Box<dyn Error>> {
    /*! A method that enables creating the cache file without training the first model instance.
    This is done in order to reduce building time of the cache and running the first model instance multi threaded. */
    // We'll parse once the command line into cl and then different objects will examine it
    let input_filename = cl.value_of("data").expect("--data expected");
    let vw_namespace_map_filepath = Path::new(input_filename)
        .parent()
        .expect("Couldn't access path given by --data")
        .join("vw_namespace_map.csv");

    let vw: vwmap::VwNamespaceMap =
        vwmap::VwNamespaceMap::new_from_csv_filepath(vw_namespace_map_filepath)?;
    let mut cache = cache::RecordCache::new(input_filename, true, &vw);
    let input = File::open(input_filename)?;
    let mut aa;
    let mut bb;
    let mut bufferred_input: &mut dyn BufRead = match input_filename.ends_with(".gz") {
        true => {
            aa = io::BufReader::new(MultiGzDecoder::new(input));
            &mut aa
        }
        false => {
            bb = io::BufReader::new(input);
            &mut bb
        }
    };
    let mut pa = parser::VowpalParser::new(&vw);
    let mut example_num = 0;
    loop {
        let reading_result;
        let buffer: &[u32];
        if !cache.reading {
            reading_result = pa.next_vowpal(&mut bufferred_input);
            buffer = match reading_result {
                Ok([]) => break, // EOF
                Ok(buffer2) => buffer2,
                Err(_e) => return Err(_e),
            };
            if cache.writing {
                cache.push_record(buffer)?;
            }
        } else {
            reading_result = cache.get_next_record();
	    match reading_result {
                Ok([]) => break, // EOF
                Ok(buffer) => buffer,
                Err(_e) => return Err(_e),
            };
        }
        example_num += 1;
    }

    log::info!("Built cache only, exiting.");
    cache.write_finish()?;
    Ok(())
}

fn main2() -> Result<(), Box<dyn Error>> {
    // We'll parse once the command line into cl and then different objects will examine it
    let cl = cmdline::parse();
    if cl.is_present("build_cache_without_training") {
        return build_cache_without_training(cl);
    }
    // Where will we be putting perdictions (if at all)
    let mut predictions_file = match cl.value_of("predictions") {
        Some(filename) => Some(BufWriter::new(File::create(filename)?)),
        None => None,
    };

    let testonly = cl.is_present("testonly");

    let final_regressor_filename = cl.value_of("final_regressor");
    let output_pred_sto: bool = cl.is_present("predictions_stdout");
    if let Some(filename) = final_regressor_filename {
        if !cl.is_present("save_resume") {
            return Err("You need to use --save_resume with --final_regressor, for vowpal wabbit compatibility")?;
        }
        log::info!("final_regressor = {}", filename);
    };

    let inference_regressor_filename = cl.value_of("convert_inference_regressor");
    if let Some(filename) = inference_regressor_filename {
        log::info!("inference_regressor = {}", filename);
    };

    /* setting up the pipeline, either from command line or from existing regressor */
    // we want heal-allocated objects here

    if cl.is_present("daemon") {
        let filename = cl
            .value_of("initial_regressor")
            .expect("Daemon mode only supports serving from --initial regressor");
        log::info!("initial_regressor = {}", filename);
        let (mi2, vw2, re_fixed) =
            persistence::new_regressor_from_filename(filename, true, Option::Some(&cl))?;

        let mut se = serving::Serving::new(&cl, &vw2, Box::new(re_fixed), &mi2)?;
        se.serve()?;
    } else if cl.is_present("convert_inference_regressor") {
        let filename = cl
            .value_of("initial_regressor")
            .expect("Convert mode requires --initial regressor");
        let (mut mi2, vw2, re_fixed) =
            persistence::new_regressor_from_filename(filename, true, Option::Some(&cl))?;
        mi2.optimizer = model_instance::Optimizer::SGD;
        if let Some(filename1) = inference_regressor_filename {
            persistence::save_regressor_to_filename(filename1, &mi2, &vw2, re_fixed).unwrap()
        }
    } else {
        let vw: vwmap::VwNamespaceMap;
        let mut re: regressor::Regressor;
        let mut sharable_regressor: BoxedRegressorTrait;
        let mi: model_instance::ModelInstance;

        if let Some(filename) = cl.value_of("initial_regressor") {
            log::info!("initial_regressor = {}", filename);
            (mi, vw, re) =
                persistence::new_regressor_from_filename(filename, testonly, Option::Some(&cl))?;
            sharable_regressor = BoxedRegressorTrait::new(Box::new(re));
        } else {
            // We load vw_namespace_map.csv just so we know all the namespaces ahead of time
            // This is one of the major differences from vowpal

            let input_filename = cl.value_of("data").expect("--data expected");
            let vw_namespace_map_filepath = Path::new(input_filename)
                .parent()
                .expect("Couldn't access path given by --data")
                .join("vw_namespace_map.csv");
            vw = vwmap::VwNamespaceMap::new_from_csv_filepath(vw_namespace_map_filepath)?;
            mi = model_instance::ModelInstance::new_from_cmdline(&cl, &vw)?;
            re = regressor::get_regressor_with_weights(&mi);
            sharable_regressor = BoxedRegressorTrait::new(Box::new(re));
        };

        let input_filename = cl.value_of("data").expect("--data expected");
        let mut cache = cache::RecordCache::new(input_filename, cl.is_present("cache"), &vw);
        let mut fbt = feature_buffer::FeatureBufferTranslator::new(&mi);
        let mut pb = sharable_regressor.new_portbuffer();

        let predictions_after: u64 = match cl.value_of("predictions_after") {
            Some(examples) => examples.parse()?,
            None => 0,
        };

        let holdout_after_option: Option<u64> =
            cl.value_of("holdout_after").map(|s| s.parse().unwrap());

        let hogwild_training = cl.is_present("hogwild_training");
        let mut hogwild_trainer = if hogwild_training {
            let hogwild_threads = match cl.value_of("hogwild_threads") {
                Some(hogwild_threads) => hogwild_threads
                    .parse()
                    .expect("hogwild_threads should be integer"),
                None => 16,
            };
            HogwildTrainer::new(sharable_regressor.clone(), &mi, hogwild_threads)
        } else {
            HogwildTrainer::default()
        };

        let prediction_model_delay: u64 = match cl.value_of("prediction_model_delay") {
            Some(delay) => delay.parse()?,
            None => 0,
        };

        let mut delayed_learning_fbs: VecDeque<feature_buffer::FeatureBuffer> =
            VecDeque::with_capacity(prediction_model_delay as usize);

        // Setup Parser, is rust forcing this disguisting way to do it, or I just don't know the pattern?
        let input = File::open(input_filename)?;
        let mut aa;
        let mut bb;
        let mut bufferred_input: &mut dyn BufRead = match input_filename.ends_with(".gz") {
            true => {
                aa = io::BufReader::new(MultiGzDecoder::new(input));
                &mut aa
            }
            false => {
                bb = io::BufReader::new(input);
                &mut bb
            }
        };

        let mut pa = parser::VowpalParser::new(&vw);

        let now = Instant::now();
        let mut example_num = 0;
        loop {
            let reading_result;
            let buffer: &[u32];
            if !cache.reading {
                reading_result = pa.next_vowpal(&mut bufferred_input);
                buffer = match reading_result {
                    Ok([]) => break, // EOF
                    Ok(buffer2) => buffer2,
                    Err(_e) => return Err(_e),
                };
                if cache.writing {
                    cache.push_record(buffer)?;
                }
            } else {
                reading_result = cache.get_next_record();
                buffer = match reading_result {
                    Ok([]) => break, // EOF
                    Ok(buffer) => buffer,
                    Err(_e) => return Err(_e),
                };
            }
            example_num += 1;
            let mut prediction: f32 = 0.0;

            if prediction_model_delay == 0 {
                let update = match holdout_after_option {
                    Some(holdout_after) => !testonly && example_num < holdout_after,
                    None => !testonly,
                };
                if hogwild_training && update {
                    hogwild_trainer.digest_example(Vec::from(buffer));
                } else {
                    fbt.translate(buffer, example_num);
                    prediction = sharable_regressor.learn(&fbt.feature_buffer, &mut pb, update);
                }
            } else {
                fbt.translate(buffer, example_num);
                if example_num > predictions_after {
                    prediction = sharable_regressor.learn(&fbt.feature_buffer, &mut pb, false);
                }
                delayed_learning_fbs.push_back(fbt.feature_buffer.clone());
                if (prediction_model_delay as usize) < delayed_learning_fbs.len() {
                    let delayed_buffer = delayed_learning_fbs.pop_front().unwrap();
                    sharable_regressor.learn(&delayed_buffer, &mut pb, !testonly);
                }
            }

            if example_num > predictions_after {
                if output_pred_sto {
                    println!("{:.6}", prediction);
                }

                match predictions_file.as_mut() {
                    Some(file) => writeln!(file, "{:.6}", prediction)?,
                    None => {}
                }
            }
        }
        cache.write_finish()?;

        if hogwild_training {
            hogwild_trainer.block_until_workers_finished();
        }
        let elapsed = now.elapsed();
        log::info!("Elapsed: {:.2?} rows: {}", elapsed, example_num);

        if let Some(filename) = final_regressor_filename {
            persistence::save_sharable_regressor_to_filename(filename, &mi, &vw, sharable_regressor)
                .unwrap()
        }
    }

    Ok(())
}
