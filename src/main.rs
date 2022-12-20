#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(non_snake_case)]
#![allow(redundant_semicolons)]
use flate2::read::MultiGzDecoder;
use std::collections::VecDeque;
use std::error::Error;
use std::f32;
use std::fs::File;
use std::fs;
use std::io;
use std::io::BufRead;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[macro_use]
extern crate nom;

//use std::collections::HashMap;
use rustc_hash::FxHashMap;
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

fn main() {
    match main2() {
        Err(e) => {
            println!("Global error: {:?}", e);
            std::process::exit(1)
        }
        Ok(()) => {}
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
    let vw: vwmap::VwNamespaceMap;
    vw = vwmap::VwNamespaceMap::new_from_csv_filepath(vw_namespace_map_filepath)?;
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
            buffer = match reading_result {
                Ok([]) => break, // EOF
                Ok(buffer) => buffer,
                Err(_e) => return Err(_e),
            };
        }
        example_num += 1;
    }
    cache.write_finish()?;
    Ok(())
}


fn main2() -> Result<(), Box<dyn Error>> {
    // We'll parse once the command line into cl and then different objects will examine it
    let cl = cmdline::parse();
    if cl.is_present("build_cache_without_training") {
        return build_cache_without_training(cl)
    }
    // Where will we be putting perdictions (if at all)
    let mut predictions_file = match cl.value_of("predictions") {
        Some(filename) => Some(BufWriter::new(File::create(filename)?)),
        None => None,
    };

    let testonly = cl.is_present("testonly");
	
    let final_regressor_filename = cl.value_of("final_regressor");
    match final_regressor_filename {
        Some(filename) => {
            if !cl.is_present("save_resume") {
                return Err("You need to use --save_resume with --final_regressor, for vowpal wabbit compatibility")?;
            }
            println!("final_regressor = {}", filename);
        }
        None => {}
    };

    let inference_regressor_filename = cl.value_of("convert_inference_regressor");
    match inference_regressor_filename {
        Some(filename1) => {
            println!("inference_regressor = {}", filename1);
        }
        None => {}
    };

    /* setting up the pipeline, either from command line or from existing regressor */
    // we want heal-allocated objects here

    if cl.is_present("daemon") {
        let filename = cl
            .value_of("initial_regressor")
            .expect("Daemon mode only supports serving from --initial regressor");
        println!("initial_regressor = {}", filename);
        let (mi2, vw2, re_fixed) = persistence::new_regressor_from_filename(filename, true, Option::Some(&cl))?;
		
        let mut se = serving::Serving::new(&cl, &vw2, Box::new(re_fixed), &mi2)?;
        se.serve()?;
    } else if cl.is_present("convert_inference_regressor") {
        let filename = cl
            .value_of("initial_regressor")
            .expect("Convert mode requires --initial regressor");
        let (mut mi2, vw2, re_fixed) = persistence::new_regressor_from_filename(filename, true, Option::Some(&cl))?;
        mi2.optimizer = model_instance::Optimizer::SGD;
        match inference_regressor_filename {
            Some(filename1) => {
                persistence::save_regressor_to_filename(filename1, &mi2, &vw2, re_fixed).unwrap()
            }
            None => {}
        }
    } else {
		
        let vw: vwmap::VwNamespaceMap;
        let mut re: regressor::Regressor;
        let mi: model_instance::ModelInstance;

        if let Some(filename) = cl.value_of("initial_regressor") {
			
            println!("initial_regressor = {}", filename);
            (mi, vw, re) = persistence::new_regressor_from_filename(filename, testonly, Option::Some(&cl))?;

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
        };

        let input_filename = cl.value_of("data").expect("--data expected");
        let mut cache = cache::RecordCache::new(input_filename, cl.is_present("cache"), &vw);
        let mut fbt = feature_buffer::FeatureBufferTranslator::new(&mi);

        let predictions_after: u64 = match cl.value_of("predictions_after") {
            Some(examples) => examples.parse()?,
            None => 0,
        };

		let mut max_freq_hash_storage: FxHashMap<u32, u32> = FxHashMap::default();
		let mut max_hashed_index: u32 = 0;
		
		if cl.is_present("rehash_prior_counts"){

			let hash_storage_input = cl.value_of("rehash_prior_counts").expect("Prior count values expected ..").to_string();
			let hash_space: String = fs::read_to_string(hash_storage_input)?;
			let mut subsequent_index: u32 = 0;
			let right_shift_constant: u32 = (mi.ffm_fields.len() as u32) * (mi.ffm_k as u32);
			let hash_frequency_lower_bound: u32 = cl.value_of("hash_freq_lower_bound").expect("Please provide the lower count bound for rehashing.").parse().unwrap();
				
			for line in hash_space.lines() {
				let instruction_sub_parts = line.split("\t");
				let vec: Vec<&str> = instruction_sub_parts.collect();
				let hash_entry: u32 = vec[0].parse().unwrap();
				let hash_value: u32 = vec[1].parse().unwrap();
				
				// only insert ones > certain frequency
				if hash_value > hash_frequency_lower_bound {
					subsequent_index += 1;
					let new_value = subsequent_index * right_shift_constant;
					max_freq_hash_storage.insert(hash_entry, new_value);
					max_hashed_index = new_value;
				}
			}
		}

        let holdout_after_option: Option<u64> =
            cl.value_of("holdout_after").map(|s| s.parse().unwrap());

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
            fbt.translate(buffer, example_num);

			// Increment counts if specified as part of CLI
			if cl.is_present("count_frequent_hashes"){
				fbt.increment_common_hash(&mut max_freq_hash_storage);
				continue
			}
			
			// Conduct the rehashing step based on priors if specified
			if cl.is_present("rehash_prior_counts") {
				fbt.max_freq_rehash(&mut max_freq_hash_storage, max_hashed_index);
			} else {
				fbt.apply_generic_mask();
			}
			
            let mut prediction: f32 = 0.0;

            if prediction_model_delay == 0 {
                let update = match holdout_after_option {
                    Some(holdout_after) => !testonly && example_num < holdout_after,
                    None => !testonly,
                };
                prediction = re.learn(&fbt.feature_buffer, update);
            } else {
                if example_num > predictions_after {
                    prediction = re.learn(&fbt.feature_buffer, false);
                }
                delayed_learning_fbs.push_back(fbt.feature_buffer.clone());
                if (prediction_model_delay as usize) < delayed_learning_fbs.len() {
                    let delayed_buffer = delayed_learning_fbs.pop_front().unwrap();
                    re.learn(&delayed_buffer, !testonly);
                }
            }

            if example_num > predictions_after {
                match predictions_file.as_mut() {
                    Some(file) => write!(file, "{:.6}\n", prediction)?,
                    None => {}
                }
            }
        }
        cache.write_finish()?;

		if cl.is_present("count_frequent_hashes"){

			let mut vec_holder = Vec::<String>::new();
			let mut all_hash_intervals: Vec<Vec<u32>> = Vec::new();
			let right_shift_constant: u32 = (mi.ffm_fields.len() as u32) * (mi.ffm_k as u32);
			
			for (key, value) in max_freq_hash_storage.into_iter() {
				let some_string = format!("{}\t{}", key.to_string(), value.to_string());
				vec_holder.push(some_string.to_string());
			}
			let final_hash_dump = vec_holder.join("\n");
			let out_file_arg = cl.value_of("count_frequent_hashes").expect("Out file for hashes not given.");
			println!("Writing frequent hashes to {} ..", out_file_arg);
			fs::write(out_file_arg, final_hash_dump).expect("Unable to write file");
			
		} else {
			match final_regressor_filename {
				Some(filename) => {
					persistence::save_regressor_to_filename(filename, &mi, &vw, re).unwrap()
				}
				None => {}
			}
		}
		
        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?} rows: {}", elapsed, example_num);

    }

    Ok(())
}
