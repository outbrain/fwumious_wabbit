#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(non_snake_case)]
use std::error::Error;
use std::path::Path;
use std::fs::File;
use std::io;
use std::io::BufWriter;
use std::io::Write;
use std::io::BufRead;
use std::f32;
use std::collections::VecDeque;
use std::time::Instant;
use flate2::read::MultiGzDecoder;


mod vwmap;
mod parser;
mod model_instance;
mod feature_buffer;
mod regressor;
mod cmdline;
mod cache;
mod persistence;
mod serving;
mod optimizer;
mod version;

//use crate::regressor::RegressorTrait;


fn main() {
    match main2() {
        Err(e) => {println!("Global error: {:?}", e); std::process::exit(1)},
        Ok(()) => {}
    }    
}

fn main2() -> Result<(), Box<dyn Error>>  {
    // We'll parse once the command line into cl and then different objects will examine it
    let cl = cmdline::parse();

    // Where will we be putting perdictions (if at all)
    let mut predictions_file = match cl.value_of("predictions") {
        Some(filename) => Some(BufWriter::new(File::create(filename)?)),
        None => None      
    };

    let testonly = cl.is_present("testonly");


    let final_regressor_filename = cl.value_of("final_regressor");
    match final_regressor_filename {
        Some(filename) => {
            if !cl.is_present("save_resume") {
                return Err("You need to use --save_resume if you are using --final_regressor, so there is compatility with vowpal")?;
            }
            println!("final_regressor = {}", filename);
        },
        None => {}
    };
    
    
    /* setting up the pipeline, either from command line or from existing regressor */
    // we want heal-allocated objects here
    let vw: vwmap::VwNamespaceMap;
    let mut re: Box<dyn regressor::RegressorTrait>;
    let mi: model_instance::ModelInstance;


    if cl.is_present("daemon") {
        let filename = cl.value_of("initial_regressor").expect("Daemon mode only supports serving from --initial regressor");
        println!("initial_regressor = {}", filename);
        println!("WARNING: Command line model parameters will be ignored");
        let (mi2, vw2, re_fixed) = persistence::new_immutable_regressor_from_filename(filename)?;
        mi = mi2; vw = vw2;
        let mut se = serving::Serving::new(&cl, &vw, re_fixed, &mi)?;
        se.serve()?;
    } else {
        if let Some(filename) = cl.value_of("initial_regressor") {
            println!("initial_regressor = {}", filename);
            println!("WARNING: Command line model parameters will be ignored");
            let (mi2, vw2, re2) = persistence::new_regressor_from_filename(filename, testonly)?;
            mi = mi2; vw = vw2; re = re2;
        } else {
            // We load vw_namespace_map.csv just so we know all the namespaces ahead of time
            // This is one of the major differences from vowpal
            let input_filename = cl.value_of("data").expect("--data expected");
            let vw_namespace_map_filepath = Path::new(input_filename).parent().expect("Couldn't access path given by --data").join("vw_namespace_map.csv");
            vw = vwmap::VwNamespaceMap::new_from_csv_filepath(vw_namespace_map_filepath)?;
            mi = model_instance::ModelInstance::new_from_cmdline(&cl, &vw)?;
            re = regressor::get_regressor(&mi);
        };
        let input_filename = cl.value_of("data").expect("--data expected");
        let mut cache = cache::RecordCache::new(input_filename, cl.is_present("cache"), &vw);
        let mut fbt = feature_buffer::FeatureBufferTranslator::new(&mi);

        let predictions_after:u32 = match cl.value_of("predictions_after") {
            Some(examples) => examples.parse()?,
            None => 0
        };
        let prediction_model_delay:u32 = match cl.value_of("prediction_model_delay") {
            Some(delay) => delay.parse()?,
            None => 0
        };

        if prediction_model_delay != 0 && testonly {
            println!("--prediction_model_delay cannot be used with --testonly");
            return Err("Error")?
        }
        
        let mut delayed_learning_fbs: VecDeque<feature_buffer::FeatureBuffer> = VecDeque::with_capacity(prediction_model_delay as usize);

        // Setup Parser, is rust forcing this disguisting way to do it, or I just don't know the pattern?
        let input = File::open(input_filename)?;
        let mut aa;
        let mut bb;
        let mut bufferred_input: &mut dyn BufRead = match input_filename.ends_with(".gz") {
            true =>  { aa = io::BufReader::new(MultiGzDecoder::new(input)); &mut aa },
            false => { bb = io::BufReader::new(input); &mut bb}
        };

        let mut pa = parser::VowpalParser::new(&vw);

        let now = Instant::now();
        let mut example_num = 0;
        loop {

            let reading_result;
            let buffer:&[u32];
            if !cache.reading {
                reading_result = pa.next_vowpal(&mut bufferred_input);
                buffer = match reading_result {
                        Ok([]) => break, // EOF
                        Ok(buffer2) => buffer2,
                        Err(_e) => return Err("Error")?
                };
                if cache.writing {
                        cache.push_record(buffer)?;
                }
            } else {
                reading_result = cache.get_next_record();
                buffer = match reading_result {
                        Ok([]) => break, // EOF
                        Ok(buffer) => buffer,
                        Err(_e) => return Err("Error")?
                };
            }
            example_num += 1;
            fbt.translate(buffer);
            let mut prediction: f32 = 0.0;

            if prediction_model_delay == 0 {
                prediction = re.learn(&fbt.feature_buffer, !testonly, example_num);
            } else {
                if example_num > predictions_after {
                    prediction = re.learn(&fbt.feature_buffer, false, example_num);
                }
                delayed_learning_fbs.push_back(fbt.feature_buffer.clone());
                if (prediction_model_delay as usize) < delayed_learning_fbs.len() {
                    let delayed_buffer = delayed_learning_fbs.pop_front().unwrap();
                    re.learn(&delayed_buffer, !testonly, example_num);
                }
            } 
            
            if example_num > predictions_after {
                match predictions_file.as_mut() {
                    Some(file) =>  write!(file, "{:.6}\n", prediction)?,
                    None => {}
                }
            }
            
        }
        cache.write_finish()?;
        match final_regressor_filename {
            Some(filename) => persistence::save_regressor_to_filename(filename, &mi, &vw, re).unwrap(),
            None => {}
        }
    
        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?} rows: {}", elapsed, example_num);
    }

    Ok(())
}



