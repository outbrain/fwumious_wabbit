#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
use std::io::Error as IOError;
use std::io::ErrorKind;
use std::error::Error;
use std::env;
use std::path::Path;
use std::process::exit;
use std::fs::File;
use std::io;
use std::io::BufWriter;
use std::fmt;
use std::str;
use std::io::Write;
use std::io::BufRead;
use std::f32;
use std::cmp::min;
//use std::io::ErrorKind;
//use std::iter::Peekable;
use fasthash::xx;
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
    let mut vw: vwmap::VwNamespaceMap;
    let mut re: regressor::Regressor;
    let mut mi: model_instance::ModelInstance;
    
    if let Some(filename) = cl.value_of("initial_regressor") {
            println!("initial_regressor = {}", filename);
            println!("WARNING: Command line model parameters will be ignored");
            let (mi2, vw2, re2) = regressor::Regressor::new_from_filename(filename)?;
            mi = mi2; vw = vw2; re = re2;
    } else {
            // We load vw_namespace_map.csv just so we know all the namespaces ahead of time
            // This is one of the major differences from vowpal
            let input_filename = cl.value_of("data").expect("--data expected");
            let vw_namespace_map_filepath = Path::new(input_filename).parent().expect("--data expected").join("vw_namespace_map.csv");
            vw = vwmap::VwNamespaceMap::new_from_csv_filepath(vw_namespace_map_filepath)?;
            mi = model_instance::ModelInstance::new_from_cmdline(&cl, &vw)?;
            re = regressor::Regressor::new(&mi);
    };
    if cl.is_present("daemon") {
        let mut se = serving::Serving::new()?;
        let s = se.serve(&vw, &re);
    }

    let input_filename = cl.value_of("data").expect("--data expected");
    let mut cache = cache::RecordCache::new(input_filename, cl.is_present("cache"), &vw);
    let mut fb = feature_buffer::FeatureBuffer::new(&mi);

    // Setup Parser, is rust forcing this disguisting way to do it, or I just don't know the pattern?
    let input = File::open(input_filename)?;
    let mut aa;
    let mut bb;
    let mut bufferred_input: &mut dyn BufRead = match input_filename.ends_with(".gz") {
        true =>  { aa = io::BufReader::new(MultiGzDecoder::new(input)); &mut aa },
        false => { bb = io::BufReader::new(input); &mut bb}
    };

    let mut pa = parser::VowpalParser::new(bufferred_input, &vw);


    let now = Instant::now();
    let mut i = 0;
    loop {
        let reading_result;
        let mut buffer:&[u32];
        if !cache.reading {
            reading_result = pa.next_vowpal();
            buffer = match reading_result {
                    Ok([]) => break, // EOF
                    Ok(buffer2) => buffer2,
                    Err(e) => return Err("Error")?
            };
            if cache.writing {
                    cache.push_record(buffer)?;
            }
        } else {
            reading_result = cache.get_next_record();
            buffer = match reading_result {
                    Ok([]) => break, // EOF
                    Ok(buffer) => buffer,
                    Err(e) => return Err("Error")?
            };
        }

        fb.translate_vowpal(buffer);
        let p = re.learn(&fb.output_buffer, true, i);
        match predictions_file.as_mut() {
            Some(file) =>  write!(file, "{:.6}\n", p)?,
            None => {}
        };
        i += 1;
    }
    cache.write_finish()?;
    match final_regressor_filename {
        Some(filename) => re.save_to_filename(filename, &mi, &vw)?,
        None => {}
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?} rows: {}", elapsed, i);


    Ok(())
}




