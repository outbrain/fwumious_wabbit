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

const RECBUF_LEN:usize = 4096 * 2;

/* 
organization of records buffer 
(offset u16, len u16)[number_of_features]
(u32)[dynamic_number_of_hashes] 
*/
/*
fn read_into_records_buffer(lines: std::io::BufReader<File>, rb: &RecordsBuffer) -> Result<(), Box<dyn Error>> {
    
    

    Ok(())
}

*/
// This is the main function
fn main() {
    match main2() {
        Err(e) => {println!("{:?}", e); std::process::exit(1)},
        Ok(()) => {}
    }    
}

fn main2() -> Result<(), Box<dyn Error>>  {
    // We'll parse once the command line into cl and then different objects will examine it
//    println!("AA {:?}", env::args());
    let cl = cmdline::parse();
    let input_filename = cl.value_of("data").unwrap();
    
    // We do load vw_namespace_map.csv just so we know all the namespaces ahead of time
    let vw_namespace_map_filepath = Path::new(input_filename).parent().unwrap().join("vw_namespace_map.csv");
    let vw = vwmap::get_global_map_from_json(vw_namespace_map_filepath)?;

    // Where will we be putting perdictions (if at all)
    let predictions_filename = cl.value_of("predictions");
    let mut predictions_file = match cl.value_of("predictions") {
        Some(filename) => {
            let mut file = File::create(filename)?;
            Some(BufWriter::new(file))
            },
        None => None      
    };
    
    // Setup Parser, is rust forcing this disguisting way to do it, or I just don't know the pattern?
    let input = File::open(input_filename)?;
    let mut aa;
    let mut bb;
    let mut bufferred_input: &mut dyn BufRead = match input_filename.ends_with(".gz") {
        true =>  { aa = io::BufReader::new(MultiGzDecoder::new(input)); &mut aa },
        false => { bb = io::BufReader::new(input); &mut bb}
    };
    let mut rr = parser::VowpalParser::new(bufferred_input, &vw);

    // Setup cache
       
    let mut cache = cache::RecordCache::new(input_filename, cl.is_present("cache"));

//    let mut mi = model_instance::ModelInstance::new_from_file("andraz-x2.json", &vw)?;
    let mut mi = model_instance::ModelInstance::new_from_cmdline(&cl, &vw)?;
    let mut fb = feature_buffer::FeatureBuffer::new(&mi);
    let mut re = regressor::Regressor::new(&mi);
    

    let now = Instant::now();
    let mut i = 0;
    loop {
        let reading_result;
        let mut buffer:&[u32];
        if !cache.reading {
            reading_result = rr.next_vowpal();
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
        let p = re.learn(&fb.output_buffer, true);
        match predictions_file.as_mut() {
              Some(file) => {
                  write!(file, "{:.6}\n", p)?;
//                    let printed = float_to_string_buffer.format(finalized_p);
//                    file.write(&printed[0..min(printed.len(), 8)].as_bytes());
//                    write!(file, "\n");
              },
            None => {}
        };
        i += 1;
    }
    cache.write_finish()?;
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?} rows: {}", elapsed, i);


    Ok(())
}




