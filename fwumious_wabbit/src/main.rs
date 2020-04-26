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

//use std::io::ErrorKind;
//use std::iter::Peekable;
use fasthash::xx;
use std::time::Instant;

use flate2::read::MultiGzDecoder;


mod vwmap;
mod record_reader;
mod model_instance;
mod feature_buffer;
mod regressor;
mod cmdline;

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
    match m2() {
        Err(e) => println!("{:?}", e),
        Ok(()) => {}
    }    
}

fn m2() -> Result<(), Box<dyn Error>>  {
    // Statements here are executed when the compiled binary is called
    let cl = cmdline::parse();
    let input_filename = cl.value_of("data").unwrap();
    // This is a bit of implicit logic
    let vw_namespace_map_filepath = Path::new(input_filename).parent().unwrap().join("vw_namespace_map.csv");
    let vw = vwmap::get_global_map_from_json(vw_namespace_map_filepath)?;

    let predictions_filename = cl.value_of("predictions");
    let mut predictions_file = match cl.value_of("predictions") {
        Some(filename) => {
            let mut file = File::create(filename)?;
            Some(BufWriter::new(file))
            },
        None => None      
    };
    let input = File::open(input_filename)?;
    let mut aa;
    let mut bb;
    let mut bufferred_input: &mut dyn BufRead = match input_filename.ends_with(".gz") {
        true =>  { aa = io::BufReader::new(MultiGzDecoder::new(input).unwrap()); &mut aa },
        false => { bb = io::BufReader::new(input); &mut bb}
    };
    let mut rr = record_reader::RecordReader::new(bufferred_input, &vw);
//    let mut mi = model_instance::ModelInstance::new_from_file("andraz-x2.json", &vw)?;
    let mut mi = model_instance::ModelInstance::new_from_cmdline(cl, &vw)?;
    let mut fb = feature_buffer::FeatureBuffer::new(&mi);
    let mut re = regressor::Regressor::new(&mi);
    

    let now = Instant::now();
    let mut i = 0;
    loop {
        let rx = rr.next_vowpal();
        match rx {
            record_reader::NextRecordResult::End => {break;},
            record_reader::NextRecordResult::Error => println!("Error from parsing records, row: {}", i),
            record_reader::NextRecordResult::Ok => ()
        }
//        println!("{:?}", rr.output_buffer);
        fb.translate_vowpal(&rr.output_buffer);
//        println!("{:?}", &fb.output_buffer);
        let p = re.learn(&fb.output_buffer, true);
        match predictions_file.as_mut() {
            Some(file) => {write!(file, "{:.6}\n", p)?;},
            None => {}
        };
        i += 1;
  //      println!("{}", p);
//        rr.print();
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?} rows: {}", elapsed, i);


//    println!("{:?}", args);
  //  println!("{}", input_filename);
    
    // Print text to the console
    //println!("Hello World!");

//    let example: i32 = 1;
    
    Ok(())

}
