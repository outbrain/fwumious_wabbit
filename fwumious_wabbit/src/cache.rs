
use std::{mem,slice};
use std::io;
use std::fs;
use std::error::Error;
use std::path;
use flate2::write::GzEncoder;
use flate2::Compression;
use flate2::read::GzDecoder;

use crate::parser;

const HEADER_MAGIC_STRING: &[u8; 4] = b"FWCA";    // Fwumious Wabbit CAche
const HEADER_VERSION:u32 = 2;

const READBUF_LEN:usize = 1024*100;


pub struct RecordCache {
    output_bufwriter: Box<dyn io::Write>,
    input_bufreader: Box<dyn io::Read>,
    temporary_filename: String,
    final_filename: String,
    pub writing: bool,
    pub reading: bool,
//    pub output_buffer: Vec<u32>,
    pub byte_buffer: Vec<u8>,//[u8; READBUF_LEN],
    start_pointer: usize,
    end_pointer: usize,
    total_read: usize,
}


impl RecordCache {
    pub fn new(input_filename: &str, enabled: bool) -> RecordCache {
        let temporary_filename: String;
        let final_filename: String;
        let gz: bool;
        temporary_filename = format!("{}.fwcache.writing", input_filename);
        final_filename = format!("{}.fwcache", input_filename);
        if !input_filename.ends_with("gz") {
            gz = false;
        } else
        {
            gz = true;
        }
        
        let mut rc = RecordCache {
            output_bufwriter: Box::new(io::BufWriter::new(io::sink())),
            input_bufreader: Box::new(io::empty()),
            temporary_filename: temporary_filename.to_string(),
            final_filename: final_filename.to_string(),
            writing: false,
            reading: false,
            byte_buffer: Vec::new(),
            start_pointer: 0,
            end_pointer: 0,
            total_read: 0,
        };
        
        if enabled {
            if path::Path::new(&final_filename).exists() {
                rc.reading = true;
                if !gz {
                    rc.input_bufreader = Box::new(fs::File::open(&final_filename).unwrap());
                } else {
                    rc.input_bufreader = Box::new(GzDecoder::new(fs::File::open(&final_filename).unwrap()));
                }
                println!("using cache_file = {}", final_filename );
                println!("ignoring text input in favor of cache input");
                rc.byte_buffer.resize(READBUF_LEN, 0);
                match rc.verify_header() {
                    Ok(()) => {},
                    Err(e) => {
                        println!("Couldn't use the existing cache file");
                        rc.reading = false;
                    }
                }
            }
            if !rc.reading {
                rc.writing = true;
                println!("creating cache file = {}", final_filename );
                if !gz {
                    rc.output_bufwriter = Box::new(io::BufWriter::new(fs::File::create(temporary_filename).unwrap()));
                } else {
                    rc.output_bufwriter = Box::new(io::BufWriter::new(GzEncoder::new(fs::File::create(temporary_filename).unwrap(),
                                                                    Compression::fast())));
                }
                rc.write_header().unwrap();
            }
        }        
        rc
    }
    
    pub fn push_record(&mut self, vp: &parser::VowpalParser) -> Result<(), Box<dyn Error>> {
        if self.writing {
            let element_size = mem::size_of::<u32>();
            let mut vv:&[u8];
            unsafe { 
                vv = slice::from_raw_parts(vp.output_buffer.as_ptr() as *const u8, 
                                            vp.output_buffer.len() * element_size) ;
                self.output_bufwriter.write(&vv)?;
            }
        }
        Ok(())
    }
    
    pub fn write_finish(&mut self)  -> Result<(), Box<dyn Error>> {
        if self.writing {
            self.output_bufwriter.flush()?;
            fs::rename(&self.temporary_filename, &self.final_filename)?;
        }
        Ok(())
    }

    pub fn write_header(&mut self) -> Result<(), Box<dyn Error>> {
        // we will write magic string FWFW
        // And then 32 bit unsigned version of the cache
        self.output_bufwriter.write(HEADER_MAGIC_STRING)?;
        self.output_bufwriter.write(&HEADER_VERSION.to_le_bytes())?;
        Ok(())
    }

    pub fn verify_header(&mut self) -> Result<(), Box<dyn Error>> {
        let mut magic_string: [u8; 4] = [0;4];
        self.input_bufreader.read(&mut magic_string)?;
        if &magic_string != HEADER_MAGIC_STRING {
            return Err("Cache header does not begin with magic bytes FWFW")?;
        }
        
        let mut version_bytes: [u8; 4] = [0;4];
        self.input_bufreader.read(&mut version_bytes)?;
        if HEADER_VERSION != u32::from_le_bytes(version_bytes) {
            println!("Cache file is of different version than supported by this fw");
            return Err("Different cache version")?;
        }
        Ok(())
    }
    

    pub fn get_next_record(&mut self) -> Result<&[u32], Box<dyn Error>> {
        if !self.reading {
            return Err("next_recrod() called on reading cache, when not opened in reading mode")?;
        }
        unsafe { 
            // We're going to cast another view over the data, so we can read it as u32
            let buf_view:&[u32];
            buf_view = slice::from_raw_parts(self.byte_buffer.as_ptr() as *const u32, READBUF_LEN) ;
            
            loop {
                // Classical buffer strategy:
                // Return if you have full record in buffer,
                // Otherwise shift the buffer and backfill it
                if self.end_pointer - self.start_pointer >= 4 {
                    let record_len = buf_view[self.start_pointer /4 ] as usize;
                    if self.start_pointer + record_len * 4 <= self.end_pointer {
                        let ret_buf = &buf_view[self.start_pointer/4..self.start_pointer/4 + record_len];
                        self.start_pointer += record_len * 4;
                        return Ok(ret_buf);
                    }
                } 
                self.byte_buffer.copy_within(self.start_pointer..self.end_pointer, 0);
                self.end_pointer -= self.start_pointer;
                self.start_pointer = 0;

                let read_len = match self.input_bufreader.read(&mut self.byte_buffer[self.end_pointer..READBUF_LEN])  {
                    Ok(n) => n,
                    Err(e) => Err(e)?          
                };
                if read_len == 0 {
                    return Ok(&[])
                }
                self.end_pointer += read_len;
                self.total_read += read_len;
            }            
        }
    }
    
    
    
    
}