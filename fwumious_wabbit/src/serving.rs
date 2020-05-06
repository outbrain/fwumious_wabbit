use std::error::Error;
use std::net;
use std::io::{BufReader,BufWriter};
use std::io::Write;

use crate::parser;
use crate::vwmap;
use crate::regressor;
use crate::feature_buffer;
use crate::model_instance;



pub struct Serving {
    listening_interface: String,
}




impl Serving {
    pub fn new<'a>(cl: &clap::ArgMatches<'a>) -> Result<Serving, Box<dyn Error>> {
        let port = match cl.value_of("port") {
            Some(port) => port.parse().expect("Port should be integer"),
            None => 26542
        };
        let listening_interface = format!("127.0.0.1:{}", port);
        println!("Starting to listen on {}", listening_interface); 
        let s = Serving {
            listening_interface:listening_interface.to_string(),
        };
        Ok(s)
    }
    
    fn handle_client(&self, stream: net::TcpStream, 
                        vw: &vwmap::VwNamespaceMap, 
                        re: &regressor::Regressor, 
                        mi: &model_instance::ModelInstance) -> Result<(), Box<dyn Error>> {
        let mut reader = BufReader::new(&stream);
        let mut writer = BufWriter::new(&stream);
        let mut pa = parser::VowpalParser::new(&mut reader, vw);
        let mut fb = feature_buffer::FeatureBuffer::new(mi);

        println!("New connection");
        let mut i = 0u32;
        loop {
            let reading_result = pa.next_vowpal();
            let mut buffer: &[u32] = match reading_result {
                    Ok([]) => return Ok(()), // EOF
                    Ok(buffer2) => buffer2,
                    Err(e) => return Err("Error")?
            };
            fb.translate_vowpal(buffer);
            let p = re.predict(&(fb.output_buffer), i);
            writer.write_all(format!("{:.6}\n", p).as_bytes())?;
            writer.flush()?; // This is definitely not the most efficient way of doing this, but let's make it perdictable first, fast second
            i += 1;
        }

        // ...
    }


    pub fn serve(&mut self, vw: &vwmap::VwNamespaceMap, re: &regressor::Regressor, mi: &model_instance::ModelInstance)  -> Result<(), Box<dyn Error>> {
        let listener = net::TcpListener::bind(&self.listening_interface)?;
        for stream in listener.incoming() {
            self.handle_client(stream?, vw, re, mi)?;
        }
        Ok(())
    }
    
}