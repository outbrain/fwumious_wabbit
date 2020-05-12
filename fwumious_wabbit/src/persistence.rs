use backtrace::Backtrace;
use std::{mem,slice};

use std::str;
use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;

use std::io::{Read, Write};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Take;
use std::fs::File;
use std::io;
use std::fs;

use crate::model_instance;
use crate::regressor;
use crate::vwmap;

const REGRESSOR_HEADER_MAGIC_STRING: &[u8; 4] = b"FWRE";    // Fwumious Wabbit REgressor
const REGRESSOR_HEADER_VERSION:u32 = 3;

impl model_instance::ModelInstance {
    pub fn save_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        let serialized = serde_json::to_vec_pretty(&self)?;
        output_bufwriter.write_u64::<LittleEndian>(serialized.len() as u64)?;
        output_bufwriter.write(&serialized)?;
        Ok(())
    }
    pub fn new_from_buf(input_bufreader: &mut dyn io::Read) -> Result<model_instance::ModelInstance, Box<dyn Error>> {
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let mi:model_instance::ModelInstance = serde_json::from_reader(input_bufreader.take(len as u64))?;
        Ok(mi)
    }

}

impl vwmap::VwNamespaceMap {
    pub fn save_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        let serialized = serde_json::to_vec_pretty(&self.vw_source)?;
        output_bufwriter.write_u64::<LittleEndian>(serialized.len() as u64)?;
        output_bufwriter.write(&serialized)?;
        Ok(())
    }

    pub fn new_from_buf(input_bufreader: &mut dyn io::Read) -> Result<vwmap::VwNamespaceMap, Box<dyn Error>> {
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        let vw_source:vwmap::VwNamespaceMapSource = serde_json::from_reader(input_bufreader.take(len as u64))?;
        let vw = vwmap::VwNamespaceMap::new_from_source(vw_source)?;
        Ok(vw)
    }
}


impl regressor::Regressor {
    pub fn save_to_filename(&self, 
                        filename: &str, 
                        model_instance: &model_instance::ModelInstance,
                        vwmap: &vwmap::VwNamespaceMap) -> Result<(), Box<dyn Error>> {
        let mut output_bufwriter = &mut io::BufWriter::new(fs::File::create(filename).unwrap());
        regressor::Regressor::write_header(output_bufwriter)?;    
        vwmap.save_to_buf(output_bufwriter)?;
        model_instance.save_to_buf(output_bufwriter)?;
        self.write_weights_to_buf(output_bufwriter)?;
        Ok(())
    }
    
    pub fn new_from_filename(
                        filename: &str, 
                        ) -> Result<(model_instance::ModelInstance,
                                     vwmap::VwNamespaceMap,
                                     regressor::Regressor), Box<dyn Error>> {
        let mut input_bufreader = &mut io::BufReader::new(fs::File::open(filename).unwrap());
        regressor::Regressor::verify_header(input_bufreader).expect("Regressor header error");    
        let vw = vwmap::VwNamespaceMap::new_from_buf(input_bufreader).expect("Loading vwmap from regressor failed");
        let mi = model_instance::ModelInstance::new_from_buf(input_bufreader).expect("Loading model instance from regressor failed");
        let mut re = regressor::Regressor::new(&mi);
        re.overwrite_weights_from_buf(&mut input_bufreader)?;
        Ok((mi, vw, re))
    }


    pub fn write_weights_to_buf(&self, output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        // It's OK! I am a limo driver!
        output_bufwriter.write_u64::<LittleEndian>(self.weights.len() as u64)?;
        unsafe {
             let buf_view:&[u8] = slice::from_raw_parts(self.weights.as_ptr() as *const u8, 
                                              self.weights.len() *mem::size_of::<f32>());
             output_bufwriter.write(buf_view)?;
        }
        
        output_bufwriter.write_u64::<LittleEndian>(self.ffm_weights.len() as u64)?;
        unsafe {
             let buf_view:&[u8] = slice::from_raw_parts(self.ffm_weights.as_ptr() as *const u8, 
                                              self.ffm_weights.len() *mem::size_of::<f32>());
             output_bufwriter.write(buf_view)?;
        }
        
        
        Ok(())
    }
    pub fn overwrite_weights_from_buf(&mut self, input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        let len = input_bufreader.read_u64::<LittleEndian>()?;
        if len != self.weights.len() as u64 {
            return Err(format!("Lenghts of weights array in regressor file differ: got {}, expected {}", len, self.weights.len()))?;
        }
        unsafe {
            let mut buf_view:&mut [u8] = slice::from_raw_parts_mut(self.weights.as_mut_ptr() as *mut u8, 
                                             self.weights.len() *mem::size_of::<f32>());
            input_bufreader.read_exact(&mut buf_view)?;
        }

        let len = input_bufreader.read_u64::<LittleEndian>()?;
        if len != self.ffm_weights.len() as u64 {
            return Err(format!("Lenghts of ffm_weights array in regressor file differ: got {}, expected {}", len, self.ffm_weights.len()))?;
        }
        unsafe {
            let mut buf_view:&mut [u8] = slice::from_raw_parts_mut(self.ffm_weights.as_mut_ptr() as *mut u8, 
                                             self.ffm_weights.len() *mem::size_of::<f32>());
            input_bufreader.read_exact(&mut buf_view)?;
        }


        Ok(())
    }

    pub fn write_header(output_bufwriter: &mut dyn io::Write) -> Result<(), Box<dyn Error>> {
        // we will write magic string FWFW
        // And then 32 bit unsigned version of the cache
        output_bufwriter.write(REGRESSOR_HEADER_MAGIC_STRING)?;
        output_bufwriter.write_u32::<LittleEndian>(REGRESSOR_HEADER_VERSION)?;
        Ok(())
    }
    
    pub fn verify_header(input_bufreader: &mut dyn io::Read) -> Result<(), Box<dyn Error>> {
        let mut magic_string: [u8; 4] = [0;4];
        input_bufreader.read(&mut magic_string)?;
        if &magic_string != REGRESSOR_HEADER_MAGIC_STRING {
            return Err("Cache header does not begin with magic bytes FWFW")?;
        }
        
        let version = input_bufreader.read_u32::<LittleEndian>()?;
        if REGRESSOR_HEADER_VERSION != version {
            return Err(format!("Cache file version of this binary: {}, version of the cache file: {}", REGRESSOR_HEADER_VERSION, version))?;
        }
        Ok(())
    }        
    
    
}
    
