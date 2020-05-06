use std::error::Error;


use crate::vwmap;
use crate::regressor;


pub struct Serving {
}


impl Serving {
    pub fn new() -> Result<Serving, Box<dyn Error>> {
        let s = Serving {};
        Ok(s)
    }
    
    pub fn serve(&mut self, vw: &vwmap::VwNamespaceMap, re: &regressor::Regressor)  -> Result<Serving, Box<dyn Error>> {
        loop {
        }
    }
    
}