use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::io::prelude::*;
use std::fs;
use serde::{Serialize,Deserialize};//, Deserialize};



pub struct VwNamespaceMap {
    pub num_namespaces: usize,
    pub map_name_to_index: HashMap <std::string::String, usize>,
    pub map_char_to_name: HashMap <char, std::string::String>,
    pub map_char_to_index: HashMap <char, usize>,
    pub lookup_char_to_index: [usize; 256], 
    pub vw_source: VwNamespaceMapSource,    // this is the source from which VwNamespaceMap can be constructed - for persistence
}

// this is serializible source from which VwNamespaceMap can be constructed
#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct VwNamespaceMapEntry {
    namespace_char: char,
    namespace_name: std::string::String,
    namespace_index: usize,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct VwNamespaceMapSource {
    entries: Vec<VwNamespaceMapEntry>,
}

impl VwNamespaceMap {
    pub fn new_from_source(vw_source: VwNamespaceMapSource)  -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut vw = VwNamespaceMap {
                                num_namespaces:0, 
                                map_name_to_index:HashMap::new(),
                                map_char_to_index:HashMap::new(),
                                map_char_to_name:HashMap::new(),
                                lookup_char_to_index: [0; 256],
                                vw_source: vw_source,
                                };
        for vw_entry in &vw.vw_source.entries {
            //let record = result?;
            let name_str = &vw_entry.namespace_name;
            let char = &vw_entry.namespace_char;
            let i = &vw_entry.namespace_index;
            
            vw.map_name_to_index.insert(String::from(name_str), *i as usize);
            vw.map_char_to_index.insert(*char, *i as usize);
            vw.map_char_to_name.insert(*char, String::from(name_str));
            vw.lookup_char_to_index[*char as usize] = *i as usize;
            if *i > vw.num_namespaces {
                vw.num_namespaces = *i;
            } 
            
        }
        Ok(vw)
    }

    pub fn new_from_csv_filepath(path: PathBuf) -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut input_bufreader = fs::File::open(&path)?;
        let mut s = String::new();
        input_bufreader.read_to_string(&mut s)?;
        VwNamespaceMap::new(&s)   
    }


    pub fn new(data: &str) -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(data.as_bytes());
        let mut vw_source = VwNamespaceMapSource { entries: vec![]};
        for (i, record_w) in rdr.records().enumerate() {
            let record = record_w?;
            let char_str = &record[0];
            let name_str = &record[1];
            if char_str.len() != 1 {
                panic!("Can't decode {:?}", record);
            }
            let char = char_str.chars().next().unwrap();
            
            vw_source.entries.push(VwNamespaceMapEntry {
                namespace_char: char,
                namespace_name: name_str.to_string(),
                namespace_index: i,
            });
        }
        
        
        VwNamespaceMap::new_from_source(vw_source)
    }

}


