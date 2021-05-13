use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::io::prelude::*;
use std::fs;
use serde::{Serialize,Deserialize};//, Deserialize};


#[derive(Clone)]
pub struct VwNamespaceMap {
    pub num_namespaces: usize,
    pub map_name_to_index: HashMap <std::string::String, usize>,
    pub map_vwname_to_name: HashMap <Vec<u8>, std::string::String>,
    pub map_vwname_to_index: HashMap <Vec<u8>, usize>,
    pub vw_source: VwNamespaceMapSource,    // this is the source from which VwNamespaceMap can be constructed - for persistence
}

// this is serializible source from which VwNamespaceMap can be constructed
#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct VwNamespaceMapEntry {
    pub namespace_vwname: std::string::String,
    namespace_name: std::string::String,
    namespace_index: usize,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct VwNamespaceMapSource {
    pub entries: Vec<VwNamespaceMapEntry>,
}

impl VwNamespaceMap {
    pub fn new_from_source(vw_source: VwNamespaceMapSource)  -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut vw = VwNamespaceMap {
                                num_namespaces:0, 
                                map_name_to_index:HashMap::new(),
                                map_vwname_to_index:HashMap::new(),
                                map_vwname_to_name:HashMap::new(),
                                vw_source: vw_source,
                                };
        for vw_entry in &vw.vw_source.entries {
            //let record = result?;
            let name_str = &vw_entry.namespace_name;
            let vwname_str = &vw_entry.namespace_vwname;
            let i = &vw_entry.namespace_index;
            
            vw.map_name_to_index.insert(String::from(name_str), *i as usize);
            vw.map_vwname_to_index.insert(vwname_str.as_bytes().to_vec(), *i as usize);
            vw.map_vwname_to_name.insert(vwname_str.as_bytes().to_vec(), String::from(name_str));
            if *i > vw.num_namespaces {
                vw.num_namespaces = *i;
            } 
        }
        vw.num_namespaces += 1;
        Ok(vw)
    }

    pub fn new_from_csv_filepath(path: PathBuf) -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut input_bufreader = fs::File::open(&path).expect("Could not find vw_namespace_map.csv in input dataset directory");
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
            let vwname_str = &record[0];
            let name_str = &record[1];
            if vwname_str.as_bytes().len() != 1 {
                println!("Warning: multi-byte namespace names are not compatible with old style namespace arguments");
            }
            
            vw_source.entries.push(VwNamespaceMapEntry {
                namespace_vwname: vwname_str.to_string(),
                namespace_name: name_str.to_string(),
                namespace_index: i,
            });
//            println!("Char: {}, name: {}, index: {}", char, name_str, i);
        }
        VwNamespaceMap::new_from_source(vw_source)
    }

}


