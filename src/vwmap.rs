use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::io::prelude::*;
use std::fs;
use serde::{Serialize,Deserialize};
use std::io::ErrorKind;
use std::io::Error as IOError;


#[derive(Clone)]
pub struct VwNamespaceMap {
    pub num_namespaces: usize,
    pub map_verbose_to_index: HashMap <std::string::String, usize>,
    pub map_vwname_to_name: HashMap <Vec<u8>, std::string::String>,
    pub map_vwname_to_index: HashMap <Vec<u8>, usize>,
    pub map_index_to_save_as_float: [bool; 256],
    pub vw_source: VwNamespaceMapSource,    // this is the source from which VwNamespaceMap can be constructed - for persistence
}

// this is serializible source from which VwNamespaceMap can be constructed
#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct VwNamespaceMapEntry {
    namespace_char: char,
    pub namespace_name: std::string::String,
    pub namespace_index: u32,
    namespace_save_as_float: bool, 
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct VwNamespaceMapSource {
    pub float_namespaces_skip_prefix: u32,
    pub entries: Vec<VwNamespaceMapEntry>,
}

impl VwNamespaceMap {
    pub fn new_from_source(vw_source: VwNamespaceMapSource)  -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut vw = VwNamespaceMap {
                                num_namespaces:0, 
                                map_verbose_to_index:HashMap::new(),
                                map_index_to_save_as_float: [false;256],
                                map_vwname_to_index: HashMap::new(),
                                map_vwname_to_name: HashMap::new(),
                                vw_source: vw_source,
                                };

        for vw_entry in &vw.vw_source.entries {
            //let record = result?;
            let name_str = &vw_entry.namespace_verbose;
            let vwname_str = &vw_entry.namespace_vwname;
            let i = &vw_entry.namespace_index;
            
            vw.map_verbose_to_index.insert(String::from(name_str), *i as usize);
            vw.map_vwname_to_index.insert(vwname_str.as_bytes().to_vec(), *i as usize);
            vw.map_vwname_to_name.insert(vwname_str.as_bytes().to_vec(), String::from(name_str));
            vw.map_index_to_save_as_float[*i as usize] = vw_entry.namespace_save_as_float;
            if *i > vw.num_namespaces as u32 {
                vw.num_namespaces = *i as usize;
            } 
        }
        vw.num_namespaces += 1;
        Ok(vw)
    }

    pub fn new_from_csv_filepath(path: PathBuf, float_namespaces: (Vec<String>, u32)) -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut input_bufreader = fs::File::open(&path).expect("Could not find vw_namespace_map.csv in input dataset directory");
        let mut s = String::new();
        input_bufreader.read_to_string(&mut s)?;
        VwNamespaceMap::new(&s, float_namespaces)
    }

    pub fn new(data: &str, float_namespaces: (Vec<String>, u32)) -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(data.as_bytes());
        let mut vw_source = VwNamespaceMapSource { entries: vec![], float_namespaces_skip_prefix: 0};
        for (i, record_w) in rdr.records().enumerate() {
            let record = record_w?;
            let vwname_str = &record[0];
            let name_str = &record[1];
            if vwname_str.as_bytes().len() != 1 {
                println!("Warning: multi-byte namespace names are not compatible with old style namespace arguments");
            }
            
            vw_source.entries.push(VwNamespaceMapEntry {
                namespace_vwname: vwname_str.to_string(),
                namespace_verbose: name_str.to_string(),
                namespace_index: i as u32,
                namespace_save_as_float: false,
            });
        }

        // It is a bit fugly that we need this passed out-of-band from command line parameters
        // But we need to know which input params to mark with 'save_as_float' flag
        for float_namespace_verbose in float_namespaces.0 {
            // create an list of indexes from list of verbose namespaces
            // Find index:
            let from_index:Vec<&VwNamespaceMapEntry> = vw_source.entries.iter().filter(|e| e.namespace_verbose == float_namespace_verbose).collect(); 
            if from_index.len() != 1 {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown or ambigious verbose namespace passed by --float_namespaces: {}", float_namespace_verbose))))
            }
            let from_index = from_index[0].namespace_index;
            //println!("From index {}", from_index);
            vw_source.entries[from_index as usize].namespace_save_as_float = true;
        }
        vw_source.float_namespaces_skip_prefix = float_namespaces.1;

        VwNamespaceMap::new_from_source(vw_source)
    }

}
