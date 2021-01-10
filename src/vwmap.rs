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
    pub num_namespaces: u32,
    pub map_name_to_index: HashMap <std::string::String, u32>,
    pub map_char_to_name: HashMap <char, std::string::String>,
    pub map_char_to_index: HashMap <char, u32>,
    pub lookup_char_to_index: [u32; 256], 
    pub lookup_char_to_save_as_float: [bool; 256],
    pub vw_source: VwNamespaceMapSource,    // this is the source from which VwNamespaceMap can be constructed - for persistence
}

// this is serializible source from which VwNamespaceMap can be constructed
#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct VwNamespaceMapEntry {
    namespace_char: char,
    namespace_name: std::string::String,
    namespace_index: u32,
    namespace_save_as_float: bool, 
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct VwNamespaceMapSource {
    entries: Vec<VwNamespaceMapEntry>,
    pub float_namespaces_skip_prefix: u32,
}

impl VwNamespaceMap {
    pub fn new_from_source(vw_source: VwNamespaceMapSource)  -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut vw = VwNamespaceMap {
                                num_namespaces:0, 
                                map_name_to_index:HashMap::new(),
                                map_char_to_index:HashMap::new(),
                                map_char_to_name:HashMap::new(),
                                lookup_char_to_index: [0; 256],
                                lookup_char_to_save_as_float: [false; 256],
                                vw_source: vw_source,
                                };

        for vw_entry in &vw.vw_source.entries {
            //let record = result?;
            let name_str = &vw_entry.namespace_name;
            let char = vw_entry.namespace_char;
            let i = vw_entry.namespace_index;
            
            vw.map_name_to_index.insert(String::from(name_str), i as u32);
            vw.map_char_to_index.insert(char, i as u32);
            vw.map_char_to_name.insert(char, String::from(name_str));
            vw.lookup_char_to_index[char as usize] = i as u32;
            vw.lookup_char_to_save_as_float[char as usize] = vw_entry.namespace_save_as_float;
            if i > vw.num_namespaces {
                vw.num_namespaces = i;
            } 
        }
        vw.num_namespaces += 1;
        Ok(vw)
    }

    pub fn new_from_csv_filepath(path: PathBuf, float_namespaces: (String, u32)) -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut input_bufreader = fs::File::open(&path).expect("Could not find vw_namespace_map.csv in input dataset directory");
        let mut s = String::new();
        input_bufreader.read_to_string(&mut s)?;
        VwNamespaceMap::new(&s, float_namespaces)
    }

    pub fn new(data: &str, float_namespaces: (String, u32)) -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(data.as_bytes());
        let mut vw_source = VwNamespaceMapSource { entries: vec![], float_namespaces_skip_prefix: 0};
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
                namespace_index: i as u32,
                namespace_save_as_float: false,
            });
//            println!("Char: {}, name: {}, index: {}", char, name_str, i);
        }

        // It is a bit fugly that we need this passed out-of-band from command line parameters
        // But we need to know which input params to mark with 'save_as_float' flag
        for char in float_namespaces.0.chars() {
            // create an list of indexes dfrom list of namespace chars
            // Find index:
            let from_index:Vec<&VwNamespaceMapEntry> = vw_source.entries.iter().filter(|e| e.namespace_char == char).collect(); 
            if from_index.len() != 1 {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown or ambigious namespace char passed by --float_namespaces: {}", char))))
            }
            let from_index = from_index[0].namespace_index;
            //println!("From index {}", from_index);
            vw_source.entries[from_index as usize].namespace_save_as_float = true;
        }
        vw_source.float_namespaces_skip_prefix = float_namespaces.1;

        VwNamespaceMap::new_from_source(vw_source)
    }

}
