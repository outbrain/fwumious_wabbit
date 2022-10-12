use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::io::prelude::*;
use std::fs;
use serde::{Serialize,Deserialize};
use std::io::ErrorKind;
use std::io::Error as IOError;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Eq)]
pub enum NamespaceType {
    Primitive = 0,
    Transformed = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Eq)]
pub enum NamespaceFormat {
    Categorical = 0, // categorical (binary) features encoding (we have the hash and weight of each feature, value of the feature is assumed to be 1.0 (binary))
    F32 = 1,	// f32 features encoding (we have the hash and value of each feature, weight is assumed to be 1.0)
}


#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Copy)]
pub struct NamespaceDescriptor {
    pub namespace_index: u16,
    pub namespace_type: NamespaceType,
    pub namespace_format: NamespaceFormat,
}


#[derive(Clone, Debug)]
pub struct VwNamespaceMap {
    pub num_namespaces: usize,
    pub map_verbose_to_namespace_descriptor: HashMap <std::string::String, NamespaceDescriptor>,
    pub map_vwname_to_namespace_descriptor: HashMap <Vec<u8>, NamespaceDescriptor>,
    pub map_vwname_to_name: HashMap <Vec<u8>, std::string::String>,
    pub vw_source: VwNamespaceMapSource,    // this is the source from which VwNamespaceMap can be constructed - for persistence
}

// this is serializible source from which VwNamespaceMap can be constructed
#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct VwNamespaceMapEntry {
    pub namespace_vwname: std::string::String,
    namespace_verbose: std::string::String,
    namespace_index: u16,
    namespace_format: NamespaceFormat, 
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct VwNamespaceMapSource {
    pub namespace_skip_prefix: u32,
    pub entries: Vec<VwNamespaceMapEntry>,
}

impl VwNamespaceMap {
    pub fn new_from_source(vw_source: VwNamespaceMapSource)  -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut vw = VwNamespaceMap {
                                num_namespaces:0, 
                                map_verbose_to_namespace_descriptor:HashMap::new(),
                                map_vwname_to_namespace_descriptor: HashMap::new(),
                                map_vwname_to_name: HashMap::new(),
                                vw_source: vw_source,
                                };

        for vw_entry in &vw.vw_source.entries {
            //let record = result?;
            let name_str = &vw_entry.namespace_verbose;
            let vwname_str = &vw_entry.namespace_vwname;
            
            let namespace_descriptor = NamespaceDescriptor{
                                        namespace_index: vw_entry.namespace_index, 
                                        namespace_type: NamespaceType::Primitive,
                                        namespace_format: vw_entry.namespace_format,
                                    };
            
            vw.map_vwname_to_name.insert(vwname_str.as_bytes().to_vec(), String::from(name_str));
            vw.map_vwname_to_namespace_descriptor.insert(vwname_str.as_bytes().to_vec(), namespace_descriptor.clone());
            vw.map_verbose_to_namespace_descriptor.insert(String::from(name_str), namespace_descriptor.clone());

            if vw_entry.namespace_index as usize > vw.num_namespaces {
                vw.num_namespaces = vw_entry.namespace_index as usize;
            } 
        }
        vw.num_namespaces += 1;
        Ok(vw)
    }

    pub fn new_from_csv_filepath(path: PathBuf) -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut input_bufreader = fs::File::open(&path).expect(&format!("Could not find vw_namespace_map.csv in input dataset directory of {:?}", path).to_string());
        let mut s = String::new();
        input_bufreader.read_to_string(&mut s)?;
        VwNamespaceMap::new(&s)
    }

    pub fn new(data: &str) -> Result<VwNamespaceMap, Box<dyn Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_reader(data.as_bytes()
            );
        let mut vw_source = VwNamespaceMapSource { entries: vec![], namespace_skip_prefix: 0};
        for (i, record_w) in rdr.records().enumerate() {
            let record = record_w?;
            let vwname_str = &record[0];
            if vwname_str.as_bytes().len() != 1 && i ==0 {
                println!("Warning: multi-byte namespace names are not compatible with old style namespace arguments");
            }
            
            if vwname_str == "_namespace_skip_prefix" {
                let namespace_skip_prefix = record[1].parse().expect("Couldn't parse _namespace_skip_prefix in vw_namespaces_map.csv");
                println!("_namespace_skip_prefix set in vw_namespace_map.csv is {}", namespace_skip_prefix);
                vw_source.namespace_skip_prefix = namespace_skip_prefix;
                continue               
            }
            
            let name_str = &record[1];
            let namespace_format = match &record.get(2) {
                Some("f32") => NamespaceFormat::F32,
                Some("") => NamespaceFormat::Categorical,
                None => NamespaceFormat::Categorical,
                Some(unknown_type) => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown type used for the feature in vw_namespace_map.csv: \"{}\". Only \"f32\" is possible.", unknown_type))))
            };
            
            vw_source.entries.push(VwNamespaceMapEntry {
                namespace_vwname: vwname_str.to_string(),
                namespace_verbose: name_str.to_string(),
                namespace_index: i as u16,
                namespace_format: namespace_format,
            });
        }

        VwNamespaceMap::new_from_source(vw_source)
    }

}


#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_simple() {
        let vw_map_string = r#"
A,featureA
B,featureB
C,featureC
"#;
        let vw = VwNamespaceMap::new(vw_map_string).unwrap();
        assert_eq!(vw.vw_source.entries.len(), 3);
        assert_eq!(vw.vw_source.namespace_skip_prefix, 0);
        assert_eq!(vw.vw_source.entries[0], 
            VwNamespaceMapEntry {
                namespace_vwname: "A".to_string(),
                namespace_verbose: "featureA".to_string(),
                namespace_index: 0,
                namespace_format: NamespaceFormat::Categorical});         

        assert_eq!(vw.vw_source.entries[1], 
            VwNamespaceMapEntry {
                namespace_vwname: "B".to_string(),
                namespace_verbose: "featureB".to_string(),
                namespace_index: 1,
                namespace_format: NamespaceFormat::Categorical});         

        assert_eq!(vw.vw_source.entries[2], 
            VwNamespaceMapEntry {
                namespace_vwname: "C".to_string(),
                namespace_verbose: "featureC".to_string(),
                namespace_index: 2,
                namespace_format: NamespaceFormat::Categorical});         
    }


    #[test]
    fn test_f32() {
        {
            let vw_map_string = "A,featureA,f32\n_namespace_skip_prefix,2";
            let vw = VwNamespaceMap::new(vw_map_string).unwrap();
            assert_eq!(vw.vw_source.entries[0], 
                VwNamespaceMapEntry {
                    namespace_vwname: "A".to_string(),
                    namespace_verbose: "featureA".to_string(),
                    namespace_index: 0,
                    namespace_format: NamespaceFormat::F32});         
            assert_eq!(vw.vw_source.namespace_skip_prefix, 2);
        }
        {
            let vw_map_string = "A,featureA,blah\n";
            let result = VwNamespaceMap::new(vw_map_string);
            assert!(result.is_err());
            assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Unknown type used for the feature in vw_namespace_map.csv: \\\"blah\\\". Only \\\"f32\\\" is possible.\" })");
        }
    }




}











