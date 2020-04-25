use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;

pub struct VwNamespaceMap {
    pub num_namespaces: u32,
    pub map_name_to_index: HashMap <std::string::String, usize>,
    pub map_char_to_name: HashMap <char, std::string::String>,
    pub map_char_to_index: HashMap <char, usize>,
    pub lookup_char_to_index: [usize; 256]
}

pub fn get_global_map(records: Vec<csv::StringRecord>)  -> Result<VwNamespaceMap, Box<dyn Error>> {
    let mut i: u32 = 0;
    let mut vw = VwNamespaceMap {
                            num_namespaces:0, 
                            map_name_to_index:HashMap::new(),
                            map_char_to_index:HashMap::new(),
                            map_char_to_name:HashMap::new(),
                            lookup_char_to_index: [0; 256],
                            };
                            
    for record in records {
        //let record = result?;
        let char_str = &record[0];
        let name_str = &record[1];
        if char_str.len() != 1 {
            panic!("Can't decode {:?}", record);
        }
        let char = char_str.chars().next().unwrap();
        vw.map_name_to_index.insert(String::from(name_str), i as usize);
        vw.map_char_to_index.insert(char, i as usize);
        vw.map_char_to_name.insert(char, String::from(name_str));
        vw.lookup_char_to_index[char as usize] = i as usize;
        i += 1;
        
//        println!("{:?}", record);
                
        
    }
    vw.num_namespaces = i;
    Ok(vw)
}

pub fn get_global_map_from_json(path: PathBuf) -> Result<VwNamespaceMap, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)?;
    let records = rdr.records().map(|it| it.unwrap()).collect();
    get_global_map(records)
}

pub fn get_global_map_from_string(data: &str) -> Result<VwNamespaceMap, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(data.as_bytes());
    let records = rdr.records().map(|it| it.unwrap()).collect();
    get_global_map(records)
}




