
use std::path::Path;
use std::error::Error;
use std::io::Cursor;
use crate::vwmap;
use crate::regressor;
use crate::model_instance;
use crate::persistence;
use crate::cmdline;
use crate::feature_buffer;
use crate::parser;
extern crate shell_words;



pub struct FWSession {
    pub vw: vwmap::VwNamespaceMap,
    pub re: regressor::Regressor,
    pub mi: model_instance::ModelInstance,
}

pub fn session_from_cl(cl: &clap::ArgMatches) -> Result<FWSession, Box<dyn Error>> {
    let vw: vwmap::VwNamespaceMap;
    let mut re: regressor::Regressor;
    let mi: model_instance::ModelInstance;

    let testonly = cl.is_present("testonly");
    if let Some(filename) = cl.value_of("initial_regressor") {
        println!("initial_regressor = {}", filename);
        println!("WARNING: Command line model parameters will be ignored");
        let (mi2, vw2, re2) = persistence::new_regressor_from_filename(filename, testonly)?;
        mi = mi2; vw = vw2; re = re2;
    } else {
        // We load vw_namespace_map.csv just so we know all the namespaces ahead of time
        // This is one of the major differences from vowpal
        if cl.is_present("data") {
            let input_filename = cl.value_of("data").unwrap();
            let vw_namespace_map_filepath = Path::new(input_filename).parent().expect("Couldn't access path given by --data").join("vw_namespace_map.csv");
            vw = vwmap::VwNamespaceMap::new_from_csv_filepath(vw_namespace_map_filepath)?;
        } else {
            let namespaces_string = cl.value_of("namespaces").expect("either --initial_regressor, --data or --namespaces have to be declared, so list of namesapces is known");
            vw = vwmap::VwNamespaceMap::new_from_cl_string(namespaces_string)?;
        }
        mi = model_instance::ModelInstance::new_from_cmdline(&cl, &vw)?;
        re = regressor::get_regressor(&mi);
    };
    Ok(FWSession { vw: vw,
                mi: mi,
                re: re})
}

pub fn session_from_cl_string(cl_string: &str) -> Result<FWSession, Box<dyn Error>> {
    let mut args = vec!["fw".to_string()];
    let mut parsed_shell = shell_words::split(cl_string).expect("failed to parse passed command line");
    args.append(&mut parsed_shell);
    let cl = cmdline::parse(args);
    session_from_cl(&cl)
}

// Used for Java bindings
impl FWSession {
    pub fn new(cl_string: &str) -> FWSession {
        let session = session_from_cl_string(cl_string).expect("failed to parse session")        ;
        println!("Session done");
        session
    }
}

pub struct FWPort {
    pub fbt:feature_buffer::FeatureBufferTranslator,
    pub pa: parser::VowpalParser,
}


// Stateful "port" that does predictions - needs to be used one at a time, NOT thread safe
impl FWPort {
    pub fn new(fws: &FWSession) -> FWPort {
        let mut fbt = feature_buffer::FeatureBufferTranslator::new(&fws.mi);
        let mut pa = parser::VowpalParser::new(&fws.vw);
        FWPort {fbt: fbt, pa: pa}
    }

    // Function returns -1.0 if parsing is not successful, as this is an impossible result, caller can consider it an error
    pub fn learn_internal(&mut self, fws: &mut FWSession, input_buffer: &str, update: bool) -> f32 {
        let mut buffered_input = Cursor::new(input_buffer);
        let reading_result = self.pa.next_vowpal(&mut buffered_input);
        let buffer = match reading_result {
                Ok([]) => return -1.0, // EOF
                Ok(buffer2) => buffer2,
                Err(_e) => return -1.0
       };

       self.fbt.translate(buffer, 0);
       //println!("FB: {:?}", self.fbt.feature_buffer);
       return fws.re.learn(&self.fbt.feature_buffer, update)
    }    
    
    pub fn learn(&mut self, fws: &mut FWSession, input_buffer: &str) -> f32 {
        self.learn_internal(fws, input_buffer, true)
    }
    pub fn predict(&mut self, fws: &mut FWSession, input_buffer: &str) -> f32 {
        self.learn_internal(fws, input_buffer, false)
    }
    
    
    
}



