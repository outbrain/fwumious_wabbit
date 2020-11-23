
use std::path::Path;
use std::error::Error;
use crate::vwmap;
use crate::regressor;
use crate::model_instance;
use crate::persistence;
use crate::cmdline;
extern crate shell_words;



pub struct FWSession {
    pub vw: vwmap::VwNamespaceMap,
    pub re: Box<dyn regressor::RegressorTrait>,
    pub mi: model_instance::ModelInstance,
}

pub fn session_from_cl(cl: &clap::ArgMatches) -> Result<FWSession, Box<dyn Error>> {
    let vw: vwmap::VwNamespaceMap;
    let mut re: Box<dyn regressor::RegressorTrait>;
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
        let input_filename = cl.value_of("data").expect("--data expected");
        let vw_namespace_map_filepath = Path::new(input_filename).parent().expect("Couldn't access path given by --data").join("vw_namespace_map.csv");
        vw = vwmap::VwNamespaceMap::new_from_csv_filepath(vw_namespace_map_filepath)?;
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

impl FWSession {
    pub fn new(cl_string: &str) -> FWSession {
        let session = session_from_cl_string(cl_string).expect("failed to parse session")        ;
        println!("Session done");
        session
    }
}







