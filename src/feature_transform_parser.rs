//#[macro_use]
//extern crate nom;

use crate::model_instance;
use crate::parser;
use crate::vwmap;
use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;
use std::collections::HashMap;
use std::mem::replace;
use std::cell::Cell;
use fasthash::murmur3;
use serde::{Serialize,Deserialize};


use crate::feature_transform_executor;

pub const TRANSFORM_NAMESPACE_MARK: u32 = 1<< 31;


#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Namespace {
    pub namespace_descriptor: vwmap::NamespaceDescriptor,
    pub namespace_verbose: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NamespaceTransform {
    pub to_namespace: Namespace,
    pub from_namespaces: Vec<Namespace>,
    pub function_name: String,
    pub function_parameters: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NamespaceTransforms {
    pub v: Vec<NamespaceTransform>,
}

struct NSStage1Parse {
    name: String,
    definition: String,
    from_namespaces: Vec<std::string::String>,
    processing: Cell<bool>, 
    done: Cell<bool>,
}


pub struct NamespaceTransformsParser {
    denormalized: HashMap <std::string::String, NSStage1Parse>, // to_namespace_str -> list of from_namespace_str   

}

impl NamespaceTransformsParser {
    // Parses transformed namespaces
    // First we add them to parser
    // Then we call resolve, which among other things:
    // - checks for cyclic dependencies
    // - processes transformations in the right order, so from namespaces are available when to namespace is being processed
    pub fn new() -> NamespaceTransformsParser {
        NamespaceTransformsParser {denormalized: HashMap::new()}
    }

    pub fn add_transform_namespace(&mut self, vw: &vwmap::VwNamespaceMap, s: &str) -> Result<(), Box<dyn Error>> {
        let rr = parse_namespace_statement(s);
        if rr.is_err() {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Error parsing {}\n{:?}", s, rr))));            
        }
        let (_, (to_namespace_verbose, function_name, from_namespaces_verbose, function_parameters)) = rr.unwrap();

        // Here we just check for clashes with namespaces from input file
        let namespace_descriptor = vw.map_verbose_to_namespace_descriptor.get(&to_namespace_verbose);
        if namespace_descriptor.is_some() {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("To namespace of {} already exists as primitive namespace: {:?}", s, to_namespace_verbose))));
        }
        
        self.denormalized.insert(to_namespace_verbose.to_owned(), NSStage1Parse {
                                                                    name: to_namespace_verbose.to_owned(),
                                                                    definition: s.to_string(),
                                                                    from_namespaces: from_namespaces_verbose,
                                                                    processing: Cell::new(false),
                                                                    done: Cell::new(false),
                                                                });
        Ok(())
    }
    
    pub fn resolve(&mut self, vw: &vwmap::VwNamespaceMap) -> Result<NamespaceTransforms, Box<dyn Error>> {
        let mut nst = NamespaceTransforms::new();
        let mut namespaces:Vec<&String> = self.denormalized.keys().collect();
        namespaces.sort();	// ensure determinism
        for key in &namespaces {
            self.depth_first_search(vw, &mut nst, key)?;
        }
        Ok(nst)
    }
    
    pub fn depth_first_search(&self, 
                                vw: &vwmap::VwNamespaceMap,
                                nst: &mut NamespaceTransforms,
                                verbose_name: &str) 
                                -> Result<(), Box<dyn Error>> {
        // If feature is primitive feature, we don't need to dive deeper
        if vw.map_verbose_to_namespace_descriptor.get(verbose_name).is_some() {
            return Ok(())
        }

        let n = match self.denormalized.get(verbose_name) {
            Some(n) => n,
            None => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Could not find namespace {:?}", verbose_name))))
        };

        if n.done.get() {
            return Ok(())
        }
        
        if n.processing.get() {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Cyclic dependency detected, one of the namespaces involved is {:?}", verbose_name))));
        }    
        
        n.processing.set(true);
        for from_namespace in &n.from_namespaces {
            self.depth_first_search(vw, nst, &from_namespace)?;
        }
        nst.add_transform(vw, &self.denormalized[verbose_name].definition)?;

        n.processing.set(false);
        n.done.set(true);
        Ok(())
                            
    }

}

// We need two-stage parsing of the transforms, so user doesn't need to specify them in order
// first stage we simply figure out namespace dependencies
// then we do directed graph search and only process transformations where we have all inputs defined
// and give error on circular dependencies



impl NamespaceTransforms {
    pub fn new() -> NamespaceTransforms {
        NamespaceTransforms { v: Vec::new(),
                            }
    }
    
    
    
    fn add_transform(&mut self, vw: &vwmap::VwNamespaceMap, s: &str) -> Result<(), Box<dyn Error>> {
        let rr = parse_namespace_statement(s);
        if rr.is_err() {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Error parsing {}\n{:?}", s, rr))));            
        }
        let (_, (to_namespace_verbose, function_name, from_namespaces_verbose, function_parameters)) = rr.unwrap();
        let to_namespace_descriptor = get_namespace_descriptor_verbose(self, vw, &to_namespace_verbose);
        if to_namespace_descriptor.is_ok() {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("To namespace of {} already exists: {:?}", s, to_namespace_verbose))));
        }

        let to_namespace_descriptor = vwmap::NamespaceDescriptor {
                                    namespace_index: self.v.len() as u16,
                                    namespace_type: vwmap::NamespaceType::Transformed,
                                    namespace_format: vwmap::NamespaceFormat::Categorical, // For now all to-namespaces are categorical
                                    };
        
        let to_namespace = Namespace {
            namespace_descriptor: to_namespace_descriptor,
            namespace_verbose: to_namespace_verbose.to_owned(),
        };
        let mut from_namespaces: Vec<Namespace> = Vec::new();
        for from_namespace_verbose in &from_namespaces_verbose {
            let from_namespace_descriptor = get_namespace_descriptor_verbose(self, vw, from_namespace_verbose)?;
            from_namespaces.push(Namespace{ namespace_descriptor: from_namespace_descriptor, 
                                            namespace_verbose: from_namespace_verbose.to_string()
                                            });
        }
        
        // Quadratic for loop... this never goes wrong! 
        for (i, from_namespace_1) in from_namespaces.iter().enumerate() {
            for from_namespace_2 in &from_namespaces[i+1..] {
                if from_namespace_1.namespace_descriptor == from_namespace_2.namespace_descriptor {
                    return Err(Box::new(IOError::new(ErrorKind::Other, format!("Using the same from namespace in multiple arguments to a function is not supported: {:?}", from_namespace_1.namespace_verbose))));
                }
            }
        }


        let nt = NamespaceTransform {
            from_namespaces: from_namespaces,
            to_namespace: to_namespace,
            function_name: function_name,
            function_parameters: function_parameters,
        };
        
         // Now we try to setup a function and then throw it away - for early validation
        let _ = feature_transform_executor::TransformExecutor::from_namespace_transform(&nt)?;
    
        self.v.push(nt);

        Ok(())
        
    }
    


}

pub fn get_namespace_descriptor(transform_namespaces: &NamespaceTransforms, vw: &vwmap::VwNamespaceMap, namespace_char: char) 
    -> Result<vwmap::NamespaceDescriptor, Box<dyn Error>> {
   // Does not support transformed names
   let namespace_descriptor = match vw.map_vwname_to_namespace_descriptor.get(&vec![namespace_char as u8]) {
       Some(namespace_descriptor) => return Ok(*namespace_descriptor),
       None => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown namespace char in command line: {}", namespace_char))))
   };
}

pub fn get_namespace_descriptor_verbose(transform_namespaces: &NamespaceTransforms, vw: &vwmap::VwNamespaceMap, namespace_verbose: &str) 
    -> Result<vwmap::NamespaceDescriptor, Box<dyn Error>> {
   let namespace_descriptor = match vw.map_verbose_to_namespace_descriptor.get(namespace_verbose) {
       Some(namespace_descriptor) => return Ok(*namespace_descriptor),
       None => {
           // Yes, we do linear search, we only call this couple of times. It's fast enough
           let f:Vec<&NamespaceTransform> = transform_namespaces.v.iter().filter(|x| x.to_namespace.namespace_verbose == namespace_verbose).collect();
           if f.len() == 0 {
               return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown verbose namespace in command line: {}", namespace_verbose))));
           } else {
               return Ok(f[0].to_namespace.namespace_descriptor);
           }
       }
   };   
}


use nom::IResult;
use nom::number::complete::be_u16;
use nom::character::complete;
use nom::bytes::complete::take_while;
use nom::AsChar;
use nom::sequence::tuple;
use nom::branch;
use nom::number;
use nom::character;
use nom;
use nom::multi;
use nom::combinator::complete;


pub fn name_char(c:char) -> bool {
    if AsChar::is_alphanum(c) || c == '_' {
        return true;
    } else {
        return false;
    }
            
}

// identifier = namespace or function name
pub fn parse_identifier(input: &str) -> IResult<&str, String> {
    let (input, (_, first_char, rest, _)) = tuple((
                                                character::complete::space0,
                                                complete::one_of("abcdefghijklmnopqrstuvwzxyABCDEFGHIJKLMNOPQRSTUVWZXY_"),
                                                take_while(name_char), 
                                                character::complete::space0
                                                ))(input)?;
    let mut s = first_char.to_string();
    s.push_str(rest);

    Ok((input, s))
}

pub fn parse_function_params_namespaces(input: &str) -> IResult<&str, Vec<String>> {
    let take_open = complete::char('('); 
    let take_close = complete::char(')'); 
    let take_separator = complete::char(','); 
    let (input, (_, namespaces_str, _)) = tuple((take_open, nom::multi::separated_list1(take_separator, parse_identifier), take_close))(input)?;
    Ok((input, namespaces_str))
}

pub fn parse_float(input: &str) -> IResult<&str, f32> {
    let (input, (_, f, _)) = tuple((character::complete::space0,
                                    number::complete::float,
                                    character::complete::space0
                                    ))(input)?;
    Ok((input, f))
}

pub fn parse_function_params_floats(input: &str) -> IResult<&str, Vec<f32>> {
    let take_open = complete::char('('); 
    let take_close = complete::char(')'); 
    let take_separator = complete::char(','); 
    let (input, (_, namespaces_str, _)) = tuple((take_open, nom::multi::separated_list0(take_separator, parse_float), take_close))(input)?;
    Ok((input, namespaces_str))
}


pub fn parse_namespace_statement(input: &str) -> IResult<&str, (String, String, Vec<String>, Vec<f32>)> {

    let (input, (to_namespace_verbose, _, function_name, from_namespace_verbose, parameters)) = 
        tuple((
            parse_identifier,
            complete::char('='),
            parse_identifier,
            parse_function_params_namespaces,
            parse_function_params_floats
            ))(input)?;
    
    Ok((input, (to_namespace_verbose, function_name, from_namespace_verbose, parameters)))
}



mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::parser::{NO_FEATURES, IS_NOT_SINGLE_MASK, MASK31};
    use crate::vwmap::{NamespaceType, NamespaceFormat, NamespaceDescriptor, VwNamespaceMap};

    fn ns_desc(i: u16) -> NamespaceDescriptor {
        NamespaceDescriptor {namespace_index: i, 
                             namespace_type: NamespaceType::Primitive,
                             namespace_format: NamespaceFormat::Categorical,
                             }
    }

    fn ns_desc_trans(i: u16) -> NamespaceDescriptor {
        NamespaceDescriptor {namespace_index: i, 
                             namespace_type: NamespaceType::Transformed,
                             namespace_format: NamespaceFormat::Categorical,
                             }
    }

    fn ns_desc_f32(i: u16) -> NamespaceDescriptor {
        NamespaceDescriptor {namespace_index: i, 
                             namespace_type: NamespaceType::Primitive,
                             namespace_format: NamespaceFormat::F32,
                             }
    }
                        

    #[test]
    fn test_namespace_transforms() {
        let vw_map_string = r#"
A,featureA
B,featureB,f32
C,featureC,f32
"#;
        let vw = VwNamespaceMap::new(vw_map_string).unwrap();

        {
            let mut nstp = NamespaceTransformsParser::new();
            let result = nstp.add_transform_namespace(&vw, "new=Combine(featureA,featureB)()");
            assert!(result.is_ok());
            let nst = nstp.resolve(&vw).unwrap();
            assert_eq!(nst.v[0].to_namespace.namespace_descriptor, ns_desc_trans(0));
            assert_eq!(nst.v[0].from_namespaces[0].namespace_descriptor, ns_desc(0));
            assert_eq!(nst.v[0].from_namespaces[1].namespace_descriptor, ns_desc_f32(1));
        }

        {
            let mut nstp = NamespaceTransformsParser::new();
            let result = nstp.add_transform_namespace(&vw, "new=unknown(featureA,featureB)()");
            assert!(result.is_ok());
            let result = nstp.resolve(&vw);
            assert!(result.is_err());
            assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Unknown transformer function: unknown\" })");
        }

        {
            let mut nstp = NamespaceTransformsParser::new();
            let result = nstp.add_transform_namespace(&vw, "featureA=Combine(featureA,featureB)()"); // unknown function
            let nst = nstp.resolve(&vw).unwrap();
            assert!(result.is_err());
            assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"To namespace of featureA=Combine(featureA,featureB)() already exists as primitive namespace: \\\"featureA\\\"\" })");
        }

        {
            let mut nstp = NamespaceTransformsParser::new();
            let result = nstp.add_transform_namespace(&vw, "new=Combine(featureA,featureA)()"); // unknown function
            let result = nstp.resolve(&vw);
            assert!(result.is_err());
            assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Using the same from namespace in multiple arguments to a function is not supported: \\\"featureA\\\"\" })");
        }


        {
            let mut nstp = NamespaceTransformsParser::new();
            nstp.add_transform_namespace(&vw, "new=unknown(nonexistent,featureB)()").unwrap(); // unknown function
            let result = nstp.resolve(&vw);
            assert!(result.is_err());
            assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Could not find namespace \\\"nonexistent\\\"\" })");
        }


        {
            // Now we test dependencies
            let mut nstp = NamespaceTransformsParser::new();
            let result = nstp.add_transform_namespace(&vw, "new1=Combine(featureA,featureB)()");
            assert!(result.is_ok());
            let result = nstp.add_transform_namespace(&vw, "new2=Combine(new1,featureB)()");
            assert!(result.is_ok());
            let nst = nstp.resolve(&vw).unwrap();
            assert_eq!(nst.v[0].to_namespace.namespace_descriptor, ns_desc_trans(0));
            assert_eq!(nst.v[0].from_namespaces[0].namespace_descriptor, ns_desc(0));
            assert_eq!(nst.v[0].from_namespaces[1].namespace_descriptor, ns_desc_f32(1));
            assert_eq!(nst.v[1].to_namespace.namespace_descriptor, ns_desc_trans(1));
            assert_eq!(nst.v[1].from_namespaces[0].namespace_descriptor, ns_desc_trans(0));
            assert_eq!(nst.v[1].from_namespaces[1].namespace_descriptor, ns_desc_f32(1));
        }

        {
            // Now reverse order. We use new1 before it is declared. 
            let mut nstp = NamespaceTransformsParser::new();
            let result = nstp.add_transform_namespace(&vw, "new2=Combine(new1,featureB)()");
            assert!(result.is_ok());
            let result = nstp.add_transform_namespace(&vw, "new1=Combine(featureA,featureB)()");
            assert!(result.is_ok());
            let nst = nstp.resolve(&vw).unwrap();
            assert_eq!(nst.v[0].to_namespace.namespace_descriptor, ns_desc_trans(0));
            assert_eq!(nst.v[0].from_namespaces[0].namespace_descriptor, ns_desc(0));
            assert_eq!(nst.v[0].from_namespaces[1].namespace_descriptor, ns_desc_f32(1));
            
            assert_eq!(nst.v[1].to_namespace.namespace_descriptor, ns_desc_trans(1));
            assert_eq!(nst.v[1].from_namespaces[0].namespace_descriptor, ns_desc_trans(0));
            assert_eq!(nst.v[1].from_namespaces[1].namespace_descriptor, ns_desc_f32(1));
        }

    }
    #[test]
    fn test_namespace_transforms_cycle() {
        let vw_map_string = r#"
A,featureA
B,featureB,f32
C,featureC,f32
"#;
        let vw = VwNamespaceMap::new(vw_map_string).unwrap();


        {
            // Now create a cycle 
            let mut nstp = NamespaceTransformsParser::new();
            let result = nstp.add_transform_namespace(&vw, "new2=Combine(new1,featureB)()");
            assert!(result.is_ok());
            let result = nstp.add_transform_namespace(&vw, "new1=Combine(new2,featureB)()");
            assert!(result.is_ok());
            let nst = nstp.resolve(&vw);
            assert!(nst.is_err());
            assert_eq!(format!("{:?}", nst), "Err(Custom { kind: Other, error: \"Cyclic dependency detected, one of the namespaces involved is \\\"new1\\\"\" })");


        }

        {
            // Now create a cycle 
            let mut nstp = NamespaceTransformsParser::new();
            let result = nstp.add_transform_namespace(&vw, "new1=Combine(new1,featureB)()");
            assert!(result.is_ok());
            let nst = nstp.resolve(&vw);
            assert!(nst.is_err());
            assert_eq!(format!("{:?}", nst), "Err(Custom { kind: Other, error: \"Cyclic dependency detected, one of the namespaces involved is \\\"new1\\\"\" })");


        }


        
    }


    #[test]
    fn test_parser1() {
        let r = parse_identifier("a");
        assert_eq!(r.unwrap().1, "a");
        let r = parse_identifier("ab");
        assert_eq!(r.unwrap().1, "ab");
        let r = parse_identifier("_a_b3_");
        assert_eq!(r.unwrap().1, "_a_b3_");
        let r = parse_identifier("#");
        assert_eq!(r.is_err(), true);
        let r = parse_identifier("3a");  // they have to start with alphabetic character or underscore
        assert_eq!(r.is_err(), true);
        
        let r = parse_function_params_namespaces("(a)");
        assert_eq!(r.unwrap().1, vec!["a"]);
        let r = parse_function_params_namespaces("(a,b)");
        assert_eq!(r.unwrap().1, vec!["a", "b"]);
        let r = parse_function_params_namespaces("( a ,  b )");
        assert_eq!(r.unwrap().1, vec!["a", "b"]);
        let r = parse_function_params_namespaces("((a)");
        assert_eq!(r.is_err(), true);
        let r = parse_function_params_namespaces("()"); // empty list of namespaces is not allowed
        assert_eq!(r.is_err(), true);

        
        let r = parse_float("0.2");
        assert_eq!(r.unwrap().1, 0.2);
        let r = parse_float(" 0.2");
        assert_eq!(r.unwrap().1, 0.2);
        let r = parse_float("(a)");
        assert_eq!(r.is_err(), true);


        let r = parse_function_params_floats("(0.1)");
        assert_eq!(r.unwrap().1, vec![0.1]);
        let r = parse_function_params_floats("(0.1,0.2)");
        assert_eq!(r.unwrap().1, vec![0.1, 0.2]);
        let r = parse_function_params_floats("( 0.1 ,  0.2 )");
        assert_eq!(r.unwrap().1, vec![0.1, 0.2]);
        let r = parse_function_params_floats("()"); // empty list of floats is allowed
        let fv:Vec<f32>=Vec::new();
        assert_eq!(r.unwrap().1, fv);

        let r = parse_namespace_statement("a=sqrt(B)(3,1,2.0)");
        let (o, rw) = r.unwrap();
        assert_eq!(rw.0, "a");
        assert_eq!(rw.1, "sqrt");
        assert_eq!(rw.2, vec!["B"]);
        assert_eq!(rw.3, vec![3f32, 1f32, 2.0]);
        
        let r = parse_namespace_statement("abc=sqrt(BDE,CG)(3,1,2.0)");
        let (o, rw) = r.unwrap();
        assert_eq!(rw.0, "abc");
        assert_eq!(rw.1, "sqrt");
        assert_eq!(rw.2, vec!["BDE", "CG"]);
        assert_eq!(rw.3, vec![3f32, 1f32, 2.0]);

        let r = parse_namespace_statement("a_bcw=s_qrt(_BD_E_,C_G)(3,1,2.0)");
        let (o, rw) = r.unwrap();
        assert_eq!(rw.0, "a_bcw");
        assert_eq!(rw.1, "s_qrt");
        assert_eq!(rw.2, vec!["_BD_E_", "C_G"]);
        assert_eq!(rw.3, vec![3f32, 1f32, 2.0]);
        
        
    
    }
}







