//#[macro_use]
//extern crate nom;

use crate::model_instance;
use crate::parser;
use crate::vwmap;
use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;

use fasthash::murmur3;
use serde::{Serialize,Deserialize};


use crate::feature_transform_executor;

pub const TRANSFORM_NAMESPACE_MARK: u32 = 1<< 31;


#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Namespace {
    pub namespace_index: u32,
    pub namespace_verbose: String,
    pub namespace_is_float: bool,
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
    pub v: Vec<NamespaceTransform>
}

impl NamespaceTransforms {
    pub fn new() -> NamespaceTransforms {
        NamespaceTransforms {v: Vec::new()}
    }

    pub fn add_transform_namespace(&mut self, vw: &vwmap::VwNamespaceMap, s: &str) -> Result<(), Box<dyn Error>> {
        let rr = parse_namespace_statement(s);
        if rr.is_err() {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Error parsing {}\n{:?}", s, rr))));            
        }
        let (_, (to_namespace_verbose, function_name, from_namespaces_verbose, function_parameters)) = rr.unwrap();

        let to_namespace_index = get_namespace_id_verbose(self, vw, &to_namespace_verbose);
        if to_namespace_index.is_ok() {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("To namespace of {} already exists: {:?}", s, to_namespace_verbose))));
        }
        let to_namespace = Namespace {
            namespace_index: self.v.len() as u32 | TRANSFORM_NAMESPACE_MARK, // mark it as special transformed namespace
            namespace_verbose: to_namespace_verbose,
            namespace_is_float: false,
        };
        
        
        let mut from_namespaces: Vec<Namespace> = Vec::new();
        for from_namespace_verbose in &from_namespaces_verbose {
            let from_namespace_index = get_namespace_id_verbose(self, vw, from_namespace_verbose)?;
         //   println!("from namespace verbose: {} from namespace index: {}",  from_namespace_verbose, from_namespace_index);
            let mut namespace_is_float:bool;
            if from_namespace_index & TRANSFORM_NAMESPACE_MARK != 0 {
                // Currently if the namespace is a transformed namespace, it cannot be float
                namespace_is_float = false;
            } else {
                namespace_is_float= vw.map_index_to_save_as_float[from_namespace_index as usize]
            }
            
            from_namespaces.push(Namespace{ namespace_index: from_namespace_index, 
                                            namespace_verbose: from_namespace_verbose.to_string(),
                                            namespace_is_float: namespace_is_float });
        }
        
        // Quadratic for loop... this never goes wrong! 
        for (i, from_namespace_1) in from_namespaces.iter().enumerate() {
            for from_namespace_2 in &from_namespaces[i+1..] {
                if from_namespace_1.namespace_index == from_namespace_2.namespace_index {
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

pub fn get_namespace_id(transform_namespaces: &NamespaceTransforms, vw: &vwmap::VwNamespaceMap, namespace_char: char) -> Result<u32, Box<dyn Error>> {
   // Does not support transformed names
   let index = match vw.map_vwname_to_index.get(&vec![namespace_char as u8]) {
       Some(index) => return Ok(*index as u32),
       None => return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown namespace char in command line: {}", namespace_char))))
   };   
}

pub fn get_namespace_id_verbose(transform_namespaces: &NamespaceTransforms, vw: &vwmap::VwNamespaceMap, namespace_verbose: &str) -> Result<u32, Box<dyn Error>> {
   let index = match vw.map_verbose_to_index.get(namespace_verbose) {
       Some(index) => return Ok(*index as u32),
       None => {
           // Yes, we do linear search, we only call this couple of times. It's fast enough
           let f:Vec<&NamespaceTransform> = transform_namespaces.v.iter().filter(|x| x.to_namespace.namespace_verbose == namespace_verbose).collect();
           if f.len() == 0 {
               return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown namespace char in command line: {}", namespace_verbose))));
           } else {
               return Ok(f[0].to_namespace.namespace_index as u32);
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
    use crate::parser::{NO_FEATURES, IS_NOT_SINGLE_MASK, IS_FLOAT_NAMESPACE_MASK, MASK31};


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







