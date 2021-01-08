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
    pub namespace_char: char,
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
        let (_, (to_namespace_char, function_name, from_namespaces_chars, function_parameters)) = rr.unwrap();

        let to_namespace_index = get_namespace_id(self, vw, to_namespace_char);
        if to_namespace_index.is_ok() {
            return Err(Box::new(IOError::new(ErrorKind::Other, format!("To namespace of {} already exists: {:?}", s, to_namespace_char))));
        }
        let to_namespace = Namespace {
            namespace_index: self.v.len() as u32 | TRANSFORM_NAMESPACE_MARK, // mark it as special transformed namespace
            namespace_char: to_namespace_char,
        };
        
        let mut from_namespaces: Vec<Namespace> = Vec::new();
        for from_namespace_char in &from_namespaces_chars {
            if !from_namespace_char.is_ascii() {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("Issue in parsing {}: From namespace ({}) has to be an ascii character", s, from_namespace_char))));
            }
            let from_namespace_index = get_namespace_id(self, vw, *from_namespace_char)?;
            println!("from namespace char: {} from namespace index: {}",  from_namespace_char, from_namespace_index);
            if from_namespace_index & TRANSFORM_NAMESPACE_MARK != 0 {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("Issue in parsing {}: From namespace ({}) cannot be an already transformed namespace", s, from_namespace_char))));
            }
            if !vw.lookup_char_to_save_as_float[*from_namespace_char as usize] {
                return Err(Box::new(IOError::new(ErrorKind::Other, format!("Issue in parsing {}: From namespace ({}) has to be defined as --float_namespaces", s, from_namespace_char))));
            }
            from_namespaces.push(Namespace{namespace_index: from_namespace_index, namespace_char: *from_namespace_char});
         }

         // TODO ... function name & parameter verification step needs to happen here

        /*let function = match &function_name[..] {
            "sqrt" => TransformFunction::Sqrt,
            _ => return Err(Box::new(IOError::new(ErrorKind::Other, format!("to namespace of {} has unknown transform function {}", s, function_name)))),
        };*/

        let nt = NamespaceTransform {
            from_namespaces: from_namespaces,
            to_namespace: to_namespace,
            function_name: function_name,
            function_parameters: function_parameters,
        };
    
        self.v.push(nt);

        Ok(())
    }
}

pub fn get_namespace_id(transform_namespaces: &NamespaceTransforms, vw: &vwmap::VwNamespaceMap, namespace_char: char) -> Result<u32, Box<dyn Error>> {
   let index = match vw.map_char_to_index.get(&namespace_char) {
       Some(index) => return Ok(*index as u32),
       None => {
           let f:Vec<&NamespaceTransform> = transform_namespaces.v.iter().filter(|x| x.to_namespace.namespace_char == namespace_char).collect();
           if f.len() == 0 {
               return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown namespace char in command line: {}", namespace_char))));
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


pub fn parse_namespace_char(input: &str) -> IResult<&str, char>{
    let take_char = complete::one_of("abcdefghijklmnopqrstuvzxyABCDEFGHIJKLMNOPQRSTUVZXY0123456789");
    let (input, output) = take_char(input)?;
    Ok((input, output))
}

pub fn parse_namespace_quoted_char(input: &str) -> IResult<&str, char> {
    // TODO: What about quoted ' char ... so parsing '\''
    let (input, (_, namespace_char, _)) = tuple ((
                                            complete::char('\''), 
                                            complete::none_of("\'"), 
                                            complete::char('\'')
                                        ))(input)?;
    Ok((input, namespace_char))
}

pub fn parse_namespace(input: &str) -> IResult<&str, char> {
    // TODO add ability to escape namespaces like \0x32 ?
    
    let (input, (_, namespace_char, _)) = tuple((character::complete::space0, 
                                            branch::alt((complete(parse_namespace_char), complete(parse_namespace_quoted_char))),
                                            character::complete::space0,
                                          ))(input)?;
    Ok((input, namespace_char))
}

pub fn parse_function_params_namespaces(input: &str) -> IResult<&str, Vec<char>> {
    let take_open = complete::char('('); 
    let take_close = complete::char(')'); 
    let take_separator = complete::char(','); 
    let (input, (_, namespaces_str, _)) = tuple((take_open, nom::multi::separated_list1(take_separator, parse_namespace), take_close))(input)?;
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

pub fn parse_function_name(input: &str) -> IResult<&str, String> {
    let (input, (_, letter1, rest, _)) = tuple((
                                                character::complete::space0, 
                                                complete::one_of("abcdefghijklmnopqrstuvzxyABCDEFGHIJKLMNOPQRSTUVZXY"),
                                                take_while(AsChar::is_alphanum), 
                                                character::complete::space0
                                                ))(input)?;
    let mut s = letter1.to_string();
    s.push_str(rest);
    Ok((input, s))
}


pub fn parse_namespace_statement(input: &str) -> IResult<&str, (char, String, Vec<char>, Vec<f32>)> {

    let (input, (to_namespace_char, _, function_name, from_namespace_chars, parameters)) = 
        tuple((
            parse_namespace,
            complete::char('='),
            parse_function_name,
            parse_function_params_namespaces,
            parse_function_params_floats
            ))(input)?;
    
    Ok((input, (to_namespace_char, function_name, from_namespace_chars, parameters)))
}



mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::parser::{NULL, IS_NOT_SINGLE_MASK, IS_FLOAT_NAMESPACE_MASK, MASK31};


    #[test]
    fn test_parser1() {
        let r = parse_namespace_char("a");
        assert_eq!(r.unwrap().1, 'a');
        let r = parse_namespace_char("ab");
        assert_eq!(r.unwrap().1, 'a');
        let r = parse_namespace_char("#");
        assert_eq!(r.is_err(), true);

        let r = parse_namespace_quoted_char("a");
        assert_eq!(r.is_err(), true);
        let r = parse_namespace_quoted_char("'a'");
        assert_eq!(r.unwrap().1, 'a');
    
    
        let r = parse_namespace("a");
        assert_eq!(r.unwrap().1, 'a');
        let r = parse_namespace("'a'");
        assert_eq!(r.unwrap().1, 'a');
        let r = parse_namespace(" a ");
        assert_eq!(r.unwrap().1, 'a');
        
        
        let r = parse_function_params_namespaces("(a)");
        assert_eq!(r.unwrap().1, vec!['a']);
        let r = parse_function_params_namespaces("(a,b)");
        assert_eq!(r.unwrap().1, vec!['a', 'b']);
        let r = parse_function_params_namespaces("( a ,  b )");
        assert_eq!(r.unwrap().1, vec!['a', 'b']);
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

        let r = parse_function_name("");
        assert_eq!(r.is_err(), true);
        let r = parse_function_name("04a");
        assert_eq!(r.is_err(), true);
        let r = parse_function_name("sqrt4");
        assert_eq!(r.unwrap().1, "sqrt4");
        let r = parse_function_name(" sqrt4 ");
        assert_eq!(r.unwrap().1, "sqrt4");

        let r = parse_namespace_statement("a=sqrt(B)(3,1,2.0)");
        let (o, rw) = r.unwrap();
        assert_eq!(rw.0, 'a');
        assert_eq!(rw.1, "sqrt");
        assert_eq!(rw.2, vec!['B']);
        assert_eq!(rw.3, vec![3f32, 1f32, 2.0]);
        
        
    
    }
}







