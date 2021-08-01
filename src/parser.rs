use std::fmt;
use fasthash::murmur3;
use std::io::BufRead;
use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;
use std::str;
use std::string::String;
use crate::vwmap;

const RECBUF_LEN:usize = 2048;
pub const HEADER_LEN:usize = 3;
pub const NAMESPACE_DESC_LEN:usize = 1;
pub const LABEL_OFFSET:usize = 1;
pub const EXAMPLE_IMPORTANCE_OFFSET:usize = 2;
pub const IS_NOT_SINGLE_MASK : u32 = 1u32 << 31;
pub const MASK31: u32 = !IS_NOT_SINGLE_MASK;
pub const NULL: u32= IS_NOT_SINGLE_MASK; // null is just an exact IS_NOT_SINGLE_MASK
pub const NO_LABEL: u32 = 0xff;
pub const FLOAT32_ONE: u32 = 1065353216;  // 1.0f32.to_bits()

#[derive (Clone)]
pub struct VowpalParser {
    vw_map: vwmap::VwNamespaceMap,
    tmp_read_buf: Vec<u8>,
    namespace_hash_seeds: [u32; 256],     // Each namespace has its hash seed
    pub output_buffer: Vec<u32>,
}


#[derive(Debug)]
pub struct FlushCommand;  // Parser returns FlushCommand to signal flush message
impl Error for FlushCommand {}
impl fmt::Display for FlushCommand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Not really an error: a flush command from client")
    }
}


/* 
organization of records buffer 
(u32) length of the output record
(u32) label
(f32) Example importance (default: 1.0)
(union_u u32)[number of features], where:
    -- if the most significant bit is zero
            - this is a a namespace with a single feature
            - bits 1-31 are a feature hash
            - feature value is assumed to be 1.0
    -- if the most significant bit is one
            - 15 next bits are the start offset, and lower 16 bits are the end offset of features beyond initial map
            - the dynamic buffer consists of (hash of the feature name, f32 value of the feature) 
[dynamic buffer of (u32 hash, f32 value of the feature]
*/

impl VowpalParser {
    pub fn new(vw: &vwmap::VwNamespaceMap) -> VowpalParser {
        let mut rr = VowpalParser {  
                            vw_map: (*vw).clone(),
                            tmp_read_buf: Vec::with_capacity(RECBUF_LEN),
                            output_buffer: Vec::with_capacity(RECBUF_LEN*2),
                            namespace_hash_seeds: [0; 256],
                        };
        rr.output_buffer.resize(vw.num_namespaces as usize * NAMESPACE_DESC_LEN + HEADER_LEN, 0);
        for i in 0..=255 {
            rr.namespace_hash_seeds[i as usize] = murmur3::hash32([i;1]);
        }
        rr
    }
    
    pub fn print(&self) -> () {
        println!("item out {:?}", self.output_buffer);
    }

    #[inline(always)]
    pub fn parse_float_or_error(&self, i_start: usize, i_end :usize, error_str: &str) -> Result<f32, Box<dyn Error>> {
        unsafe {
            match str::from_utf8_unchecked(&self.tmp_read_buf[i_start..i_end]).parse::<f32>() {
                Ok(f) => return Ok(f),
                Err(_e) => return Err(Box::new(IOError::new(ErrorKind::Other, format!("{}: {}", error_str, String::from_utf8_lossy(&self.tmp_read_buf[i_start..i_end])))))
            };
        };
    }

    pub fn next_vowpal(&mut self, input_bufread: &mut impl BufRead) -> Result<&[u32], Box<dyn Error>> {
            self.tmp_read_buf.truncate(0);
            let rowlen1 = match input_bufread.read_until(0x0a, &mut self.tmp_read_buf) {
                Ok(0) => return Ok(&[]),
                Ok(n) => n,
                Err(e) => Err(e)?
            };

            let bufpos: usize = (self.vw_map.num_namespaces as usize) + HEADER_LEN;
            self.output_buffer.truncate(bufpos);
            for i in &mut self.output_buffer[0..bufpos] { *i = NULL };

            let mut current_char_num_of_features = 0;

            unsafe {
                let p = self.tmp_read_buf.as_ptr();
                let mut i_start:usize;
                let mut i_end:usize = 0;

                // first token is a label or "flush" command
                match *p.add(0) {
                    0x31 => self.output_buffer[LABEL_OFFSET] = 1,    // 1
                    0x2d => self.output_buffer[LABEL_OFFSET] = 0,    // -1
                    0x7c => self.output_buffer[LABEL_OFFSET] = NO_LABEL, // when first character is |, this means there is no label
                    _ => {
                        // "flush" ascii 66, 6C, 75, 73, 68
                        if rowlen1 >= 5 && *p.add(0) == 0x66  && *p.add(1) == 0x6C && *p.add(2) == 0x75 && *p.add(3) == 0x73 && *p.add(4) == 0x68 {
                            return Err(Box::new(FlushCommand))
                        } else {
                            return Err(Box::new(IOError::new(ErrorKind::Other, format!("Unknown first character of the label: ascii {:?}", *p.add(0)))))
                        }
                    }
                };
                
                let rowlen = rowlen1 - 1; // ignore last newline byte
                if self.output_buffer[LABEL_OFFSET] != NO_LABEL {
                        // if we have a label, let's check if we also have label weight
                        while *p.add(i_end) != 0x20 && i_end < rowlen {i_end += 1;}; // find space
                        while *p.add(i_end) == 0x20 && i_end < rowlen {i_end += 1;}; // find first non-space
                        //if next character is not "|", we assume it's a example importance
                        //i_end +=1;
                        if *p.add(i_end) != 0x7c { // this token does not start with "|", so it has to be example improtance floating point
                                i_start = i_end;
                                while *p.add(i_end) != 0x20 && i_end < rowlen {i_end += 1;}; // find end of token (space)
                                let importance = self.parse_float_or_error(i_start, i_end, "Failed parsing example importance")?;
                                if importance < 0.0  {
                                    return Err(Box::new(IOError::new(ErrorKind::Other, format!("Example importance cannot be negative: {:?}! ", importance))));
                                }
                                self.output_buffer[EXAMPLE_IMPORTANCE_OFFSET] = importance.to_bits();
                        } else {
                                self.output_buffer[EXAMPLE_IMPORTANCE_OFFSET] = FLOAT32_ONE;
                        }                                              
                } else {
                        self.output_buffer[EXAMPLE_IMPORTANCE_OFFSET] = FLOAT32_ONE;
                }                
                // Then we look for first namespace
                while *p.add(i_end) != 0x7c && i_end < rowlen { i_end += 1;};
                
                let mut current_char:usize = 0;
                let mut current_char_index:usize = HEADER_LEN;
                let mut bufpos_namespace_start = 0;
                let mut current_namespace_weight:f32 = 1.0;
//                print!("AAAAAAAA\n");
                while i_end < rowlen {
                    // <letter>[:<weight>]
                    
                    // First skip spaces
                    while *p.add(i_end) == 0x20 && i_end < rowlen {i_end += 1;}
                    i_start = i_end;
                    while *p.add(i_end) != 0x20 && *p.add(i_end) != 0x3a && i_end < rowlen {i_end += 1;}
                    let i_end_first_part = i_end;
                    while *p.add(i_end) != 0x20 && i_end < rowlen {i_end += 1; }
                    
                    //println!("item out {:?}", std::str::from_utf8(&rr.tmp_read_buf[i_start..i_end]));
                    if *p.add(i_start) == 0x7c { // "|"
                        // new namespace index
                        i_start += 1;
                        if i_end - i_start != 1 {
                            // Namespace that has more than one character is either an error, or it defines the 
                            if i_end_first_part - i_start != 1 {
                                return Err(Box::new(IOError::new(ErrorKind::Other, format!("Only single letter namespaces are allowed, however namespace string is: {:?}", String::from_utf8_lossy(&self.tmp_read_buf[i_start..i_end_first_part])))));
                            }
                            current_namespace_weight = self.parse_float_or_error(i_end_first_part+1, i_end, "Failed parsing namespace weight")?;
                        } else {
                            current_namespace_weight = 1.0;
                        }
                     //   print!("Only single letter namespaces are allowed, however namespace string is: {:?}\n", String::from_utf8_lossy(&self.tmp_read_buf[i_start..i_end_first_part]));
                        current_char = *p.add(i_start) as usize;
                        current_char_index = self.vw_map.lookup_char_to_index[current_char] * NAMESPACE_DESC_LEN + HEADER_LEN;
                        current_char_num_of_features = 0;
                        bufpos_namespace_start = self.output_buffer.len(); // this is only used if we will have multiple values
                    } else { 
                        // We have a feature! Let's hash it and write it to the buffer
                        // println!("item out {:?}", std::str::from_utf8(&rr.tmp_read_buf[i_start..i_end]));
                     //   print!("F {:?}\n", String::from_utf8_lossy(&self.tmp_read_buf[i_start..i_end_first_part]));
                        let h = murmur3::hash32_with_seed(&self.tmp_read_buf[i_start..i_end_first_part], 
                                                          *self.namespace_hash_seeds.get_unchecked(current_char)) & MASK31;  

                        let feature_weight:f32 = match i_end - i_end_first_part {
                            0 => 1.0,
                            _ => self.parse_float_or_error(i_end_first_part + 1, i_end, "Failed parsing feature weight")?
                        };
                        
                        // We have three options: 
                        // - first feature, no weights -> put it in-place
                        // - if it's second feature and first one was "simple", then promote it
                        // -- and then just add feature to the end of the buffer
                        
                        if current_namespace_weight == 1.0 && feature_weight == 1.0 && current_char_num_of_features == 0 {
                            *self.output_buffer.get_unchecked_mut(current_char_index) = h;
                        } else {
                            if (current_char_num_of_features == 1) && (*self.output_buffer.get_unchecked(current_char_index) & IS_NOT_SINGLE_MASK) == 0 {
                                // We need to promote feature currently written in-place to out of place
                                self.output_buffer.push(*self.output_buffer.get_unchecked(current_char_index));
                                self.output_buffer.push(FLOAT32_ONE);
                            }
                            self.output_buffer.push(h);
                            self.output_buffer.push((current_namespace_weight * feature_weight).to_bits());
                            *self.output_buffer.get_unchecked_mut(current_char_index) = IS_NOT_SINGLE_MASK | (((bufpos_namespace_start<<16) + self.output_buffer.len()) as u32);
                        }
                        current_char_num_of_features += 1;
                    }
                    i_end += 1;
                    
                }
            }
            
//            println!("item out {:?} {}", self.output_buffer, bufpos);
            self.output_buffer[0] = self.output_buffer.len() as u32;
            Ok(&self.output_buffer)
        }

}



#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use vwmap;
    use std::io::Cursor;
        
    fn nd(start: u32, end: u32) -> u32 {
        return (start << 16) + end;
    }


    #[test]
    fn test_vowpal() {
        // Test for perfect vowpal-compatible hashing
        let vw_map_string = r#"
A,featureA
B,featureB
C,featureC
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();

        fn str_to_cursor(s: &str) -> Cursor<Vec<u8>> {
          Cursor::new(s.as_bytes().to_vec())
        }

        let mut rr = VowpalParser::new(&vw);
        // we test a single record, single namespace
        let mut buf = str_to_cursor("1 |A a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6,  1, FLOAT32_ONE,  
                                                        2988156968 & MASK31, 
                                                        NULL, 
                                                        NULL]);
 
        // we test a single record, single namespace, space at the end
        let mut buf = str_to_cursor("1 |A a \n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6,  1, FLOAT32_ONE,  
                                                        2988156968 & MASK31, 
                                                        NULL, 
                                                        NULL]);
                                                        

        // we test a single record, single namespace, space after label
        let mut buf = str_to_cursor("1  |A a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6,  1, FLOAT32_ONE,  
                                                        2988156968 & MASK31, 
                                                        NULL, 
                                                        NULL]);
                                                        
        // we test a single record, single namespace, space between namespace and label
        let mut buf = str_to_cursor("1 |A  a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6,  1, FLOAT32_ONE,  
                                                        2988156968 & MASK31, 
                                                        NULL, 
                                                        NULL]);
                                                        
                                                        
                                                        
                                                         
                                                        
        let mut buf = str_to_cursor("-1 |B b\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6, 0, FLOAT32_ONE,
                                                        NULL, 
                                                        2422381320 & MASK31, 
                                                        NULL]);
        // single namespace with two features
        let mut buf = str_to_cursor("1 |A a b\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [10, 1, FLOAT32_ONE,  
                                                        nd(6,10) | IS_NOT_SINGLE_MASK, 	// |A
                                                        NULL, 				// |B 
                                                        NULL, 				// |C
                                                        2988156968 & MASK31, FLOAT32_ONE,   // |A a
                                                        3529656005 & MASK31, FLOAT32_ONE]); // |A b
        // two namespaces
        let mut buf = str_to_cursor("-1 |A a |B b\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6, 0, FLOAT32_ONE,
                                                        2988156968 & MASK31, 
                                                        2422381320 & MASK31, 
                                                        NULL]);

        // two namespaces, double space
        let mut buf = str_to_cursor("-1 |A a  |B b\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6, 0, FLOAT32_ONE,
                                                        2988156968 & MASK31, 
                                                        2422381320 & MASK31, 
                                                        NULL]);
        
        // only single letter namespaces are allowed
        let mut buf = str_to_cursor("1 |MORE_THAN_A_LETTER a\n");
        assert!(rr.next_vowpal(&mut buf).is_err());

        let mut buf = str_to_cursor("1 |MORE_THAN_A_LETTER a\n");
        let result = rr.next_vowpal(&mut buf);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Only single letter namespaces are allowed, however namespace string is: \\\"MORE_THAN_A_LETTER\\\"\" })");
 
        // namespace weight test
        let mut buf = str_to_cursor("1 |A:1.0 a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6, 1, FLOAT32_ONE,
                                                        2988156968 & MASK31, 
                                                        NULL, 
                                                        NULL]);
        // not a parsable number
        let mut buf = str_to_cursor("1 |A:not_a_parsable_number a\n");
        let result = rr.next_vowpal(&mut buf);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Failed parsing namespace weight: not_a_parsable_number\" })");

        // double weight
        let mut buf = str_to_cursor("1 |A:1:1 a\n");
        let result = rr.next_vowpal(&mut buf);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Failed parsing namespace weight: 1:1\" })");

        // namespace weight test
        let mut buf = str_to_cursor("1 |A:2.0 a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [8, 1, FLOAT32_ONE, 
                                                        nd(6, 8) | IS_NOT_SINGLE_MASK, 
                                                        NULL, 
                                                        NULL, 
                                                        2988156968 & MASK31, 2.0f32.to_bits()]);
       // feature weight
        let mut buf = str_to_cursor("1 |A a:2.0\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [8, 1, FLOAT32_ONE, 
                                                        nd(6, 8) | IS_NOT_SINGLE_MASK, 
                                                        NULL, 
                                                        NULL, 
                                                        2988156968 & MASK31, 2.0f32.to_bits()]);

       // two feature weights
        let mut buf = str_to_cursor("1 |A a:2.0 b:3.0\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [10, 1, FLOAT32_ONE, 
                                                        nd(6, 10) | IS_NOT_SINGLE_MASK, 
                                                        NULL, 
                                                        NULL, 
                                                        2988156968 & MASK31, 2.0f32.to_bits(),
                                                        3529656005 & MASK31, 3.0f32.to_bits(),
                                                        ]);

       // feature weight + namespace weight
        let mut buf = str_to_cursor("1 |A:3 a:2.0\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [8, 1, FLOAT32_ONE, 
                                                        nd(6, 8) | IS_NOT_SINGLE_MASK, 
                                                        NULL, 
                                                        NULL, 
                                                        2988156968 & MASK31, 6.0f32.to_bits()]);

       // bad feature weight
        let mut buf = str_to_cursor("1 |A a:2x0\n");
        let result = rr.next_vowpal(&mut buf);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Failed parsing feature weight: 2x0\" })");


       // first no weight, then two weighted features
        let mut buf = str_to_cursor("1 |A a b:2.0 c:3.0\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [12, 1, FLOAT32_ONE, 
                                                        nd(6, 12) | IS_NOT_SINGLE_MASK, 
                                                        NULL, 
                                                        NULL, 
                                                        2988156968 & MASK31, 1.0f32.to_bits(),
                                                        3529656005 & MASK31, 2.0f32.to_bits(),
                                                        906509 & MASK31, 3.0f32.to_bits(),
                                                        ]);

 
 
                
        // LABEL TESTS
        // without label
        let mut buf = str_to_cursor("|A a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6, NO_LABEL, FLOAT32_ONE,
                                                        2988156968 & MASK31, 
                                                        NULL, 
                                                        NULL]);

        /* Should we support this ? 
        let mut buf = str_to_cursor(" |A a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6, NO_LABEL, FLOAT32_ONE,
                                                        2988156968 & MASK31, 
                                                        NULL, 
                                                        NULL]);
        */
                
        //println!("{:?}", rr.output_buffer);
        // now we test if end-of-stream works correctly
        str_to_cursor("");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap().len(), 0);
        
        // flush should return [999]
        let mut buf = str_to_cursor("flush");
        assert_eq!(rr.next_vowpal(&mut buf).err().unwrap().is::<FlushCommand>(), true);

        // Unrecognized label -> Error
        let mut buf = str_to_cursor("$1");
        let result = rr.next_vowpal(&mut buf);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Unknown first character of the label: ascii 36\" })");

        // Example importance is negative -> Error
        let mut buf = str_to_cursor("1 -0.1 |A a\n");
        let result = rr.next_vowpal(&mut buf);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Example importance cannot be negative: -0.1! \" })");

        // After label, there is neither namespace definition (|) nor example importance float
        let mut buf = str_to_cursor("1 fdsa |A a\n");
        let result = rr.next_vowpal(&mut buf);
        assert!(result.is_err());
        assert_eq!(format!("{:?}", result), "Err(Custom { kind: Other, error: \"Failed parsing example importance: fdsa\" })");
        
        // Example importance
        let mut buf = str_to_cursor("1 0.1 |A a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6, 1, 0.1f32.to_bits(),
                                                        2988156968 & MASK31, 
                                                        NULL, 
                                                        NULL]);

        // Example importance with bunch of spaces
        let mut buf = str_to_cursor("1  0.1  |A  a \n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [6, 1, 0.1f32.to_bits(),
                                                        2988156968 & MASK31, 
                                                        NULL, 
                                                        NULL]);
 
 
 
    }
}