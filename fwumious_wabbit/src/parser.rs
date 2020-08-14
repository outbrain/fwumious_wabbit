use std::fs::File;
use fasthash::murmur3;
use fasthash::xx;
use std::io;
use std::io::BufRead;
use std::error::Error;
use std::io::Error as IOError;
use std::io::ErrorKind;
use std::str;
use crate::vwmap;

const RECBUF_LEN:usize = 2048;
pub const HEADER_LEN:usize = 2;
pub const NAMESPACE_DESC_LEN:usize = 1;
pub const LABEL_OFFSET:usize = 1;
pub const IS_NOT_SINGLE_MASK : u32 = 1u32 << 31;
pub const MASK31: u32 = !IS_NOT_SINGLE_MASK;
pub const NULL: u32= IS_NOT_SINGLE_MASK; // null is just an exact IS_NOT_SINGLE_MASK
pub const NO_LABEL: u32 = 0xff;
pub const FLOAT32_ONE: u32 = 1065353216;  // 1.0f32.to_bits()

pub struct VowpalParser<'a> {
    vw_map: &'a vwmap::VwNamespaceMap,
    tmp_read_buf: Vec<u8>,
    namespace_hash_seeds: [u32; 256],     // Each namespace has its hash seed
    pub output_buffer: Vec<u32>,
}


/* 
organization of records buffer 
(u32) length of the output record
(union_u u32)[number of features], where:
    -- if the most significant bit is zero
            - this is a a namespace with a single feature
            - bits 1-31 are a feature hash
            - feature value is assumed to be 1.0
    -- if the most significant bit is one
            - 15 next bits are the start offset, and lower 16 bits are the end offset of features beyond initial map
            - the dynamic buffer consists of (hash of the feature name, f32 value of the feature) 
(u32)[dynamic buffer]
*/

impl<'a> VowpalParser<'a> {
    pub fn new(vw: &'a vwmap::VwNamespaceMap) -> VowpalParser<'a> {
        let mut rr = VowpalParser {  
                            vw_map: vw,
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


    pub fn next_vowpal(&mut self, input_bufread: &mut dyn BufRead) -> Result<&[u32], Box<dyn Error>>  {

            // flush is kw to return OK(&[999])
            // Output item
            self.tmp_read_buf.truncate(0);
            let rowlen1 = match input_bufread.read_until(0x0a, &mut self.tmp_read_buf) {
                Ok(0) => return Ok(&[]),
                Ok(n) => n,
                Err(e) => Err(e)?          
            };
            {
                let mut bufpos: usize = (self.vw_map.num_namespaces as usize) + HEADER_LEN;
                self.output_buffer.truncate(bufpos);
                for i in &mut self.output_buffer[0..bufpos] { *i = NULL };
            }
            let mut current_char_num_of_features = 0;

            unsafe {
                let p = self.tmp_read_buf.as_ptr();
                let mut buf = self.output_buffer.as_mut_ptr();

                // first token, either 1 or -1
                match *p.add(0) {
                    0x31 => self.output_buffer[LABEL_OFFSET] = 1,    // 1
                    0x2d => self.output_buffer[LABEL_OFFSET] = 0,    // -1
                    _ => {
                        // flush asci 66, 6C, 75, 73, 68
                        if rowlen1 >= 5 && *p.add(0) == 0x66  && *p.add(1) == 0x6C && *p.add(2) == 0x75 && *p.add(3) == 0x73 && *p.add(4) == 0x68 {
                            return Ok(&[999])
                        }
                        self.output_buffer[LABEL_OFFSET] = NO_LABEL;
                    }
                };
                let mut current_char_index:usize = 0 * 2 + HEADER_LEN;
                let mut i_start:usize;
                let mut i_end:usize = 0;

                while *p.add(i_end) != 0x7c { i_end += 1;};
                i_start = i_end;
                let rowlen = rowlen1 - 1; // ignore last newline byte
                
                let mut current_char:usize = 0;
                let mut bufpos_namespace_start = 0;
                let mut current_namespace_weight:f32 = 1.0;
                while i_end < rowlen {
                    while i_end < rowlen && *p.add(i_end) != 0x20 {
                        i_end += 1;
                    }
                    //println!("item out {:?}", std::str::from_utf8(&rr.tmp_read_buf[i_start..i_end]));
                                    
                    if *p.add(i_start) == 0x7c { // "|"
                        // new namespace index
                        i_start += 1;
                        current_char = *p.add(i_start) as usize;
                        current_char_index = self.vw_map.lookup_char_to_index[current_char] * NAMESPACE_DESC_LEN + HEADER_LEN;
                        current_char_num_of_features = 0;
                        current_namespace_weight = 1.0;
                        bufpos_namespace_start = self.output_buffer.len(); // this is only used if we will have multiple values
                        if i_end - i_start != 1 {
                            // COMPLEX NAMESPACE DEFINITION
                            // namespace more than a single letter? the only allowed format is: letter:float_value, for example  "A:2.4"
                            let mut splitter = self.tmp_read_buf[i_start..i_end].split(|char| *char == 0x3a); // ":"
                            let namespace_v = splitter.next().unwrap();
                            if namespace_v.len() != 1 {
                                //println!("a {:?}", str::from_utf8(namespace_v);
                                return Err(Box::new(IOError::new(ErrorKind::Other, format!("Only single letter namespaces are allowed"))));
                            }
                            match splitter.next() {
                                Some(v) => {current_namespace_weight = str::from_utf8_unchecked(v).parse()?},
                                None => {}
                            }
                            match splitter.next() {
                                Some(v) => {return Err(Box::new(IOError::new(ErrorKind::Other, format!("Double weight for a namespace is not allowed"))))},
                                None => {}
                            }
                        }
                    } else { 
                        // We have a feature! Let's hash it and write it to the buffer
                        // println!("item out {:?}", std::str::from_utf8(&rr.tmp_read_buf[i_start..i_end]));
                        let h = murmur3::hash32_with_seed(&self.tmp_read_buf[i_start..i_end], 
                                                          *self.namespace_hash_seeds.get_unchecked(current_char)) & MASK31;  
                        //self.output_buffer.push(h);
                        if current_namespace_weight == 1.0 {
                            if current_char_num_of_features == 0 {
                                *buf.add(current_char_index) = h;
                            } else if current_char_num_of_features == 1 {
                                // We need to promote feature currently written in-place to out of place
                                self.output_buffer.push(*buf.add(current_char_index));
                                self.output_buffer.push(FLOAT32_ONE);
                                // Then add also out of place character
                                self.output_buffer.push(h);
                                self.output_buffer.push(FLOAT32_ONE);
                                // Now we store current pointers to in-place location, with IS_NOT_SINGLE_MASK marker
                                let bufpos = self.output_buffer.len();
                                *buf.add(current_char_index) = IS_NOT_SINGLE_MASK | (((bufpos_namespace_start<<16) + bufpos) as u32);
                            } else {
                                // Now push a new value and store the new pointer to in_place
                                self.output_buffer.push(h);
                                self.output_buffer.push(FLOAT32_ONE);
                                let bufpos = self.output_buffer.len();
                                *buf.add(current_char_index) = IS_NOT_SINGLE_MASK | (((bufpos_namespace_start<<16) + bufpos) as u32);
                            }
                        } else
                        {
                                // Now push a new value and store the new pointer to in_place
                                self.output_buffer.push(h);
                                self.output_buffer.push(current_namespace_weight.to_bits());
                                let bufpos = self.output_buffer.len();
                                *buf.add(current_char_index) = IS_NOT_SINGLE_MASK | (((bufpos_namespace_start<<16) + bufpos) as u32);
                        }
                        current_char_num_of_features += 1;
                    }
                    i_end += 1;
                    i_start = i_end ;
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
    use std::io::{Write,Seek};
        
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
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [5, 1, 2988156968 & MASK31, NULL, NULL]);
        let mut buf = str_to_cursor("-1 |B b\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [5, 0, NULL, 2422381320 & MASK31, NULL]);
        // single namespace with two features
        let mut buf = str_to_cursor("1 |A a b\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [9, 1, 
                                                        nd(5,9) | IS_NOT_SINGLE_MASK, 	// |A
                                                        NULL, 				// |B 
                                                        NULL, 				// |C
                                                        2988156968 & MASK31, FLOAT32_ONE,   // |A a
                                                        3529656005 & MASK31, FLOAT32_ONE]); // |A b
        // two namespaces
        let mut buf = str_to_cursor("-1 |A a |B b\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [5, 0, 2988156968 & MASK31, 2422381320 & MASK31, NULL]);
        
        // only single letter namespaces are allowed
        let mut buf = str_to_cursor("1 |MORE_THAN_A_LETTER a\n");
        assert!(rr.next_vowpal(&mut buf).is_err());
        // namespace weight test
        let mut buf = str_to_cursor("1 |A:1.0 a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [5, 1, 2988156968 & MASK31, NULL, NULL]);
        // not a parsable number
        let mut buf = str_to_cursor("1 |A:not_a_parsable_number a\n");
        assert!(rr.next_vowpal(&mut buf).is_err());
        // double weight
        let mut buf = str_to_cursor("1 |A:1:1 a\n");
        assert!(rr.next_vowpal(&mut buf).is_err());
        // namespace weight test
        let mut buf = str_to_cursor("1 |A:2.0 a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [7, 1, 
                                                        nd(5, 7) | IS_NOT_SINGLE_MASK, 
                                                        NULL, 
                                                        NULL, 
                                                        2988156968 & MASK31, 2.0f32.to_bits()]);
                
        // LABEL TESTS
        // without label
        let mut buf = str_to_cursor("|A a\n");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [5, NO_LABEL, 2988156968 & MASK31, NULL, NULL]);
        
        
        //println!("{:?}", rr.output_buffer);
        // now we test if end-of-stream works correctly
        str_to_cursor("");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap().len(), 0);

        // flush should return [999]
        let mut buf = str_to_cursor("flush");
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [999]);

    }
}