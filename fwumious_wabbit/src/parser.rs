use std::fs::File;
use fasthash::murmur3;
use fasthash::xx;
use std::io;
use std::io::BufRead;
use std::error::Error;

use crate::vwmap;

const RECBUF_LEN:usize = 2048;
pub const HEADER_LEN:usize = 2;
pub const NAMESPACE_DESC_LEN:usize = 1;
pub const LABEL_OFFSET:usize = 1;
pub const IS_NOT_SINGLE_MASK : u32 = 1u32 << 31;
pub const MASK31: u32 = !IS_NOT_SINGLE_MASK;
pub const NULL: u32= IS_NOT_SINGLE_MASK; // null is just an exact IS_NOT_SINGLE_MASK
pub const NO_LABEL: u32 = 0xff;

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
    -- if the most significant bit is zero, this is a single value feature and bits 1-31 are a hash
    -- if the most significant bit is zero, then, 15 next bits are the start offset, and 16 next bits are the end offset of features beyond the buffer 
(u32)[dynamic_number_of_hashes]
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
                while i_end < rowlen {
                    while i_end < rowlen && *p.add(i_end) != 0x20 {
                        i_end += 1;
                    }
                    //println!("item out {:?}", std::str::from_utf8(&rr.tmp_read_buf[i_start..i_end]));
                                    
                    if *p.add(i_start) == 0x7c { // "|"
                        // As we get new namespace index, we store end of the last one
                        //println!("Closing index {} {}", current_char_index, bufpos);
                        current_char = *p.add(i_start+1) as usize;
                        current_char_index = self.vw_map.lookup_char_to_index[current_char] * NAMESPACE_DESC_LEN + HEADER_LEN;
                        current_char_num_of_features = 0;
                    } else { 
                        // We have a feature! Let's hash it and write it to the buffer
                        // println!("item out {:?}", std::str::from_utf8(&rr.tmp_read_buf[i_start..i_end]));
                        let h = murmur3::hash32_with_seed(&self.tmp_read_buf[i_start..i_end], 
                                                          *self.namespace_hash_seeds.get_unchecked(current_char)) & MASK31;  
                        //self.output_buffer.push(h);
                        if current_char_num_of_features == 0 {
                            *buf.add(current_char_index) = h;
                        } else if current_char_num_of_features == 1 {
                            // We need to promote feature currently written in-place to out of place
                            bufpos_namespace_start = self.output_buffer.len();
                            self.output_buffer.push(*buf.add(current_char_index));
                            // Then add also out of place character
                            self.output_buffer.push(h);
                            // Now we store current pointers to in-place location, with IS_NOT_SINGLE_MASK marker
                            let bufpos = self.output_buffer.len();
                            *buf.add(current_char_index) = IS_NOT_SINGLE_MASK | (((bufpos_namespace_start<<16) + bufpos) as u32);
                        } else {
                            // Now push a new value and store the new pointer to in_place
                            self.output_buffer.push(h);
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
        fn str_to_cursor(c: &mut Cursor<Vec<u8>>, s: &str) -> () {
          c.seek(std::io::SeekFrom::Start(0)).unwrap();
          c.write(s.as_bytes()).unwrap();
          c.seek(std::io::SeekFrom::Start(0)).unwrap();
        }

        let mut buf = Cursor::new(Vec::new());
        str_to_cursor(&mut buf, 
r#"1 |A a
-1 |B b
1 |A a b
-1 |A a |B b
|A a
"#);
        let mut rr = VowpalParser::new(&vw);
        // we test a single record
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [5, 1, 2988156968 & MASK31, NULL, NULL]);
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [5, 0, NULL, 2422381320 & MASK31, NULL]);
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [7, 1, nd(5,7) | IS_NOT_SINGLE_MASK, NULL, NULL, 2988156968 & MASK31, 3529656005 & MASK31]);
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [5, 0, 2988156968 & MASK31, 2422381320 & MASK31, NULL]);
        // without label
        assert_eq!(rr.next_vowpal(&mut buf).unwrap(), [5, NO_LABEL, 2988156968 & MASK31, NULL, NULL]);
        
        
        //println!("{:?}", rr.output_buffer);
        // now we test if end-of-stream works correctly
        assert_eq!(rr.next_vowpal(&mut buf).unwrap().len(), 0);
    }
}