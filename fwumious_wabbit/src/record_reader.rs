use std::fs::File;
use fasthash::xx;
use std::io;
use std::io::BufRead;
//mod vwmap;

use crate::vwmap;

const RECBUF_LEN:usize = 2048;
pub const HEADER_LEN:usize = 2;
pub const LABEL_OFFSET:usize = 1;

pub struct RecordReader<'a> {
    input_bufreader: &'a mut dyn BufRead,
    vw_map: &'a vwmap::VwNamespaceMap,
    tmp_read_buf: Vec<u8>,
    pub output_buffer: Vec<u32>
}

#[derive(Debug)]
#[derive(PartialEq)]
pub enum NextRecordResult {
    Ok,
    End,
    Error,
}



/* 
organization of records buffer 
(offset u16, len u16)[number_of_features]
(u32)[dynamic_number_of_hashes] 
*/

impl<'a> RecordReader<'a> {
//    pub fn new(buffered_input: io::BufReader<File>, vw: &'a vwmap::VwNamespaceMap) -> RecordReader {
    pub fn new(buffered_input: &'a mut dyn BufRead, vw: &'a vwmap::VwNamespaceMap) -> RecordReader<'a> {
        let mut rr = RecordReader {  
                            input_bufreader: buffered_input, 
                            vw_map: vw,
                            tmp_read_buf: Vec::with_capacity(RECBUF_LEN),
                            output_buffer: Vec::with_capacity(RECBUF_LEN*2)
                        };
        rr.output_buffer.resize(vw.num_namespaces as usize * 2 + HEADER_LEN, 0);
        rr
    }
    
    pub fn print(&self) -> () {
        println!("item out {:?}", self.output_buffer);
    }


    pub fn next(&mut self) -> NextRecordResult {
            // Output item
            self.tmp_read_buf.truncate(0);
            //println!("{:?}", self.tmp_read_buf);
            let rowlen1 = match self.input_bufreader.read_until(0x0a, &mut self.tmp_read_buf) {
                Err(e) => return NextRecordResult::Error,
                Ok(0) => {
                        self.output_buffer.truncate(0);
                        return NextRecordResult::End
                        },
                Ok(len) => len,
            };
            //println!("{:?}", self.tmp_read_buf);
            // Fields are: 
            // u32 length of the output record
            // u32 label
            // [u32, u32] * number of namespaces - beginnings and ends of the indexes 
            let mut bufpos: usize = (self.vw_map.num_namespaces as usize) * 2 + HEADER_LEN;
            self.output_buffer.truncate(bufpos);
            for i in &mut self.output_buffer[0..bufpos] { *i = 0 }     

            unsafe {
                let p = self.tmp_read_buf.as_ptr();
                let mut buf = self.output_buffer.as_mut_ptr();
                
                // first token, either 1 or -1
                match *p.add(0) {
                    0x31 => self.output_buffer[LABEL_OFFSET] = 1,    // 1
                    0x2d => self.output_buffer[LABEL_OFFSET] = 0,    // -1
                    _ => return NextRecordResult::Error
                };
                let mut current_char_index:usize = 0 * 2 + HEADER_LEN;
                let mut i_start:usize;
                let mut i_end:usize = 0;

                while *p.add(i_end) != 0x7c { i_end += 1;};
                i_start = i_end;
                let rowlen = rowlen1 - 1; // ignore last newline byte
                
                while i_end < rowlen {
                    while i_end < rowlen && *p.add(i_end) != 0x20 {
                        i_end += 1;
                    }
                    //println!("item out {:?}", std::str::from_utf8(&rr.tmp_read_buf[i_start..i_end]));
                                    
                    if *p.add(i_start) == 0x7c { // "|"
                        // As we get new namespace index, we store end of the last one
                        //println!("Closing index {} {}", current_char_index, bufpos);
                        *buf.add(current_char_index + 1) = bufpos as u32;
                        current_char_index = self.vw_map.lookup_char_to_index[*p.add(i_start+1) as usize] * 2 + HEADER_LEN;
                        //println!("Opening index {} {}", current_char_index, bufpos);
                        *buf.add(current_char_index) = bufpos as u32;
                    } else { 
                        // We have a feature! Let's hash it and write it to the buffer
                        // println!("item out {:?}", std::str::from_utf8(&rr.tmp_read_buf[i_start..i_end]));
                        let h = xx::hash32(&self.tmp_read_buf[i_start..i_end]);  
                        // TODO: this might invalidate "buf" ... we need to work on that
                        self.output_buffer.push(h);
                        bufpos += 1;
                    }
                    
                    i_end += 1;
                    i_start = i_end ;
                    
                }
                *buf.add(current_char_index+1) = bufpos as u32;
                *buf = bufpos as u32;
            }
            
            //println!("item out {:?}", rr.output_buffer);
            NextRecordResult::Ok
        }
}


#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use vwmap;
    use std::io::Cursor;
    use std::io::{Write,Seek};
        
    #[test]
    fn test_all() {
        let vw_map_string = r#"
A,featureA
B,featureB
C,featureC
"#;
        let vw = vwmap::get_global_map_from_string(vw_map_string).unwrap();
        fn str_to_cursor(c: &mut Cursor<Vec<u8>>, s: &str) -> () {
          c.seek(std::io::SeekFrom::Start(0)).unwrap();
          c.write(s.as_bytes()).unwrap();
          c.seek(std::io::SeekFrom::Start(0)).unwrap();
        }

        let mut buf = Cursor::new(Vec::new());
        str_to_cursor(&mut buf, 
r#"1 |A a
-1 |B b
1 |C b c
"#);
        let mut rr = RecordReader::new(&mut buf, &vw);
        // we test a single record
        assert_eq!(rr.next(), NextRecordResult::Ok);
        assert_eq!(rr.output_buffer, vec![9, 1, 8, 9, 0, 0, 0, 0, 1426945110]);
        assert_eq!(rr.next(), NextRecordResult::Ok);
        assert_eq!(rr.output_buffer, vec![9, 0, 0, 8, 8, 9, 0, 0, 2718739903]);
        assert_eq!(rr.next(), NextRecordResult::Ok);
        assert_eq!(rr.output_buffer, vec![10, 1, 0, 8, 0, 0, 8, 10, 2718739903, 4004515611]);
        //println!("{:?}", rr.output_buffer);
        // now we test if end-of-stream works correctly
        assert_eq!(rr.next(), NextRecordResult::End);
        assert_eq!(rr.output_buffer.len(), 0);

  //      assert_eq!(rr.next(), NextRecordResult::Ok);
    //    assert_eq!(rr.output_buffer, vec![9, 1, 8, 9, 0, 0, 0, 0, 1426945110]);




//        println!("rr {:?}", rr.output_buffer);
  //      str_to_cursor(&mut buf, "1 |\n");
  //       buf.write("ABC".as_bytes()).unwrap();
  //      assert_eq!(rr.next(), NextRecordResult::End);
        
        
        
//        assert_eq!(fb.output_buffer, vec![1, 0xfea, 0xfeb]);
    }



}

