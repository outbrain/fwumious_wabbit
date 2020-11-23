#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(non_snake_case)]

 
mod java_glue;
mod jni_c_header;
pub use crate::java_glue::*;


mod vwmap;
mod parser;
mod model_instance;
mod feature_buffer;
mod regressor;
mod cmdline;
mod cache;
mod persistence;
mod serving;
mod optimizer;
mod version;
mod session;



// src/lib.rs
struct ASession {
    a: i32,
}

impl ASession {
    pub fn new() -> ASession {
        ASession { a: 2 }
    }

    pub fn add_and1(&self, val: i32) -> i32 {
        self.a + val + 1
    }

    // Greeting with full, no-runtime-cost support for newlines and UTF-8
    pub fn greet(to: &str) -> String {
        format!("Hello {} ✋\nIt's a pleasure to meet you!", to)
    }
}


