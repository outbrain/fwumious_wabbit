use std::error::Error;
use std::net;
use std::io::{BufReader, BufWriter};
use std::io;
use std::io::Write;
use std::thread;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::time;
use std::fs::File;

use daemonize::Daemonize;

use crate::parser;
use crate::vwmap;
use crate::regressor;
use crate::feature_buffer;
use crate::model_instance;


pub struct Serving {
    listening_interface: String,
    worker_threads: Vec<thread::JoinHandle<u32>>,
    sender: mpsc::Sender<net::TcpStream>,
    foreground: bool,
}

pub struct WorkerThread {
    id: u32,
    receiver: Arc<Mutex<mpsc::Receiver<net::TcpStream>>>,
    re: Arc<regressor::FixedRegressor>,
    fbt: feature_buffer::FeatureBufferTranslator,
    vw: vwmap::VwNamespaceMap,
}

impl WorkerThread {
    pub fn new(
        id: u32,
        receiver: Arc<Mutex<mpsc::Receiver<net::TcpStream>>>,
        vw: &vwmap::VwNamespaceMap,
        re: Arc<regressor::FixedRegressor>,
        mi: &model_instance::ModelInstance,
    ) -> Result<thread::JoinHandle<u32>, Box<dyn Error>> {
        let mut mi = WorkerThread {
            id: id,
            receiver: receiver,
            re: re,   // THIS IS NOT A SMALL STRUCTURE
            fbt: feature_buffer::FeatureBufferTranslator::new(mi),
            vw: (*vw).clone(),

        };
        let thread = thread::spawn(move || {
            mi.start();
            1u32
        });
        Ok(thread)
    }

    pub fn start(&mut self) -> () {
        let mut pa = parser::VowpalParser::new(&self.vw);

        loop {
            let tcp_stream = self.receiver.lock().unwrap().recv().unwrap();
            let mut writer = BufWriter::new(&tcp_stream);
            let mut reader = BufReader::new(&tcp_stream);

            //        println!("New connection");
            let mut i = 0u32;
            loop {
                let reading_result = pa.next_vowpal(&mut reader);
                let mut p_res:String;
                match reading_result {
                    Ok([]) => break, // EOF
                    Ok([999]) => {
                        match writer.flush() {
                            Ok(_) => {},
                            Err(e) => { /*println!("Flushing socket failed, dropping it");*/ break; }
                        }
                    },
                    Ok(buffer2) => {
                        self.fbt.translate_vowpal(buffer2);
                        let p = self.re.predict(&(self.fbt.feature_buffer), i);
                        p_res = format!("{:.6}\n", p);
                        match writer.write_all(p_res.as_bytes()) {
                            Ok(_) => {},
                            Err(e) => { /*println!("Write to socket failed, dropping it"); */break; }
                        };
                    },
                    Err(e) =>
                        {
                            p_res = format!("ERR: {}\n", e.to_string());
                            match writer.write_all(p_res.as_bytes()) {
                                Ok(_) => match writer.flush() {
                                    Ok(_) => {},
                                    Err(e) => { /*println!("Flushing socket failed, dropping it");*/ break; }
                                },
                                Err(e) => { /*println!("Write to socket failed, dropping it"); */break; }
                            };
                            break
                        },
                };

                // not the smartest
                // this is lazy flushing
                if reader.buffer().is_empty() {
                    match writer.flush() {
                        Ok(_) => {},
                        Err(e) => { /*println!("Flushing socket failed, dropping it");*/ break; }
                    };
                }
                i += 1;
            }
        }
    }

//    pub fn handle_connection(&mut self, tcp_stream: net::TcpStream, mut pa: &mut parser::VowpalParser) {
}


impl Serving {
    pub fn new<'a>(cl: &clap::ArgMatches<'a>,
                   vw: &vwmap::VwNamespaceMap,
                   re: regressor::Regressor,
                   mi: &model_instance::ModelInstance,
    ) -> Result<Serving, Box<dyn Error>> {
        let port = match cl.value_of("port") {
            Some(port) => port.parse().expect("Port should be integer"),
            None => 26542
        };
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));


        let re_fixed = Arc::new(regressor::FixedRegressor::new(re));
//        println!("Stage2");

        let listening_interface = format!("127.0.0.1:{}", port);
        println!("Starting to listen on {}", listening_interface);
        let mut s = Serving {
            listening_interface: listening_interface.to_string(),
            worker_threads: Vec::new(),
            sender: sender,
            foreground: cl.is_present("foreground"),
        };

        let num_children = match cl.value_of("num_children") {
            Some(num_children) => num_children.parse().expect("num_children should be integer"),
            None => 10
        };
        println!("Number of threads {}", num_children);

        if !s.foreground {
            //  let stdout = File::create("/tmp/daemon.out").unwrap();
            //  let stderr = File::create("/tmp/daemon.err").unwrap();
            let daemonize = Daemonize::new();
            //.stdout(stdout)  // Redirect stdout to `/tmp/daemon.out`.
            //.stderr(stderr);  // Redirect stderr to `/tmp/daemon.err`.;
            match daemonize.start() {
                Ok(_) => println!("Success, daemonized"),
                Err(e) => return Err(e)?,
            }
        }

        for i in 0..num_children {
            let newt = WorkerThread::new(i,
                                         Arc::clone(&receiver),
                                         vw,
                                         re_fixed.clone(),
                                         mi,
            )?;
            s.worker_threads.push(newt);
        }
        Ok(s)
    }

    pub fn serve(&mut self) -> Result<(), Box<dyn Error>> {
        let listener = net::TcpListener::bind(&self.listening_interface).expect("Cannot bind to the interface");
        println!("Bind done, deamonizing and calling accept");
        for stream in listener.incoming() {
            self.sender.send(stream?)?;
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_exception() {
        let v = "abc5";
        match v.parse::<i32>() {
            Ok(pVal) => {},
            Err(e) =>
                {
                    let msg = e.to_string();
                    let msg = format!("ERR:{}", msg);
                    println!("{}", msg);
                    assert_eq! ("ERR:invalid digit found in string", msg);
                }
            ,
        };
    }



}

