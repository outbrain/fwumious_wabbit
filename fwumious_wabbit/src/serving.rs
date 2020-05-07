use std::error::Error;
use std::net;
use std::io::{BufReader,BufWriter};
use std::io::Write;
use std::thread;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::time;

use crate::parser;
use crate::vwmap;
use crate::regressor;
use crate::feature_buffer;
use crate::model_instance;



pub struct Serving {
    listening_interface: String,
    worker_threads: Vec<thread::JoinHandle<u32>>,
    sender: mpsc::Sender<net::TcpStream>,

}

pub struct WorkerThread {
    id: u32,
    receiver: Arc<Mutex<mpsc::Receiver<net::TcpStream>>>,
    vw: vwmap::VwNamespaceMap, 
    re: Arc<regressor::FixedRegressor>, 
    mi: model_instance::ModelInstance,
}

impl WorkerThread {
    pub fn new(
            id: u32, 
            receiver: Arc<Mutex<mpsc::Receiver<net::TcpStream>>>,
            vw: &vwmap::VwNamespaceMap, 
            re: Arc<regressor::FixedRegressor>, 
            mi: &model_instance::ModelInstance
    
    ) -> Result<thread::JoinHandle<u32>, Box<dyn Error>> {
        let mut mi = WorkerThread {	
            id: id,
            receiver: receiver,
            vw: (*vw).clone(),   // these are small structures and we are ok with cloning
            re: re,   // THIS IS NOT A SMALL STRUCTURE
            mi: (*mi).clone(),
        };
        let thread = thread::spawn(move || {mi.start(); 1u32});
        Ok(thread)
    }
    
    pub fn start(&self) -> () {
        loop {
            let tcp_stream = self.receiver.lock().unwrap().recv().unwrap();
            self.handle_connection(tcp_stream);
        // wait for work and do it 
        }
    }
    
    pub fn handle_connection(&self, tcp_stream: net::TcpStream) {
        let mut reader = BufReader::new(&tcp_stream);
        let mut writer = BufWriter::new(&tcp_stream);
        let mut pa = parser::VowpalParser::new(&mut reader, &self.vw);
        let mut fb = feature_buffer::FeatureBuffer::new(&self.mi);

        println!("New connection");
        let mut i = 0u32;
        loop {
            let reading_result = pa.next_vowpal();
            let mut buffer: &[u32] = match reading_result {
                    Ok([]) => return (), // EOF
                    Ok(buffer2) => buffer2,
                    Err(e) => return (),
            };
            fb.translate_vowpal(buffer);
            let p = self.re.predict(&(fb.output_buffer), i);
            writer.write_all(format!("{:.6}\n", p).as_bytes()).unwrap();
            writer.flush().unwrap(); // This is definitely not the most efficient way of doing this, but let's make it perdictable first, fast second
            i += 1;
        }

    }
    
}
 



impl Serving {
    pub fn new<'a>(cl: &clap::ArgMatches<'a>,
        vw: &vwmap::VwNamespaceMap, 
        re: regressor::Regressor, 
        mi: &model_instance::ModelInstance
    
    ) -> Result<Serving, Box<dyn Error>> {
        let port = match cl.value_of("port") {
            Some(port) => port.parse().expect("Port should be integer"),
            None => 26542
        };
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        
        
        thread::sleep(time::Duration::from_millis(10000));
        
        let re_fixed = Arc::new(regressor::FixedRegressor::new(re));
        
        let listening_interface = format!("127.0.0.1:{}", port);
        println!("Starting to listen on {}", listening_interface); 
        let mut s = Serving {
            listening_interface:listening_interface.to_string(),
            worker_threads: Vec::new(),
            sender: sender,
            
        };
 
        
        let num_children = match cl.value_of("num_children") {
            Some(num_children) => num_children.parse().expect("num_children should be integer"),
            None => 10
        };
        
        println!("Number of threads {}", num_children);
        for i in 1..10 {
            let newt =WorkerThread::new(1, 
                                Arc::clone(&receiver),
                                vw,
                                re_fixed.clone(),
                                mi,                    
                                )?;
            s.worker_threads.push(newt);
        }
        Ok(s)
    }
    
    
    pub fn serve(&mut self)  -> Result<(), Box<dyn Error>> {
        let listener = net::TcpListener::bind(&self.listening_interface)?;
        println!("Bind done, calling accept");
        for stream in listener.incoming() {
            self.sender.send(stream?)?;
        }
        Ok(())
    }
    
}