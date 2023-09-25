#![allow(dead_code,unused_imports)]

use daemonize::Daemonize;
use std::error::Error;
use std::io;
use std::io::{BufReader, BufWriter};
use std::net;
use std::ops::DerefMut;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use crate::feature_buffer;
use crate::model_instance;
use crate::multithread_helpers::BoxedRegressorTrait;
use crate::parser;
use crate::persistence;
use crate::port_buffer;
use crate::regressor;
use crate::vwmap;

pub struct Serving {
    listening_interface: String,
    worker_threads: Vec<thread::JoinHandle<u32>>,
    sender: mpsc::Sender<net::TcpStream>,
    foreground: bool,
}

pub struct WorkerThread {
    id: u32,
    re_fixed: BoxedRegressorTrait,
    fbt: feature_buffer::FeatureBufferTranslator,
    pa: parser::VowpalParser,
    pb: port_buffer::PortBuffer,
}

pub trait IsEmpty {
    fn is_empty(&mut self) -> bool;
}
impl IsEmpty for io::BufReader<&net::TcpStream> {
    fn is_empty(&mut self) -> bool {
        return self.buffer().is_empty();
    }
}

// These are used only for unit-tests
#[derive(Debug, PartialEq)]
pub enum ConnectionEnd {
    EndOfStream,
    StreamWriteError,
    StreamFlushError,
    ParseError,
}

impl WorkerThread {
    pub fn new(
        id: u32,
        re_fixed: BoxedRegressorTrait,
        fbt: feature_buffer::FeatureBufferTranslator,
        pa: parser::VowpalParser,
        pb: port_buffer::PortBuffer,
        receiver: Arc<Mutex<mpsc::Receiver<net::TcpStream>>>,
    ) -> Result<thread::JoinHandle<u32>, Box<dyn Error>> {
        let mut wt = WorkerThread {
            id,
            re_fixed,
            fbt,
            pa,
            pb,
        };
        let thread = thread::spawn(move || {
            wt.start(receiver);
            1u32
        });
        Ok(thread)
    }

    pub fn handle_connection(
        &mut self,
        reader: &mut (impl io::BufRead + IsEmpty),
        writer: &mut impl io::Write,
    ) -> ConnectionEnd {
        let mut i = 0u64; // This is per-thread example number
        loop {
            let reading_result = self.pa.next_vowpal(reader);

            match reading_result {
                Ok([]) => return ConnectionEnd::EndOfStream, // EOF
                Ok(buffer2) => {
                    self.fbt.translate(buffer2, i);
                    let p = self
                        .re_fixed
                        .predict(&(self.fbt.feature_buffer), &mut self.pb);
                    let p_res = format!("{:.6}\n", p);
                    match writer.write_all(p_res.as_bytes()) {
                        Ok(_) => {}
                        Err(_e) => {
                            return ConnectionEnd::StreamWriteError;
                        }
                    };
                }
                Err(e) => {
                    if e.is::<parser::FlushCommand>() {
                        // FlushCommand just causes us to flush, not to break
                        match writer.flush() {
                            Ok(_) => {}
                            Err(_e) => {
                                /*println!("Flushing the socket failed, dropping it");*/
                                return ConnectionEnd::StreamFlushError;
                            }
                        }
                    } else if e.is::<parser::HogwildLoadCommand>() {
                        // FlushCommand just causes us to flush, not to break
                        let hogwild_command =
                            e.downcast_ref::<parser::HogwildLoadCommand>().unwrap();
                        match persistence::hogwild_load(
                            self.re_fixed.deref_mut(),
                            &hogwild_command.filename,
                        ) {
                            Ok(_) => {
                                let p_res = "hogwild_load success\n".to_string();
                                match writer.write_all(p_res.as_bytes()) {
                                    Ok(_) => {}
                                    Err(_e) => {
                                        return ConnectionEnd::StreamWriteError;
                                    }
                                };
                            }
                            Err(_e) => {
                                // TODO This kind of error should fold the whole daemon...
                                let p_res = "ERR: hogwild_load fail\n".to_string();
                                match writer.write_all(p_res.as_bytes()) {
                                    Ok(_) => {}
                                    Err(_e) => {
                                        return ConnectionEnd::StreamWriteError;
                                    }
                                };
                                return ConnectionEnd::StreamWriteError;
                            }
                        }
                    } else {
                        let p_res = format!("ERR: {}\n", e);
                        match writer.write_all(p_res.as_bytes()) {
                            Ok(_) => match writer.flush() {
                                Ok(_) => {}
                                Err(_e) => {
                                    return ConnectionEnd::StreamFlushError;
                                }
                            },
                            Err(_e) => {
                                return ConnectionEnd::StreamWriteError;
                            }
                        };
                        return ConnectionEnd::ParseError;
                    }
                }
            };

            // lazy flushing
            if reader.is_empty() {
                match writer.flush() {
                    Ok(_) => {}
                    Err(_e) => {
                        return ConnectionEnd::StreamFlushError;
                    }
                };
            }
            i += 1;
        }
    }

    pub fn start(&mut self, receiver: Arc<Mutex<mpsc::Receiver<net::TcpStream>>>) {
        // Simple endless serving loop: receive new connection and serve it
        // when handle_connection exits, the connection is dropped
        loop {
            let tcp_stream = receiver.lock().unwrap().recv().unwrap();
            let mut reader = BufReader::new(&tcp_stream);
            let mut writer = BufWriter::new(&tcp_stream);
            self.handle_connection(&mut reader, &mut writer);
        }
    }
}

impl Serving {
    pub fn new(
        cl: &clap::ArgMatches<'_>,
        vw: &vwmap::VwNamespaceMap,
        re_fixed: Box<regressor::Regressor>,
        mi: &model_instance::ModelInstance,
    ) -> Result<Serving, Box<dyn Error>> {
        let port = match cl.value_of("port") {
            Some(port) => port.parse().expect("Port should be integer"),
            None => 26542,
        };
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));

        let listening_interface = format!("127.0.0.1:{}", port);
        log::info!("Starting to listen on {}", listening_interface);
        let mut s = Serving {
            listening_interface,
            worker_threads: Vec::new(),
            sender,
            foreground: cl.is_present("foreground"),
        };

        let num_children = match cl.value_of("num_children") {
            Some(num_children) => num_children
                .parse()
                .expect("num_children should be integer"),
            None => 10,
        };
        log::info!("Number of threads {}", num_children);

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

        let re_fixed2 = BoxedRegressorTrait::new(re_fixed);
        let pb = re_fixed2.new_portbuffer();
        let fbt = feature_buffer::FeatureBufferTranslator::new(mi);
        let pa = parser::VowpalParser::new(vw);
        for i in 0..num_children {
            let newt = WorkerThread::new(
                i,
                re_fixed2.clone(),
                fbt.clone(),
                pa.clone(),
                pb.clone(),
                Arc::clone(&receiver),
            )?;
            s.worker_threads.push(newt);
        }
        Ok(s)
    }

    pub fn serve(&mut self) -> Result<(), Box<dyn Error>> {
        let listener = net::TcpListener::bind(&self.listening_interface)
            .expect("Cannot bind to the interface");
        log::info!("Bind done, deamonizing and calling accept");
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
    use crate::feature_buffer;
    use crate::regressor;
    use mockstream::{FailingMockStream, SharedMockStream};
    use std::io::ErrorKind;
    use std::str;
    use tempfile::tempdir;

    impl IsEmpty for std::io::BufReader<mockstream::SharedMockStream> {
        fn is_empty(&mut self) -> bool {
            true
        }
    }
    impl IsEmpty for std::io::BufReader<mockstream::FailingMockStream> {
        fn is_empty(&mut self) -> bool {
            true
        }
    }

    #[test]
    fn test_handle_connection() {
        let vw_map_string = r#"
A,featureA
B,featureB
C,featureC
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.optimizer = model_instance::Optimizer::AdagradLUT;
        let mut re = regressor::Regressor::new(&mi);
        mi.optimizer = model_instance::Optimizer::SGD;
        let re_fixed = BoxedRegressorTrait::new(Box::new(re.immutable_regressor(&mi).unwrap()));
        let fbt = feature_buffer::FeatureBufferTranslator::new(&mi);
        let pa = parser::VowpalParser::new(&vw);
        let pb = re_fixed.new_portbuffer();

        let mut newt = WorkerThread {
            id: 1,
            fbt,
            pa,
            re_fixed,
            pb,
        };

        {
            // WORKING STREAM TEST
            let mut mocked_stream = SharedMockStream::new();
            let mut reader = BufReader::new(mocked_stream.clone());
            let mut writer = BufWriter::new(mocked_stream.clone());
            // Just passes through, as the stream is empty
            newt.handle_connection(&mut reader, &mut writer);

            // now let's start playing
            mocked_stream.push_bytes_to_read(b"|A 0 |A 0");
            assert_eq!(
                ConnectionEnd::EndOfStream,
                newt.handle_connection(&mut reader, &mut writer)
            );
            let x = mocked_stream.pop_bytes_written();
            assert_eq!(x, b"0.500000\n");

            mocked_stream.push_bytes_to_read(b"1 |A 0 |A 0");
            assert_eq!(
                ConnectionEnd::EndOfStream,
                newt.handle_connection(&mut reader, &mut writer)
            );
            let x = mocked_stream.pop_bytes_written();
            assert_eq!(x, b"0.500000\n");

            mocked_stream.push_bytes_to_read(b"! exclamation mark is not a valid label");
            assert_eq!(
                ConnectionEnd::ParseError,
                newt.handle_connection(&mut reader, &mut writer)
            );
            let x = mocked_stream.pop_bytes_written();
            assert_eq!(&x[..], &b"ERR: Cannot parse an example\n"[..]);
        }

        // Non Working stream test

        {
            let mut mocked_stream_ok = SharedMockStream::new();
            let mocked_stream_error = FailingMockStream::new(ErrorKind::Other, "Failing", 3);
            let mut reader = BufReader::new(mocked_stream_ok.clone());
            let mut writer = BufWriter::new(mocked_stream_error.clone());
            mocked_stream_ok.push_bytes_to_read(b"|A 0 |A 0");
            // Now there will be an error
            assert_eq!(
                ConnectionEnd::StreamFlushError,
                newt.handle_connection(&mut reader, &mut writer)
            );
            let mut reader = BufReader::new(mocked_stream_error.clone());
            let mut writer = BufWriter::new(mocked_stream_error);
            // Now there will be an error
            assert_eq!(
                ConnectionEnd::StreamFlushError,
                newt.handle_connection(&mut reader, &mut writer)
            );
        }
    }

    #[test]
    fn test_hogwild() {
        let vw_map_string = r#"
A,featureA
B,featureB
C,featureC
"#;
        let vw = vwmap::VwNamespaceMap::new(vw_map_string).unwrap();
        let mut mi = model_instance::ModelInstance::new_empty().unwrap();
        mi.learning_rate = 0.1;
        mi.power_t = 0.0;
        mi.bit_precision = 18;
        mi.ffm_k = 1;
        mi.ffm_bit_precision = 18;
        mi.ffm_power_t = 0.0;
        mi.ffm_learning_rate = 0.1;
        mi.ffm_fields = vec![vec![], vec![]];
        mi.optimizer = model_instance::Optimizer::AdagradLUT;
        let re_1 = regressor::Regressor::new(&mi);
        mi.optimizer = model_instance::Optimizer::SGD;
        let re_2 = regressor::Regressor::new(&mi);

        let dir = tempdir().unwrap();
        let regressor_filepath_1 = dir
            .path()
            .join("test_regressor1.fw")
            .to_str()
            .unwrap()
            .to_owned();
        persistence::save_regressor_to_filename(&regressor_filepath_1, &mi, &vw, re_1).unwrap();

        let regressor_filepath_2 = dir
            .path()
            .join("test_regressor2.fw")
            .to_str()
            .unwrap()
            .to_owned();
        persistence::save_regressor_to_filename(&regressor_filepath_2, &mi, &vw, re_2).unwrap();

        // OK NOW EVERYTHING IS READY... Let's start
        mi.optimizer = model_instance::Optimizer::AdagradLUT;
        let mut re = regressor::Regressor::new(&mi);
        mi.optimizer = model_instance::Optimizer::SGD;
        let re_fixed = BoxedRegressorTrait::new(Box::new(re.immutable_regressor(&mi).unwrap()));
        let fbt = feature_buffer::FeatureBufferTranslator::new(&mi);
        let pa = parser::VowpalParser::new(&vw);
        let pb = re_fixed.new_portbuffer();

        let mut newt = WorkerThread {
            id: 1,
            fbt,
            pa,
            re_fixed,
            pb,
        };

        {
            // WORKING STREAM TEST
            let mut mocked_stream = SharedMockStream::new();
            let mut reader = BufReader::new(mocked_stream.clone());
            let mut writer = BufWriter::new(mocked_stream.clone());
            // Just passes through, as the stream is empty
            newt.handle_connection(&mut reader, &mut writer);

            // now let's start playing
            mocked_stream
                .push_bytes_to_read(format!("hogwild_load {}", &regressor_filepath_1).as_bytes());
            assert_eq!(
                ConnectionEnd::EndOfStream,
                newt.handle_connection(&mut reader, &mut writer)
            );
            let x = mocked_stream.pop_bytes_written();
            assert_eq!(
                str::from_utf8(&x),
                str::from_utf8(b"hogwild_load success\n")
            );

            // now incompatible regressor - should return error
            mocked_stream
                .push_bytes_to_read(format!("hogwild_load {}", &regressor_filepath_2).as_bytes());
            assert_eq!(
                ConnectionEnd::EndOfStream,
                newt.handle_connection(&mut reader, &mut writer)
            );
            let x = mocked_stream.pop_bytes_written();
            assert_eq!(
                str::from_utf8(&x),
                str::from_utf8(b"hogwild_load success\n")
            );
            /*

                        assert_eq!(ConnectionEnd::StreamWriteError, newt.handle_connection(&mut reader, &mut writer));
                        let x = mocked_stream.pop_bytes_written();
                        assert_eq!(str::from_utf8(&x), str::from_utf8(b""));
            */
            // file does not exist
            mocked_stream.push_bytes_to_read("hogwild_load /fba/baba/ba".as_bytes());
            assert_eq!(
                ConnectionEnd::StreamWriteError,
                newt.handle_connection(&mut reader, &mut writer)
            );
            let x = mocked_stream.pop_bytes_written();
            assert_eq!(str::from_utf8(&x), str::from_utf8(b""));
        }
    }
}
