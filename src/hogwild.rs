use std::error::Error;
use std::sync::{Arc, mpsc, Mutex};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::thread::JoinHandle;
use crate::feature_buffer::FeatureBuffer;
use crate::multithread_helpers::BoxedRegressorTrait;
use crate::port_buffer::PortBuffer;
use crate::regressor::Regressor;

pub struct HogwildTrainer {
    workers: Vec<JoinHandle<()>>,
    sender: Sender<FeatureBuffer>,
}

pub struct HogwildWorker {
    regressor: BoxedRegressorTrait,
    port_buffer: PortBuffer
}

impl HogwildTrainer {
    pub fn new(sharable_regressor: BoxedRegressorTrait, numWorkers: u32) -> HogwildTrainer {
        let (sender, receiver): (Sender<FeatureBuffer>, Receiver<FeatureBuffer>) = mpsc::channel();
        let mut trainer = HogwildTrainer {
            workers: Vec::new(),
            sender,
        };
        let receiver: Arc<Mutex<Receiver<FeatureBuffer>>> = Arc::new(Mutex::new(receiver));
        let port_buffer = sharable_regressor.new_portbuffer();
        for i in 0..numWorkers {
            let worker = HogwildWorker::new(
                sharable_regressor.clone(), 
                port_buffer.clone(), 
                Arc::clone(&receiver)
            );
            trainer.workers.push(worker);
        }
        trainer
    }
    
    pub fn new_dummy() -> HogwildTrainer {
        let (sender, receiver): (Sender<FeatureBuffer>, Receiver<FeatureBuffer>) = mpsc::channel();
        return HogwildTrainer {
            workers: vec![],
            sender,
        };
    }

    pub fn digest_example(&self, feature_buffer: FeatureBuffer) {
        self.sender.send(feature_buffer).unwrap();
    }

    pub fn block_until_workers_finished(self) {
        drop(self.sender);
        for worker in self.workers {
            worker.join().unwrap();
        } 
    }
}

impl HogwildWorker {
    pub fn new(
        regressor: BoxedRegressorTrait,
        port_buffer: PortBuffer,
        receiver: Arc<Mutex<Receiver<FeatureBuffer>>>
    ) -> JoinHandle<()> {
        let mut worker = HogwildWorker {
            regressor,
            port_buffer
        };
        let thread = thread::spawn(move || {
            worker.train(receiver)
        });
        thread
    }

    pub fn train(&mut self, receiver: Arc<Mutex<Receiver<FeatureBuffer>>>) {
        loop {
            let feature_buffer = match receiver.lock().unwrap().recv() {
                Ok(feature_buffer) => feature_buffer,
                Err(RecvError) => break // channel was closed
            };
            self.regressor.learn(&feature_buffer, &mut self.port_buffer, true);
        }
    }
}
