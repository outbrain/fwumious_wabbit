use std::error::Error;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

use crate::feature_buffer::{FeatureBuffer, FeatureBufferTranslator};
use crate::model_instance::ModelInstance;
use crate::multithread_helpers::BoxedRegressorTrait;
use crate::port_buffer::PortBuffer;
use crate::regressor::Regressor;

static CHANNEL_CAPACITY: usize = 100_000;

pub struct HogwildTrainer {
    workers: Vec<JoinHandle<()>>,
    sender: SyncSender<Vec<u32>>,
}

pub struct HogwildWorker {
    regressor: BoxedRegressorTrait,
    feature_buffer_translator: FeatureBufferTranslator,
    port_buffer: PortBuffer,
}

impl HogwildTrainer {
    pub fn new(
        sharable_regressor: BoxedRegressorTrait,
        model_instance: &ModelInstance,
        numWorkers: u32,
    ) -> HogwildTrainer {
        let (sender, receiver): (SyncSender<Vec<u32>>, Receiver<Vec<u32>>) =
            mpsc::sync_channel(CHANNEL_CAPACITY);
        let mut trainer = HogwildTrainer {
            workers: Vec::with_capacity(numWorkers as usize),
            sender,
        };
        let receiver: Arc<Mutex<Receiver<Vec<u32>>>> = Arc::new(Mutex::new(receiver));
        let feature_buffer_translator = FeatureBufferTranslator::new(model_instance);
        let port_buffer = sharable_regressor.new_portbuffer();
        for i in 0..numWorkers {
            let worker = HogwildWorker::new(
                sharable_regressor.clone(),
                feature_buffer_translator.clone(),
                port_buffer.clone(),
                Arc::clone(&receiver),
            );
            trainer.workers.push(worker);
        }
        trainer
    }

    pub fn digest_example(&self, feature_buffer: Vec<u32>) {
        self.sender.send(feature_buffer).unwrap();
    }

    pub fn block_until_workers_finished(self) {
        drop(self.sender);
        for worker in self.workers {
            worker.join().unwrap();
        }
    }
}

impl Default for HogwildTrainer {
    fn default() -> Self {
        let (sender, receiver) = mpsc::sync_channel(0);
        HogwildTrainer {
            workers: vec![],
            sender,
        }
    }
}

impl HogwildWorker {
    pub fn new(
        regressor: BoxedRegressorTrait,
        feature_buffer_translator: FeatureBufferTranslator,
        port_buffer: PortBuffer,
        receiver: Arc<Mutex<Receiver<Vec<u32>>>>,
    ) -> JoinHandle<()> {
        let mut worker = HogwildWorker {
            regressor,
            feature_buffer_translator,
            port_buffer,
        };

        thread::spawn(move || worker.train(receiver))
    }

    pub fn train(&mut self, receiver: Arc<Mutex<Receiver<Vec<u32>>>>) {
        let mut some_num = 0u64;
        loop {
            let buffer = match receiver.lock().unwrap().recv() {
                Ok(feature_buffer) => feature_buffer,
                Err(RecvError) => break, // channel was closed
            };
            self.feature_buffer_translator
                .translate(buffer.as_slice(), some_num);
            self.regressor.learn(
                &self.feature_buffer_translator.feature_buffer,
                &mut self.port_buffer,
                true,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hogwild_trainer_new_creates_workers() {
        let num_workers = 4;
        let mut model_instance = ModelInstance::new_empty().unwrap();
        let mut regressor = Regressor::new(&model_instance);
        let sharable_regressor: BoxedRegressorTrait = BoxedRegressorTrait::new(Box::new(regressor));
        let trainer = HogwildTrainer::new(sharable_regressor, &model_instance, num_workers);

        assert_eq!(trainer.workers.len(), num_workers as usize);
    }
}
