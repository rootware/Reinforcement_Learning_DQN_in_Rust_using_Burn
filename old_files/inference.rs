use crate::training::TrainingConfig;
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: Tensor<B,2>) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let output = model.forward(item.clone());
    let record = model.into_record();
    println!("{}",     record.linear1.weight.to_data());
    //println!("{}",     record.linear2.weight.to_data());


    println!("Predicted {} Expected {}",  output.into_data(), item.into_data());
}