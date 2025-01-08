use core::f64;

use crate::model::{MnistBatch, MnistBatcher, Model, ModelConfig};
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric}, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep
    },
};

/// 
impl<B: Backend> Model<B> {
    pub fn loss_calculation(
        &self,
        target: Tensor<B, 2>,
        prediction: Tensor<B,2>
    ) -> RegressionOutput<B> {
        let loss = prediction.clone().flatten(0, 1).sub(target.clone()).powf(Tensor::from([2])).mean();

        RegressionOutput::new(loss, prediction, target)
    }
}



impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.loss_calculation(batch.input);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> RegressionOutput<B> {
        self.loss_calculation(batch.input)
    }
}


#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 100)]
    pub num_epochs: usize,
    #[config(default = 50)]
    pub batch_size: usize,
    #[config(default = 5)]
    pub num_workers: usize,
    #[config(default = 1)]
    pub seed: u64,
    #[config(default = 1.0e-2)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}
use burn::data::dataset::Dataset;



// Define the dataset
#[derive(Debug)]
struct TensorDataset<B: Backend> {
    data: Vec<Tensor<B, 1>>, // Input-output pairs
}

impl<B: Backend> TensorDataset<B> {
    // Constructor
    fn new(data: Vec<Tensor<B, 1>>) -> Self {
        Self { data }
    }
}

impl<B: Backend> Dataset<Tensor<B, 1>> for TensorDataset<B> {
    fn get(&self, index: usize) -> std::option::Option<Tensor<B, 1>> {
        std::option::Option::from(self.data[index].clone())
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}




pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device)  {//where TensorDataset<B>: Dataset<burn::tensor::Tensor<<B as AutoDiff>::InnerBackend, 1>> {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = MnistBatcher::<B>::new(device.clone());
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());



    
    let mut data = vec![
        Tensor::<B::InnerBackend, 1>::from( [ 1.0, 2.0]),
        Tensor::<B::InnerBackend, 1>::from([3.0,4.0]),
  
    ];

    for i in 0..100 {
        for j in 0..100 {
            data.push(Tensor::<B::InnerBackend,1>::from( [-10.0 + i as f64 ,-10.0 + j as f64]) );
        }
    }


    let mut data2 = Vec::<Tensor<B,1>>::new();//vec![

    for i in 0..100 {
        for j in 0..100 {
            data2.push(Tensor::<B,1>::from([i as f64-50.0 ,j as f64 - 50.0]) );
        }
    }
    
    // Create the dataset
    let dataset = TensorDataset::new(data2.clone());
    let dataset2 = TensorDataset::new(data);

  
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset);

   let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset2);
    

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train.clone(), dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}