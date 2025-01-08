pub mod replay_buffer;
pub mod dqn;
pub mod environment;
pub mod model;
pub mod training;
pub mod utils;

use burn::{backend::{Autodiff, Wgpu}, module::Module, nn::loss::Reduction, optim::{Adam, AdamConfig, Optimizer}, tensor::Tensor};
use dqn::DQN;
use replay_buffer::ReplayBuffer;
use utils::{MyAutodiffBackend};
use burn::optim::{ GradientsParams};



fn main() {
    let input_size = 4; // Example input size (e.g., state size)
    let hidden_size = 64;
    let output_size = 2; // Number of actions (Q-values)

    let device = Default::default();
    let model = model::ModelConfig::new(2, 2,2).init::<MyAutodiffBackend>(&device);

    // Initialize DQN model and optimizer
    let mut dqn_model = DQN::new(model,  ReplayBuffer::new(5));
    let myweights = dqn_model.nn_model.clone().into_record();


    // Experience replay buffer
    let mut replay_buffer = ReplayBuffer::new(10000);

    // Training parameters
    let batch_size = 32;
    let gamma = 0.99; // Discount factor
    let epsilon = 0.1; // Epsilon for exploration-exploitation trade-off
    let episodes = 1;
    let mut total_timesteps = 0;

   // let data = Tensor::<MyAutodiffBackend, 2>::from( [[1.0, 2.0], [3.0, 4.0]]);
   let data = Tensor::<MyAutodiffBackend, 2>::from( [[1.0, 2.0], [2.0, 3.0], [3.0,4.0], [5.0, 6.0]]);

   for i in 0..100000{
    let output = dqn_model.forward(data.clone());
   // println!("{}", output.clone());
    let target = Tensor::<MyAutodiffBackend, 2>::from( [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [10.0,12.0]]);//Tensor::<MyAutodiffBackend, 2>::from( [[1.0, 2.0], [3.0, 5.0]]);
    
   // println!("{}", "hello");
    let loss = (target - output).abs();//burn::nn::loss::MseLoss::new().forward( output, target, Reduction::Sum);// 
    let grads = loss.backward();
    let grad_params = GradientsParams::from_grads(grads, &dqn_model.nn_model);

  //  println!("{:?}", grad_params);
 //   println!("{:?}", dqn_model.nn_model.clone());
    let opt_config = AdamConfig::new();
    let mut opt = opt_config.init();    

    dqn_model.nn_model = opt.step(1.0e-2, dqn_model.nn_model, grad_params);
   }
   // println!("{:?}", myweights.linear3.weight.to_data());
   // println!("{:?}", myweights.linear3.bias.unwrap().to_data());

    println!("{}", dqn_model.nn_model.forward(data));
   // let myweights2 = dqn_model.nn_model.into_record();
  //  println!("{:?}", myweights2.linear3.weight.to_data());
   // println!("{:?}", myweights2.linear3.bias.unwrap().to_data());



   
}
