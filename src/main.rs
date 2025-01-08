pub mod replay_buffer;
pub mod dqn;
pub mod environment;
pub mod model;
pub mod training;
pub mod utils;

use burn::{backend::{Autodiff, Wgpu}, optim::Adam, tensor::Tensor};
use dqn::DQN;
use replay_buffer::ReplayBuffer;
use utils::{MyAutodiffBackend};




fn main() {
    let input_size = 4; // Example input size (e.g., state size)
    let hidden_size = 64;
    let output_size = 2; // Number of actions (Q-values)

    let device = Default::default();
    let model = model::ModelConfig::new(2, 2,2).init::<MyAutodiffBackend>(&device);

    // Initialize DQN model and optimizer
    let dqn_model = DQN::new(model,  ReplayBuffer::new(5));

    // Experience replay buffer
    let mut replay_buffer = ReplayBuffer::new(10000);

    // Training parameters
    let batch_size = 32;
    let gamma = 0.99; // Discount factor
    let epsilon = 0.1; // Epsilon for exploration-exploitation trade-off
    let episodes = 1;
    let mut total_timesteps = 0;

    let data = Tensor::<MyAutodiffBackend, 2>::from( [[1.0, 2.0], [3.0, 4.0]]);
    let output = dqn_model.forward(data);
    println!("{}", output);


   
}
