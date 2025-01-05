pub mod replay_buffer;
pub mod dqn;
pub mod environment;
pub mod model;
pub mod training;
use burn::{backend::Wgpu, optim::Adam, tensor::Tensor};
use dqn::DQN;
use replay_buffer::ReplayBuffer;


type MyBackend = Wgpu<f32, i32>;


fn main() {
    let input_size = 4; // Example input size (e.g., state size)
    let hidden_size = 64;
    let output_size = 2; // Number of actions (Q-values)

    let device = Default::default();
    let model = model::ModelConfig::new(2, 2,2).init::<MyBackend>(&device);

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

   
}
