pub mod replay_buffer;
pub mod dqn;
pub mod environment;
pub mod model;
//pub mod training;
pub mod utils;

use dqn::DQN;
use replay_buffer::ReplayBuffer;
use utils::{MyAutodiffBackend, MyConfig};



fn main() {
    let input_size = 4; // Example input size (e.g., state size)
    let hidden_size = 64;
    let output_size = 2; // Number of actions (Q-values)

    let device = Default::default();
    let model = model::ModelConfig::new(10, 20,4).init::<MyAutodiffBackend>(&device);

    let myconfig = MyConfig{ gamma: 0.1, lr: 1.0e-2, epsilon: 0.8, tau: 0.1};
    // Initialize DQN model and optimizer
    let mut dqn_model = DQN::new(model,  ReplayBuffer::new(5), myconfig);
    dqn_model.train(2,2);
    





   
}
