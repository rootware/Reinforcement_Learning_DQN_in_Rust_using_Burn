pub mod dqn;
pub mod environment;
pub mod model;
pub mod replay_buffer;
//pub mod training;
pub mod utils;

use crate::model::Model;
use dqn::DQN;
use replay_buffer::ReplayBuffer;
use utils::*;

fn main() {
    let device = Default::default();
    let model = model::ModelConfig::new(STATE_SIZE, HIDDEN_SIZE, NUM_ACTIONS)
        .init::<MyAutodiffBackend>(&device);

    let myconfig = MyConfig {
        gamma: 0.99,
        lr: 1.0e-1,
        epsilon: 0.9,
        tau: 0.1,
    };
    // Initialize DQN model and optimizer
    let mut dqn_model = DQN::new(model.clone(), ReplayBuffer::new(200), myconfig.clone());

    dqn_model.train(1000, 10);
    dqn_model.extract_policy();
}
