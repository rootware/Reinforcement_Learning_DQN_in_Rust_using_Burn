pub mod agent;
pub mod environment;
pub mod model;
pub mod replay_buffer;
//pub mod training;
pub mod utils;
use burn_tch::LibTorchDevice;
use environment::onedgrid::*;
//use burn_cuda::CudaDevice;
use crate::agent::{ddqn::DDQN, dqn::DQN};
use replay_buffer::ReplayBuffer;
use utils::*;

fn main() {
    let device = LibTorchDevice::Cpu;
    let model = model::ModelConfig::new(STATE_SIZE, HIDDEN_SIZE, NUM_ACTIONS)
        .init::<MyAutodiffBackend>(&device);
    let myconfig = MyConfig {
        gamma: 0.9,
        lr: 1.0e-2,
        epsilon: 1.0,
        tau: 0.1,
    };

    // Initialize DQN model and optimizer
    let mut dqn_model = DQN::new(
        model.clone(),
        model.clone(),
        ReplayBuffer::new(1000),
        myconfig.clone(),
    );

    // Initialize DDQN model and optimizer
    let mut ddqn_model = DDQN::new(
        model.clone(),
        model.clone(),
        ReplayBuffer::new(1000),
        myconfig.clone(),
    );

    dqn_model.train(20, 5);
    println!("zero epsilon policy");
    dqn_model.extract_policy_zero_epsilon();
    println!("best ever policy");
    dqn_model.extract_policy();

    ddqn_model.train(20, 5);
    println!("zero epsilon policy");
    ddqn_model.extract_policy_zero_epsilon();
    println!("best ever policy");
    ddqn_model.extract_policy();
}
