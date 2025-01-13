pub mod agent;
pub mod environment;
pub mod model;
pub mod replay_buffer;
//pub mod training;
pub mod utils;
use burn::tensor::Tensor;
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
        gamma: 0.99,
        lr: 1.0,
        epsilon: 1.0,
        tau: 0.2,
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
    dqn_model.train(50, 50);
    //ddqn_model.train(1, 1);

    dqn_model.update_target();

    let mut state = [0.,0.];
    for i in 0..23{
        state[0] = ((i as i32 - 11 )% 11) as f64;
        println!("{i}\t{:?}\t{}\t{}\t{}\t{}\t{}", 
            state, 
            dqn_model.target_model.forward( Tensor::<MyAutodiffBackend,1>::from(state )).argmax(0).to_data(),
            ddqn_model.target_model.forward(Tensor::<MyAutodiffBackend,1>::from(state )).argmax(0).to_data() ,
            dqn_model.env.reward_calc(&state),
            dqn_model.target_model.forward( Tensor::<MyAutodiffBackend,1>::from(state )).max().to_data(),
            ddqn_model.target_model.forward(Tensor::<MyAutodiffBackend,1>::from(state )).max().to_data() ,
            );
    }


    /* 
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
    */
}
