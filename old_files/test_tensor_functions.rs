pub mod dqn;
pub mod environment;
pub mod model;
pub mod replay_buffer;
//pub mod training;
pub mod utils;
use burn::tensor::Tensor;
use burn_tch::LibTorchDevice;

//use burn_cuda::CudaDevice;
use dqn::DQN;
use replay_buffer::ReplayBuffer;
use utils::*;


fn main() {
    //let device = Default::default();
   // let device = CudaDevice::default();
   let device = LibTorchDevice::Cpu;
    let model = model::ModelConfig::new(STATE_SIZE, HIDDEN_SIZE, NUM_ACTIONS)
        .init::<MyAutodiffBackend>(&device);

    let myconfig = MyConfig {
        gamma: 0.99,
        lr: 1.0e-3,
        epsilon: 1.0,
        tau: 0.1,
    };
    // Initialize DQN model and optimizer
    let mut dqn_model = DQN::new(model.clone(), model.clone(),ReplayBuffer::new(100000), myconfig.clone());

   // dqn_model.train(100, 1000);
    //println!("zero epsilon policy");
    //dqn_model.extract_policy_zero_epsilon();
    //println!("best ever policy");
   // dqn_model.extract_policy();

   let q_val = dqn_model.forward(Tensor::<MyAutodiffBackend, 1>::from(dqn_model.env.current_state));
   println!("{}",  q_val.clone());
   let max_q:Result<Vec<i64>, _> = q_val.clone().argmax(0).to_data().to_vec();
   println!("{:?}", max_q.unwrap());
   println!("{}", q_val.max().mul_scalar(dqn_model.config.gamma));

}
