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
use environment::Environment;
use model::Model;
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
        lr: 1.0e-2,
        epsilon: 1.0,
        tau: 0.1,
    };
    
    // Initialize DQN model and optimizer
    let mut dqn_model = DQN::new(model.clone(), model.clone(),ReplayBuffer::new(1000), myconfig.clone());
    let mut state : State  = [0.0, 0.0];

    for i in 0..5 {
        for j in 0..5{
            state[0] = i as f64;
            state[1] = j as f64;
            println!("{}\t{}\t{}", state[0], state[1], dqn_model.forward(Tensor::<MyAutodiffBackend,1>::from(state)).to_data().to_string() );
        }
    }
    
    dqn_model.train(   100, 100);
    // dqn_model.update_target();
 //   println!("zero epsilon policy");
  //  dqn_model.extract_policy_zero_epsilon();
  //  println!("best ever policy");
  //  dqn_model.extract_policy();

   // println!("{}", dqn_model.forward(Tensor::<MyAutodiffBackend,1>::from( [3.0, 5.0])) );
   // println!("{}", dqn_model.forward(Tensor::<MyAutodiffBackend,1>::from( [3.0, 4.0])) );
   // println!("{}", dqn_model.forward(Tensor::<MyAutodiffBackend,1>::from( [2.0, 5.0])) );
    
    
    let mut state : State  = [0.0, 0.0];

    for i in 0..5 {
        for j in 0..5{
            state[0] = i as f64;
            state[1] = j as f64;
            println!("{}\t{}\t{}\t{}", state[0], state[1], dqn_model.forward(Tensor::<MyAutodiffBackend,1>::from(state)).to_data().to_string() ,
            dqn_model.target_model.forward(Tensor::<MyAutodiffBackend,1>::from(state)).to_data().to_string() );
        }
    }
    /* 
    let mut first_model = model::ModelConfig::new(STATE_SIZE, HIDDEN_SIZE, NUM_ACTIONS)
    .init::<MyAutodiffBackend>(&device);
    let mut second_model = model::ModelConfig::new(STATE_SIZE, HIDDEN_SIZE, NUM_ACTIONS)
    .init::<MyAutodiffBackend>(&device);
    let mut state : State  = [0.0, 0.0];

    println!("{}",first_model.forward(Tensor::<MyAutodiffBackend,1>::from(state)).to_data().to_string() );
    println!("{}",second_model.forward(Tensor::<MyAutodiffBackend,1>::from(state)).to_data().to_string() );
    second_model = Model::copy_model(second_model, &first_model.clone());
    println!("{}",first_model.forward(Tensor::<MyAutodiffBackend,1>::from(state)).to_data().to_string() );
    println!("{}",second_model.forward(Tensor::<MyAutodiffBackend,1>::from(state)).to_data().to_string() );

*/




}
