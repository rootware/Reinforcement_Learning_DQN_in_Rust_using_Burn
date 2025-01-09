pub mod replay_buffer;
pub mod dqn;
pub mod environment;
pub mod model;
//pub mod training;
pub mod utils;
use burn::module::{Module, ModuleMapper, Param};
use burn::backend::autodiff::checkpoint::state;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use dqn::DQN;
use replay_buffer::ReplayBuffer;
use utils::*;
use crate::model::Model;


fn main() {


    let device = Default::default();
    let model = model::ModelConfig::new(STATE_SIZE, HIDDEN_SIZE, NUM_ACTIONS).init::<MyAutodiffBackend>(&device);
    let model2 = model::ModelConfig::new(STATE_SIZE, HIDDEN_SIZE, NUM_ACTIONS).init::<MyAutodiffBackend>(&device);
    let model3 = model::ModelConfig::new(STATE_SIZE, HIDDEN_SIZE, NUM_ACTIONS).init::<MyAutodiffBackend>(&device);


    let myconfig = MyConfig{ gamma: 0.99, lr: 1.0e-1, epsilon: 0.9, tau: 0.1};
    // Initialize DQN model and optimizer
    let mut dqn_model_1 = DQN::new(model.clone(),  ReplayBuffer::new(200), myconfig.clone());
    let mut dqn_model_2 = DQN::new(model2.clone(),  ReplayBuffer::new(200), myconfig.clone());
    let mut dqn_model_3 = DQN::new(model3,  ReplayBuffer::new(200), myconfig);

    Model::soft_update(dqn_model_3.nn_model, &dqn_model_1.nn_model, &dqn_model_2.nn_model, 0.5);

    //dqn_model.train(1000,10);
   // dqn_model.extract_policy();



    /*
    println!("{}",dqn_model.nn_model.linear1.weight.clone().to_data());
    dqn_model.nn_model.linear1.weight = Param::from_tensor(Tensor::<MyAutodiffBackend,2>::from(dqn_model.nn_model.linear1.weight.to_data()).sub(
      Tensor::<MyAutodiffBackend,2>::from(dqn_model.nn_model.linear1.weight.clone().to_data())
    ) );

    println!("{}",dqn_model.nn_model.linear1.weight.clone().to_data());

  */

   
}
