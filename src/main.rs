pub mod dqn;
pub mod environment;
pub mod model;
pub mod replay_buffer;
//pub mod training;
pub mod utils;
use burn_tch::LibTorchDevice;

//use burn_cuda::CudaDevice;
use dqn::DQN;
use model::Model;
use replay_buffer::ReplayBuffer;
use utils::*;

fn main() {
    //let device = Default::default();
    // let device = CudaDevice::default();
    let device = LibTorchDevice::Cpu;
    let model = model::ModelConfig::new(STATE_SIZE, HIDDEN_SIZE, NUM_ACTIONS)
        .init::<MyAutodiffBackend>(&device);

}
