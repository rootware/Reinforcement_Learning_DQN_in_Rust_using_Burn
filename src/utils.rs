//use burn::backend::{Autodiff, Wgpu};
// use burn_cuda::{Cuda, CudaDevice};

use burn::backend::Autodiff;
use burn_tch::LibTorch;

//pub type MyBackend = Wgpu<f32, i32>;
//pub type MyAutodiffBackend = Autodiff<MyBackend>;

// pub type MyBackend = Cuda<f32, i32>;
// pub type MyAutodiffBackend = Autodiff<MyBackend>;

pub type MyBackend = LibTorch<f32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

pub const STATE_SIZE: usize = 2;
pub const HIDDEN_SIZE: usize = 2;
pub const NUM_ACTIONS: usize = 4;
pub type State = [f64; STATE_SIZE];

#[derive(Clone, Copy)]
pub struct MyConfig {
    pub gamma: f64,
    pub lr: f64,
    pub epsilon: f64,
    pub tau: f64,
}

pub fn epsilon_greed(current: i32, total: i32, decay: f64) -> f64 {
    return f64::exp(-(current / total) as f64 / decay);
}
