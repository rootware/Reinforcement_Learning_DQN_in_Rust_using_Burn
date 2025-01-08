use burn::backend::{Autodiff, Wgpu};

pub type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

pub const STATE_SIZE : usize = 2;
pub const HIDDEN_SIZE : usize = 2;
pub const NUM_ACTIONS : usize = 4;
pub type State = [f64; STATE_SIZE];


pub struct MyConfig {
    pub gamma: f64,
    pub lr : f64,
    pub epsilon : f64,
    pub tau : f64
}