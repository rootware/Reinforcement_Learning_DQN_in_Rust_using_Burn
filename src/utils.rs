use burn::backend::{Autodiff, Wgpu};

pub type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

pub struct MyConfig {
    pub gamma: f64,
    pub lr : f64,
    pub epsilon : f64,
    pub tau : f64
}