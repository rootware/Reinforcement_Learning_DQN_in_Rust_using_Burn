use burn::backend::{Autodiff, Wgpu};

pub type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;