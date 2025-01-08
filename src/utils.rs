use burn::backend::{Autodiff, Wgpu};

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;