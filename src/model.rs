use burn::{
    nn::{
        Linear, LinearConfig, Relu,
    }, prelude::*
};
use burn::data::dataloader::batcher::Batcher;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub linear3: Linear<B>,
    pub activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    state_size: usize,
    hidden_size: usize,
    num_actions:usize,

}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(self.state_size, self.hidden_size).with_bias(true).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size).with_bias(false).init(device),
            linear3: LinearConfig::new( self.hidden_size, self.num_actions).with_bias(false).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 1>) -> Tensor<B, 1> {
        //let [batch_size, height, width] = images.dims();
       // let [batch_size, length ] = images.dims();

        // Create a channel at the second dimension.
        //let x = images.reshape([batch_size,  length]);
        let x = images;
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x); // [batch_size, num_classes]
        let x = self.activation.forward(x) ;
        let x = self.linear3.forward(x);
        let x = self.activation.forward(x);
        x
    }
}

/* 

#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}



#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub input: Tensor<B, 2>,
   // pub targets: Tensor<B, 1, Int>,
}

// replace MnistItem with f64

impl<B: Backend> Batcher<Tensor<B,1>, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<Tensor<B,1>>) -> MnistBatch<B> {
  
        let myitems = items.iter().map( |tensor| tensor.clone().reshape([1, 2])).collect();
        let myinput = Tensor::cat(myitems, 0).to_device(&self.device);

        MnistBatch { input: myinput}
    }
}
 */