use burn::{
    module::{Param, ParamId},
    nn::{Linear, LinearConfig, Relu, Sigmoid},
    prelude::*,
    tensor::backend::AutodiffBackend,
};

use crate::MyAutodiffBackend;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub linear3: Linear<B>,
    pub activation2: Sigmoid,
    pub activation1: Sigmoid,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    state_size: usize,
    hidden_size: usize,
    num_actions: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(self.state_size, self.hidden_size)
                .with_bias(true)
                .with_initializer(nn::Initializer::XavierUniform { gain: 10.0 })
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .with_initializer(nn::Initializer::XavierUniform { gain: 10.0 })
                .init(device),
            linear3: LinearConfig::new(self.hidden_size, self.num_actions)
                .with_bias(true)
                .with_initializer(nn::Initializer::XavierUniform { gain: 10.0 })
                .init(device),
            activation2: Sigmoid::new(),
            activation1: Sigmoid::new(),
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
        let x = self.activation2.forward(x);
        let x = self.linear2.forward(x); // [batch_size, num_classes]
        let x = self.activation2.forward(x);
        let x = self.linear3.forward(x);
        let x = self.activation1.forward(x);
        x
    }
}

impl<B: AutodiffBackend> Model<B> {
    pub fn copy_model(mut a: Model<B>, b: &Model<B>) -> Model<B> {


        a.linear1.weight = a.linear1.weight.map(|_| b.linear1.weight.val());

        a.linear1.bias = match (&b.linear1.bias) {
            (Some(b_bias)) => Some(Param::initialized(ParamId::new(), b_bias.val())),
            _ => None,
        };

        //let mut bias  = *(a.linear1.bias.as_mut().unwrap());
        //*bias = (*bias)
        //  .map(|_| b.linear1.bias.as_ref().unwrap().val().mul_scalar(beta) + c.linear1.bias.as_ref().unwrap().val().mul_scalar(1. - beta));

        a.linear2.weight = a.linear2.weight.map(|_| b.linear2.weight.val());

        a.linear2.bias = match (&b.linear2.bias) {
            (Some(b_bias)) => Some(Param::initialized(ParamId::new(), b_bias.val())),
            _ => None,
        };

        a.linear3.weight = a.linear3.weight.map(|_| b.linear3.weight.val());

        a.linear3.bias = match (&b.linear3.bias) {
            (Some(b_bias)) => Some(Param::initialized(ParamId::new(), b_bias.val())),
            _ => None,
        };
        //  println!("a after: {}", a.linear1.weight.val());

        a
    }
}

impl<B: AutodiffBackend> Model<B> {
    pub fn soft_copy_model(mut a: Model<B>, b: &Model<B>, weight: f64) -> Model<B> {

        if weight < 0. || weight > 1. {
            println!("Weight is outside bounds: must be within 0 and 1.");
        }


        a.linear1.weight = a.linear1.weight.map(|w| w.mul_scalar(1.-weight) + b.linear1.weight.val().mul_scalar(weight));
        let bias =  match (&b.linear1.bias) {
            (Some(b_bias)) => b_bias.val(),
            _ => Tensor::<B, 1>::from([0.0]),
        };
        a.linear1.bias = match (&a.linear1.bias) {
            (Some(a_bias)) => Some(Param::initialized(ParamId::new(), a_bias.val().mul_scalar(1.0 - weight) + bias.mul_scalar(weight) )),
            _ => None,
        };
      
        a.linear2.weight = a.linear2.weight.map(|w| w.mul_scalar(1.-weight) + b.linear2.weight.val().mul_scalar(weight));
        let bias =  match (&b.linear2.bias) {
            (Some(b_bias)) => b_bias.val(),
            _ => Tensor::<B, 1>::from([0.0]),
        };
        a.linear2.bias = match (&a.linear2.bias) {
            (Some(a_bias)) => Some(Param::initialized(ParamId::new(), a_bias.val().mul_scalar(1.0 - weight) + bias.mul_scalar(weight) )),
            _ => None,
        };

        a.linear3.weight = a.linear3.weight.map(|w| w.mul_scalar(1.-weight) + b.linear3.weight.val().mul_scalar(weight));
        let bias =  match (&b.linear3.bias) {
            (Some(b_bias)) => b_bias.val(),
            _ => Tensor::<B, 1>::from([0.0]),
        };
        a.linear3.bias = match (&a.linear3.bias) {
            (Some(a_bias)) => Some(Param::initialized(ParamId::new(), a_bias.val().mul_scalar(1.0 - weight) + bias.mul_scalar(weight) )),
            _ => None,
        };

        a
    }
}

// impl<B: AutodiffBackend> Model<B> {
//     pub fn weighted_copy(
//         mut a: Model<B>,
//         b: &Model<B>,
//         c: &Model<B>,
//         beta: f32, // e.g., 0.1
//     ) -> Model<B> {
//         // Assuming no bias in linear layers
//         println!("b: {}", b.linear1.weight.val());
//         println!("c: {}", c.linear1.weight.val());
//         println!("a before: {}", a.linear1.weight.val());
//
//         a.linear1.weight = a.linear1.weight.map(|_| {
//             b.linear1.weight.val().mul_scalar(beta) + c.linear1.weight.val().mul_scalar(1. - beta)
//         });
//
//         //let mut bias  = *(a.linear1.bias.as_mut().unwrap());
//         //*bias = (*bias)
//         //  .map(|_| b.linear1.bias.as_ref().unwrap().val().mul_scalar(beta) + c.linear1.bias.as_ref().unwrap().val().mul_scalar(1. - beta));
//
//         a.linear2.weight = a.linear2.weight.map(|_| {
//             b.linear2.weight.val().mul_scalar(beta) + c.linear2.weight.val().mul_scalar(1. - beta)
//         });
//
//         a.linear3.weight = a.linear3.weight.map(|_| {
//             b.linear3.weight.val().mul_scalar(beta) + c.linear3.weight.val().mul_scalar(1. - beta)
//         });
//
//         println!("a after: {}", a.linear1.weight.val());
//
//         a
//     }
// }
//
// */
