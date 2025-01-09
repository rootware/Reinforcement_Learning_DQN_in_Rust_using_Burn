use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    tensor::backend::AutodiffBackend,
};

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
    num_actions: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(self.state_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear3: LinearConfig::new(self.hidden_size, self.num_actions)
                .with_bias(true)
                .init(device),
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
        let x = self.activation.forward(x);
        let x = self.linear3.forward(x);
        let x = self.activation.forward(x);
        x
    }
}


impl<B: AutodiffBackend> Model<B> {
    pub fn copy_model(
        mut a: Model<B>,
        b: &Model<B>
    ) -> Model<B> {
        // Assuming no bias in linear layers
      //  println!("b: {}", b.linear1.weight.val());
      //  println!("c: {}", c.linear1.weight.val());
       // println!("a before: {}", a.linear1.weight.val());

        a.linear1.weight = a.linear1.weight.map(|_| {
            b.linear1.weight.val()
        });

        //let mut bias  = *(a.linear1.bias.as_mut().unwrap());
        //*bias = (*bias)
        //  .map(|_| b.linear1.bias.as_ref().unwrap().val().mul_scalar(beta) + c.linear1.bias.as_ref().unwrap().val().mul_scalar(1. - beta));

        a.linear2.weight = a.linear2.weight.map(|_| {
            b.linear2.weight.val()
        });

        a.linear3.weight = a.linear3.weight.map(|_| {
            b.linear3.weight.val()
        });

      //  println!("a after: {}", a.linear1.weight.val());

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