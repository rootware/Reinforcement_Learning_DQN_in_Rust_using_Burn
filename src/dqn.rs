use burn::{nn::loss::Reduction, optim::{Adam, AdamConfig, GradientsParams}, tensor::{Int, Tensor}};
use environment::Environment;
use model::Model;
use crate::utils::*;
use crate::replay_buffer::ReplayBuffer;
use crate::{environment, model}; // For experience replay
use burn::optim::Optimizer;

// Define a simple neural network for Q-function approximation
pub struct DQN{
    pub env : Environment,
    pub nn_model : Model<MyAutodiffBackend>,
    pub replay_buffer : ReplayBuffer,
    pub config : MyConfig,
   // pub optimizer: burn::optim::adaptor::OptimizerAdaptor<Adam<_>, _, _> ,
}

impl DQN {
    pub fn new(model_arg: Model<MyAutodiffBackend>, replay_buffer: ReplayBuffer , config: MyConfig) -> Self {
        
        DQN {
            env: Environment::new(),
            nn_model: model_arg,
            replay_buffer,
            config,
           // optimizer: AdamConfig::new().with_epsilon(0.1).init(),
        }
    }

    pub fn forward(&self, x: Tensor<MyAutodiffBackend, 1>) -> Tensor<MyAutodiffBackend, 1> {
        self.nn_model.forward(x)
    }

    pub fn train(&mut self, num_episodes: i32, num_trials: i32) {
        
        for i in 0..num_episodes as usize {
            let mut finish = false;
            while !finish {
                let action = self.propose_action();
                let result = self.env.step(action);
                finish = self.env.done();
                self.replay_buffer.add(result);
            }

            self.update_model(10);
        }

    }

    pub fn propose_action(&self) -> i32 {
        return 0;
    }

    pub fn update_model(&mut self, batch_size: usize) {
       let mut optimizer = AdamConfig::new().with_epsilon(0.1).init();
        // Sample a batch of experiences from the replay buffer
        let batch = self.replay_buffer.sample(batch_size);
        // DQN Q-learning update
        for mem in batch {
            // Compute target Q-value

            let tensor_state = Tensor::<MyAutodiffBackend, 1>::from(mem.current_state);
            let tensor_next_state = Tensor::<MyAutodiffBackend, 1>::from(mem.next_state);
            let done = mem.done;
            let reward = Tensor::<MyAutodiffBackend,1>::from([mem.reward]);
            let action:Tensor<MyAutodiffBackend, 1, Int>  = Tensor::<MyAutodiffBackend,1, Int>::from([mem.action]);
            let next_q_values = self.forward(tensor_next_state);
            let target = if done {
                reward
            } else {
                reward + next_q_values.max().mul_scalar(self.config.gamma  )
            };

            // Compute Q-value for the current state and action
            let q_values = self.forward(tensor_state.clone());
            let q_value = q_values.select(0,action);//q_values.select(0, action  );

            let loss = burn::nn::loss::MseLoss::new().forward( q_value, target, Reduction::Auto);// 

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads2 = GradientsParams::from_grads(grads, &self.nn_model);
            // Update the model using the optimizer.
            self.nn_model = optimizer.step(1.0e-2, self.nn_model.clone(), grads2);

        }
    }
    
}

