use std::str::FromStr;

use crate::replay_buffer::ReplayBuffer;
use crate::utils::*;
use crate::{environment, model};
use burn::optim::adaptor::OptimizerAdaptor;
// For experience replay
use burn::optim::Optimizer;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::{
    optim::{AdamConfig, GradientsParams},
    tensor::{Int, Tensor},
};
use environment::one_qubit::*;
use model::Model;
use rand::Rng;
use burn::module::Module;

// Define a simple neural network for Q-function approximation
pub struct DDQN {
    pub env: Environment,
    pub policy_model: Model<MyAutodiffBackend>,
    pub target_model: Model<MyAutodiffBackend>,
    pub replay_buffer: ReplayBuffer,
    pub config: MyConfig,
    pub action_record: Vec<i32>,
    // pub optimizer: burn::optim::adaptor::OptimizerAdaptor<Adam<_>, _, _> ,
    pub optimizer:
        OptimizerAdaptor<burn::optim::Adam<MyBackend>, Model<MyAutodiffBackend>, MyAutodiffBackend>,
}

impl DDQN {
    pub fn new(
        policy: Model<MyAutodiffBackend>,
        target: Model<MyAutodiffBackend>,
        replay_buffer: ReplayBuffer,
        config: MyConfig,
    ) -> Self {
        let optimizer = AdamConfig::new()
            //.with_epsilon(config.epsilon as f32)
            .init();
        DDQN {
            env: Environment::new(),
            policy_model: policy,
            target_model: target,
            replay_buffer,
            config,
            action_record: Vec::new(),
            optimizer,
        }
    }

    pub fn forward(&self, x: Tensor<MyAutodiffBackend, 1>) -> Tensor<MyAutodiffBackend, 1> {
        self.policy_model.forward(x)
    }

    pub fn train(&mut self, num_episodes: i32, num_trials: i32) {
        // let mut print_string = String::new();
        let mut current_reward = 0.0;
        for _j in 0..num_trials {
            //let mut step_count = 0;
            for _i in 0..num_episodes as usize {
                let mut finish = false;
                while !finish {
                    let action = self.propose_action();
                    let result = self.env.step(action);
                    finish = self.env.done();
                    self.replay_buffer.add(result);

                    //print_string = self.update_model(50);
                    self.update_model(50);
                }
                if self.env.reward() > current_reward {
                    self.action_record = self.env.action_record.clone();
                    current_reward = self.env.reward();
                }
                self.env.reset();
            }
            // println!("{}\t{}\t{}", j, self.config.epsilon, print_string);
            self.config.epsilon -= 0.8 / (num_trials as f64);
        }
    }

    pub fn propose_action(&self) -> i32 {
        let mut rng = rand::thread_rng();
        let random_float: f64 = rng.gen_range(0.0..1.0);

        if random_float < self.config.epsilon {
            return rng.gen_range(0..NUM_ACTIONS) as i32;
        } else {
            let q_val = self
                .policy_model
                .forward(Tensor::<MyAutodiffBackend, 1>::from(self.env.current_state));
            let max: Result<Vec<i64>, _> = q_val.argmax(0).to_data().to_vec();
            let mut max2 = max.unwrap();
            max2.pop().unwrap() as i32
        }
    }

    pub fn update_target(&mut self) {
        // self.target_model = Model::copy_model(self.target_model.clone(), &self.policy_model);
        self.target_model = Model::soft_copy_model(
            self.target_model.clone(),
            &self.policy_model,
            1. - self.config.tau,
        );
    }
    pub fn update_model(&mut self, batch_size: usize) {
        //  let mut optimizer = AdamConfig::new()
        //      .with_epsilon(self.config.epsilon as f32)
        //     .init();
        // Sample a batch of experiences from the replay buffer
        let batch = self.replay_buffer.sample(batch_size);
        let mut index = 0;
        let B = batch.len();
        let mut total_loss = Tensor::<MyAutodiffBackend, 1>::from([0]);
        // DQN Q-learning update
        for mem in batch {
            // Compute target Q-value

            let tensor_state = Tensor::<MyAutodiffBackend, 1>::from(mem.current_state);

            let tensor_next_state = Tensor::<MyAutodiffBackend, 1>::from(mem.next_state);
            let done = mem.done;
            let reward = Tensor::<MyAutodiffBackend, 1>::from([mem.reward]);
            let action: Tensor<MyAutodiffBackend, 1, Int> =
                Tensor::<MyAutodiffBackend, 1, Int>::from([mem.action]);

            let policy_action = self.forward(tensor_next_state.clone()).argmax(0);

            let next_q_values = self
                .target_model
                .forward(tensor_next_state.clone())
                .select(0, policy_action);
            let y_value = if done {
                reward.clone()
            } else {
                reward.clone() + next_q_values.clone().mul_scalar(self.config.gamma)
            };

            if index == -1 {
                println!(
                    "State{}\n Next State {}\n action{}\n next q{}\n target{}\n reward{}",
                    tensor_state.to_data(),
                    tensor_next_state.clone().to_data(),
                    action.to_data(),
                    next_q_values.clone().to_data(),
                    y_value.to_data(),
                    reward.to_data()
                );
            }
            index += 1;
            // Compute Q-value for the current state and action
            let q_values = self.policy_model.forward(tensor_state.clone());
            let q_value = q_values.select(0, action);
            let loss = (q_value - y_value)
                .abs()
                .powi(Tensor::<MyAutodiffBackend, 1>::from([2])); //.require_grad();
            total_loss = total_loss.add(loss);
        }
        // Gradients for the current backward pass
        total_loss = total_loss.div(Tensor::<MyAutodiffBackend, 1>::from([B as f64]));
        let grads = total_loss.backward();
        // Gradients linked to each parameter of the model.
        let grads2 = GradientsParams::from_grads(grads, &self.policy_model);
        // Update the model using the optimizer.
        self.policy_model = self
            .optimizer
            .step(self.config.lr, self.policy_model.clone(), grads2);
        self.update_target();
    }

    pub fn extract_policy(&mut self) {
        self.env.reset();
        self.config.epsilon = 0.0;

        while !self.action_record.is_empty() {
            let action = self.action_record.pop().unwrap();
            self.env.step(action);
            println!("{:?}", self.env.current_state);
        }
    }

    pub fn extract_policy_zero_epsilon(&mut self) {
        self.env.reset();
        self.config.epsilon = 0.0;

        println!("Step No. \t Policy Q Values \t Target Policy \t Action \t Fidelity Post Action");
        while !self.env.done() {
            let action = self.propose_action();

            let target_qval = self
                .target_model
                .forward(Tensor::<MyAutodiffBackend, 1>::from(self.env.current_state))
                .to_data();
            let policy_qval = self
                .policy_model
                .forward(Tensor::<MyAutodiffBackend, 1>::from(self.env.current_state))
                .to_data();
            self.env.step(action);
            let fidelity = self.env.fidelity(&self.env.current_state);
            println!(
                "{}\t  {}\t \t{}\t \t{}\t {}",
                self.env.current_steps, policy_qval, target_qval, action, fidelity
            );
        }
    }

    pub fn save_nets(&self, path : String) {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.target_model.clone()
            .save_file(path.clone() +"target", &recorder)
            .expect("Should be able to save the model");
        self.policy_model.clone()
            .save_file(path + "policy", &recorder)
            .expect("Should be able to save the model");
    }
}
