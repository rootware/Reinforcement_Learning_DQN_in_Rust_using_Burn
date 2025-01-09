use crate::replay_buffer::ReplayBuffer;
use crate::utils::*;
use crate::{environment, model}; // For experience replay
use burn::optim::Optimizer;
use burn::{
    optim::{AdamConfig, GradientsParams},
    tensor::{Int, Tensor},
};
use environment::Environment;
use model::Model;
use rand::Rng;

// Define a simple neural network for Q-function approximation
pub struct DQN {
    pub env: Environment,
    pub policy_model: Model<MyAutodiffBackend>,
    pub target_model: Model<MyAutodiffBackend>,
    pub replay_buffer: ReplayBuffer,
    pub config: MyConfig,
    pub action_record: Vec<i32>,
    // pub optimizer: burn::optim::adaptor::OptimizerAdaptor<Adam<_>, _, _> ,
}

impl DQN {
    pub fn new(
        policy: Model<MyAutodiffBackend>,
        target: Model<MyAutodiffBackend>,
        replay_buffer: ReplayBuffer,
        config: MyConfig,
    ) -> Self {
        DQN {
            env: Environment::new(),
            policy_model: policy,
            target_model : target,
            replay_buffer,
            config,
            action_record: Vec::new(), // optimizer: AdamConfig::new().with_epsilon(0.1).init(),
        }
    }

    pub fn forward(&self, x: Tensor<MyAutodiffBackend, 1>) -> Tensor<MyAutodiffBackend, 1> {
        self.policy_model.forward(x)
    }

    pub fn train(&mut self, num_episodes: i32, num_trials: i32) {
        let mut print_string = String::new();
        let mut current_reward = 0.0;
        for j in 0..num_trials{
            let mut step_count = 0;
            let target_update_period = 5;
            for i in 0..num_episodes as usize {
                let mut finish = false;
                while !finish {
 
                    let action = self.propose_action();
                    let result = self.env.step(action);
                    finish = self.env.done();
                    self.replay_buffer.add(result);
                }


                print_string = self.update_model(50);

                step_count = i;
                if step_count % target_update_period == 0 && i!=0 {
                    self.update_target();
                   // println!("target updated");
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

        if random_float > self.config.epsilon {
            return rng.gen_range(0..NUM_ACTIONS) as i32;
        } else {
            let q_val = self.target_model.forward(Tensor::<MyAutodiffBackend, 1>::from(self.env.current_state));
            let max: Result<Vec<i64>, _> = q_val.argmax(0).to_data().to_vec();
            let mut max2 = max.unwrap();
            max2.pop().unwrap() as i32
        }
    }

    pub fn update_target(&mut self){
       // self.target_model = Model::copy_model(self.target_model.clone(), &self.policy_model);
       self.target_model = Model::copy_model(self.target_model.clone(), &self.policy_model);
    }
    pub fn update_model(&mut self, batch_size: usize) -> String {
        let mut optimizer = AdamConfig::new()
            .with_epsilon(self.config.epsilon as f32)
            .init();
        // Sample a batch of experiences from the replay buffer
        let batch = self.replay_buffer.sample(batch_size);
        let mut loss_string = String::new();
        // DQN Q-learning update
        for mem in batch {
            // Compute target Q-value

            let tensor_state = Tensor::<MyAutodiffBackend, 1>::from(mem.current_state);
            let tensor_next_state = Tensor::<MyAutodiffBackend, 1>::from(mem.next_state);
            let done = mem.done;
            let reward = Tensor::<MyAutodiffBackend, 1>::from([mem.reward]);
            let action: Tensor<MyAutodiffBackend, 1, Int> =
                Tensor::<MyAutodiffBackend, 1, Int>::from([mem.action]);
            let next_q_values = self.forward(tensor_next_state);
            let target = if done {
                reward
            } else {
                reward + next_q_values.max().mul_scalar(self.config.gamma)
            };

            // Compute Q-value for the current state and action
            let q_values = self.forward(tensor_state.clone());
            let q_value = q_values.select(0, action); //q_values.select(0, action  );

            // let loss = burn::policy::loss::MseLoss::new().forward( q_value, target, Reduction::Sum);//
            let loss = (q_value - target).abs(); //.require_grad();
                                                 // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads2 = GradientsParams::from_grads(grads, &self.policy_model);
            // Update the model using the optimizer.
            self.policy_model = optimizer.step(self.config.lr, self.policy_model.clone(), grads2);

            if self.config.epsilon <= 0.5 && self.config.epsilon >= 0.45 {
              //  println!("{}", self.policy_model.forward(Tensor::<MyAutodiffBackend,1>::from( [3.,5.])) );
            }
            loss_string = loss.to_data().to_string();
        }
        loss_string
    }

    pub fn extract_policy(&mut self) {
        self.env.reset();
        self.config.epsilon = 0.0;

        while !self.action_record.is_empty() {
            //let action = self.propose_action();
            let action = self.action_record.pop().unwrap();
            self.env.step(action);
        //    println!("{}, {:?}", action, self.env.current_state.clone());
        }
    }

    pub fn extract_policy_zero_epsilon(&mut self) {
        self.env.reset();
        self.config.epsilon = 0.0;

        while !self.env.done() {
            let action = self.propose_action();
            //let action = self.action_record.pop().unwrap();
            self.env.step(action);
          //  println!("{}, {:?}", action, self.env.current_state.clone());
        }
    }
}
