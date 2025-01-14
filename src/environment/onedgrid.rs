use rand::Rng;

use crate::replay_buffer::Memory;

<<<<<<< Updated upstream
pub const STATE_SIZE: usize = 2;
=======
<<<<<<< HEAD
pub const STATE_SIZE: usize = 1;
=======
pub const STATE_SIZE: usize = 2;
>>>>>>> e583fbe140ff9b9e7509b3a2dda6ebfbb4eeb60f
>>>>>>> Stashed changes
pub const HIDDEN_SIZE: usize = 6;
pub const NUM_ACTIONS: usize = 2;
pub type State = [f64; STATE_SIZE];

<<<<<<< Updated upstream
pub const TARGET: State = [-2.01, 0.];
=======
<<<<<<< HEAD
pub const TARGET: State = [4.01];
=======
pub const TARGET: State = [-2.01, 0.];
>>>>>>> e583fbe140ff9b9e7509b3a2dda6ebfbb4eeb60f
>>>>>>> Stashed changes
pub struct Environment {
    pub current_state: State,
    current_steps: i32,
    maxsteps: i32,
    pub action_record: Vec<i32>,
}

impl Environment {
    pub fn step(&mut self, action: i32) -> Memory {
        self.current_steps += 1;

        let prev_state = self.current_state.clone();
        if action == 0 {
            self.current_state[0] += 1.0;
<<<<<<< Updated upstream
=======
<<<<<<< HEAD

            if self.current_state[0] >= 11. {
                self.current_state[0] = -10.
            }

            if self.current_state[0] <= -11. {
                self.current_state[0] = 10.
            }
            //self.current_state[0] = (self.current_state[0] as i32 % 11) as f64;
        }
        if action == 1 {
            self.current_state[0] -= 1.0;
            //self.current_state[0] = (self.current_state[0] as i32 % 11) as f64;
=======
>>>>>>> Stashed changes
            self.current_state[0] = (self.current_state[0] as i32 % 11) as f64;
        }
        if action == 1 {
            self.current_state[0] -= 1.0;

            self.current_state[0] = (self.current_state[0] as i32 % 11) as f64;
<<<<<<< Updated upstream
=======
>>>>>>> e583fbe140ff9b9e7509b3a2dda6ebfbb4eeb60f
>>>>>>> Stashed changes

        }

        self.action_record.push(action);
        Memory {
            current_state: prev_state,
            next_state: self.current_state,
            action,
            reward: self.reward(),
            done: self.done(),
        }
    }

    pub fn distance2(&self, state: &State) -> f64 {
<<<<<<< Updated upstream
        (state[0] - TARGET[0]).powi(2) + (state[1] - TARGET[1]).powi(2)
=======
<<<<<<< HEAD
        (state[0] - TARGET[0]).powi(2)
=======
        (state[0] - TARGET[0]).powi(2) + (state[1] - TARGET[1]).powi(2)
>>>>>>> e583fbe140ff9b9e7509b3a2dda6ebfbb4eeb60f
>>>>>>> Stashed changes
    }
    pub fn reward(&self) -> f64 {
       // let f = self.distance2(&self.current_state) / self.distance2(&[-3., 5.]);
        //f64::exp((1. - f) / (0.1 + f))
        if self.done() {
            1./self.distance2(&self.current_state)
        }
        else {
            0.
        }
    }

    pub fn reward_calc(&self, state: &State) -> f64 {
        // let f = self.distance2(&self.current_state) / self.distance2(&[-3., 5.]);
         //f64::exp((1. - f) / (0.1 + f))
         if (state[0] - TARGET[0]).abs() <= 0.5 {
             1./self.distance2(&state)
         }
         else {
             0.
         }
     }

    pub fn done(&self) -> bool {
<<<<<<< Updated upstream
        if self.current_steps >= self.maxsteps || self.distance2(&self.current_state) <= 0.01 {
=======
<<<<<<< HEAD
        if self.current_steps >= self.maxsteps || self.distance2(&self.current_state) <= 0.4 {
=======
        if self.current_steps >= self.maxsteps || self.distance2(&self.current_state) <= 0.01 {
>>>>>>> e583fbe140ff9b9e7509b3a2dda6ebfbb4eeb60f
>>>>>>> Stashed changes
            true
        } else {
            false
        }
    }

    pub fn new() -> Environment {
        Environment {
<<<<<<< Updated upstream
            current_state: [0.0, 0.0],
            current_steps: 0,
            maxsteps: 20,
=======
<<<<<<< HEAD
            current_state: [0.0],
            current_steps: 0,
            maxsteps: 40,
=======
            current_state: [0.0, 0.0],
            current_steps: 0,
            maxsteps: 20,
>>>>>>> e583fbe140ff9b9e7509b3a2dda6ebfbb4eeb60f
>>>>>>> Stashed changes
            action_record: Vec::new(),
        }
    }

    pub fn reset(&mut self) {

        let mut rng = rand::thread_rng();
    
        let min = -11; // Lower bound
        let max = 11; // Upper bound

        let random_number = rng.gen_range(min..=max);
<<<<<<< Updated upstream
        self.current_state = [random_number as f64, 0.0];
=======
<<<<<<< HEAD
        self.current_state = [0.];//[random_number as f64];
=======
        self.current_state = [random_number as f64, 0.0];
>>>>>>> e583fbe140ff9b9e7509b3a2dda6ebfbb4eeb60f
>>>>>>> Stashed changes
        self.current_steps = 0;
        self.action_record = Vec::new();
    }
}
