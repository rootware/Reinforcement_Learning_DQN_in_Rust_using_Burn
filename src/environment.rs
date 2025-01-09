use crate::replay_buffer::Memory;
use crate::utils::*;

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
        }
        if action == 1 {
            self.current_state[1] += 1.0;
        }
        if action == 2 {
            self.current_state[0] -= 1.0;
        }
        if action == 3 {
            self.current_state[1] -= 1.0;
        }

        self.action_record.push(action);
        Memory {
            current_state: prev_state,
            // 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            next_state: self.current_state,
            //] 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] ,
            action,
            reward: self.reward(),
            done: self.done(),
        }
    }

    pub fn reward(&self) -> f64 {
        1.0 / (self.current_state[0] + self.current_state[1] - 24.000000001).abs()
    }

    pub fn done(&self) -> bool {
        if self.current_steps >= self.maxsteps
            || (self.current_state[0] + self.current_state[1] - 24.000000001).abs() <= 0.01
        {
            true
        } else {
            false
        }
    }

    pub fn new() -> Environment {
        Environment {
            current_state: [0.0, 0.0],
            current_steps: 0,
            maxsteps: 30,
            action_record: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.current_state = [0.0, 0.0];
        self.current_steps = 0;
        self.action_record = Vec::new();
    }
}
