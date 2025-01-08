use crate::replay_buffer::{ Memory};


pub struct Environment{
    current_steps: i32,
    maxsteps : i32,
}

impl Environment {
    pub fn step(&mut self, action: i32) -> Memory {
        self.current_steps+=1;
        Memory{ current_state: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], next_state:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] ,action: 1, reward: 1.0, done: false}
    }

    pub fn reward(&self) -> f64 {
        1.0
    }

    pub fn done(&self) -> bool {
        if self.current_steps >= self.maxsteps {
            true
        } else {
            false
        }
    }

    pub fn new() -> Environment {
        Environment{current_steps: 0,maxsteps: 10}
    }
}