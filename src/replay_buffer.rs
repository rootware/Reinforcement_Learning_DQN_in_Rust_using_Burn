use rand::seq::SliceRandom; // For experience replay

use crate::utils::*;
#[derive(Clone)]
pub struct Memory {
    pub current_state : State,
    pub next_state : State,
    pub action : i32,
    pub reward : f64,
    pub done : bool,
}





pub struct ReplayBuffer {
    capacity: usize,
    buffer: Vec<Memory>, // (state, action, reward, next_state, done)
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            capacity,
            buffer: Vec::new(),
        }
    }

    pub fn add_component_wise(&mut self, state: State, next_state: State, action: i32, reward: f64, done: bool) {
        if self.buffer.len() == self.capacity {
            self.buffer.remove(0); // Remove oldest experience if full
        }
        self.buffer.push(Memory{current_state: state, next_state, action, reward, done});
    }
    
    pub fn add(&mut self, mem: Memory){
        self.buffer.push(mem);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Memory> {
        let mut rng = rand::thread_rng();
        self.buffer.choose_multiple(&mut rng, batch_size).cloned().collect()
    }
}