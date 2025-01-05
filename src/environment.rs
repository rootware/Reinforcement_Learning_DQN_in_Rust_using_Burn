use replay_buffer::{ Memory};


pub struct Environment{
    maxsteps : i32,
}

impl Environment {
    pub fn step(&mut self) -> Memory {
        Memory{ current_state: Vec::<f64>::new(), next_state: Vec::<f64>::new(), action: 1, reward: 1.0, done: false}
    }

    pub fn reward(&self) -> f64 {
        1.0
    }

    pub fn done(&self) -> bool {
        false
    }

    pub fn new() -> Environment {
        Environment{maxsteps: 10}
    }
}